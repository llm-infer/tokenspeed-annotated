# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# CacheBatchMetadata：一次前向的 block table（每个 cache group 一张），校验并上传到 GPU 后的封装。
#
# 你在这里：ModelExecutor.execute_forward_op 开头（第 4 阶段）调用 from_forward_op 构造它，
# 随后把 tables(...) 交给注意力后端（CacheGroupRouter）换算成 kernel 页表。
#
# 设计要点：
#   1. 只能通过工厂方法 from_forward_op 构造（直接构造会报错），保证每份表都经过校验
#   2. 持有生成它的那个 forward_op 的强引用，取表时检查“是不是当前这次前向的”，
#      防止 overlap 调度下误用上一轮或下一轮的旧表（stale metadata）
#   3. 有 extend 的批次额外保留一份 Host 副本（tables_cpu），后端规划 extend 时读它，不必等 GPU
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from tokenspeed.runtime.engine.scheduler_utils import (
    packed_block_tables_from_forward_op,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
    require_positive_int,
)


@dataclass(frozen=True, init=False)
class CacheBatchMetadata:
    """Factory-only views tied to one scheduler forward operation.

    Attributes:
        group_ids: Cache group IDs in runtime-contract order.
    """

    # 译：只能由工厂创建的视图，绑定到一次调度器前向操作。
    #
    #     属性：
    #         group_ids：按运行时契约顺序排列的 cache group ID。

    group_ids: tuple[str, ...]
    _group_tables: Mapping[str, torch.Tensor] = field(repr=False, compare=False)
    # None unless the factory was asked for host tables.
    # 译：除非工厂被要求生成 Host 表，否则为 None。
    _group_tables_cpu: Mapping[str, torch.Tensor] | None = field(
        repr=False, compare=False
    )
    _forward_op: Any = field(repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("CacheBatchMetadata is factory-only; use from_forward_op()")

    @classmethod
    def from_forward_op(
        cls,
        forward_op: Any,
        *,
        device: torch.device | str,
        contract: CacheRuntimeContract,
        num_requests: int,
        validate_page_ids: bool,
        host_tables: bool,
    ) -> CacheBatchMetadata:
        """Check the tables against the contract, pack once, and retain
        operation provenance.

        The group set, dtype, shape and row count are always checked. With
        ``validate_page_ids`` every page ID is also scanned against its
        group's virtual block count: tables built outside the scheduler
        (prefill-graph capture) always ask for it, scheduler-built ones under
        ``TOKENSPEED_VALIDATE_PAGE_IDS``. ``host_tables`` also keeps a host
        copy for ``tables_cpu``: a forward with extends plans from it, a pure
        decode never reads it.
        """
        # 译：对照契约检查这些表，打包一次，并保留生成它的操作。
        #
        #     组集合、dtype、形状和行数总是会检查。开启 ``validate_page_ids`` 时，还会对照每个组的虚拟 block 数扫描每个页 ID：
        #     在调度器之外构建的表（prefill graph 捕获）总是要求扫描，调度器构建的表在 ``TOKENSPEED_VALIDATE_PAGE_IDS`` 下扫描。
        #     ``host_tables`` 还会为 ``tables_cpu`` 保留一份 Host 副本：带 extend 的前向根据它做规划，纯 decode 从不读它。
        if forward_op is None:
            raise ValueError("forward_op must not be None")
        require_positive_int("num_reqs", num_requests)
        # 每个组的最大合法页号 = 该组虚拟 block 数 - 1（第 0 页是保留的空页，也计入总数）
        group_ids = tuple(spec.group_id for spec in contract.group_specs)
        counts = contract.virtual_block_counts
        max_page_ids = {
            group_id: require_positive_int(
                f"max page ID for {group_id!r}",
                counts[group_id] - 1,
            )
            for group_id in group_ids
        }
        if (
            not group_ids
            or any(
                not isinstance(group_id, str) or not group_id for group_id in group_ids
            )
            or len(group_ids) != len(set(group_ids))
        ):
            raise ValueError(
                "runtime contract must provide ordered nonempty unique group IDs"
            )
        tables = packed_block_tables_from_forward_op(
            forward_op,
            device,
            host_tables=host_tables,
            num_reqs=num_requests,
            # State groups pack last and end to end, so their rows pad in one launch.
            # 译：state 组排在最后，首尾相接，所以它们的行可以在一次启动中填充。
            expected_group_ids=tuple(
                spec.group_id
                for spec in sorted(
                    contract.group_specs, key=lambda spec: spec.attn_kind == "state"
                )
            ),
            # The packer scans every page ID against bounds it is given.
            # 译：打包器会对照给定的边界扫描每个页 ID。
            max_page_ids=max_page_ids if validate_page_ids else None,
        )
        return cls._from_validated_tables(
            group_ids=group_ids,
            group_tables={group_id: tables.tables[group_id] for group_id in group_ids},
            group_tables_cpu=(
                {group_id: tables.tables_cpu[group_id] for group_id in group_ids}
                if host_tables
                else None
            ),
            num_requests=num_requests,
            forward_op=forward_op,
        )

    @classmethod
    # 结构校验：每张表都是 int32、二维、行数等于请求数、宽度大于 0、位于同一设备，
    # 并且所有组的表共享同一块打包存储（router 一次启动填充所有组的前提）。
    def _from_validated_tables(
        cls,
        *,
        group_ids: tuple[str, ...],
        group_tables: Mapping[str, torch.Tensor],
        group_tables_cpu: Mapping[str, torch.Tensor] | None,
        num_requests: int,
        forward_op: Any,
    ) -> CacheBatchMetadata:
        if tuple(group_tables) != group_ids or (
            group_tables_cpu is not None and tuple(group_tables_cpu) != group_ids
        ):
            raise ValueError(
                "cache group table mapping must exactly match contract order"
            )
        table_device: torch.device | None = None
        ordered = dict(group_tables)
        for group_id, table in ordered.items():
            if not isinstance(table, torch.Tensor):
                raise ValueError(f"cache group {group_id!r} must be a tensor")
            if table.dtype != torch.int32:
                raise ValueError(f"cache group {group_id!r} must use int32")
            if table.ndim != 2 or table.shape[0] != num_requests:
                raise ValueError(f"cache group {group_id!r} has invalid shape")
            if table.shape[1] == 0:
                raise ValueError(f"cache group {group_id!r} has zero width")
            if table.device.type not in ("cpu", "cuda", "npu"):
                raise ValueError(
                    f"cache group {group_id!r} must be on CPU, CUDA, or NPU"
                )
            if table_device is None:
                table_device = table.device
            elif table.device != table_device:
                raise ValueError("cache group tables must use one CPU/CUDA/NPU device")
        nonempty = [table for table in ordered.values() if table.numel()]
        pointers = {table.untyped_storage().data_ptr() for table in nonempty}
        if len(pointers) != 1:
            raise ValueError("cache group tables must share packed storage")

        for group_id, table in (group_tables_cpu or {}).items():
            if not isinstance(table, torch.Tensor) or table.device.type != "cpu":
                raise ValueError(
                    f"CPU table for cache group {group_id!r} must be on the CPU"
                )
            if tuple(table.shape) != tuple(ordered[group_id].shape):
                raise ValueError(f"CPU and device tables for {group_id!r} disagree")

        metadata = object.__new__(cls)
        object.__setattr__(metadata, "group_ids", group_ids)
        object.__setattr__(metadata, "_group_tables", MappingProxyType(ordered))
        object.__setattr__(
            metadata,
            "_group_tables_cpu",
            (
                None
                if group_tables_cpu is None
                else MappingProxyType(dict(group_tables_cpu))
            ),
        )
        # A strong reference makes Python/nanobind object identity safe against
        # id reuse until all metadata views become unreachable.
        # 译：强引用让 Python/nanobind 对象的身份比较在所有元数据视图都不可达之前，不受 id 复用的影响。
        object.__setattr__(metadata, "_forward_op", forward_op)
        return metadata

    # 检查传入的前向操作就是生成本元数据的那一个（按对象身份比较）
    def _validate_active_forward_op(self, active_forward_op: Any) -> None:
        if active_forward_op is not self._forward_op:
            raise RuntimeError(
                "stale cache metadata does not match the active forward operation"
            )

    def tables(self, *, active_forward_op: Any) -> Mapping[str, torch.Tensor]:
        """Return every group's table view, in a read-only mapping, after
        freshness validation. The views are this step's own upload, with the
        state groups' tables end to end at its tail: the executor pads
        stopped requests' state rows in them in place."""
        # 译：在新鲜度校验之后，以只读映射的形式返回每个组的表视图。这些视图是本步自己的上传，state 组的表首尾相接地
        #     放在末尾：执行器在其中原地填充已停止请求的 state 行。
        self._validate_active_forward_op(active_forward_op)
        return self._group_tables

    def tables_cpu(self, *, active_forward_op: Any) -> Mapping[str, torch.Tensor]:
        """The same tables as CPU views of one host copy (pinned for a device)."""
        # 译：同样的表，作为一份 Host 副本（对设备而言是 pinned 内存）的 CPU 视图。
        self._validate_active_forward_op(active_forward_op)
        if self._group_tables_cpu is None:
            raise RuntimeError("this batch's metadata was built without host tables")
        return self._group_tables_cpu
