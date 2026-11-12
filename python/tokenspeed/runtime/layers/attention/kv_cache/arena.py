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

"""The cache owner for ordinary allocations and mixed host/device planes."""

# 译：普通分配以及 Host/设备混合平面的 cache 所有者。

# CacheArena：一个模型唯一的 cache 显存所有者。
#
# 你在这里：启动时 layers/attention/registry.py 先做 cache 规划（第 14 阶段），再由 kv_cache/factory.py 的
# create_cache_arena 创建 arena，然后在其上创建 KV 池视图（kv_cache/base.py）。运行时它负责执行调度器下发的
# 页级 cache 操作：页清零（zero_blocks）、页拷贝（copy_blocks）。
#
# 它持有的东西：
#   regions：实际的字节缓冲区（普通情况下只有一块 GPU 上的 uint8 缓冲区 "arena"；KV 卸载时还有 Host 上的 pinned 区域）
#   plan：内存计划，描述每个字段在这些字节中的位置
#   _fields：每个字段的张量视图（按计划切出，见 _bind）
#   runtime_contract：发布给 C++ 调度器的契约：前缀粒度、LCM block 数、各 cache group 的 spec 和页数等，
#     调度器据此知道有多少页可分配
#
# LCM block（父块）：物理分配单位。多个 cache group 的 block 大小不同时，按最小公倍数把各组的若干个 block
# 打包进同一个父块。单个 cache group 的模型（Qwen3）里，一个父块就是“所有层的一个页”。
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from tokenspeed_kernel.ops.kvcache import zero_byte_ranges
from tokenspeed_kernel.ops.kvcache.triton import copy_page_fields, zero_page_fields

from tokenspeed.runtime.layers.attention.kv_cache.offload import SparseKVOffload
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheMemoryPlan
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.storage import (
    CacheStoragePlan,
    plan_cache_storage,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


# 页拷贝的安排：把 (源页, 目标页) 对分成若干“波”，每一波内的拷贝可以并行执行，且保证任何一页在被覆盖之前，
# 所有要读它的拷贝都已经完成。不同 cache group 可能共用同一个父块，所以按字节范围判断重叠，而不是按页号。
def _copy_waves(
    pairs_by_group: Mapping[str, np.ndarray],
    field_layouts: Mapping[str, Sequence[tuple[int, int, int]]],
) -> list[dict[str, np.ndarray]]:
    """Read each source before any group overwrites its field byte ranges.

    Groups can reuse the same LCM parent with different packing. Block ids
    alone cannot express that aliasing, while ordering whole parents would
    invent dependencies between disjoint child blocks. Use the plan's field
    ranges, which also retain their identity when storage is split into
    separate Host and Device regions.

    Raises:
        ValueError: The pairs copy blocks in a cycle.
    """
    # 译：保证在任何组覆盖某个源的字段字节范围之前先读取它。
    #
    #     不同的组可以以不同的打包方式复用同一个 LCM 父块。仅凭 block id 无法表达这种共用，而按整个父块排序又会在
    #     互不相交的子 block 之间凭空制造依赖。因此使用计划中的字段范围，它们在存储被拆分成独立的 Host 和设备区域时
    #     也保持自己的身份。
    #
    #     异常：
    #         ValueError：这些拷贝对形成了环。
    waves: list[dict[str, np.ndarray]] = []
    remaining = {group: pairs for group, pairs in pairs_by_group.items() if len(pairs)}
    sources: dict[str, np.ndarray] = {}
    destinations: dict[str, np.ndarray] = {}
    for group, pairs in remaining.items():
        fields = np.asarray(field_layouts[group], dtype=np.int64)
        ids = pairs.astype(np.int64, copy=False)
        source_start = fields[:, 0] + ids[:, :1] * fields[:, 1]
        destination_start = fields[:, 0] + ids[:, 1:] * fields[:, 1]
        sources[group] = np.stack((source_start, source_start + fields[:, 2]), axis=-1)
        destinations[group] = np.stack(
            (destination_start, destination_start + fields[:, 2]), axis=-1
        )
    while remaining:
        reads = np.concatenate([ranges.reshape(-1, 2) for ranges in sources.values()])
        order = np.argsort(reads[:, 0])
        starts = reads[order, 0]
        # For a destination [a, b), a source overlaps iff a source starting
        # before b ends after a. Prefix maxima answer all fields together
        # without constructing a quadratic matrix of copy dependencies.
        # 译：对一个目标范围 [a, b)，一个源与它重叠，当且仅当某个在 b 之前开始的源在 a 之后结束。前缀最大值可以一次回答
        #     所有字段的问题，而不必构造一个二次规模的拷贝依赖矩阵。
        past_ends = np.maximum.accumulate(reads[order, 1])
        wave: dict[str, np.ndarray] = {}
        for group, pairs in tuple(remaining.items()):
            writes = destinations[group]
            last = np.searchsorted(starts, writes[..., 1], side="left") - 1
            overlaps = (last >= 0) & (past_ends[np.maximum(last, 0)] > writes[..., 0])
            ready = ~overlaps.any(axis=1)
            if ready.any():
                wave[group] = pairs[ready]
            if ready.all():
                del remaining[group], sources[group], destinations[group]
            else:
                remaining[group] = pairs[~ready]
                sources[group] = sources[group][~ready]
                destinations[group] = writes[~ready]
        if not wave:
            remaining_pairs = {
                group: pairs.tolist() for group, pairs in remaining.items()
            }
            raise ValueError(
                f"block pairs copy in a cycle: {remaining_pairs}; "
                "no order reads every byte range before it is overwritten"
            )
        waves.append(wave)
    return waves


class CacheArena:
    """Own the cache allocation, its typed field views, and its contract.

    One model, one arena. Every ``CachePool`` is a compute view onto an
    arena and owns no memory of its own: target and draft views of one
    merged plan share this object, so "who allocates" and "who may clear"
    stop being per-pool modes and become properties of the single owner.

    The arena is the sole publisher of the scheduler-facing runtime
    contract, so a secondary view cannot republish or diverge from it.
    Publication is unconditional: an arena without group specs would be an
    arena no scheduler can address, and ModelExecutor rejects it anyway.
    """

    # 译：拥有 cache 分配、它的类型化字段视图以及它的契约。
    #
    #     一个模型，一个 arena。每个 ``CachePool`` 都是 arena 上的一个计算视图，自己不拥有显存：同一个合并计划的目标视图
    #     和草稿视图共享这个对象，所以“谁来分配”“谁可以清零”不再是每个池各自的模式，而是唯一所有者的属性。
    #
    #     arena 是面向调度器的运行时契约的唯一发布者，所以次要的视图无法重新发布或与之不一致。发布是无条件的：
    #     没有组 spec 的 arena 是调度器无法寻址的 arena，ModelExecutor 无论如何都会拒绝它。

    def __init__(
        self,
        plan: CacheMemoryPlan,
        device: str,
        *,
        cache_group_specs: tuple[CacheGroupSpec, ...],
        token_capacity: int | None = None,
        enable_memory_saver: bool = False,
        storage_plan: CacheStoragePlan | None = None,
    ):
        if not cache_group_specs:
            raise ValueError(
                "cache arena requires at least one cache group spec to publish"
            )
        self.plan = plan
        self.device = device
        # Materialize immutable byte geometry at setup, not on first hand-out.
        # zero_blocks and copy_blocks ship only page ids per call; each group's
        # field table (page-0 offset, page stride, payload bytes) lives on the
        # device. The kernels trust these ranges, so their bound is checked
        # once here.
        # A group may have no fields on this rank: pipeline parallelism keeps
        # every group but drops the fields of layers another stage owns.
        # 译：在初始化时实体化不可变的字节几何，而不是在第一次分发时。zero_blocks 和 copy_blocks 每次调用只传页 id；
        #     每个组的字段表（第 0 页的偏移、页步长、有效字节数）放在设备上。kernel 信任这些范围，所以它们的边界在这里检查一次。
        #     在本 rank 上一个组可能没有字段：流水线并行保留每个组，但丢弃其他 stage 拥有的层的字段。
        self._zero_field_tables: dict[str, tuple[int, torch.Tensor, int]] = {}
        for group in plan.groups:
            page_count, fields = plan.page_field_layout(group.group_id)
            for base, stride, size in fields:
                if base + (page_count - 1) * stride + size > plan.arena_bytes:
                    raise ValueError(
                        f"cache group {group.group_id!r} field geometry reaches "
                        f"past the {plan.arena_bytes}-byte arena"
                    )
            self._zero_field_tables[group.group_id] = (
                page_count,
                torch.tensor(fields, dtype=torch.int64, device=device).reshape(
                    len(fields), 3
                ),
                max((size for _, _, size in fields), default=0),
            )
        self._cache_group_specs_by_id = {
            spec.group_id: spec for spec in cache_group_specs
        }
        # One adapter for the whole arena. The allocation happens inside its
        # region, so the sleep/wake lifetime is a property of the arena.
        # Tag as "kv_cache", no CPU backup: cache bytes are discarded on
        # sleep and rebuilt after wake (paging overwrites; clear() zeros the
        # remapped pages).
        # 译：整个 arena 一个适配器。分配发生在它的区域内，所以 sleep/wake 的生命周期是 arena 的属性。
        #     标记为 "kv_cache"，不做 CPU 备份：sleep 时丢弃 cache 字节，wake 之后再重建（分页会覆盖；clear() 会把
        #     重新映射的页清零）。
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.storage_plan = storage_plan or plan_cache_storage(
            plan,
            cache_group_specs,
            offload=None,
        )
        expected_storage = plan_cache_storage(
            plan,
            cache_group_specs,
            offload=self.storage_plan.offload,
        )
        if self.storage_plan != expected_storage:
            raise ValueError("cache physical binding does not match its memory plan")
        self.offload_config = self.storage_plan.offload
        self.offload: SparseKVOffload | None = None
        self.regions: dict[str, torch.Tensor] = {}
        # Only device regions live in the memory saver's pool; pinned Host
        # regions (sparse offload history) are ordinary Host allocations and
        # stay mapped across sleep. Wake zeroes both (clear()).
        # 译：只有设备区域位于 memory saver 的池中；pinned 的 Host 区域（稀疏卸载的历史）是普通的 Host 分配，
        #     在 sleep 期间保持映射。wake 时两者都清零（clear()）。
        # 真正分配显存的地方：每个区域一个 torch.zeros 的 uint8 张量（Qwen3 只有一个 GPU 区域）。
        # 随后为计划中的每个字段切出视图（_bind），并在需要时创建 KV 卸载对象。
        with self.memory_saver_adapter.region(tag="kv_cache", enable_cpu_backup=False):
            for region in self.storage_plan.regions:
                host = region.placement == "host"
                self.regions[region.region_id] = torch.zeros(
                    region.size_bytes,
                    dtype=torch.uint8,
                    device="cpu" if host else device,
                    pin_memory=host,
                )
        self._field_bindings = {b.field_id: b for b in self.storage_plan.fields}
        self._fields: dict[str, torch.Tensor] = {
            field.field_id: self._bind(field) for field in plan.fields
        }
        if self.offload_config is not None:
            self.offload = SparseKVOffload(
                self, self.offload_config, workspaces=self.storage_plan.workspaces
            )
        expected_device = (
            self.storage_plan.device_history_bytes
            + self.storage_plan.persistent_workspace_bytes
        )
        if self.allocated_device_bytes != expected_device:
            raise RuntimeError(
                "cache allocation differs from its physical workspace plan"
            )
        plan_groups = {group.group_id: group for group in plan.groups}
        # The contract joins the recipe's logical specs with the plan's
        # physical facts for the same groups. The plan owns page counts and
        # packing; the contract carries them beside the specs rather than
        # copying them in, and the recipe packs the plan from the same
        # (spec, fields) pairs these specs come from, so both name one group
        # set by construction.
        # 译：契约把 recipe 的逻辑 spec 与计划中相同组的物理事实结合起来。计划拥有页数和打包方式；契约把它们放在 spec 旁边
        #     携带，而不是复制进去。recipe 根据这些 spec 来自的同一组 (spec, fields) 对来打包计划，所以两者按构造就指向
        #     同一个组集合。
        self.runtime_contract = CacheRuntimeContract(
            # The identity axis comes from the plan, never read back out of
            # view state. Per-group CacheBlock spans live in the group specs
            # as block_granularity.
            # 译：身份轴来自计划，从不从视图状态中读回。每个组的 CacheBlock 跨度作为 block_granularity 存放在组 spec 中。
            prefix_granularity=plan.prefix_granularity,
            num_lcm_blocks=plan.num_lcm_blocks,
            token_capacity=(
                token_capacity if token_capacity is not None else self.size
            ),
            group_specs=tuple(cache_group_specs),
            group_page_counts={
                spec.group_id: plan_groups[spec.group_id].page_count
                for spec in cache_group_specs
            },
            group_packing={
                spec.group_id: plan_groups[spec.group_id].cache_blocks_per_lcm_block
                for spec in cache_group_specs
            },
        )
        logger.info(
            f"Allocated cache arena: device_bytes={self.allocated_device_bytes:d}, "
            f"host_bytes={self.allocated_host_bytes:d}, logical_plan_bytes={plan.arena_bytes:d}, prefix_granularity="
            f"{plan.prefix_granularity:d}, num_lcm_blocks={plan.num_lcm_blocks:d}, "
            f"device {device!s}",
        )
        if self.offload_config is not None:
            logger.info(
                f"Sparse KV hot pool: rows={self.offload_config.device_rows}, "
                f"buffer_rows={self.offload_config.buffer_tokens}, "
                f"request_slots={self.offload_config.request_slots}, "
                f"persistent_bytes={self.storage_plan.persistent_workspace_bytes}, "
                f"temporary_budget_bytes={self.storage_plan.temporary_workspace_bytes}"
            )

    @property
    def cache_group_specs(self) -> tuple[CacheGroupSpec, ...]:
        """The published group specs. Stored once, in the contract."""
        # 译：已发布的组 spec。只在契约中存储一次。
        return self.runtime_contract.group_specs

    @property
    def cache_group_page_counts(self) -> Mapping[str, int]:
        """Per-group CacheBlock counts. Stored once, in the contract."""
        # 译：每个组的 CacheBlock 数。只在契约中存储一次。
        return self.runtime_contract.group_page_counts

    @property
    def prefix_granularity(self) -> int:
        """The identity grain, from the plan that defined this arena."""
        # 译：身份粒度，来自定义本 arena 的计划。
        return self.plan.prefix_granularity

    @property
    def kv_page_size(self) -> int:
        """KV arena page span for paged consumers.

        Pinned 1:1 to the identity grain -- this property is the single point
        of the prefix-page <-> KV-page convention, and geometry math must read
        it rather than ``prefix_granularity``.
        """
        # 译：分页使用方看到的 KV arena 页跨度。
        #
        #     与身份粒度 1:1 绑定——这个属性是“前缀页 <-> KV 页”约定的唯一所在，几何计算必须读它，而不是读 ``prefix_granularity``。
        # 也就是说：前缀缓存按多少 token 一段计算哈希，KV 页就跨多少 token（主线中即页大小 P）
        return self.plan.prefix_granularity

    @property
    def size(self) -> int:
        """Token slots the most finely packed group addresses.

        The plan's parent count times the tightest packing. One definition
        here, so compute views no longer re-derive it and disagree.
        """
        # 译：打包最紧密的组所寻址的 token 槽位数。
        #
        #     计划的父块数乘以最紧密的打包数。在这里只定义一次，这样各计算视图不再各自推导而得出不同的结果。
        max_packing = max(
            group.cache_blocks_per_lcm_block for group in self.plan.groups
        )
        return self.plan.num_lcm_blocks * max_packing * self.plan.prefix_granularity

    # 按计划切出一个字段的张量视图：
    #   1. 用 as_strided 在字节缓冲区上构造 [页数, *字段形状] 的视图：页与页之间相隔 page_stride_bytes，
    #      起点是字段在 arena 中的偏移
    #   2. 如果字段形状的第 0 维就是“一页的 token 数”（历史组中按 token 存放的字段，例如 MHA 的 K/V [P, Hkv, D]），
    #      就把页维折叠掉，得到按 token 槽位索引的 [页数 × P, Hkv, D]；否则（state 组等）保留按页索引的视图
    #   Qwen3-8B：layer.3.k 的视图为 [NP*P, 8, 128]
    def _bind(self, field) -> torch.Tensor:
        """Stride one planned field's view over the arena bytes.

        Pages are the physical unit, so the stride math is page-major; a
        per-token field then folds that axis into its entries (see
        :meth:`field` for why the planned shape decides).
        """
        # 译：在 arena 字节上为一个计划中的字段构造带步长的视图。
        #
        #     页是物理单位，所以步长计算以页为主序；一个按 token 存放的字段随后把该轴折叠进它的条目中（为什么由计划中的
        #     形状决定，见 :meth:`field`）。
        dtype = getattr(torch, field.dtype)
        group = self.plan.group(field.group_id)
        element_strides = []
        stride = 1
        for extent in reversed(field.shape):
            element_strides.append(stride)
            stride *= extent
        binding = self._field_bindings[field.field_id]
        buffer = self.regions[binding.region_id]
        offset = binding.offset_bytes
        pages = buffer.view(dtype).as_strided(
            (group.page_count, *field.shape),
            (
                field.page_stride_bytes // field.element_size,
                *reversed(element_strides),
            ),
            offset // field.element_size,
        )
        spec = self._cache_group_specs_by_id[field.group_id]
        # Keep ordinary fields' addressing contract. History groups can also
        # contain page-indexed compressor state whose shape starts with rows.
        # 译：保持普通字段的寻址约定。历史组中也可能包含形状以行开头的、按页索引的压缩器状态。
        token_rows = self.plan.prefix_granularity
        if (
            self.offload_config is not None
            and field.field_id in self.offload_config.field_ids
        ):
            token_rows = spec.rows_per_page
        if spec.attn_kind == "state" or field.shape[0] != token_rows:
            return pages
        return pages.view(-1, *field.shape[1:])

    def field(self, field_id: str) -> torch.Tensor:
        """Return one planned field's view into the arena.

        The view is shaped the way the field is addressed. A history field whose
        planned shape leads with the prefix grain stores an entry per token,
        so its page axis is folded away and consumers index entries directly.
        Offloaded fields use their group's physical row grain for this fold.
        Other fields (recurrent state, compressor state, block-scaled scale
        planes, V4's byte-shaped planes) retain page-indexed views.
        The plan decides which, so no compute view restates
        its kernel's geometry, and the dtype likewise comes from the plan.

        Args:
            field_id: A field id named by the memory plan.

        Returns:
            The field's view over its arena bytes.

        Raises:
            ValueError: The plan does not name this field.
        """
        # 译：返回计划中一个字段在 arena 中的视图。
        #
        #     视图的形状就是字段被访问的方式。规划形状以前缀粒度开头的历史字段为每个 token 存一个条目，所以它的页轴被折叠掉，
        #     使用方直接索引条目。卸载的字段使用其组的物理行粒度做这种折叠。其他字段（递归状态、压缩器状态、分块缩放的
        #     scale 平面、V4 的按字节成形的平面）保留按页索引的视图。由计划决定是哪一种，所以没有计算视图重复说明它的 kernel
        #     几何，dtype 同样来自计划。
        #
        #     参数：
        #         field_id：内存计划中命名的一个字段 id。
        #
        #     返回：
        #         该字段在其 arena 字节上的视图。
        #
        #     异常：
        #         ValueError：计划中没有这个字段。
        try:
            return self._fields[field_id]
        except KeyError:
            raise ValueError(f"cache field {field_id!r} is not planned") from None

    def field_address(self, field_id: str) -> int:
        """Address of physical block zero, including host placement."""
        # 译：物理 block 0 的地址，包括 Host 上的放置。
        return self._fields[field_id].data_ptr()

    @property
    def buffer(self) -> torch.Tensor:
        """Compatibility view for consumers requiring a single device slab."""
        # 译：为需要单一设备字节块的使用方提供的兼容视图。
        if not self.storage_plan.monolithic:
            raise ValueError("this cache uses physical regions, not a device slab")
        return self.regions["arena"]

    def registration_regions(self) -> tuple[tuple[int, int], ...]:
        return tuple((b.data_ptr(), b.nbytes) for b in self.regions.values())

    @property
    def allocated_device_bytes(self) -> int:
        history = sum(
            self.regions[r.region_id].nbytes
            for r in self.storage_plan.regions
            if r.placement == "device"
        )
        return history + (
            0
            if self.offload is None
            else sum(state.device_nbytes for state in self.offload.fields.values())
        )

    @property
    def allocated_host_bytes(self) -> int:
        return sum(
            self.regions[r.region_id].nbytes
            for r in self.storage_plan.regions
            if r.placement == "host"
        )

    # 计算用的字段视图：开启 KV 卸载时返回 GPU 上的计算缓冲区，否则就是 arena 中的字段本身
    def compute_field(self, field_id: str) -> torch.Tensor:
        if self.offload is not None and field_id in self.offload.fields:
            return self.offload.fields[field_id].device
        return self.field(field_id)

    def field_ids(self) -> frozenset[str]:
        """Every field the plan names, all of them materialized."""
        # 译：计划中命名的每个字段，它们都已经实体化。
        return frozenset(self._fields)

    def field_block_byte_offset(self, field_id: str, block_id: int) -> int:
        return self.plan.field_page_byte_offset(field_id, block_id)

    def _validate_block_ids(
        self, group_id: str, ids: np.ndarray, *, expected_ndim: int
    ) -> bool:
        """Check one group's local block ids; False when this rank holds no bytes."""
        # 译：检查一个组的本地 block id；当本 rank 不持有任何字节时返回 False。
        page_count, fields, _ = self._zero_field_tables[group_id]
        if (
            not isinstance(ids, np.ndarray)
            or ids.ndim != expected_ndim
            or ids.dtype.kind not in "iu"
        ):
            raise TypeError(
                f"block IDs for group {group_id!r} must be a {expected_ndim}-D "
                "integer array"
            )
        if ids.size == 0:
            return False
        low, high = int(ids.min()), int(ids.max())
        if low < 0 or high >= page_count:
            raise IndexError(
                f"local block ID {low if low < 0 else high} outside "
                f"[0, {page_count}) for group {group_id!r}"
            )
        # Validated, but this rank may hold none of the group's bytes.
        # 译：已校验，但本 rank 可能不持有该组的任何字节。
        return fields.shape[0] > 0

    def _upload_block_ids(
        self, arrays: Sequence[np.ndarray], *, columns: int
    ) -> tuple[torch.Tensor, list[int]]:
        """Stage every group's ids through one pinned copy, 16-byte aligned spans."""
        # 译：把每个组的 id 通过一次 pinned 拷贝上传，每段起点按 16 字节对齐。
        spans: list[int] = []
        total = 0
        for ids in arrays:
            spans.append(total)
            # Each group's span starts 16-byte aligned, as the kernels require
            # of the pointer they receive.
            # 译：每个组的那段起点按 16 字节对齐，这是 kernel 对它接收的指针的要求。
            total += -(-ids.shape[0] * columns // 4) * 4
        # The caching host allocator tracks the pinned block's pending copy,
        # so the next call cannot overwrite it early.
        # 译：缓存式的 Host 分配器会跟踪这块 pinned 内存上未完成的拷贝，所以下一次调用不会过早覆盖它。
        staging = torch.empty(total, dtype=torch.int32, pin_memory=True)
        host = staging.numpy()
        for start, ids in zip(spans, arrays, strict=True):
            host[start : start + ids.size] = ids.reshape(-1)
        return staging.to(self.buffer.device, non_blocking=True), spans

    # 页清零（调度器计划中的 cache 操作）：新分配给请求、又不会被整页覆盖的页，先清零，避免读到上一个使用者的数据。
    # 参数是“每个 cache group 的本地页号数组”；页号一次性上传到 GPU，由一个 kernel 完成“页 × 字段”的展开清零，
    # 避免长 prompt 一次分配数千页时在 Python 里逐页循环。
    def zero_blocks(self, block_ids_by_group: Mapping[str, np.ndarray]) -> None:
        """Clear local physical blocks after validating every group's IDs.

        Host work stays O(pages) with no per-page Python object: the ids are
        validated as one array, staged through one pinned copy, and each
        group's page x field expansion runs in the kernel. A long prompt's
        admission hands out thousands of pages, so per-range Python here
        would stall the forward thread.

        Args:
            block_ids_by_group: Local block IDs per group as 1-D integer
                arrays, each in [0, group.page_count).

        Raises:
            IndexError: A block ID is outside its group's physical range.
            TypeError: A group's IDs are not a 1-D integer array.
        """
        # 译：在校验每个组的 ID 之后，清零本地的物理 block。
        #
        #     Host 端的工作保持 O(页数)，不为每页创建 Python 对象：id 作为一个数组校验，通过一次 pinned 拷贝上传，
        #     每个组的“页 × 字段”展开在 kernel 中完成。一个长 prompt 的准入会分发数千个页，所以这里如果逐个范围用 Python
        #     处理，就会卡住 forward 线程。
        #
        #     参数：
        #         block_ids_by_group：每个组的本地 block ID，为一维整数数组，每个都在 [0, group.page_count) 内。
        #
        #     异常：
        #         IndexError：某个 block ID 超出了其组的物理范围。
        #         TypeError：某个组的 ID 不是一维整数数组。
        groups: list[str] = []
        arrays: list[np.ndarray] = []
        for group_id, ids in block_ids_by_group.items():
            if self._validate_block_ids(group_id, ids, expected_ndim=1):
                groups.append(group_id)
                arrays.append(ids)
        if not arrays:
            return
        if not self.storage_plan.monolithic:
            # Mixed regions use physical bindings; Host clears run on the
            # execution stream through the CuTeDSL kernel boundary.
            # 译：混合区域使用物理绑定；Host 上的清零通过 CuTeDSL kernel 边界在执行 stream 上运行。
            segments = [
                segment
                for group_id, ids in zip(groups, arrays, strict=True)
                for segment in self.block_byte_segments(group_id, ids.tolist())
            ]
            for region_id, ranges in self.storage_plan.split_ranges(segments).items():
                zero_byte_ranges(self.regions[region_id], ranges, device=self.device)
            return
        pages, spans = self._upload_block_ids(arrays, columns=1)
        for group_id, start, ids in zip(groups, spans, arrays, strict=True):
            _, fields, max_field_bytes = self._zero_field_tables[group_id]
            zero_page_fields(
                self.buffer,
                pages[start : start + ids.size],
                fields,
                max_field_bytes=max_field_bytes,
            )

    # 页拷贝（调度器计划中的 cache 操作）：把整页复制到同组的另一页。用于前缀缓存命中后复制边界所在的页
    # （命中的前缀最后一页只用了一部分，请求要在私有副本上继续写）等场景（第 14 阶段）。
    def copy_blocks(self, block_pairs_by_group: Mapping[str, np.ndarray]) -> None:
        """Copy local physical blocks whole onto other blocks of their group.

        One pinned upload of the ``(source, destination)`` pairs, one page x
        field expansion per group and wave in the kernel. Waves order all
        groups by their source and destination byte ranges: a published
        block evicted and re-granted to another group is read before the
        new owner's copy overwrites it. Pairs of one wave copy in parallel.

        Args:
            block_pairs_by_group: Local ``[n, 2]`` integer arrays per group of
                ``(source, destination)`` block IDs, each in
                [0, group.page_count).

        Raises:
            IndexError: A block ID is outside its group's physical range.
            TypeError: A group's pairs are not an ``[n, 2]`` integer array.
            ValueError: A pair copies a block onto itself, two pairs share a
                destination, or the byte ranges form a copy cycle.
        """
        # 译：把本地的物理 block 整块拷贝到同组的其他 block 上。
        #
        #     ``(源, 目标)`` 对通过一次 pinned 上传完成，每个组、每一波在 kernel 中做一次“页 × 字段”的展开。各波按源和目标的
        #     字节范围给所有组排序：一个被驱逐、又被重新分配给另一个组的已发布 block，会在新所有者的拷贝覆盖它之前被读取。
        #     同一波中的各对并行拷贝。
        #
        #     参数：
        #         block_pairs_by_group：每个组的本地 ``[n, 2]`` 整数数组，内容是 ``(源, 目标)`` block ID，每个都在
        #             [0, group.page_count) 内。
        #
        #     异常：
        #         IndexError：某个 block ID 超出了其组的物理范围。
        #         TypeError：某个组的对不是 ``[n, 2]`` 的整数数组。
        #         ValueError：某一对把一个 block 拷贝到它自己，两对共用一个目标，或者字节范围形成拷贝环。
        validated: dict[str, np.ndarray] = {}
        for group_id, pairs in block_pairs_by_group.items():
            if not self._validate_block_ids(group_id, pairs, expected_ndim=2):
                continue
            if pairs.shape[1] != 2:
                raise TypeError(
                    f"block pairs for group {group_id!r} must be an [n, 2] array"
                )
            destinations = pairs[:, 1]
            if (pairs[:, 0] == destinations).any():
                raise ValueError(
                    f"block pairs for group {group_id!r} copy a block onto itself"
                )
            if np.unique(destinations).size != destinations.size:
                raise ValueError(
                    f"block pairs for group {group_id!r} write one destination "
                    "twice; the copies of a wave run in parallel"
                )
            validated[group_id] = pairs
        if not validated:
            return
        waves = _copy_waves(
            validated,
            {group: self.plan.page_field_layout(group)[1] for group in validated},
        )
        groups: list[str] = []
        arrays: list[np.ndarray] = []
        for wave in waves:
            for group_id, pairs in wave.items():
                groups.append(group_id)
                arrays.append(pairs)
        if not self.storage_plan.monolithic:
            self._copy_block_byte_ranges(groups, arrays)
            return
        pairs, spans = self._upload_block_ids(arrays, columns=2)
        for group_id, start, ids in zip(groups, spans, arrays, strict=True):
            _, fields, max_field_bytes = self._zero_field_tables[group_id]
            copy_page_fields(
                self.buffer,
                pairs[start : start + ids.size].view(-1, 2),
                fields,
                max_field_bytes=max_field_bytes,
            )

    def _copy_block_byte_ranges(
        self, groups: list[str], waves: list[np.ndarray]
    ) -> None:
        """Copy blocks of a region-backed cache field by field, in wave order.

        A device region copies on the current stream. A Host region is pinned
        memory that GPU kernels write on that stream, so the stream drains
        once before the first host copy; the copies themselves are host
        memcpys that complete before the thread enqueues the next forward.
        Block copies are rare (a tail hit, an owner's move off a published
        block), so the per-field Python here is affordable.
        """
        # 译：按波的顺序，逐字段拷贝一个由区域支撑的 cache 字段的 block。
        #
        #     设备区域在当前 stream 上拷贝。Host 区域是 GPU kernel 在该 stream 上写入的 pinned 内存，所以在第一次 Host 拷贝之前
        #     先排空 stream 一次；拷贝本身是 Host 上的 memcpy，在线程把下一次前向入队之前完成。block 拷贝很少见（尾部命中、
        #     所有者从已发布 block 上移走），所以这里逐字段的 Python 处理是可以承受的。
        drained = False
        for group_id, pairs in zip(groups, waves, strict=True):
            for source, destination in pairs.tolist():
                source_ranges = self.storage_plan.split_ranges(
                    self.block_byte_segments(group_id, [source])
                )
                destination_ranges = self.storage_plan.split_ranges(
                    self.block_byte_segments(group_id, [destination])
                )
                for region_id, region in self.regions.items():
                    ranges = zip(
                        source_ranges[region_id],
                        destination_ranges[region_id],
                        strict=True,
                    )
                    if (
                        region.device.type == "cpu"
                        and not drained
                        and source_ranges[region_id]
                    ):
                        torch.cuda.current_stream(self.device).synchronize()
                        drained = True
                    for (src, size), (dst, _) in ranges:
                        region[dst : dst + size].copy_(region[src : src + size])

    def block_byte_segments(
        self, group_id: str, block_ids: list[int]
    ) -> list[tuple[int, int]]:
        return self.plan.block_byte_segments(group_id, block_ids)

    @property
    def supports_disaggregation(self) -> bool:
        """Whether the arena exposes complete inputs for cache transfer."""
        # 译：arena 是否为 cache 传输暴露了完整的输入。
        return all(spec.transfer_policy is not None for spec in self.cache_group_specs)

    def contract_binding(self) -> torch.Tensor:
        """Return the arena buffer the cache contract is bound to.

        Field dtypes are not returned: the plan carries them, and the PD
        contract reads them from there.
        """
        # 译：返回 cache 契约所绑定的 arena 缓冲区。
        #
        #     不返回字段的 dtype：计划携带了它们，PD 契约从那里读取。
        if not self.supports_disaggregation:
            raise RuntimeError(
                "cache arena has no transfer policy in its runtime contract"
            )
        return self.buffer

    def synchronize(self) -> None:
        """Wait for the arena's own copy streams (sparse KV offload).

        A sleep release calls this before the device regions are unmapped, so
        no prefetch or writeback still reads or writes them.
        """
        # 译：等待 arena 自己的拷贝 stream（稀疏 KV 卸载）。
        #
        #     sleep 释放在取消设备区域的映射之前调用它，确保没有预取或写回仍在读写它们。
        if self.offload is not None:
            self.offload.synchronize()

    @torch.no_grad()
    def clear(self) -> None:
        """Zero the arena after sleep/wake remaps its storage.

        Exactly one owner, so callers may fan this out over every compute
        view without double-clearing or needing a "views skip" rule. The
        sparse offload's hot-pool tags and replacement state are reset with
        it, and its Host history is zeroed like any other region.
        """
        # 译：在 sleep/wake 重新映射其存储之后把 arena 清零。
        #
        #     只有一个所有者，所以调用方可以在每个计算视图上都调用它，而不会重复清零，也不需要“视图跳过”的规则。
        #     稀疏卸载的热池标记和替换状态随之重置，它的 Host 历史像其他区域一样被清零。
        if self.offload is not None:
            self.offload.synchronize()
            self.offload.clear()
        for buffer in self.regions.values():
            zero_byte_ranges(buffer, [(0, buffer.nbytes)], device=self.device)
