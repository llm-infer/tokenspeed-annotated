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

"""The cache-group router: the one place scheduler blocks become kernel pages.

``CacheGroupRouter`` is the runner-facing attention backend for every model
whose attention is paged KV. It holds one ``PagedAttentionBackend`` leaf per
attention (history) cache group and is the only object between the
scheduler bridge and the kernels that expands raw group tables:

* it learns each group's block granularity from the pool's published specs
  (``CacheGroupGeometry``) and each leaf's kernel page size, and expands the
  bridge's raw per-group block tables into kernel page tables — the single
  logical -> physical mapping point cache-concepts.md Principle 5 asks for;
* it derives every KV write location (extend spans, the decode / verify
  window, and the draft steps the drafters publish) from those same tables,
  so slot math has exactly one home;
* it dispatches a layer's forward to the leaf of ``layer.group_id`` with
  that group's write locations.

Leaves see ``page_table`` / ``seq_lens`` / ``out_cache_loc``. Indexers consume
resolved ``group_view`` results for cross-group indexing. Their verification
state is registered at startup on the outermost backend and committed through
its side-state hook, independently of these leaves. A single-group model is a
router with one leaf; there is no single-table special case.
"""

# 译：cache group 路由器：调度器 block 变成 kernel 页的唯一地点。
#
#     ``CacheGroupRouter`` 是所有注意力为分页 KV 的模型面向 runner 的注意力后端。它为每个注意力（历史）cache group
#     持有一个 ``PagedAttentionBackend`` 叶子后端，并且是调度器桥接层与 kernel 之间唯一展开原始组表的对象：
#
#     * 它从 pool 发布的规格（``CacheGroupGeometry``）得知每个组的 block 粒度，从每个叶子得知其 kernel 页大小，
#       并把桥接层的原始按组 block table 展开为 kernel 页表——这正是 cache-concepts.md 原则 5 要求的
#       唯一的逻辑 -> 物理映射点；
#     * 它从同样这些表推导出每一个 KV 写入位置（extend 段、decode / verify 窗口，以及草稿模型发布的草稿步），
#       所以槽位计算只有一个归宿；
#     * 它把一层的前向分派给 ``layer.group_id`` 对应的叶子，并附上该组的写入位置。
#
#     叶子看到的是 ``page_table`` / ``seq_lens`` / ``out_cache_loc``。索引器（indexer）消费已解析的 ``group_view``
#     结果来做跨组索引。它们的验证状态在启动时注册在最外层后端上，并通过其旁路状态（side-state）钩子提交，
#     与这些叶子无关。单组模型就是只有一个叶子的 router；没有单表的特殊情况。

# 本文件是第 3 阶段的核心：注意力后端的“总入口”。
#
# 你在这里：ModelExecutor 每次前向开始时调用 router 的 init_forward_metadata（extend / mixed）或
# refresh_decode_metadata（decode）准备元数据；随后每一层 PagedAttention（layers/paged_attention.py）
# 先由 prologue 按 router 给出的写入位置把 K/V 写进 KV 池，再调用 router.forward 做核心注意力。
#
# router 自己不算注意力，它做三件事：
#   1. 页表换算：调度器给的是按 cache group 的“block 号”表，叶子 kernel 要的是“kernel 页号”表。
#      一个 block 可以包含多个 kernel 页（ratio = block 粒度 / kernel 页大小），换算在 GroupTableStacks 里
#   2. 写入位置：所有 slot 计算都在这里（经 GroupTableStacks 调用 write_locations.py 的 Triton kernel）
#   3. 分派：按 layer.group_id 找到叶子后端（Qwen3 只有一个 group，叶子是 MHA 后端，见 paged/mha.py）
#
# 为什么需要“cache group”：有些模型不同层的 KV 形态不同（例如滑动窗口层与全注意力层、MLA 与索引器的 K），
# 它们按组分别分配、分别前缀匹配。每个组有自己的 block table。Qwen3-8B 只有一个组，router 只有一个叶子。
#
# 数据流（一次前向）：
#   调度器 block table {gid: [bs, num_blocks] int32}（CacheBatchMetadata 上传，见 backends/cache_metadata.py）
#       │ stacks.fill：block -> kernel 页，padding 行 / 列置 0（哑页）
#       ▼
#   stacks.tables [G, max_bs, max_pages] int32 ──► 每个叶子拿 [bs, max_num_pages] 视图建自己的元数据
#       │ extend_locations / compute_decode_locations（Triton）
#       ▼
#   写入位置：extend 段 [sum(新 token)]、decode 窗口 [bs * N] ──► prologue 写 KV；forward 交给叶子
#
# 贯穿例子（Qwen3-8B，单卡，P=64，一个 cache group，block 粒度 = kernel 页大小 = 64，ratio = 1）：
#   MIXED 批：A extend 300 个新 token（无前缀），B decode（本步写位置 1000），bs=2，num_extends=1，T=301。
#   A 的 block table [12, 3, 40, 7, 21] -> kernel 页表行 [12, 3, 40, 7, 21, 0, 0, ...]（多出的列补 0）
#   extend 段：A 的 300 个槽位，最后一个 = 21×64 + 43 = 1387
#   decode 窗口：B 的 1 个槽位 = B 的页表[15]×64 + 40（N=1，无投机解码）
#   prologue 拿到的 301 个槽位 = extend 段 300 个 + decode 行 1 个（顺序与 q/k/v 的 token 顺序一致）
#
# 文件中与草稿模型（Eagle / MTP / DFlash）、draft tree、KV offload、DSA 相关的部分属于第 11 阶段及以后，
# 第一次读可以跳过。
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.kvcache.triton import compact_window_rows

from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    reject_bounded_replay,
)
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
    CacheGroupGeometry,
    learn_cache_group_geometry,
)
from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)
from tokenspeed.runtime.layers.attention.backends.support import TreeSupport

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan
    from tokenspeed.runtime.layers.attention.backends.paged.offload_adapter import (
        KVOffloadAdapter,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
        TreeVerifyInputs,
    )
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        HistoryGatherWorkspace,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

_PREWRITTEN = "paged attention KV is written by the prologue before core attention"


@dataclass(frozen=True)
class DraftHistoryView:
    """The full-history table as the draft chain's write-location source.

    Attributes:
        table: ``[max_bs, stack_max_num_pages]`` contiguous kernel-page table
            (one group's table in the router's stack; address-stable for the
            graph's lifetime).
        page_size: Tokens per kernel page.
        max_tokens: Per-request token capacity of the group's own columns.
    """

    # 译：作为草稿链写入位置来源的全历史（full-history）表。
    #
    #     属性：
    #         table：``[max_bs, stack_max_num_pages]`` 连续的 kernel 页表
    #             （router 堆叠中某一个组的表；在 graph 生命周期内地址稳定）。
    #         page_size：每个 kernel 页的 token 数。
    #         max_tokens：该组自身各列所对应的每请求 token 容量。

    table: torch.Tensor
    page_size: int
    max_tokens: int


@dataclass
class RouterDecodeWriteLocations:
    """Per-group decode write-slot views published for one padded ``bs``.

    Pointer-stable views of the router's location stack; the captured graph
    records them through the leaves' KV writes, so the pointer guard walks
    this object (``decode_write_locations`` slot).
    """

    # 译：为某个 padding 后的 ``bs`` 发布的按组 decode 写入槽位视图。
    #
    #     是 router 位置堆叠上指针稳定的视图；捕获的 graph 通过叶子的 KV 写入记录了它们，
    #     所以指针守卫（pointer guard）会遍历这个对象（``decode_write_locations`` 槽）。

    tokens_per_req: int
    by_group: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PagedGroupView:
    """Resolved pages for a backend runtime consuming one cache group.

    The table aliases the router's persistent stack, with its kernel page size.
    """

    # 译：为消费某个 cache group 的后端运行时解析好的页。
    #
    #     该表是 router 持久堆叠的别名（同一块存储），并附带其 kernel 页大小。

    page_table: torch.Tensor
    kernel_page_size: int


# 生命周期：
#   启动：set_cache_pool -> _publish_cache_pool：按 pool 的 cache group 规格用 leaf_factory 建叶子
#         init_cuda_graph_state：分配 GroupTableStacks（按最大 decode 批大小，一次分配、原地复用）
#   每次前向（二选一）：
#         extend / mixed / idle -> init_forward_metadata
#         decode               -> refresh_decode_metadata（eager 与 CUDA Graph 回放共用这一条路径）
#   每一层：prologue 取 padded_write_locations 写 KV -> forward -> 叶子的 forward_extend / forward_decode
class CacheGroupRouter(AttentionBackend):
    """One paged leaf per attention cache group; see the module docstring."""

    # 译：每个注意力 cache group 一个分页叶子；见模块 docstring。

    def __init__(
        self,
        leaf_factory: Callable[[str, int], PagedAttentionBackend],
        *,
        is_draft: bool,
        spec_num_tokens: int,
        device,
        consumed_group_ids: tuple[str, ...] | None,
        query_sharded: bool = False,
    ) -> None:
        """Args:
        leaf_factory: ``(group_id, block_granularity) -> leaf``; called once
            per paged group of the bound pool's view at :meth:`set_cache_pool`
            (the pool is the only source of the group set and geometry).
        is_draft: Whether this is the draft side (selects the draft-hook
            behavior; write locations are router-owned on both sides).
        spec_num_tokens: Verify width ``N`` (1 without speculation); the
            target's decode write window and the location stack's
            per-request capacity.
        device: Buffer device.
        consumed_group_ids: Groups served by attention leaves, or None to
            serve every local paged group. Other groups go to peer consumers.
        query_sharded: Whether this side's config carries a query-context
            shard; False without ``--query-context-parallel-size``.
        """
        # 译：参数：
        #     leaf_factory：``(group_id, block_granularity) -> 叶子``；在 :meth:`set_cache_pool` 时
        #         对所绑定 pool 视图中的每个分页组调用一次（pool 是组集合与几何信息的唯一来源）。
        #     is_draft：是否为草稿侧（选择草稿钩子的行为；两侧的写入位置都由 router 持有）。
        #     spec_num_tokens：验证宽度 ``N``（无投机解码时为 1）；即目标模型的 decode 写入窗口，
        #         也是位置堆叠的每请求容量。
        #     device：缓冲区所在设备。
        #     consumed_group_ids：由注意力叶子服务的组；为 None 时服务所有本地分页组。其他组交给对等的消费者。
        #     query_sharded：本侧配置是否带有 query-context 分片；没有 ``--query-context-parallel-size`` 时为 False。
        self._leaf_factory = leaf_factory
        self._consumed_group_ids = consumed_group_ids
        self.leaves: dict[str, PagedAttentionBackend] = {}
        self._geometry: CacheGroupGeometry | None = None
        self._offload_adapter: KVOffloadAdapter | None = None
        self.is_draft = bool(is_draft)
        self.spec_num_tokens = max(int(spec_num_tokens or 1), 1)
        self.device = device
        self._query_sharded = bool(query_sharded)
        # Draft-tree verify: compaction's K/V address table, rebuilt on every pool bind.
        # 译：draft tree 验证：压缩（compaction）用的 K/V 地址表，每次绑定 pool 时重建。
        self._tree_verify_armed = False
        self._tree_window_rows: dict[str, tuple[torch.Tensor, int]] = {}
        self._init_pool_binding()
        self._forget_bound_pool_state()

    def bind(
        self,
        geometry: CacheGroupGeometry,
        leaves: Mapping[str, PagedAttentionBackend],
    ) -> None:
        """Install the pool-derived geometry and one leaf per paged group.

        ``set_cache_pool`` is the production caller; unit fixtures bind
        hand-built leaves directly.
        """
        # 译：安装由 pool 推导出的几何信息，并为每个分页组安装一个叶子。
        #
        #     生产环境的调用方是 ``set_cache_pool``；单元测试夹具直接绑定手工构造的叶子。
        self._check_routable(geometry, tuple(leaves))
        self._geometry = geometry
        self.leaves = dict(sorted(leaves.items()))
        self._forget_bound_pool_state()

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    # 译：结构

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(self.leaves)

    @property
    def query_sharded(self) -> bool:
        """Whether this side computes a query-context shard of the extend span.

        Taken from this side's attention config by :func:`create_paged_router`.
        A draft planning its own attention family is configured ``qcp_size`` 1
        even under a sharded target (``model_wide_kwargs``'s ``draft_unsharded``).
        """
        # 译：本侧是否计算 extend 段的 query-context 分片。
        #
        #     由 :func:`create_paged_router` 从本侧的注意力配置中取得。即使目标模型是分片的，自己规划注意力家族的草稿模型
        #     也被配置为 ``qcp_size`` 1（``model_wide_kwargs`` 的 ``draft_unsharded``）。
        return self._query_sharded

    def child_backends(self) -> tuple[PagedAttentionBackend, ...]:
        return tuple(self.leaves.values())

    def _forget_bound_pool_state(self) -> None:
        """The table stacks and write locations built for a bound pool; init rebuilds them."""
        # 译：为所绑定 pool 构建的表堆叠和写入位置；初始化时会重建它们。
        self._offload_adapter = None
        self._stacks: GroupTableStacks | None = None
        self._decode_views: dict[tuple[int, int], RouterDecodeWriteLocations] = {}
        # Published write locations: the decode slot (graph-recorded views,
        # refreshed in place) and the extend span (the stack's buffer, or fresh past it).
        # 译：已发布的写入位置：decode 槽（graph 记录的视图，原地刷新）和 extend 段
        #     （堆叠的缓冲区，超出时则是新分配的张量）。
        self.decode_write_locations: RouterDecodeWriteLocations | None = None
        self._extend_write_locations: dict[str, torch.Tensor] | None = None
        self._decode_request_offset = 0
        # Groups whose pages are dealt over KVP owners (configure_runtime).
        # 译：页面分发到多个 KVP 所有者上的组（configure_runtime 中设置）。
        self._page_sharded_groups: frozenset[str] = frozenset()

    def _check_routable(
        self, geometry: CacheGroupGeometry, group_ids: tuple[str, ...]
    ) -> None:
        if not group_ids:
            raise ValueError(
                f"{type(self).__name__}: the bound pool view has no paged "
                "(history) cache groups to route"
            )
        for gid in group_ids:
            if geometry.attn_kinds[gid] == "state":
                raise ValueError(
                    f"cache group {gid!r} is a state group; the router serves "
                    "paged history groups only"
                )

    def _derive(
        self, cache_pool: CachePool
    ) -> tuple[CacheGroupGeometry, tuple[str, ...]]:
        """The geometry and paged group ids ``cache_pool`` publishes."""
        # 译：``cache_pool`` 发布的几何信息和分页组 id。
        geometry = learn_cache_group_geometry(cache_pool.arena.cache_group_specs)
        group_ids = cache_pool.paged_group_ids
        if self._consumed_group_ids is not None:
            group_ids = tuple(
                gid for gid in group_ids if gid in self._consumed_group_ids
            )
        return geometry, tuple(sorted(group_ids))

    def _learn(
        self, cache_pool: CachePool
    ) -> tuple[CacheGroupGeometry, tuple[str, ...]]:
        """The geometry and paged group ids of ``cache_pool``, or raise on a change."""
        # 译：``cache_pool`` 的几何信息和分页组 id；若有变化则报错。
        geometry, group_ids = self._derive(cache_pool)
        self._check_routable(geometry, group_ids)
        if self._geometry is not None and (
            geometry != self._geometry or group_ids != tuple(self.leaves)
        ):
            raise RuntimeError(
                "CacheGroupRouter cannot rebind a pool of a different geometry"
            )
        return geometry, group_ids

    def validate_cache_pool(self, cache_pool: CachePool) -> None:
        super().validate_cache_pool(cache_pool)
        self._learn(cache_pool)

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        """A first bind builds and binds the leaves; every bind drops the tables."""
        # 译：首次绑定时构建并绑定叶子；每次绑定都丢弃旧表。
        geometry, group_ids = self._derive(cache_pool)
        if self._geometry is not None:
            self._geometry = geometry
            self._forget_bound_pool_state()
        else:
            self.bind(
                geometry,
                {
                    gid: self._leaf_factory(gid, geometry.granularity_of(gid))
                    for gid in group_ids
                },
            )
            for leaf in self.leaves.values():
                leaf.set_cache_pool(cache_pool)
        super()._publish_cache_pool(cache_pool)
        if self._tree_verify_armed:
            self._bind_tree_window_rows()

        if cache_pool.arena.offload is not None and not self.is_draft:
            from tokenspeed.runtime.layers.attention.backends.paged.offload_adapter import (
                KVOffloadAdapter,
            )

            self._offload_adapter = KVOffloadAdapter(self, cache_pool)

    def configure_runtime(self, **kwargs) -> None:
        if self._offload_adapter is not None:
            self._offload_adapter.validate_slots(kwargs["request_slot_capacity"])
        specs = {
            spec.group_id: spec for spec in self.cache_pool.arena.cache_group_specs
        }
        self._page_sharded_groups = frozenset(
            group_id for group_id in self.leaves if specs[group_id].shard_count > 1
        )
        for group_id, leaf in self.leaves.items():
            leaf.configure_runtime(
                block_granularity=self.geometry.granularity_of(group_id),
                shard_count=specs[group_id].shard_count,
                virtual_block_count=self.cache_pool.arena.runtime_contract.virtual_block_counts[
                    group_id
                ],
                **kwargs,
            )

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        for leaf in self.leaves.values():
            leaf.init_prefill_graph_state(max_num_tokens, max_bs)

    def preallocate_history_gather_workspace(self, max_model_len: int) -> int:
        return sum(
            leaf.preallocate_history_gather_workspace(max_model_len)
            for leaf in self.leaves.values()
        )

    def history_gather_workspace(self) -> HistoryGatherWorkspace | None:
        workspaces = [
            workspace
            for leaf in self.leaves.values()
            if (workspace := leaf.history_gather_workspace()) is not None
        ]
        if len(workspaces) > 1:
            raise RuntimeError(
                "CacheGroupRouter holds more than one history gather workspace"
            )
        return workspaces[0] if workspaces else None

    def adopt_history_gather_workspace(self, workspace: HistoryGatherWorkspace) -> None:
        for leaf in self.leaves.values():
            leaf.adopt_history_gather_workspace(workspace)

    def register_step_counter(self, step_counter) -> None:
        # The MLA leaves record the PD layerwise step inside their chunked
        # prefill themselves; everything else records at the router's forward.
        # 译：MLA 叶子在自己的分块 prefill 内部记录 PD 逐层（layerwise）步；
        #     其他情况都在 router 的 forward 处记录。
        self.step_counter = step_counter
        for leaf in self.leaves.values():
            leaf.step_counter = step_counter

    def _leaf_for(self, layer: PagedAttention) -> PagedAttentionBackend:
        try:
            return self.leaves[layer.group_id]
        except KeyError:
            raise KeyError(
                f"layer {getattr(layer, 'layer_id', '?')} names cache group "
                f"{layer.group_id!r}; this router serves {self.group_ids}"
            ) from None

    def _sole_leaf(self, what: str) -> PagedAttentionBackend:
        if len(self.leaves) != 1:
            raise RuntimeError(
                f"{what} assumes a single attention cache group; this router "
                f"serves {self.group_ids}"
            )
        return next(iter(self.leaves.values()))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    # 译：生命周期

    @property
    def geometry(self) -> CacheGroupGeometry:
        if self._geometry is None:
            raise RuntimeError(
                f"{type(self).__name__} is not bound to a cache pool yet"
            )
        return self._geometry

    def _table_specs(self) -> list[GroupTableSpec]:
        return [
            GroupTableSpec(
                group_id=gid,
                block_granularity=self.geometry.granularity_of(gid),
                kernel_page_size=leaf.kernel_page_size,
                max_num_pages=leaf.max_num_pages,
            )
            for gid, leaf in self.leaves.items()
        ]

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        """Allocate the group stacks and every leaf's persistent buffers.

        Sized by the max decode batch (never the capture ladder); runs
        unconditionally at wrapper construction so eager decode refreshes the
        same buffers a graph would.
        """
        # 译：分配组堆叠以及每个叶子的持久缓冲区。
        #
        #     按最大 decode 批大小确定尺寸（从不按捕获阶梯）；在包装器构造时无条件运行，
        #     这样 eager decode 刷新的就是 graph 会刷新的同一批缓冲区。
        # Qwen3-8B 例子：max_bs=256、上下文 40960 -> max_num_pages = 640，
        # tables 为 [1, 256, 640] int32（640 KiB），decode_locs 为 [1, 256 * N] int32。
        # 注意：即使不开 CUDA Graph 也会调用；eager 与 graph 用同一套持久缓冲区，这是“统一 decode 路径”的前提。
        self._stacks = GroupTableStacks(
            self._table_specs(),
            max_bs=max_bs,
            max_tokens_per_req=self.spec_num_tokens,
            max_extend_tokens=int(kwargs.get("max_extend_tokens", 0)),
            device=self.device,
        )
        self._decode_views = {}
        for leaf in self.leaves.values():
            leaf.init_cuda_graph_state(max_bs)

    @property
    def stacks(self) -> GroupTableStacks:
        if self._stacks is None:
            raise RuntimeError(
                "CacheGroupRouter.init_cuda_graph_state must run before any "
                "metadata call"
            )
        return self._stacks

    def group_view(self, group_id: str, bs: int) -> PagedGroupView:
        """Return a group's resolved kernel pages for the current batch.

        Args:
            group_id: A paged group published by the bound cache pool.
            bs: Number of batch rows, including graph padding.

        Returns:
            A view over the router's table with its kernel page size.
            The router remains the only geometry owner.
        """
        # 译：返回某个组在当前批次下已解析的 kernel 页。
        #
        #     参数：
        #         group_id：所绑定 cache pool 发布的某个分页组。
        #         bs：批次行数，包含 graph padding。
        #
        #     返回：
        #         router 表上的一个视图，附带其 kernel 页大小。
        #         router 仍是唯一的几何信息持有者。
        page_size = self.stacks.group_kernel_page_size(group_id)
        return PagedGroupView(
            page_table=self.stacks.table(group_id, bs),
            kernel_page_size=page_size,
        )

    # ------------------------------------------------------------------
    # Write locations
    # ------------------------------------------------------------------
    # 译：写入位置

    @property
    def _decode_tokens_per_req(self) -> int:
        # The refresh-time window: the target's verify width, and the draft
        # step-0 window (the verify-shaped round writes the same N slots).
        # Draft steps 1+ republish their own windows through the draft hooks.
        # 译：刷新时的窗口：目标模型的验证宽度，以及草稿第 0 步的窗口（形状同验证的那一轮写入相同的 N 个槽位）。
        #     草稿第 1 步及以后通过草稿钩子重新发布各自的窗口。
        return self.spec_num_tokens

    # 按 (bs, N) 缓存视图对象：同一个 padding 后的 bs 每次拿到的是同一块存储的同一个视图，
    # CUDA Graph 捕获时记录的地址在回放时仍然有效。
    def _publish_decode_locations(self, bs: int, tokens_per_req: int) -> None:
        key = (bs, tokens_per_req)
        views = self._decode_views.get(key)
        if views is None:
            views = RouterDecodeWriteLocations(
                tokens_per_req=tokens_per_req,
                by_group={
                    gid: self.stacks.decode_locations(gid, bs, tokens_per_req)
                    for gid in self.leaves
                },
            )
            self._decode_views[key] = views
        self.decode_write_locations = views

    # decode 写入位置：每个请求写末尾 N 个位置，slot = 页表[pos // P] × P + pos % P。
    # 例：B 的 seq_len=1001（含本步新 token），N=1 -> 位置 1000 -> 页序号 15、偏移 40。
    def _refresh_decode_locations(self, bs: int, seq_lens: torch.Tensor) -> None:
        n = self._decode_tokens_per_req
        self.stacks.compute_decode_locations(bs, seq_lens, n)
        self._publish_decode_locations(bs, n)

    def decode_window_locations(self) -> torch.Tensor:
        """The full-history group's current decode write window view
        (``[decode_bs * tokens_per_req]``, address-stable; a MIXED round's
        extend requests are skipped). A block drafter writes its KV through the
        window of ITS OWN router: the target's full-history group when the
        draft's fields are the target's shape (an MLA depth), a group of its own
        when its family declares another (a DFlash GQA stack). The page-id
        spaces differ, so the side asking must be the side writing."""
        # 译：全历史组当前的 decode 写入窗口视图（``[decode_bs * tokens_per_req]``，地址稳定；跳过 MIXED 轮中的
        #     extend 请求）。块草稿模型通过它“自己的” router 的窗口写 KV：当草稿的字段与目标模型同形状时（一个 MLA 深度），
        #     用的是目标模型的全历史组；当它的家族声明了别的形状时（DFlash 的 GQA 堆叠），用它自己的组。
        #     两者的页 id 空间不同，所以发问的一侧必须就是写入的一侧。
        return self._decode_window_of(self.group_ids[self._draft_history_index()])

    def _decode_window_of(self, gid: str) -> torch.Tensor:
        """Group ``gid``'s current decode write window view (see ``decode_window_locations``)."""
        # 译：组 ``gid`` 当前的 decode 写入窗口视图（见 ``decode_window_locations``）。
        published = self.decode_write_locations
        if published is None:
            raise RuntimeError("decode window requested before any decode refresh")
        locs = published.by_group[gid]
        if self._decode_request_offset:
            locs = locs[self._decode_request_offset * published.tokens_per_req :]
        return locs

    def extend_span_locations(self) -> torch.Tensor:
        """The full-history group's extend write span of the current round
        (``[sum(extend_seq_lens)]``, request-major)."""
        # 译：全历史组在当前这一轮的 extend 写入段（``[sum(extend_seq_lens)]``，以请求为主序）。
        if self._extend_write_locations is None:
            raise RuntimeError("extend spans requested before init_forward_metadata")
        gid = self.group_ids[self._draft_history_index()]
        return self._extend_write_locations[gid]

    def tree_support(self) -> TreeSupport:
        unsupported = []
        for gid in self.group_ids:
            attn_kind = self.geometry.attn_kinds[gid]
            _, entry_stride_tokens = self.geometry.row_geometry[gid]
            if attn_kind != "full_history" or entry_stride_tokens not in (None, 1):
                unsupported.append(gid)
        if not unsupported:
            return TreeSupport(verify_blocker=None, draft_blocker=None)
        blocker = (
            "draft trees need full-history cache groups of one row per token; "
            f"{', '.join(unsupported)} slide or pack several tokens per row"
        )
        return TreeSupport(verify_blocker=blocker, draft_blocker=blocker)

    def bind_tree_verify(self, inputs: TreeVerifyInputs) -> None:
        super().bind_tree_verify(inputs)
        self._tree_verify_armed = True
        if self.cache_pool is not None:
            self._bind_tree_window_rows()

    def _bind_tree_window_rows(self) -> None:
        """Per group, the address table of the K/V token-row buffers compaction moves rows in."""
        # 译：按组，压缩在其中移动行的 K/V token 行缓冲区的地址表。
        pool = self.cache_pool
        buffers_by_group: dict[str, list[torch.Tensor]] = {}
        for layer, gid in sorted(pool.history_group_by_layer().items()):
            if gid in self.leaves:
                buffers_by_group.setdefault(gid, []).extend(pool.get_kv_buffer(layer))
        self._tree_window_rows = {}
        for gid, buffers in buffers_by_group.items():
            # MLA's value is a view of its latent key row at the same address; the key row covers it.
            # 译：MLA 的 value 是其潜在（latent）key 行在同一地址上的视图；key 行已覆盖它。
            widest: dict[int, torch.Tensor] = {}
            for buf in buffers:
                kept = widest.get(buf.data_ptr())
                if kept is None or buf[0].numel() > kept[0].numel():
                    widest[buf.data_ptr()] = buf
            buffers = list(widest.values())
            row_bytes = {buf[0].numel() * buf.element_size() for buf in buffers}
            contiguous = all(buf.is_contiguous() for buf in buffers)
            if len(row_bytes) != 1 or not contiguous:
                raise NotImplementedError(
                    "draft-tree compaction needs contiguous K/V token rows of one "
                    f"width per cache group; {gid} has widths {sorted(row_bytes)}, "
                    f"contiguous={contiguous}"
                )
            # Layers may alias one region through the memory plan; move each region once.
            # 译：多个层可能通过内存规划别名到同一区域；每个区域只移动一次。
            addresses = torch.tensor(
                sorted(widest),
                dtype=torch.int64,
                device=self.device,
            )
            self._tree_window_rows[gid] = (addresses, row_bytes.pop())
        if not self._tree_window_rows:
            raise NotImplementedError(
                f"draft-tree compaction found no K/V rows in cache groups {self.group_ids}"
            )

    def compact_verify_window(self, path: torch.Tensor) -> None:
        for gid, (addresses, row_bytes) in self._tree_window_rows.items():
            compact_window_rows(
                addresses, self._decode_window_of(gid), path, row_bytes=row_bytes
            )

    def prepare_cache_batch(self, request_slots, *, num_extends, stream):
        """Begin target KV offloading residency; ordinary arenas do nothing."""
        # 译：开始目标模型的 KV offload 驻留；普通 arena 什么都不做。
        if self._offload_adapter is not None:
            self._offload_adapter.begin(
                request_slots, num_extends=num_extends, stream=stream
            )

    def writeback_accepted_kv(self, accept_lengths):
        """Write accepted offloaded KV to Host and join before completion."""
        # 译：把已接受的 offload KV 写回 Host，并在完成前等待其结束。
        if self._offload_adapter is not None:
            self._offload_adapter.engine.commit(accept_lengths)

    def invalidate_cache_residency(self, request_slots, *, stream):
        """Fence offloading streams and discard recycled slots' hot tags."""
        # 译：对 offload 流做栅栏同步，并丢弃被回收槽位的热标签。
        if self._offload_adapter is not None:
            self._offload_adapter.engine.reset_requests(request_slots, stream=stream)

    def prepare_sparse_kv_access(self, layer, selection, positions, *, forward_mode):
        """Return offloading compute writes after loading or joining prefetch.

        Offloaded fields accept decode only. Other fields keep their
        ordinary history addresses.
        """
        # 译：在加载或等待预取完成后，返回 offload 计算用的写入位置。
        #
        #     offload 的字段只接受 decode。其他字段保持其普通的历史地址。
        if self._offload_adapter is not None:
            return self._offload_adapter.prepare(
                layer, selection, positions, forward_mode
            )
        return self.write_locations(layer, forward_mode)

    def prefetch_sparse_kv(self, layer, selection, positions):
        """Prefetch offloaded consumer fields sharing this layer's selection.

        This producer hook does not prepare the consumer's compute access.
        """
        # 译：为共享本层选择结果的 offload 消费者字段做预取。
        #
        #     这个生产者钩子不负责准备消费者的计算访问。
        if self._offload_adapter is not None:
            self._offload_adapter.prefetch(layer, selection, positions)

    def write_locations(
        self, layer: PagedAttention, forward_mode: ForwardMode
    ) -> torch.Tensor:
        """Return history IDs without entering the KV offloading access path.

        Index-K writers query these before selection exists. IDs belong to
        layer.group_id and never refer to hot/staging storage; metadata must
        already have been initialized/refreshed. No KV is copied here.
        Decode (and the decode half of a MIXED round) returns the token-major
        ``[bs * N]`` window view over the location stack; extend returns the
        ``[sum(extend_seq_lens)]`` span computed at ``init_forward_metadata``.
        """
        # 译：返回历史 ID，不进入 KV offload 的访问路径。
        #
        #     索引 K 的写入者在选择结果存在之前就查询这些位置。ID 属于 layer.group_id，从不指向热存储 / 暂存存储；
        #     元数据必须已经初始化 / 刷新过。这里不拷贝任何 KV。
        #     decode（以及 MIXED 轮的 decode 部分）返回位置堆叠上以 token 为主序的 ``[bs * N]`` 窗口视图；
        #     extend 返回 ``init_forward_metadata`` 时算出的 ``[sum(extend_seq_lens)]`` 段。
        # 例：decode 模式返回 decode_locs 的视图；MIXED 时 _decode_request_offset = num_extends，
        # 跳过前面 extend 请求在窗口里占的位置，只留 decode 请求的。extend / MIXED 模式返回 extend 段（A 的 300 个）。
        self._leaf_for(layer)
        gid = layer.group_id
        if forward_mode.is_decode():
            published = self.decode_write_locations
            if published is None:
                raise RuntimeError(
                    "decode write locations requested before any decode refresh"
                )
            locs = published.by_group[gid]
            if self._decode_request_offset:
                locs = locs[self._decode_request_offset * published.tokens_per_req :]
            return locs
        if self._extend_write_locations is None:
            raise RuntimeError(
                "extend write locations requested before init_forward_metadata"
            )
        return self._extend_write_locations[gid]

    def padded_write_locations(
        self, layer: PagedAttention, forward_mode: ForwardMode, rows: int
    ) -> torch.Tensor:
        """The forward's rows padded with the dummy slot 0 to the rows a
        graph-padded forward carries: the extend span, then a MIXED round's
        decode rows, then the stack buffer's zero tail; decode rows carry their
        own slots. A forward wider than the buffer runs eager and gets a fresh
        tensor."""
        # 译：本次前向的行，用哑槽位 0 补齐到 graph padding 后的前向所携带的行数：先是 extend 段，然后是 MIXED 轮的
        #     decode 行，再是堆叠缓冲区的零尾巴；decode 行携带的就是它们自己的槽位。比缓冲区更宽的前向以 eager 方式运行，
        #     得到一个新分配的张量。
        # prologue（layers/paged_attention.py）写 KV 时调用的就是这个函数，rows = 本次前向的 token 行数。
        # 例：MIXED 批 rows=301，extend 段只有 300 个 -> 返回 [A 的 300 个槽位, B 的 1 个槽位]；
        # 若 prefill graph 把 301 行 padding 到 512 行，则后 211 个是 0（写到哑页，不影响任何请求）。
        locations = self.forward_write_locations(layer, forward_mode)
        if locations.numel() == rows:
            return locations
        if not forward_mode.is_extend_or_mixed():
            raise ValueError(
                f"{locations.numel()} {forward_mode.name.lower()} write slots for {rows} rows"
            )
        return self.stacks.padded_extend_span(layer.group_id, rows)

    def _append_decode_rows(self, bs: int, num_extends: int) -> None:
        """A MIXED round's decode rows follow its extend span in the padded span."""
        # 译：在 padding 后的段中，MIXED 轮的 decode 行紧跟在其 extend 段之后。
        n = self._decode_tokens_per_req
        self.stacks.append_decode_rows(num_extends * n, (bs - num_extends) * n)

    def forward_write_locations(
        self, layer: PagedAttention, forward_mode: ForwardMode
    ) -> torch.Tensor:
        """A draft step 0 over a MIXED round carries the extend rows and then the
        decode rows, whether it dispatches as MIXED or locally as DECODE."""
        # 译：MIXED 轮上的草稿第 0 步携带 extend 行再接 decode 行，无论它以 MIXED 分派，还是在本地以 DECODE 分派。
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        if self._offload_adapter is not None:
            return self._offload_adapter.compute_write_locations(layer, forward_mode)
        mixed_draft_step_zero = (
            (forward_mode.is_decode() or forward_mode.is_mixed())
            and self.is_draft
            and self._decode_request_offset != 0
            and self._extend_write_locations is not None
        )
        if not mixed_draft_step_zero:
            return self.write_locations(layer, forward_mode)
        extend_locations = self._extend_write_locations[layer.group_id]
        decode_locations = self.write_locations(layer, ForwardMode.DECODE)
        if decode_locations.numel() == 0:
            return extend_locations
        return torch.cat((extend_locations, decode_locations))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    # 译：元数据

    def _check_live_delivery(self, actual_bs: int, block_tables) -> None:
        """A live batch must deliver every routed group's table — the
        persistent decode buffers would otherwise serve stale pages."""
        # 译：一个有活跃请求的批次必须提供每个被路由组的表——否则持久的 decode 缓冲区会继续服务过期的页。
        if actual_bs <= 0:
            return
        missing = [gid for gid in self.leaves if gid not in block_tables]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__}: block_tables at bs={actual_bs} is missing "
                f"cache groups {missing} (delivered: {sorted(block_tables)})"
            )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_replay_lens_cpu: torch.Tensor,
        extend_prompt_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        query_shard: QueryShardPlan | None,
        block_tables_cpu: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> None:
        """Extend / mixed / idle-warmup metadata for every leaf.

        Expands each group's raw table into the stack, computes the extend
        write spans (and, for a MIXED round, the decode requests' verify
        window), then hands every leaf its ``[bs, max_num_pages]`` kernel page
        table. ``extend_with_prefix`` (some extend request continues a cached
        or chunked prefix) travels with the extend lengths: leaves size their
        paged-prefix metadata by it, so it must reach them unchanged. A
        ``query_shard`` reaches every leaf too, with the host mirror of the
        extend rows of its kernel page table (``block_tables_cpu``, the
        runner's host mirror of ``block_tables``, expanded the way the stack
        expands the device tables), so a leaf that gathers history by page
        owner can split the gather without a device sync. A page-sharded
        group's leaf gets the mirror for an extend with a prefix too, to split
        its prefix-chunk gathers the same way, and for any pure extend, to
        bound the pages each rank owns (the DSA leaf's owned-width index
        tiles). Bounded native prefill also uses it for cold batches to keep
        the receive pages within the scheduler's budget; other forwards read
        nothing from the mirror.
        """
        # 译：为每个叶子准备 extend / mixed / idle 预热的元数据。
        #
        #     把每个组的原始表展开进堆叠，计算 extend 写入段（以及 MIXED 轮中 decode 请求的验证窗口），
        #     然后把各自的 ``[bs, max_num_pages]`` kernel 页表交给每个叶子。``extend_with_prefix``（某个 extend 请求
        #     接续了已缓存的或分块的前缀）随 extend 长度一起传递：叶子按它决定分页前缀元数据的大小，所以必须原样到达叶子。
        #     ``query_shard`` 也会到达每个叶子，同时附带其 kernel 页表中 extend 行的 Host 镜像（``block_tables_cpu``，
        #     即 runner 对 ``block_tables`` 的 Host 镜像，按堆叠展开设备表的同样方式展开），这样按页所有者收集历史的叶子
        #     可以不做设备同步就拆分收集操作。页分片组的叶子在带前缀的 extend 时也会拿到该镜像，用来以同样方式拆分前缀块的
        #     收集；在任何纯 extend 时也会拿到，用来界定每个 rank 拥有的页（DSA 叶子按拥有宽度划分的索引 tile）。
        #     有界的原生 prefill 也会对冷批次使用它，使接收页保持在调度器的预算内；其他前向不从镜像读任何东西。
        # 步骤（以 MIXED 批 A extend 300 + B decode 为例，bs=2，num_extends=1）：
        #   1. stacks.fill：两行 block table 展开成 kernel 页表（Qwen3 ratio=1，就是原样拷贝 + 补 0）
        #   2. extend_locations：只对前 num_extends 个请求（A）算 300 个写入槽位
        #   3. MIXED 才有：_refresh_decode_locations 对全部 bs 个请求按 decode 规则算窗口（A 那一个是无用的），
        #      _decode_request_offset=1 让后续读取跳过 A；_append_decode_rows 把 B 的槽位接到 extend 段之后
        #   4. 每个叶子 init_forward_metadata：MHA 叶子据此建 cu_seqlens 等 prefill kernel 需要的元数据
        # 注意：MHA 叶子不把 MIXED 批拆成两次注意力调用，而是把 B 当作 query 长度为 N 的 extend 请求，
        # 与 A 一起走一次 ragged prefill。MLA 系（FlashMLA，第 12 阶段）会拆成 prefill 半批和 decode 半批分别调用。
        del extend_prompt_lens_cpu
        reject_bounded_replay(extend_replay_lens_cpu, "CacheGroupRouter")
        del kwargs
        sharded = query_shard is not None and query_shard.size > 1
        # A new forward: the sparse layers' shared top-k is per forward.
        # 译：新的一次前向：稀疏层共享的 top-k 是按前向划分的。
        self.sparse_topk.clear()
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed/idle ({forward_mode})"
            )
        live_bs = 0 if forward_mode.is_idle() else bs
        self._check_live_delivery(live_bs, block_tables)
        self.stacks.fill(bs, live_bs, block_tables)
        self._extend_write_locations = None
        self._decode_request_offset = 0
        if forward_mode.is_extend_or_mixed() and num_extends > 0:
            total = int(sum(int(x) for x in extend_seq_lens_cpu[:num_extends].tolist()))
            self._extend_write_locations = self.stacks.extend_locations(
                extend_prefix_lens[:num_extends], extend_seq_lens[:num_extends], total
            )
        if forward_mode.is_mixed():
            # The decode requests follow the extend requests; their verify
            # window is the same math as a pure decode over every request,
            # sliced.
            # 译：decode 请求排在 extend 请求之后；它们的验证窗口与“对所有请求做一次纯 decode”是同样的计算，取其切片即可。
            self._refresh_decode_locations(bs, seq_lens)
            self._decode_request_offset = num_extends
            self._append_decode_rows(bs, num_extends)
        for gid, leaf in self.leaves.items():
            leaf.init_forward_metadata(
                bs,
                num_extends,
                seq_lens,
                self.stacks.table(gid, bs),
                forward_mode,
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                extend_prefix_lens=extend_prefix_lens,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
                extend_with_prefix=extend_with_prefix,
                query_shard=query_shard,
                page_table_cpu=(
                    self.stacks.host_table(gid, block_tables_cpu[gid], num_extends)
                    if sharded
                    or (
                        gid in self._page_sharded_groups
                        and (extend_with_prefix or forward_mode.is_extend())
                    )
                    or (num_extends > 0 and leaf.prefill_needs_host_page_table)
                    else None
                ),
            )
            leaf.set_request_slots(req_pool_indices[:bs])

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        block_tables: Mapping[str, torch.Tensor],
        num_extends: int = 0,
        for_graph_replay: bool = False,
        **kwargs,
    ) -> None:
        """The single decode path: fill the stacks from this step's tables,
        derive the decode write window, refresh every leaf in place."""
        # 译：唯一的 decode 路径：用本步的表填充堆叠，推导出 decode 写入窗口，原地刷新每个叶子。
        # decode 的关键约束：CUDA Graph 回放时 kernel 读的是捕获时记录的地址，所以元数据只能“原地刷新”，不能新建张量。
        # eager decode 也走这里（for_graph_replay=False），保证两种执行方式的元数据完全一致。
        # bs 是 padding 后的批大小（graph 的桶），actual_bs 是真实请求数；[actual_bs, bs) 的行在 fill 中置 0（哑页）。
        del kwargs
        self.sparse_topk.clear()
        if forward_mode.is_extend_or_mixed():
            raise RuntimeError(
                f"refresh_decode_metadata serves decode only ({forward_mode})"
            )
        self._check_live_delivery(actual_bs, block_tables)
        self.stacks.fill(bs, actual_bs, block_tables)
        # The window covers every request; DECODE reads skip the extend
        # requests a MIXED round's draft refresh carries (num_extends is 0 on
        # targets).
        # 译：窗口覆盖所有请求；DECODE 读取时跳过 MIXED 轮草稿刷新所携带的 extend 请求（对目标模型来说 num_extends 为 0）。
        self._decode_request_offset = num_extends
        self._refresh_decode_locations(bs, seq_lens)
        if num_extends and self._extend_write_locations is not None:
            self._append_decode_rows(bs, num_extends)
        # 快路径：叶子若只需要“seq_lens + 页表”两个缓冲区（decode_buffer_targets 返回非 None），
        # 就由 router 用一次 _foreach_copy_ 把所有这类叶子的缓冲区一起拷好，再让叶子 publish；否则叶子自己 refresh。
        targets = {
            gid: leaf.decode_buffer_targets(bs) for gid, leaf in self.leaves.items()
        }
        batched = [gid for gid, pair in targets.items() if pair is not None]
        if batched:
            # One launch fills every such leaf (plus one clamp per seq_lens
            # floor); two stream copies per leaf cost ~4 us each at bs=1.
            # 译：一次启动填好所有这类叶子（外加每种 seq_lens 下限一次 clamp）；每个叶子两次流上拷贝，在 bs=1 时每次约 4 us。
            floored = {}
            for gid in batched:
                floor = self.leaves[gid].decode_seq_lens_floor
                if floor not in floored:
                    floored[floor] = (
                        seq_lens[:bs] if floor <= 1 else seq_lens[:bs].clamp_min(floor)
                    )
            torch._foreach_copy_(
                [buf for gid in batched for buf in targets[gid]],
                [
                    src
                    for gid in batched
                    for src in (
                        floored[self.leaves[gid].decode_seq_lens_floor],
                        self.stacks.table(gid, bs),
                    )
                ],
            )
        for gid, leaf in self.leaves.items():
            if targets[gid] is None:
                leaf.refresh_decode_metadata(
                    bs,
                    actual_bs,
                    seq_lens,
                    self.stacks.table(gid, bs),
                    num_extends=num_extends,
                    for_graph_replay=for_graph_replay,
                )
            else:
                leaf.publish_decode_metadata(bs, seq_lens, num_extends=num_extends)
            leaf.set_request_slots(req_pool_indices[:bs])

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> None:
        """Capture seeding: the idle fill over the same stacks replay
        refreshes, then each leaf's own capture hook (the leaf default is
        its idle refresh; FlashMLA installs its recorded tile schedule)."""
        # 译：捕获时的初始填充：对回放时刷新的同一批堆叠做 idle 填充，然后调用每个叶子自己的捕获钩子
        #     （叶子默认是其 idle 刷新；FlashMLA 会安装它记录下来的 tile 调度）。
        del kwargs
        self.sparse_topk.clear()
        if not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"{type(self).__name__} CUDA graphs record decode only, got {forward_mode}"
            )
        self.stacks.fill(bs, 0, block_tables)
        self._decode_request_offset = 0
        self._refresh_decode_locations(bs, seq_lens)
        for gid, leaf in self.leaves.items():
            leaf.init_forward_metadata_capture_cuda_graph(
                bs, seq_lens, self.stacks.table(gid, bs)
            )
            leaf.set_request_slots(req_pool_indices[:bs])

    # ------------------------------------------------------------------
    # Draft write locations (the drafters' in-graph slot math)
    # ------------------------------------------------------------------
    # 译：草稿写入位置（草稿模型在 graph 内的槽位计算）

    # 以下到 fill_block_decode_seq_lens 为投机解码草稿模型专用（第 11 阶段），初读可跳过。
    def _draft_history_index(self) -> int:
        """Stack index of the full-history group the draft chain writes.

        The multi-step drafters commit their KV along the full-history
        table; a pool without one serves a single group, which then IS the
        history.
        """
        # 译：草稿链写入的全历史组在堆叠中的下标。
        #
        #     多步草稿模型沿全历史表提交它们的 KV；没有全历史组的 pool 只服务单个组，那个组就是历史。
        gid = self.geometry.full_history_group_id
        if gid is not None and gid in self.leaves:
            return self.stacks.index(gid)
        if len(self.leaves) == 1:
            return 0
        raise RuntimeError(
            f"{type(self).__name__}: no full-history group among {self.group_ids} "
            "to derive draft write locations from"
        )

    def draft_history_view(self) -> DraftHistoryView:
        """The address-stable full-history table for in-graph draft consumers.

        The multi-step drafters resolve their KV write slots inside the
        captured graph (inputs derive from verify's in-graph accept lengths),
        so their kernels record this table's address at capture; the router
        owns the storage and its refresh rewrites it in place each round.

        The tensor is the group's FULL contiguous table in the stack
        (``[max_bs, stack_max_num_pages]``, batch-ordered): the slot kernels
        assume row-major contiguity. Columns past the group's own
        ``max_num_pages`` are null pages by the fill contract and resolve to
        the dummy slot 0; ``max_tokens`` is the
        group's real per-request capacity, the bound block drafters clamp
        their prefix against.
        """
        # 译：供 graph 内草稿消费者使用的、地址稳定的全历史表。
        #
        #     多步草稿模型在捕获的 graph 内解析它们的 KV 写入槽位（输入来自验证在 graph 内得到的接受长度），
        #     所以它们的 kernel 在捕获时记录这张表的地址；router 持有其存储，每轮刷新时原地重写。
        #
        #     该张量是该组在堆叠中“完整的”连续表（``[max_bs, stack_max_num_pages]``，按批次排序）：槽位 kernel 假定行主序连续。
        #     超出该组自身 ``max_num_pages`` 的列按填充约定是空页，解析到哑槽位 0；``max_tokens`` 是该组真实的每请求容量，
        #     块草稿模型据此截断它们的前缀。
        i = self._draft_history_index()
        gid = self.group_ids[i]
        # Host-side spec, never ``int(self.stacks.page_sizes[i])``: the block
        # drafters call this inside the captured graph, where a device->host
        # read is a stream-capture violation.
        # 译：用 Host 端规格，绝不用 ``int(self.stacks.page_sizes[i])``：块草稿模型在捕获的 graph 内调用它，
        #     此时设备 -> Host 的读取会违反流捕获规则。
        return DraftHistoryView(
            table=self.stacks.tables[i],
            page_size=self.stacks.group_kernel_page_size(gid),
            max_tokens=self.stacks.group_capacity_tokens(gid),
        )

    def draft_write_locations_uniform(
        self, out: torch.Tensor, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Scratch-buffer slot resolve over the full-history table (side
        writes that must not clobber the published step window)."""
        # 译：在全历史表上解析暂存缓冲区的槽位（这些旁路写入不能覆盖已发布的步窗口）。
        view = self.draft_history_view()
        if view.table.is_cuda:
            from tokenspeed.runtime.execution.cache_loc_kernel import (
                compute_out_cache_loc_uniform,
            )

            compute_out_cache_loc_uniform(
                out_cache_loc_ptr=out,
                uniform_input_length=num_tokens,
                cache_start=cache_start,
                page_table=view.table,
                page_size=view.page_size,
            )
            return out
        # CPU reference (unit tests): same slot rule, overflow/holes -> 0.
        # 译：CPU 参考实现（单元测试）：同样的槽位规则，越界 / 空洞 -> 0。
        bs = cache_start.shape[0]
        steps = torch.arange(num_tokens, dtype=torch.int64, device=view.table.device)
        pos = cache_start.to(torch.int64).unsqueeze(1) + steps
        max_num_pages = view.table.shape[1]
        page_idx = (pos // view.page_size).clamp_max(max_num_pages - 1)
        overflow = (pos // view.page_size) >= max_num_pages
        pages = view.table[:bs].to(torch.int64).gather(1, page_idx)
        locs = pages * view.page_size + pos % view.page_size
        locs = torch.where(overflow | (pages <= 0), torch.zeros_like(locs), locs)
        out[: bs * num_tokens].copy_(locs.reshape(-1).to(out.dtype))
        return out

    def publish_draft_step_locations(
        self, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Publish this draft step's KV write window: ``num_tokens`` slots
        per request starting at ``cache_start``, for every routed group.

        The drafters declare the window (they own the step semantics — the
        Eagle chain's one advancing slot, MTP's re-anchored k-window, the
        DFLASH block); the math and the address-stable storage live here.
        Captured-graph friendly: the same fused launch the decode refresh
        records, over the same location stack, so ``write_locations`` /
        the leaf forwards read the step's slots at recorded addresses.

        Returns the full-history group's ``[bs * num_tokens]`` view (the
        block drafters hand it to their fused prep kernels).
        """
        # 译：发布本草稿步的 KV 写入窗口：对每个被路由的组，每个请求从 ``cache_start`` 开始的 ``num_tokens`` 个槽位。
        #
        #     草稿模型声明窗口（它们拥有步的语义——Eagle 链每步前进一个槽位、MTP 重新锚定的 k 窗口、DFLASH 的块）；
        #     计算和地址稳定的存储在这里。对捕获 graph 友好：与 decode 刷新记录的是同一个融合启动、同一个位置堆叠，
        #     所以 ``write_locations`` / 叶子的前向在记录下来的地址上读到本步的槽位。
        #
        #     返回全历史组的 ``[bs * num_tokens]`` 视图（块草稿模型把它交给自己的融合准备 kernel）。
        bs = cache_start.shape[0]
        # Window positions run cache_start .. cache_start+n-1.
        # 译：窗口位置为 cache_start .. cache_start+n-1。
        self.stacks.compute_window_locations(bs, cache_start, num_tokens)
        self._decode_request_offset = 0
        self._publish_decode_locations(bs, num_tokens)
        i = self._draft_history_index()
        return self.stacks.decode_locs[i, : bs * num_tokens]

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """Eagle chain step: republish the leaves' seq_lens, in-graph.

        Deliberately seq-lens-only: Eagle's step-0 rejected-tail correction
        fires this BEFORE the step-0 attention has consumed the verify-shaped
        write window, so window publication stays an explicit drafter-loop
        call (:meth:`publish_draft_step_locations`)."""
        # 译：Eagle 链步：在 graph 内重新发布叶子的 seq_lens。
        #
        #     刻意只改 seq_lens：Eagle 第 0 步的被拒尾部修正会在第 0 步注意力消费验证形状的写入窗口“之前”触发它，
        #     所以窗口发布保持为草稿循环中的一次显式调用（:meth:`publish_draft_step_locations`）。
        for leaf in self.leaves.values():
            leaf.advance_draft_forward_metadata(seq_lens)

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        """Vanilla MTP re-anchor: seq_lens become the committed frontier,
        in-graph. Seq-lens-only like :meth:`advance_draft_forward_metadata`;
        the drafter publishes its k-window explicitly. Each leaf's own hook
        decides whether the k-row window needs more than the seq_lens edit
        (the leaf default is that edit; DSA re-expands its per-token rows)."""
        # 译：普通 MTP 的重新锚定：seq_lens 在 graph 内变为已提交的前沿。与 :meth:`advance_draft_forward_metadata` 一样
        #     只改 seq_lens；草稿模型显式发布它的 k 窗口。每个叶子自己的钩子决定 k 行窗口是否需要比修改 seq_lens 更多的操作
        #     （叶子默认就是这个修改；DSA 会重新展开它的逐 token 行）。
        for leaf in self.leaves.values():
            leaf.update_draft_forward_metadata(frontier)

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        for leaf in self.leaves.values():
            leaf.fill_block_decode_seq_lens(bs, block_seq_lens)

    @contextmanager
    def override_num_extends(self, num_extends: int):
        with ExitStack() as stack:
            for leaf in self.leaves.values():
                stack.enter_context(leaf.override_num_extends(num_extends))
            yield

    def supports_narrowed_draft_decode(self, forward_mode: ForwardMode) -> bool:
        return all(
            leaf.supports_narrowed_draft_decode(forward_mode)
            for leaf in self.leaves.values()
        )

    def cache_placement(self, layer):
        return self._leaf_for(layer).cache_placement(layer)

    def reads_fp8_kv_scales(self, layer):
        return self._leaf_for(layer).reads_fp8_kv_scales(layer)

    def decode_kv_view(self, layer, token_to_kv_pool):
        return self._leaf_for(layer).decode_kv_view(layer, token_to_kv_pool)

    def query_dtype(self, layer, forward_mode, activation):
        return self._leaf_for(layer).query_dtype(layer, forward_mode, activation)

    # 核心注意力入口。@break_point：breakable CUDA Graph 在这里断开（第 7 阶段），注意力可以在 graph 外 eager 执行。
    # 注意：save_kv_cache 必须为 False。本层的 K/V 已由 prologue 写进 KV 池，后端只读不写（见 _PREWRITTEN）。
    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool: CachePool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool,
        record_kv_cache: bool | None = None,
        **kwargs,
    ):
        # No ambient-ctx override: a MIXED round's halves pass sub-context modes this must honor.
        # 译：不使用环境上下文覆盖：MIXED 轮的两半会传入子上下文的模式，这里必须遵守。
        assert not save_kv_cache, _PREWRITTEN
        leaf = self._leaf_for(layer)
        if self._offload_adapter is not None and "topk_indices" in kwargs:
            kwargs["topk_indices"] = self._offload_adapter.read_indices(
                layer, kwargs["topk_indices"]
            )
        out_cache_loc = self.forward_write_locations(layer, forward_mode)
        with self.record_pd_cache_step(
            forward_mode, writes_in_call=False, record_kv_cache=record_kv_cache
        ):
            if forward_mode.is_decode():
                return leaf.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    **kwargs,
                )
            return leaf.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                forward_mode=forward_mode,
                **kwargs,
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        token_to_kv_pool,
        bs,
        save_kv_cache: bool,
        **kwargs,
    ):
        """Composite hosts (hybrid GDN/KDA) dispatch decode directly."""
        # 译：复合宿主（混合 GDN/KDA）直接分派 decode。
        assert not save_kv_cache, _PREWRITTEN
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        leaf = self._leaf_for(layer)
        if self._offload_adapter is not None and "topk_indices" in kwargs:
            kwargs["topk_indices"] = self._offload_adapter.read_indices(
                layer, kwargs["topk_indices"]
            )
        out_cache_loc = self.forward_write_locations(layer, ForwardMode.DECODE)
        return leaf.forward_decode(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            bs,
            **kwargs,
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        token_to_kv_pool,
        bs,
        save_kv_cache: bool,
        forward_mode=None,
        **kwargs,
    ):
        """Composite hosts dispatch extend directly."""
        # 译：复合宿主直接分派 extend。
        assert not save_kv_cache, _PREWRITTEN
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        leaf = self._leaf_for(layer)
        out_cache_loc = self.forward_write_locations(layer, ForwardMode.EXTEND)
        return leaf.forward_extend(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            bs,
            forward_mode=forward_mode,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Single-group model-side surface (the MLA family's model code reads the
    # leaf's metadata and chunked prefill directly)
    # ------------------------------------------------------------------
    # 译：单组模型侧接口（MLA 家族的模型代码直接读叶子的元数据和分块 prefill）

    @property
    def chunked_prefill_metadata(self):
        return self._sole_leaf("chunked_prefill_metadata").chunked_prefill_metadata

    @property
    def forward_decode_metadata(self):
        return self._sole_leaf("forward_decode_metadata").forward_decode_metadata

    @property
    def forward_prefill_metadata(self):
        return self._sole_leaf("forward_prefill_metadata").forward_prefill_metadata

    @property
    def supports_mla_projected_value_decode(self) -> bool:
        return all(
            getattr(leaf, "supports_mla_projected_value_decode", False)
            for leaf in self.leaves.values()
        )

    @property
    def supports_layer_sliding_window(self) -> bool:
        # A window dropped by any leaf is a window dropped, so this is all().
        # 译：任何一个叶子丢掉窗口，窗口就被丢掉了，所以这里用 all()。
        return bool(self.leaves) and all(
            leaf.supports_layer_sliding_window for leaf in self.leaves.values()
        )

    @property
    def supports_layer_window_sink(self) -> bool:
        # A sink dropped by any leaf is a sink dropped, as for the window.
        # 译：任何一个叶子丢掉 sink，sink 就被丢掉了，与窗口同理。
        return bool(self.leaves) and all(
            leaf.supports_layer_window_sink for leaf in self.leaves.values()
        )

    @property
    def data_type(self):
        return self._sole_leaf("data_type").data_type

    @property
    def max_context_len(self) -> int:
        # Every leaf sizes from the one config.context_len; any leaf answers.
        # 译：每个叶子都按同一个 config.context_len 确定尺寸；问任何一个叶子都行。
        return next(iter(self.leaves.values())).max_context_len

    def forward_extend_chunked(self, *args, **kwargs):
        return self._sole_leaf("forward_extend_chunked").forward_extend_chunked(
            *args, **kwargs
        )

    def read_prefix_history(self, *args, **kwargs):
        return self._sole_leaf("read_prefix_history").read_prefix_history(
            *args, **kwargs
        )

    def forward_sparse_prefill(self, *args, **kwargs):
        return self._sole_leaf("forward_sparse_prefill").forward_sparse_prefill(
            *args, **kwargs
        )

    # ------------------------------------------------------------------
    # DSA query-shard surface: a sparse-attention model's own top-k over KVP
    # pages reads the sharded extend's request groups and gathers each
    # group's index-K history through the leaf (``docs/design/unified_path.md``,
    # "Query context parallelism").
    # ------------------------------------------------------------------
    # 译：DSA query 分片接口：稀疏注意力模型在 KVP 页上自己做的 top-k 会读取分片 extend 的请求组，
    #     并通过叶子收集每个组的索引 K 历史（``docs/design/unified_path.md``，“Query context parallelism”）。

    def require_query_shard_metadata(self):
        return self._sole_leaf(
            "require_query_shard_metadata"
        ).require_query_shard_metadata()

    def gather_history_index_k(self, layer_id: int, token_to_kv_pool, group):
        return self._sole_leaf("gather_history_index_k").gather_history_index_k(
            layer_id, token_to_kv_pool, group
        )

    # DSA KVP prefill: top-k over KVP pages tiles by the leaf's planned owned width.
    # 译：DSA KVP prefill：KVP 页上的 top-k 按叶子规划的拥有宽度划分 tile。

    def prefill_index_columns(self) -> int | None:
        return self._sole_leaf("prefill_index_columns").prefill_index_columns()

    # ------------------------------------------------------------------
    # DSA KPool surface (GLM-5.3-Flash): the model drives the leaf's pooled
    # indexing runtime through the router.
    # ------------------------------------------------------------------
    # 译：DSA KPool 接口（GLM-5.3-Flash）：模型通过 router 驱动叶子的池化索引运行时。

    def require_kpool_runtime(self):
        return self._sole_leaf("require_kpool_runtime").require_kpool_runtime()

    def kpool_prefill_page_table(self, num_requests: int):
        return self._sole_leaf("kpool_prefill_page_table").kpool_prefill_page_table(
            num_requests
        )

    def kpool_decode_page_table(self, row_start: int, num_requests: int):
        return self._sole_leaf("kpool_decode_page_table").kpool_decode_page_table(
            row_start, num_requests
        )
