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

"""Cache-group geometry learned from the pool's published specs.

One immutable value object answers every "what shape is this group?"
question a backend asks — group block granularities, every group's kind
(the positive-claim vocabulary ``cache_consumer_attn_kinds`` filters against),
and which group is the full history the draft chain writes along. Learned
exactly once, at ``set_cache_pool`` (the arena's published specs are the
only source, so the eager and CUDA-graph arms can never answer
differently).
"""

# 译：从池发布的 spec 中得知的 cache group 几何。
#
#     一个不可变的值对象回答后端提出的所有“这个组是什么形状？”的问题——各组的 block 粒度、每个组的类型
#     （``cache_consumer_attn_kinds`` 据以筛选的正向声明词汇），以及哪个组是草稿链沿着写入的全历史组。只在
#     ``set_cache_pool`` 时得知一次（arena 发布的 spec 是唯一来源，所以 eager 和 CUDA Graph 两条路径的回答永远不会不同）。

from __future__ import annotations

from dataclasses import dataclass, field


# cache group 的几何信息（启动时从 KV 池的 cache group spec 中学到一次）：
#   granularities：每个非 state 组的 block 粒度（一个 block 覆盖多少 token）
#   attn_kinds：每个组的类型（full_history / sliding_window / state）
#   full_history_group_id：全历史组的 id
#   row_geometry：每个组的物理行布局（每页多少行、每行覆盖多少 token）
#   windows：每个组的滑动窗口大小（全历史组为 None）
# Qwen3-8B 只有一个全历史组：granularities = {组 id: P}，attn_kinds = {组 id: "full_history"}。
@dataclass(frozen=True)
class CacheGroupGeometry:
    """Frozen per-pool group geometry.

    Attributes:
        granularities: ``group_id -> block_granularity`` for every non-state
            group (rows-per-page span; equals the group's page size for the
            row-geometry groups kept here).
        attn_kinds: ``group_id -> attn_kind`` for every published group. The
            positive-claim vocabulary: a backend keeps exactly the delivered
            groups whose kind it declared in ``cache_consumer_attn_kinds``; the
            rest (state blocks for the mamba backend, wrapper-owned conv
            groups) ride the same dict to their own consumers. Empty when
            no pool is bound (unit fixtures, pre-contract draft pools).
        full_history_group_id: The first ``attn_kind="full_history"`` group — the
            table the router's draft write locations ride — or None when the
            pool publishes no such group (a single-group pool's sole group
            then serves as the history). Chosen from the pool at every bind,
            so it takes no part in geometry equality.
        row_geometry: ``group_id -> (rows_per_page, entry_stride_tokens)``
            for every non-state group: the physical row layout the leaves'
            kernels read, so a rebind keeps it.
        windows: ``group_id -> sliding_window_tokens`` for every non-state
            group (None for full history): the scheduler's retention
            contract, which a rebind must keep as well.
    """

    # 译：冻结的、按池的组几何。
    #
    #     属性：
    #         granularities：每个非 state 组的 ``group_id -> block_granularity``（每页行的跨度；对这里保留的行几何组，
    #             它等于该组的页大小）。
    #         attn_kinds：每个已发布组的 ``group_id -> attn_kind``。正向声明的词汇：后端只保留类型在其
    #             ``cache_consumer_attn_kinds`` 中声明过的已交付组；其余的（mamba 后端的 state block、包装器拥有的 conv 组）
    #             沿着同一个字典交给它们自己的使用方。没有绑定池时为空（单元测试夹具、契约之前的草稿池）。
    #         full_history_group_id：第一个 ``attn_kind="full_history"`` 的组——router 的草稿写入位置所依赖的表——
    #             池没有发布这样的组时为 None（此时单组池的唯一组充当历史）。每次绑定时从池中选出，所以不参与几何的相等比较。
    #         row_geometry：每个非 state 组的 ``group_id -> (rows_per_page, entry_stride_tokens)``：叶子 kernel 读取的
    #             物理行布局，所以重新绑定时要保持不变。
    #         windows：每个非 state 组的 ``group_id -> sliding_window_tokens``（全历史为 None）：调度器的保留契约，
    #             重新绑定时同样要保持不变。

    granularities: dict[str, int] = field(default_factory=dict)
    attn_kinds: dict[str, str] = field(default_factory=dict)
    full_history_group_id: str | None = field(default=None, compare=False)
    row_geometry: dict[str, tuple[int | None, int | None]] = field(kw_only=True)
    windows: dict[str, int | None] = field(kw_only=True)

    def granularity_of(self, group_id: str) -> int:
        """This group's block granularity; an unknown id is a contract bug.

        Every id reaching here must name a learned row-geometry group —
        layer group ids are bound from the pool's plan at startup
        (``bind_cache_groups``), and table dicts are keyed by contract ids.
        No fallback: a miss means the geometry was never learned (pool not
        bound) or the id belongs to a state group.
        """
        # 译：该组的 block 粒度；未知的 id 是契约错误。
        #
        #     到达这里的每个 id 都必须是一个已学到的行几何组——层的组 id 在启动时从池的计划中绑定（``bind_cache_groups``），
        #     表的字典以契约 id 为键。没有回退：查不到意味着从未学到几何（池未绑定），或者该 id 属于一个 state 组。
        try:
            return self.granularities[group_id]
        except KeyError:
            raise KeyError(
                f"cache group {group_id!r} has no learned block granularity "
                f"(learned row-geometry groups: {sorted(self.granularities)})"
            ) from None


def learn_cache_group_geometry(cache_group_specs) -> CacheGroupGeometry:
    """Build the geometry from the pool's published group specs.

    Args:
        cache_group_specs: The arena's ``cache_group_specs`` tuple.

    Returns:
        The frozen geometry.
    """
    # 译：从池发布的组 spec 构建几何。
    #
    #     参数：
    #         cache_group_specs：arena 的 ``cache_group_specs`` 元组。
    #
    #     返回：
    #         冻结的几何。
    full_history = next(
        (spec for spec in cache_group_specs if spec.attn_kind == "full_history"), None
    )
    return CacheGroupGeometry(
        granularities={
            str(spec.group_id): spec.block_granularity
            for spec in cache_group_specs
            if spec.attn_kind != "state"
        },
        attn_kinds={
            str(spec.group_id): str(spec.attn_kind) for spec in cache_group_specs
        },
        full_history_group_id=(
            str(full_history.group_id) if full_history is not None else None
        ),
        row_geometry={
            str(spec.group_id): (spec.rows_per_page, spec.entry_stride_tokens)
            for spec in cache_group_specs
            if spec.attn_kind != "state"
        },
        windows={
            str(spec.group_id): spec.sliding_window_tokens
            for spec in cache_group_specs
            if spec.attn_kind != "state"
        },
    )
