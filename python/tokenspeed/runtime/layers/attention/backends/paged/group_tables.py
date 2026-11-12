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

"""The router's stacked per-group tables: the one block -> page mapping.

``GroupTableStacks`` turns the bridge's raw per-group block tables (scheduler
pages of each group's ``block_granularity``, batch-ordered, ``-1`` ragged
padding, ``0`` null holes) into one ``[G, max_bs, stack_max_num_pages]``
stack of kernel page tables — group ``g`` expanded to its leaf's
``kernel_page_size`` and padded to the leaf's ``max_num_pages``;
``stack_max_num_pages`` is the widest group's — plus the ``[G, max_bs * N]``
stack of decode write slots derived from those tables. Both fills are one
launch for every group (the bridge uploads all groups into one packed
buffer, so the unpack reads it directly); the padding contract is uniform:
request rows past the live batch and page columns past a group's
``max_num_pages`` are 0, the zero-initialized dummy page, safe to
dereference and never a live request's cache.

The stacks are allocated once (``init_cuda_graph_state``) and refilled in
place. Leaves copy their ``[bs, max_num_pages]`` view into their own
graph-recorded buffers, but three consumers read the stack storage directly
inside captured graphs and so pin its address: the decode write-slot views
the router publishes, the full-history table the block drafters get from
``draft_history_view``, and the per-group tables the QSA indexer takes from
``table()``.
"""

# 译：router 的按组堆叠表：唯一的 block -> 页映射。
#
#     ``GroupTableStacks`` 把桥接层原始的按组 block table（每个组以其 ``block_granularity`` 为单位的调度器页，
#     按批次排序，``-1`` 为不规则 padding，``0`` 为空洞）变成一个 ``[G, max_bs, stack_max_num_pages]`` 的
#     kernel 页表堆叠——组 ``g`` 被展开到其叶子的 ``kernel_page_size``，并补齐到该叶子的 ``max_num_pages``；
#     ``stack_max_num_pages`` 取最宽的那个组——外加由这些表推导出的 ``[G, max_bs * N]`` decode 写入槽位堆叠。
#     两种填充对所有组都只需一次启动（桥接层把所有组上传到一个打包缓冲区，所以解包直接读它）；
#     padding 约定是统一的：超出活跃批次的请求行和超出某组 ``max_num_pages`` 的页列都为 0，
#     即初始化为零的哑页，可以安全解引用，且从不是任何活跃请求的 cache。
#
#     堆叠只分配一次（``init_cuda_graph_state``），之后原地重填。叶子把它们的 ``[bs, max_num_pages]`` 视图拷贝进
#     自己由 graph 记录的缓冲区，但有三个消费者在捕获的 graph 内直接读堆叠的存储，因此固定了它的地址：
#     router 发布的 decode 写入槽位视图、块草稿模型从 ``draft_history_view`` 拿到的全历史表，
#     以及 QSA 索引器从 ``table()`` 取得的按组表。

# 你在这里：CacheGroupRouter（router.py）持有一个 GroupTableStacks，每次前向开始时调用 fill 填页表，
# 再调用 extend_locations / compute_decode_locations 算写入槽位。
#
# 两个“页”的概念（第一次见容易混）：
#   block（调度器页）：C++ 调度器分配与前缀匹配的单位，大小 = 该 cache group 的 block_granularity
#   kernel 页：注意力 kernel 读页表时的单位，大小 = 叶子后端的 kernel_page_size（即 P）
#   要求 block_granularity 是 kernel 页大小的整数倍，ratio = block_granularity / kernel_page_size。
#   一个 block 号 b 展开成 ratio 个连续的 kernel 页号：b*ratio, b*ratio+1, ..., b*ratio+ratio-1。
#
# 展开例子（假设 block_granularity=128，kernel 页大小 64，ratio=2）：
#   原始 block table 行  [ 6,  0,  9, -1]          （0 = 空洞，-1 = 不规则 padding）
#   kernel 页表行        [12, 13,  0,  0, 18, 19,  0,  0,  0, ...]
#   第 col 列 = block[col // ratio] * ratio + col % ratio；block <= 0 的都变成 0（哑页）。
# Qwen3-8B 的贯穿例子 ratio=1，展开就是原样拷贝，A 的行 [12, 3, 40, 7, 21] -> [12, 3, 40, 7, 21, 0, 0, ...]。
#
# 存储布局：
#   tables       [G, max_bs, stack_max_num_pages] int32   kernel 页表（G = cache group 数，Qwen3 为 1）
#   decode_locs  [G, max_bs * N] int32                     decode 写入槽位（N = 每请求写入 token 数）
#   extend_locs  [G, max_extend_tokens] int32              extend 写入槽位（尾部为 0，供 prefill graph padding）
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from tokenspeed.runtime.layers.attention.backends.paged.write_locations import (
    decode_write_locations,
    extend_write_locations,
    window_write_locations,
)
from tokenspeed.runtime.utils.triton import tl, triton


@dataclass(frozen=True)
class GroupTableSpec:
    """Geometry of one routed group's kernel page table.

    Attributes:
        group_id: The cache group id (the key in every delivered dict).
        block_granularity: Tokens per scheduler block (raw table unit).
        kernel_page_size: Tokens per kernel page of the consuming leaf; must
            divide ``block_granularity``.
        max_num_pages: Kernel pages per request the leaf reads
            (``leaf.max_num_pages``).
    """

    # 译：一个被路由组的 kernel 页表几何信息。
    #
    #     属性：
    #         group_id：cache group id（每个传入字典中的键）。
    #         block_granularity：每个调度器 block 的 token 数（原始表的单位）。
    #         kernel_page_size：消费它的叶子的每个 kernel 页的 token 数；必须整除 ``block_granularity``。
    #         max_num_pages：叶子读取的每请求 kernel 页数（``leaf.max_num_pages``）。

    group_id: str
    block_granularity: int
    kernel_page_size: int
    max_num_pages: int

    @property
    def ratio(self) -> int:
        if self.kernel_page_size <= 0 or self.block_granularity % self.kernel_page_size:
            raise ValueError(
                f"cache group {self.group_id!r}: block_granularity "
                f"{self.block_granularity} is not a positive multiple of the "
                f"kernel page size {self.kernel_page_size}"
            )
        return self.block_granularity // self.kernel_page_size


# Groups one packed fill serves; a router with more launches once per group.
# 译：一次打包填充可服务的组数；超过这个数的 router 每个组启动一次。
_PACKED_MAX_GROUPS = 8


@triton.jit
def _unpack_row(
    src_ptr,  # this group's raw table [>= actual_bs, num_blocks] int32
    # 译：本组的原始表 [>= actual_bs, num_blocks] int32
    dst_ptr,  # this group's stack [max_bs, stack_max_num_pages] int32
    # 译：本组的堆叠 [max_bs, stack_max_num_pages] int32
    b,  # the request row
    # 译：请求行
    live,  # whether the row is a live request
    # 译：该行是否为活跃请求
    num_blocks,  # scheduler blocks per request (the raw table's column count)
    # 译：每个请求的调度器 block 数（原始表的列数）
    src_stride_b,
    ratio,  # kernel pages per scheduler block
    # 译：每个调度器 block 包含的 kernel 页数
    max_num_pages,  # kernel pages per request the leaf reads
    # 译：叶子读取的每请求 kernel 页数
    dst_stride_b,
    stack_max_num_pages,  # the stack's column count (widest group)
    # 译：堆叠的列数（最宽的组）
    BLOCK_COLS: tl.constexpr,
):
    """Expand one request row of a group's raw block table into its stack."""
    # 译：把一个组原始 block table 的一个请求行展开进其堆叠。
    # 按 BLOCK_COLS 列一段循环覆盖整行（stack_max_num_pages 列）；每段内向量化处理。
    # 行偏移 b * src_stride_b 先转 int64，防止大批次 × 宽表时 int32 溢出。
    col_off = tl.arange(0, BLOCK_COLS)
    for col0 in range(0, stack_max_num_pages, BLOCK_COLS):
        col = col0 + col_off
        block = col // ratio
        # Columns past this group's own max_num_pages are the padding
        # contract's tail: always the null page 0, never expansion residue
        # (the slot kernels clamp at the stack's column count, so they can
        # read them).
        # 译：超出本组自身 max_num_pages 的列是 padding 约定的尾部：总是空页 0，从不是展开的残留
        #     （槽位 kernel 在堆叠的列数处截断，所以它们可能读到这些列）。
        in_table = live & (block < num_blocks) & (col < max_num_pages)
        raw = tl.load(
            src_ptr + b.to(tl.int64) * src_stride_b + block, mask=in_table, other=0
        )
        # Holes and ragged padding (id <= 0) collapse onto the dummy page 0;
        # a real block expands to its ratio consecutive kernel pages.
        # 译：空洞和不规则 padding（id <= 0）都归并到哑页 0；真实的 block 展开为 ratio 个连续的 kernel 页。
        page = tl.maximum(raw, 0).to(tl.int64) * ratio + col % ratio
        page = tl.where(raw > 0, page, 0)
        vals = tl.where(in_table, page, 0)
        tl.store(
            dst_ptr + b * dst_stride_b + col,
            vals.to(tl.int32),
            mask=col < stack_max_num_pages,
        )


@triton.jit(
    do_not_specialize=["num_blocks", "src_stride_b", "actual_bs"],
    do_not_specialize_on_alignment=["src_ptr"],
)
def _unpack_group_kernel(
    src_ptr,
    dst_ptr,
    num_blocks,
    src_stride_b,
    ratio,
    max_num_pages,
    dst_stride_b,
    stack_max_num_pages,
    actual_bs,
    BLOCK_COLS: tl.constexpr,
):
    """Expand one group's raw block table into its kernel-page stack, one
    request per program (arguments as :func:`_unpack_row`): the fill of
    tables that do not share the bridge's packed upload."""
    # 译：把一个组的原始 block table 展开进其 kernel 页堆叠，每个 program 处理一个请求（参数同 :func:`_unpack_row`）：
    #     用于不共享桥接层打包上传的那些表的填充。
    b = tl.program_id(0)  # grid == padded bs
    # 译：grid == padding 后的 bs
    _unpack_row(
        src_ptr,
        dst_ptr,
        b,
        b < actual_bs,
        num_blocks,
        src_stride_b,
        ratio,
        max_num_pages,
        dst_stride_b,
        stack_max_num_pages,
        BLOCK_COLS,
    )


@triton.jit(
    do_not_specialize=[
        "actual_bs",
        *(f"offset{i}" for i in range(_PACKED_MAX_GROUPS)),
        *(f"blocks{i}" for i in range(_PACKED_MAX_GROUPS)),
    ],
    do_not_specialize_on_alignment=["packed_ptr"],
)
def _unpack_packed_groups_kernel(
    packed_ptr,  # the bridge's one packed upload of every group's raw table
    # 译：桥接层对每个组原始表的一次打包上传
    dst_ptr,  # the stack [G, max_bs, stack_max_num_pages] int32
    # 译：堆叠 [G, max_bs, stack_max_num_pages] int32
    offset0,
    offset1,
    offset2,
    offset3,
    offset4,
    offset5,
    offset6,
    offset7,
    blocks0,
    blocks1,
    blocks2,
    blocks3,
    blocks4,
    blocks5,
    blocks6,
    blocks7,
    dst_stride_g,
    dst_stride_b,
    stack_max_num_pages,
    actual_bs,
    RATIOS: tl.constexpr,
    MAX_NUM_PAGES: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Every group's fill in one launch, one (request, group) per program.

    Group ``i`` reads the row-major ``[>= actual_bs, blocks_i]`` table at
    element ``offset_i`` of the packed upload. Offsets and block counts change
    every step, so they are plain scalars kept out of the compile key (Triton
    would specialize them inside a tuple); groups past ``NUM_GROUPS`` are 0.
    """
    # 译：所有组的填充在一次启动中完成，每个 program 处理一个（请求，组）。
    #
    #     组 ``i`` 读取打包上传中从元素 ``offset_i`` 开始的行主序 ``[>= actual_bs, blocks_i]`` 表。偏移和 block 数每步都变，
    #     所以它们是普通标量，不进入编译键（若放在 tuple 里 Triton 会对它们特化）；超过 ``NUM_GROUPS`` 的组为 0。
    # 为什么展开成 offset0..7、blocks0..7 这些单独参数：它们每步都变，必须是运行时参数（do_not_specialize），
    # 否则每换一个值就重新编译一次 kernel，在前向线程上卡住 100+ ms。RATIOS / MAX_NUM_PAGES 是启动后固定的
    # 模型几何，可以做 constexpr。static_range 在编译期展开对每个组的分支。
    b = tl.program_id(0)  # grid == (padded bs, groups)
    # 译：grid == (padding 后的 bs, 组数)
    g = tl.program_id(1)
    offsets = (offset0, offset1, offset2, offset3, offset4, offset5, offset6, offset7)
    blocks = (blocks0, blocks1, blocks2, blocks3, blocks4, blocks5, blocks6, blocks7)
    for i in tl.static_range(NUM_GROUPS):
        if g == i:
            _unpack_row(
                packed_ptr + offsets[i],
                dst_ptr + i * dst_stride_g,
                b,
                b < actual_bs,
                blocks[i],
                blocks[i],
                RATIOS[i],
                MAX_NUM_PAGES[i],
                dst_stride_b,
                stack_max_num_pages,
                BLOCK_COLS,
            )


def _packed_layout(
    srcs: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, list[int]] | None:
    """The shared storage and per-group element offsets of tables that are
    row-major views into one upload (the bridge packs every group into one),
    or None when any is not and each group needs its own launch."""
    # 译：若这些表都是同一次上传中的行主序视图（桥接层把所有组打包进一次上传），返回共享存储和每个组的元素偏移；
    #     只要有一个不是，就返回 None，此时每个组各需一次启动。
    if len(srcs) > _PACKED_MAX_GROUPS:
        return None
    storage = srcs[0].untyped_storage().data_ptr()
    for src in srcs:
        if (
            src.untyped_storage().data_ptr() != storage
            or src.shape[1] == 0
            or src.stride(1) != 1
            or src.stride(0) != src.shape[1]
        ):
            return None
    start = min(src.storage_offset() for src in srcs)
    end = max(src.storage_offset() + src.numel() for src in srcs)
    first = min(srcs, key=lambda src: src.storage_offset())
    base = first.as_strided((end - start,), (1,))
    return base, [src.storage_offset() - start for src in srcs]


class GroupTableStacks:
    """Stacked kernel page tables and decode write slots for a router's groups.

    Attributes:
        tables: ``[G, max_bs, stack_max_num_pages]`` int32 kernel page
            tables, group-major in ``group_ids`` order.
        decode_locs: ``[G, max_bs * max_tokens_per_req]`` int32 decode write
            slots (token-major per request) refreshed by
            :meth:`compute_decode_locations`; the router publishes per-bs
            views of it, so it is address-stable for the graph's lifetime.
        page_sizes: ``[G]`` int32 device tensor of kernel page sizes.
        page_sizes_cpu: Immutable host tuple from the same specs; avoids
            device scalar reads in the Torch write-location fallback.
    """

    # 译：一个 router 的各组堆叠 kernel 页表和 decode 写入槽位。
    #
    #     属性：
    #         tables：``[G, max_bs, stack_max_num_pages]`` int32 kernel 页表，按 ``group_ids`` 顺序以组为主序。
    #         decode_locs：``[G, max_bs * max_tokens_per_req]`` int32 decode 写入槽位（每个请求内以 token 为主序），
    #             由 :meth:`compute_decode_locations` 刷新；router 发布它的按 bs 视图，所以它在 graph 生命周期内地址稳定。
    #         page_sizes：kernel 页大小的 ``[G]`` int32 设备张量。
    #         page_sizes_cpu：来自同一份规格的不可变 Host 元组；避免在 Torch 写入位置回退路径中读取设备标量。

    def __init__(
        self,
        groups: Sequence[GroupTableSpec],
        *,
        max_bs: int,
        max_tokens_per_req: int,
        max_extend_tokens: int,
        device,
    ) -> None:
        if not groups:
            raise ValueError("GroupTableStacks needs at least one group")
        self.group_ids: tuple[str, ...] = tuple(spec.group_id for spec in groups)
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError(f"duplicate cache group ids: {self.group_ids}")
        self._specs = {spec.group_id: spec for spec in groups}
        self._index = {gid: i for i, gid in enumerate(self.group_ids)}
        self._ratios = tuple(spec.ratio for spec in groups)
        self._max_num_pages = tuple(spec.max_num_pages for spec in groups)
        self.page_sizes_cpu = tuple(int(spec.kernel_page_size) for spec in groups)
        if any(size <= 0 for size in self.page_sizes_cpu):
            raise ValueError("kernel page sizes must be positive host integers")
        self.max_bs = int(max_bs)
        self.max_tokens_per_req = max(int(max_tokens_per_req), 1)
        g = len(groups)
        stack_max_num_pages = max(self._max_num_pages)
        self.tables = torch.zeros(
            (g, self.max_bs, stack_max_num_pages), dtype=torch.int32, device=device
        )
        self.decode_locs = torch.zeros(
            (g, self.max_bs * self.max_tokens_per_req), dtype=torch.int32, device=device
        )
        # Extend spans up to this many tokens live here, tail at the dummy slot 0,
        # so a prefill graph's padded prologue can record and replay the address.
        # 译：不超过这个 token 数的 extend 段存放在这里，尾部为哑槽位 0，
        #     这样 prefill graph 的 padding 后的 prologue 可以记录并回放这个地址。
        self.extend_locs = torch.zeros(
            (g, max(int(max_extend_tokens), 0)), dtype=torch.int32, device=device
        )
        self._extend_total = 0
        self._extend_span: torch.Tensor | None = None
        # A MIXED round's forward carries the decode rows after the extend span.
        # 译：MIXED 轮的前向在 extend 段之后携带 decode 行。
        self._forward_total = 0
        self._decode_tail: torch.Tensor | None = None
        self.page_sizes = torch.tensor(
            self.page_sizes_cpu, dtype=torch.int32, device=device
        )
        if self.tables.is_cuda and g <= _PACKED_MAX_GROUPS:
            # Compile this geometry's packed fill now, not while serving.
            # 译：现在就编译这一几何下的打包填充 kernel，而不是在服务时编译。
            zeros = torch.zeros((g, 1), dtype=torch.int32, device=device)
            self.fill(1, 1, dict(zip(self.group_ids, zeros.split(1))))

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    # 译：视图

    def index(self, group_id: str) -> int:
        return self._index[group_id]

    def group_capacity_tokens(self, group_id: str) -> int:
        """Per-request token capacity of one group's own columns."""
        # 译：一个组自身各列的每请求 token 容量。
        i = self._index[group_id]
        return self._max_num_pages[i] * self._specs[group_id].kernel_page_size

    def group_kernel_page_size(self, group_id: str) -> int:
        """One group's kernel page size from the host-side spec (the device
        ``page_sizes`` mirror feeds the kernels; reading it back would sync)."""
        # 译：从 Host 端规格读取一个组的 kernel 页大小（设备上的 ``page_sizes`` 镜像供 kernel 使用；读回它会同步）。
        return int(self._specs[group_id].kernel_page_size)

    def table(self, group_id: str, bs: int) -> torch.Tensor:
        """``[bs, max_num_pages]`` kernel page table view of one group."""
        # 译：一个组的 ``[bs, max_num_pages]`` kernel 页表视图。
        i = self._index[group_id]
        return self.tables[i, :bs, : self._max_num_pages[i]]

    def host_table(
        self, group_id: str, block_table_cpu: torch.Tensor, num_requests: int
    ) -> torch.Tensor:
        """Expand one group's host block table into kernel pages: the
        ``[num_requests, blocks * ratio]`` host mirror of :meth:`table`'s
        leading rows (unpadded), the same block-to-page math as the device
        fill, for leaves that count page ownership on the host."""
        # 译：把一个组的 Host block table 展开为 kernel 页：即 :meth:`table` 开头若干行的
        #     ``[num_requests, blocks * ratio]`` Host 镜像（不 padding），与设备填充用同样的 block 到页的计算，
        #     供在 Host 上统计页所有权的叶子使用。
        i = self._index[group_id]
        ratio = self._ratios[i]
        src = block_table_cpu[:num_requests].to(torch.int64).clamp_min(0)
        if ratio != 1:
            offsets = torch.arange(ratio, dtype=torch.int64)
            src = (src.unsqueeze(-1) * ratio + offsets).reshape(num_requests, -1)
            # Page 0 stays 0 across its ratio slots.
            # 译：页 0 在它的 ratio 个位置上都保持为 0。
            src = torch.where(src >= ratio, src, torch.zeros_like(src))
        return src.to(torch.int32)

    def decode_locations(
        self, group_id: str, bs: int, tokens_per_req: int
    ) -> torch.Tensor:
        """``[bs * tokens_per_req]`` decode write-slot view of one group."""
        # 译：一个组的 ``[bs * tokens_per_req]`` decode 写入槽位视图。
        n = max(int(tokens_per_req), 1)
        if n > self.max_tokens_per_req:
            raise RuntimeError(
                f"decode write slots sized for {self.max_tokens_per_req} tokens per "
                f"request, asked for {n}"
            )
        return self.decode_locs[self._index[group_id], : bs * n]

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------
    # 译：填充

    def fill(
        self, bs: int, actual_bs: int, block_tables: Mapping[str, torch.Tensor]
    ) -> None:
        """Copy this step's raw group tables into the stack, expanded to
        kernel pages and padded.

        Args:
            bs: Requests to prepare (the padded graph batch, or ``actual_bs``).
            actual_bs: Live requests; ``block_tables`` requests past it are
                ignored and the stack's requests ``[actual_bs, bs)`` are
                zeroed.
            block_tables: ``group_id -> [>= actual_bs, num_blocks]`` int32
                raw scheduler tables covering every routed group (the
                router's ``_check_live_delivery`` guards a live batch's dict).
        """
        # 译：把本步的原始组表拷贝进堆叠，展开为 kernel 页并做 padding。
        #
        #     参数：
        #         bs：要准备的请求数（padding 后的 graph 批大小，或 ``actual_bs``）。
        #         actual_bs：活跃请求数；``block_tables`` 中超出它的请求被忽略，堆叠中 ``[actual_bs, bs)`` 的请求被置零。
        #         block_tables：``group_id -> [>= actual_bs, num_blocks]`` int32 原始调度器表，覆盖每个被路由的组
        #             （router 的 ``_check_live_delivery`` 守护活跃批次的这个字典）。
        if bs < actual_bs or actual_bs < 0:
            raise RuntimeError(f"need 0 <= actual_bs <= bs, got {actual_bs=} {bs=}")
        if bs > self.max_bs:
            raise RuntimeError(f"bs={bs} exceeds the stack capacity {self.max_bs}")
        if bs == 0:
            return
        if actual_bs == 0:
            # Idle / warmup: no live requests to read — every request's row
            # is the null page, whatever placeholder (or nothing) was
            # delivered.
            # 译：idle / 预热：没有可读的活跃请求——无论传入的是什么占位（或什么都没传），每个请求行都是空页。
            self.tables[:, :bs].zero_()
            return
        srcs = [block_tables[gid] for gid in self.group_ids]
        for gid, src in zip(self.group_ids, srcs):
            if src.ndim != 2 or src.dtype != torch.int32:
                raise RuntimeError(
                    f"cache group {gid!r} table must be a 2-D int32 tensor, got "
                    f"{tuple(src.shape)} {src.dtype}"
                )
            if src.shape[0] < actual_bs:
                raise RuntimeError(
                    f"cache group {gid!r} table has {src.shape[0]} request rows for "
                    f"a live batch of {actual_bs}"
                )
        # 三条路径：
        #   1. CUDA + 所有组的表在同一块打包存储里（常态）：一次 _unpack_packed_groups_kernel，grid = (bs, 组数)
        #   2. CUDA + 表不共享存储（例如 prefill graph 捕获时自建的表）：每组一次 _unpack_group_kernel
        #   3. CPU：_fill_torch 参考实现
        if self.tables.is_cuda:
            stack_max_num_pages = self.tables.shape[2]
            block_cols = 128 if stack_max_num_pages >= 128 else 64
            packed = _packed_layout(srcs)
            if packed is not None:
                base, offsets = packed
                pad = [0] * (_PACKED_MAX_GROUPS - len(srcs))
                _unpack_packed_groups_kernel[(bs, len(srcs))](
                    base,
                    self.tables,
                    *offsets,
                    *pad,
                    *(src.shape[1] for src in srcs),
                    *pad,
                    self.tables.stride(0),
                    self.tables.stride(1),
                    stack_max_num_pages,
                    actual_bs,
                    RATIOS=self._ratios,
                    MAX_NUM_PAGES=self._max_num_pages,
                    NUM_GROUPS=len(srcs),
                    BLOCK_COLS=block_cols,
                )
                return
            for i, src in enumerate(srcs):
                if src.stride(1) != 1:
                    src = src.contiguous()
                _unpack_group_kernel[(bs,)](
                    src,
                    self.tables[i],
                    src.shape[1],
                    src.stride(0),
                    self._ratios[i],
                    self._max_num_pages[i],
                    self.tables.stride(1),
                    stack_max_num_pages,
                    actual_bs,
                    BLOCK_COLS=block_cols,
                )
            return
        self._fill_torch(bs, actual_bs, srcs)

    def _fill_torch(
        self, bs: int, actual_bs: int, srcs: Sequence[torch.Tensor]
    ) -> None:
        """CPU reference of the unpack kernel (unit tests); ``actual_bs > 0``
        (``fill`` handles the idle case before dispatching here)."""
        # 译：解包 kernel 的 CPU 参考实现（单元测试）；``actual_bs > 0``（``fill`` 在分派到这里之前已处理 idle 情况）。
        for i, src in enumerate(srcs):
            dst = self.tables[i, :bs]
            max_num_pages = self._max_num_pages[i]
            ratio = self._ratios[i]
            # Only the scheduler blocks whose expansion lands inside the
            # leaf's max_num_pages are read.
            # 译：只读取那些展开后落在叶子 max_num_pages 之内的调度器 block。
            num_blocks = min(src.shape[1], -(-max_num_pages // ratio))
            live = src[:actual_bs, :num_blocks].clamp_min(0)
            if ratio != 1:
                offsets = torch.arange(ratio, dtype=live.dtype, device=live.device)
                live = (live.unsqueeze(-1) * ratio + offsets).reshape(actual_bs, -1)
                live = torch.where(
                    live >= ratio, live, torch.zeros_like(live)
                )  # page 0 stays 0 across its ratio slots
                # 译：页 0 在它的 ratio 个位置上都保持为 0
            num_pages = min(live.shape[1], max_num_pages)
            dst[:actual_bs, :num_pages].copy_(live[:, :num_pages])
            dst[:actual_bs, num_pages:].zero_()
            dst[actual_bs:bs].zero_()

    def compute_decode_locations(
        self, bs: int, seq_lens: torch.Tensor, tokens_per_req: int
    ) -> None:
        """Refresh ``decode_locs[:, : bs * tokens_per_req]`` from the current
        stack and the live ``seq_lens`` (one launch for every group)."""
        # 译：根据当前堆叠和活跃的 ``seq_lens`` 刷新 ``decode_locs[:, : bs * tokens_per_req]``（所有组一次启动）。
        decode_write_locations(
            self.tables,
            self.page_sizes,
            seq_lens,
            self.decode_locs,
            bs,
            tokens_per_req,
            page_sizes_cpu=self.page_sizes_cpu,
        )

    def compute_window_locations(
        self, bs: int, window_starts: torch.Tensor, tokens_per_req: int
    ) -> None:
        """:meth:`compute_decode_locations` of the windows starting at ``window_starts``."""
        # 译：对从 ``window_starts`` 开始的窗口做 :meth:`compute_decode_locations`。
        window_write_locations(
            self.tables,
            self.page_sizes,
            window_starts,
            self.decode_locs,
            bs,
            tokens_per_req,
            page_sizes_cpu=self.page_sizes_cpu,
        )

    def extend_locations(
        self,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        total_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Every group's extend write slots over the current stack
        (``group_id -> [total_tokens]``): views of the persistent buffer when the
        span fits it, with the tail zeroed, else fresh tensors."""
        # 译：当前堆叠上每个组的 extend 写入槽位（``group_id -> [total_tokens]``）：段放得下持久缓冲区时是它的视图，
        #     尾部清零，否则是新分配的张量。
        # 例：A extend 300 个 token -> total_tokens=300，返回 {gid: extend_locs[0, :300]}，[300:] 清零；
        # 若某次 prefill 有 20000 个 token 而缓冲区只有 16384 列（假设值，由 max_extend_tokens 决定），
        # 就新分配张量（这次前向不能走 prefill graph）。
        self._extend_total = self._forward_total = total_tokens
        self._decode_tail = None
        if total_tokens <= self.extend_locs.shape[1]:
            out = self.extend_locs[:, :total_tokens]
            self.extend_locs[:, total_tokens:].zero_()
        else:
            out = torch.empty(
                (len(self.group_ids), total_tokens),
                dtype=torch.int32,
                device=self.tables.device,
            )
        locs = extend_write_locations(
            self.tables,
            self.page_sizes,
            extend_prefix_lens,
            extend_seq_lens,
            total_tokens,
            out,
            page_sizes_cpu=self.page_sizes_cpu,
        )
        self._extend_span = locs
        return {gid: locs[i] for i, gid in enumerate(self.group_ids)}

    def append_decode_rows(self, first_token: int, count: int) -> None:
        """Place a MIXED round's decode rows, ``decode_locs[:, first:first + count]``
        after its extend span, retaining the tail for padding past the buffer."""
        # 译：放置 MIXED 轮的 decode 行，即 ``decode_locs[:, first:first + count]``，接在其 extend 段之后；
        #     保留这段尾巴，用于超出缓冲区时的 padding。
        # 例：MIXED 批 A extend 300 + B decode（N=1）：router 调用 append_decode_rows(first_token=1, count=1)，
        # decode_locs[:, 0] 是 A 的（无用），decode_locs[:, 1] 是 B 的 -> 拷到 extend_locs[:, 300]，前向共 301 行。
        total = self._extend_total
        self._forward_total = total + count
        tail = self.decode_locs[:, first_token : first_token + count]
        self._decode_tail = tail
        if self._forward_total <= self.extend_locs.shape[1]:
            self.extend_locs[:, total : self._forward_total].copy_(tail)
            self.extend_locs[:, self._forward_total :].zero_()

    def padded_extend_span(self, group_id: str, rows: int) -> torch.Tensor:
        """The forward's rows (the extend span, then a MIXED round's decode rows)
        widened to ``rows`` with the dummy slot 0: a view of the buffer's zero
        tail where the graph-padded forward records it, else a fresh tensor for
        the eager forward wider than the buffer."""
        # 译：本次前向的行（extend 段，然后是 MIXED 轮的 decode 行）用哑槽位 0 加宽到 ``rows``：在 graph padding 的前向
        #     记录它的地方，是缓冲区零尾巴的视图；对比缓冲区更宽的 eager 前向，则是一个新分配的张量。
        total = self._forward_total
        if rows < total:
            raise ValueError(f"a span of {total} tokens cannot pad to {rows} rows")
        index = self._index[group_id]
        if rows <= self.extend_locs.shape[1]:
            return self.extend_locs[index, :rows]
        padded = torch.zeros(rows, dtype=torch.int32, device=self.tables.device)
        padded[: self._extend_total] = self._extend_span[index]
        if self._decode_tail is not None:
            padded[self._extend_total : total] = self._decode_tail[index]
        return padded
