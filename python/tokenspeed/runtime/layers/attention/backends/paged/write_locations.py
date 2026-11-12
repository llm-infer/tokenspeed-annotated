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

"""KV write-location math over stacked per-group kernel page tables.

One invariant, three shapes: ``slot = table[req, pos // P] * P + pos % P``
with ``P`` the group's kernel page size. Decode writes a trailing window of
``n`` positions per request (``n == 1`` plain decode, ``n == N`` target
verify or a block draft), extend writes the ``[prefix, prefix + new)`` span
per request in q/k/v token order. Null and hole pages (id <= 0) and positions
past the table route to slot 0, the zero-initialized dummy page that never
aliases a live request.

Every function takes the stacked ``[G, max_bs, max_num_pages]`` table the
router fills (``group_tables.GroupTableStacks``) so all groups' locations
come from one launch. CUDA tensors run the triton kernels; CPU and NPU
tensors take the Torch path. An optional immutable host page-size mirror
avoids reading these static constants back from the device.
"""

# 译：基于堆叠的按组 kernel 页表计算 KV 写入位置。
#
#     一个不变式，三种形状：``slot = table[req, pos // P] * P + pos % P``，其中 ``P`` 是该组的 kernel 页大小。
#     decode 为每个请求写入末尾 ``n`` 个位置的窗口（``n == 1`` 为普通 decode，``n == N`` 为目标模型验证或块草稿），
#     extend 为每个请求按 q/k/v 的 token 顺序写入 ``[prefix, prefix + new)`` 这一段。空页和空洞页（id <= 0）以及
#     超出页表的位置都被导向槽位 0，即那个初始化为零、永远不会与活跃请求共用的哑页。
#
#     每个函数都接收由 router 填充的、堆叠的 ``[G, max_bs, max_num_pages]`` 页表（``group_tables.GroupTableStacks``），
#     所以所有组的位置由一次启动算出。CUDA 张量运行 triton kernel；CPU 和 NPU 张量走 Torch 路径。一个可选的、
#     不可变的 Host 端页大小镜像可以避免从设备读回这些静态常量。

# 计算每个新 token 的 KV 写到哪个槽位（slot）。
#
# 你在这里：每次前向开始时，CacheGroupRouter 准备元数据时调用（backends/paged/router.py），结果存在 router 的
# 固定缓冲区里；随后每一层的注意力 prologue 按这些槽位把本层的 K/V 写进 KV 池。
#
# slot = 页号 × P + 页内偏移 这条公式是熟悉的。本文件的不同之处：
#   1. 在 GPU 上由 Triton kernel 直接从页表和序列长度算出，不在 CPU 上逐个拼接再拷贝
#   2. 一次 kernel 启动同时算出所有 cache group 的写入位置（每个组的页大小可以不同；Qwen3 只有 1 组）
#   3. decode 支持每个请求写多个 token（投机解码的验证窗口，第 11 阶段）
#   4. 注意：没有“-1 表示不写”。页号 <= 0 的空页、以及位置超出页表的情况（例如 CUDA Graph 的 padding 行）
#      都写到 slot 0。第 0 页是保留的哑页，从不分配给真实请求，所以写进去的东西不会污染任何请求的 KV。
#
# 贯穿例子（P=64）：
#   A：extend，prefix=0，新 token 300 个 -> 位置 0..299，占 5 个页（300 = 4×64 + 44）。
#      若 A 的页表为 [12, 3, 40, 7, 21]，第 299 个 token：页序号 4、偏移 43 -> 页 21 -> slot = 21×64 + 43 = 1387
#   B：decode，seq_len=1001（含本次新 token）-> 位置 1000：页序号 15、偏移 40 -> slot = B 的页表[15]×64 + 40
from __future__ import annotations

import torch

from tokenspeed.runtime.utils.triton import tl, triton


# decode 写入位置的 kernel。grid = (组数, ceil(总 token 数 / BLOCK))，每个 program 处理一个组的 BLOCK 个 token。
# 第 tok 个输出属于第 tok // n 个请求的第 tok % n 步，位置 = seq_len + seq_len_offset - n + step。
# do_not_specialize：这几个参数每批都会变，不让 Triton 按它们的值分别编译（避免运行时反复编译）。
# 注意偏移计算先转成 int64，最后存成 int32。
@triton.jit(do_not_specialize=["num_tokens", "tokens_per_req", "seq_len_offset"])
def _decode_locs_kernel(
    tables_ptr,  # [G, max_bs, max_num_pages] int32
    page_sizes_ptr,  # [G] int32
    seq_lens_ptr,  # [bs] int32
    out_ptr,  # [G, cap] int32
    stride_g,
    stride_b,
    max_num_pages,
    out_stride_g,
    num_tokens,  # bs * tokens_per_req
    tokens_per_req,
    seq_len_offset,  # added to every loaded length
    # 译：加到每个读入的长度上
    BLOCK: tl.constexpr,
):
    g = tl.program_id(0)
    start = tl.program_id(1) * BLOCK
    tok = start + tl.arange(0, BLOCK)  # token-major index into out[g]
    # 译：out[g] 中以 token 为主序的下标
    mask = tok < num_tokens
    req = tok // tokens_per_req
    step = tok % tokens_per_req
    seq_len = tl.load(seq_lens_ptr + req, mask=mask, other=1).to(tl.int64)
    pos = tl.maximum(seq_len + seq_len_offset - tokens_per_req + step, 0)
    page_size = tl.load(page_sizes_ptr + g).to(tl.int64)
    page_idx = pos // page_size
    # 超出页表宽度的位置，以及空页（页号 <= 0），都写到 slot 0（哑页）
    overflow = page_idx >= max_num_pages
    page_idx = tl.minimum(page_idx, max_num_pages - 1)
    page = tl.load(
        tables_ptr + g * stride_g + req * stride_b + page_idx, mask=mask, other=0
    ).to(tl.int64)
    loc = page * page_size + pos % page_size
    loc = tl.where(overflow | (page <= 0), 0, loc)
    tl.store(out_ptr + g * out_stride_g + tok, loc.to(tl.int32), mask=mask)


# extend 写入位置的 kernel。grid = (组数, 请求数)，每个 program 处理一个组的一个请求，
# 按 BLOCK 循环覆盖该请求的全部新 token：位置 = prefix + 0..num_new-1，输出写到 cu_extend_lens[req] 开始的那一段。
@triton.jit(do_not_specialize=["max_num_pages", "out_stride_g"])
def _extend_locs_kernel(
    tables_ptr,  # [G, max_bs, max_num_pages] int32
    page_sizes_ptr,  # [G] int32
    prefix_lens_ptr,  # [bs] int32
    extend_lens_ptr,  # [bs] int32
    cu_extend_lens_ptr,  # [bs + 1] int32
    out_ptr,  # [G, total] int32
    stride_g,
    stride_b,
    max_num_pages,
    out_stride_g,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(0)
    req = tl.program_id(1)
    prefix = tl.load(prefix_lens_ptr + req).to(tl.int64)
    num_new = tl.load(extend_lens_ptr + req)
    out_base = tl.load(cu_extend_lens_ptr + req).to(tl.int64)
    page_size = tl.load(page_sizes_ptr + g).to(tl.int64)
    for block_start in range(0, num_new, BLOCK):
        offs = block_start + tl.arange(0, BLOCK)
        mask = offs < num_new
        pos = prefix + offs
        page_idx = pos // page_size
        overflow = page_idx >= max_num_pages
        page_idx = tl.minimum(page_idx, max_num_pages - 1)
        page = tl.load(
            tables_ptr + g * stride_g + req * stride_b + page_idx,
            mask=mask,
            other=0,
        ).to(tl.int64)
        loc = page * page_size + pos % page_size
        loc = tl.where(overflow | (page <= 0), 0, loc)
        tl.store(
            out_ptr + g * out_stride_g + out_base + offs, loc.to(tl.int32), mask=mask
        )


# 下面两个是 Torch 参考实现（CPU / NPU 使用），逻辑与上面的 Triton kernel 相同
def _decode_positions(seq_lens: torch.Tensor, tokens_per_req: int) -> torch.Tensor:
    """``[bs * n]`` token-major write positions of the trailing window."""
    # 译：末尾窗口中、以 token 为主序的 ``[bs * n]`` 个写入位置。
    n = tokens_per_req
    lens = seq_lens.to(torch.int64)
    if n == 1:
        return (lens - 1).clamp_min(0)
    steps = torch.arange(n, device=seq_lens.device, dtype=torch.int64)
    return (lens.unsqueeze(1) - n + steps).clamp_min(0).reshape(-1)


def _gather_slots(
    table: torch.Tensor, req: torch.Tensor, pos: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Torch reference of the slot rule for one group's
    ``[max_bs, max_num_pages]`` table."""
    # 译：对一个组的 ``[max_bs, max_num_pages]`` 页表，槽位规则的 Torch 参考实现。
    max_num_pages = table.shape[1]
    page_idx = pos // page_size
    overflow = page_idx >= max_num_pages
    page = table[req, page_idx.clamp_max(max_num_pages - 1)].to(torch.int64)
    loc = page * page_size + pos % page_size
    return torch.where(overflow | (page <= 0), torch.zeros_like(loc), loc).to(
        torch.int32
    )


def decode_write_locations(
    tables: torch.Tensor,
    page_sizes: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    bs: int,
    tokens_per_req: int,
    *,
    page_sizes_cpu: tuple[int, ...] | None = None,
) -> None:
    """Fill ``out[:, : bs * tokens_per_req]`` with every group's decode
    write slots.

    Args:
        tables: ``[G, max_bs >= bs, max_num_pages]`` int32 stacked kernel page
            tables.
        page_sizes: ``[G]`` int32 kernel page size per group.
        seq_lens: ``[>= bs]`` int32 lengths including the newest token(s);
            request ``b`` writes positions ``seq_lens[b] - n .. seq_lens[b] - 1``.
        out: ``[G, cap]`` int32 destination, ``cap >= bs * tokens_per_req``;
            filled token-major per request (the verify layout).
        bs: Requests to compute (padded requests read seq_len 1 and land on
            slot 0).
        tokens_per_req: Trailing window width ``n``.
        page_sizes_cpu: Optional host tuple from the same group specs as
            ``page_sizes``. Used only by the Torch fallback; absent keeps
            the original device-scalar path.
    """
    # 译：把每个组的 decode 写入槽位填进 ``out[:, : bs * tokens_per_req]``。
    #
    #     参数：
    #         tables：``[G, max_bs >= bs, max_num_pages]``，int32 的堆叠 kernel 页表。
    #         page_sizes：``[G]``，每个组的 int32 kernel 页大小。
    #         seq_lens：``[>= bs]``，int32 的长度，包含最新的 token；请求 ``b`` 写入位置 ``seq_lens[b] - n .. seq_lens[b] - 1``。
    #         out：``[G, cap]`` 的 int32 目标，``cap >= bs * tokens_per_req``；按请求、以 token 为主序填充（验证的布局）。
    #         bs：要计算的请求数（padding 的请求读到 seq_len 1，落在槽位 0）。
    #         tokens_per_req：末尾窗口的宽度 ``n``。
    #         page_sizes_cpu：可选的 Host 元组，来自与 ``page_sizes`` 相同的组 spec。只用于 Torch 回退路径；
    #             不提供时保留原来的设备标量路径。
    _write_locations(
        tables,
        page_sizes,
        seq_lens,
        out,
        bs,
        tokens_per_req,
        0,
        page_sizes_cpu=page_sizes_cpu,
    )


# 与 decode 相同，只是给出的是窗口起点而不是序列长度（投机解码的草稿步使用）
def window_write_locations(
    tables: torch.Tensor,
    page_sizes: torch.Tensor,
    window_starts: torch.Tensor,
    out: torch.Tensor,
    bs: int,
    tokens_per_req: int,
    *,
    page_sizes_cpu: tuple[int, ...] | None = None,
) -> None:
    """:func:`decode_write_locations` of the windows ``start .. start + n - 1`` for ``window_starts[:bs]``."""
    # 译：对 ``window_starts[:bs]`` 的窗口 ``start .. start + n - 1`` 执行 :func:`decode_write_locations`。
    _write_locations(
        tables,
        page_sizes,
        window_starts,
        out,
        bs,
        tokens_per_req,
        tokens_per_req,
        page_sizes_cpu=page_sizes_cpu,
    )


def _validate_cpu_page_sizes(page_sizes_cpu, num_groups):
    if page_sizes_cpu is None:
        return
    if not isinstance(page_sizes_cpu, tuple) or len(page_sizes_cpu) != num_groups:
        raise ValueError("host page sizes must be a tuple matching the cache groups")
    if any(type(size) is not int or size <= 0 for size in page_sizes_cpu):
        raise ValueError("host page sizes must contain positive integers")


# decode 写入位置的共同实现：GPU 上启动 Triton kernel；否则走 Torch 路径。
# out 是 router 的固定缓冲区，容量不够时直接报错（说明预分配的大小与本批不匹配）。
def _write_locations(
    tables: torch.Tensor,
    page_sizes: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    bs: int,
    tokens_per_req: int,
    seq_len_offset: int,
    *,
    page_sizes_cpu: tuple[int, ...] | None = None,
) -> None:
    n = max(int(tokens_per_req), 1)
    num_tokens = bs * n
    if num_tokens == 0 or tables.shape[0] == 0:
        return
    if out.shape[1] < num_tokens:
        raise RuntimeError(
            f"decode write-location buffer holds {out.shape[1]} slots per group, "
            f"need {num_tokens} (bs={bs}, tokens_per_req={n})"
        )
    if tables.is_cuda:
        block = 128
        grid = (tables.shape[0], triton.cdiv(num_tokens, block))
        _decode_locs_kernel[grid](
            tables,
            page_sizes,
            seq_lens,
            out,
            tables.stride(0),
            tables.stride(1),
            tables.shape[2],
            out.stride(0),
            num_tokens,
            n,
            seq_len_offset,
            BLOCK=block,
        )
        return
    _validate_cpu_page_sizes(page_sizes_cpu, tables.shape[0])
    pos = _decode_positions(seq_lens[:bs].to(torch.int64) + seq_len_offset, n)
    req = torch.arange(bs, device=tables.device).repeat_interleave(n)
    for g in range(tables.shape[0]):
        page_size = int(page_sizes[g]) if page_sizes_cpu is None else page_sizes_cpu[g]
        out[g, :num_tokens] = _gather_slots(tables[g], req, pos, page_size)


def extend_write_locations(
    tables: torch.Tensor,
    page_sizes: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    total_tokens: int,
    out: torch.Tensor,
    *,
    page_sizes_cpu: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Every group's extend write slots, ``[G, total_tokens]`` in q/k/v
    token order (request-major, ``cu_extend_seq_lens`` layout), into ``out``.

    Args:
        tables: ``[G, max_bs >= bs, max_num_pages]`` int32 stacked kernel page
            tables.
        page_sizes: ``[G]`` int32 kernel page size per group.
        extend_prefix_lens: ``[bs]`` int32 cached prefix per request.
        extend_seq_lens: ``[bs]`` int32 new tokens per request.
        total_tokens: ``sum(extend_seq_lens)`` from the host mirror — no
            device sync here.
        out: ``[G, total_tokens]`` int32 rows to fill; a view of the router's
            persistent buffer, so a captured graph can record its address.

    Returns:
        ``out``. The optional ``page_sizes_cpu`` host tuple replaces only
        static page-size scalar reads on the Torch fallback.
    """
    # 译：每个组的 extend 写入槽位，``[G, total_tokens]``，按 q/k/v 的 token 顺序（以请求为主序，``cu_extend_seq_lens`` 布局），
    #     写入 ``out``。
    #
    #     参数：
    #         tables：``[G, max_bs >= bs, max_num_pages]``，int32 的堆叠 kernel 页表。
    #         page_sizes：``[G]``，每个组的 int32 kernel 页大小。
    #         extend_prefix_lens：``[bs]``，int32，每个请求已缓存的前缀长度。
    #         extend_seq_lens：``[bs]``，int32，每个请求的新 token 数。
    #         total_tokens：来自 Host 镜像的 ``sum(extend_seq_lens)``——这里不做设备同步。
    #         out：要填充的 ``[G, total_tokens]`` 个 int32 行；是 router 持久缓冲区的一个视图，所以捕获的 graph 可以记录它的地址。
    #
    #     返回：
    #         ``out``。可选的 ``page_sizes_cpu`` Host 元组只替换 Torch 回退路径上对静态页大小标量的读取。
    bs = extend_seq_lens.shape[0]
    if total_tokens == 0 or tables.shape[0] == 0 or bs == 0:
        return out
    # cu_extend：新 token 数的前缀和，决定每个请求的写入位置在输出中从哪里开始。
    # 例：A 300 个新 token、只有 A 做 extend -> cu_extend = [0, 300]
    cu_extend = torch.zeros(bs + 1, dtype=torch.int32, device=tables.device)
    torch.cumsum(extend_seq_lens, dim=0, out=cu_extend[1:])
    if tables.is_cuda:
        _extend_locs_kernel[(tables.shape[0], bs)](
            tables,
            page_sizes,
            extend_prefix_lens,
            extend_seq_lens,
            cu_extend,
            out,
            tables.stride(0),
            tables.stride(1),
            tables.shape[2],
            out.stride(0),
            BLOCK=128,
        )
        return out
    _validate_cpu_page_sizes(page_sizes_cpu, tables.shape[0])
    lens = extend_seq_lens.to(torch.int64)
    req = torch.arange(bs, device=tables.device).repeat_interleave(lens)
    starts = cu_extend[:-1].to(torch.int64).repeat_interleave(lens)
    pos = extend_prefix_lens.to(torch.int64)[req] + (
        torch.arange(total_tokens, device=tables.device) - starts
    )
    for g in range(tables.shape[0]):
        page_size = int(page_sizes[g]) if page_sizes_cpu is None else page_sizes_cpu[g]
        out[g] = _gather_slots(tables[g], req, pos, page_size)
    return out
