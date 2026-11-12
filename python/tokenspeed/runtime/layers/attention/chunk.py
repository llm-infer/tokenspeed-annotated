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

# MLA 分块前缀重放（chunked prefix replay）的切块规划与槽位收集。使用者：MLA 系叶子后端
# （paged/flashmla.py、trtllm_mla.py、tokenspeed_mla.py、mla.py）。Qwen3（MHA）不经过这里，MLA 细节见第 12 阶段。
#
# 为什么要分块：MLA 的 KV cache 存的是压缩后的潜在向量（latent），prefill 时若有已缓存的前缀，
# 要把前缀的 latent 展开成逐头的 K/V 才能和新 token 的 query 做注意力。展开后的体积远大于 latent，
# 所以把所有请求的前缀切成若干“循环”（loop），每个循环最多展开 capacity 行：
#   capacity = chunked_prefill_size × mla_chunk_multiplier（默认 4）
# 每个循环算出一份部分注意力输出和 log-sum-exp，最后与新 token 部分的因果注意力结果合并（与 flash attention
# 分块合并的原理相同）。
#
# 流程：
#   plan_prefix_chunks（CPU）：决定每个循环里每个请求读前缀的哪一段 [starts, ends)
#       ▼
#   get_chunks_paged：把规划异步拷到 GPU；每个循环用 Triton kernel 按页表把位置换成 KV 槽位
#       ▼
#   build_chunked_prefill_metadata_arrays：整理成后端 prefill 元数据需要的数组
from dataclasses import dataclass
from functools import cached_property

import torch

from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.triton import tl, triton


# 对一个循环：每个 program 处理一个请求，把该请求本循环要读的前缀位置 [start, start + len) 换成 KV 槽位，
# 写到 chunk_kv_indices 中该请求的那一段（偏移为 cum_seq_lens[pid]）。
# slot = page_table[req, pos // P] × P + pos % P（与写入位置的公式相同）。
@triton.jit(
    do_not_specialize=["page_table_ptr_stride"],
    do_not_specialize_on_alignment=[
        "chunk_start_idx_ptr",
        "chunk_seq_lens_ptr",
        "chunk_cum_seq_lens_ptr",
    ],
)
def create_chunked_cache_kv_indices_paged(
    page_table_ptr,  # [bs, max_pages], row i == batch position i
    # 译：[bs, max_pages]，第 i 行就是批次位置 i
    chunk_start_idx_ptr,  # (batch_size,)
    chunk_seq_lens_ptr,  # (batch_size,)
    chunk_cum_seq_lens_ptr,  # (batch_size + 1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)
    page_table_ptr_stride,
    PAGE_SIZE: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    chunk_kv_indices_offset = tl.load(chunk_cum_seq_lens_ptr + pid)

    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        token_pos = chunk_start_pos + offset
        page_idx = token_pos // PAGE_SIZE
        page_id = tl.load(
            page_table_ptr + pid * page_table_ptr_stride + page_idx,
            mask=mask,
        )
        # 注意：page_id * PAGE_SIZE 与上面的 pid * page_table_ptr_stride 都是 int32 运算；
        # 只有 KV 池超过 2^31 个 token 槽位时才会溢出，实际不会触发，但与仓库“运行时规模的偏移用 int64”的规则不一致。
        kv_slot = page_id * PAGE_SIZE + token_pos % PAGE_SIZE
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset,
            kv_slot,
            mask=mask,
        )


def get_max_chunk_capacity():
    return (
        global_server_args_dict["chunked_prefill_size"]
        * global_server_args_dict["mla_chunk_multiplier"]
    )


@dataclass
class Chunks:
    starts: torch.Tensor
    ends: torch.Tensor
    len_in_chunk: torch.Tensor

    @cached_property
    def boundaries(self) -> torch.Tensor:
        """Per-request positions before the first loop and after each loop."""
        # 译：每个请求在第一个循环之前以及每个循环之后的位置。
        return torch.cat((self.ends.new_zeros((1, self.ends.shape[1])), self.ends))

    @cached_property
    def cum_seq_lens(self):
        num_chunks = self.starts.shape[0]
        bs = self.starts.shape[1]
        result = torch.zeros(
            num_chunks, bs + 1, device=self.starts.device, dtype=torch.int32
        )
        torch.cumsum(self.len_in_chunk, dim=1, out=result[:, 1:])
        return result


def chunking(prefix_lens: torch.Tensor, num_chunks, batch_size, chunk_len):
    """Equal-width chunks: every request reads ``chunk_len`` rows per loop.

    Loops that a request's prefix no longer reaches contribute zero rows for
    it, so the per-loop row count shrinks as short prefixes finish. Example,
    ``prefix_lens = [256, 512, 768, 1024]``, ``chunk_len = 256``,
    ``num_chunks = cdiv(1024, 256) = 4``::

                      req0  req1  req2  req3
        starts[0]        0     0     0     0     ends[0]   256  256  256  256
        starts[1]      256   256   256   256     ends[1]   256  512  512  512
        starts[2]      512   512   512   512     ends[2]   256  512  768  768
        starts[3]      768   768   768   768     ends[3]   256  512  768 1024
        len_in_chunk   [[256, 256, 256, 256], [0, 256, 256, 256],
                        [0, 0, 256, 256], [0, 0, 0, 256]]

    ``plan_prefix_chunks`` uses this layout when the per-request share of
    the capacity is small; otherwise it shares the capacity instead.
    """
    # 译：等宽的块：每个请求每个循环读 ``chunk_len`` 行。
    #
    #     一个请求的前缀已经够不到的循环为它贡献零行，所以随着短前缀读完，每个循环的行数会减少。
    #     例：``prefix_lens = [256, 512, 768, 1024]``，``chunk_len = 256``，``num_chunks = cdiv(1024, 256) = 4``：
    #
    #                       req0  req1  req2  req3
    #         starts[0]        0     0     0     0     ends[0]   256  256  256  256
    #         starts[1]      256   256   256   256     ends[1]   256  512  512  512
    #         starts[2]      512   512   512   512     ends[2]   256  512  768  768
    #         starts[3]      768   768   768   768     ends[3]   256  512  768 1024
    #         len_in_chunk   [[256, 256, 256, 256], [0, 256, 256, 256],
    #                         [0, 0, 256, 256], [0, 0, 0, 256]]
    #
    #     当每个请求分到的容量份额较小时，``plan_prefix_chunks`` 使用这种布局；否则改为共享容量。
    starts = (
        torch.arange(num_chunks, device=prefix_lens.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, batch_size)
        * chunk_len
    )
    ends = torch.min(prefix_lens.unsqueeze(0), starts + chunk_len).to(torch.int32)

    chunks = Chunks(
        starts=starts,
        ends=ends,
        len_in_chunk=(ends - starts).clamp(min=0).to(torch.int32),
    )
    return chunks


def plan_prefix_chunks(prefix_lens_cpu: torch.Tensor, capacity: int) -> Chunks:
    """Share a bounded expansion buffer among requests with remaining history.

    Args:
        prefix_lens_cpu: One nonnegative cached-prefix length per request on CPU.
        capacity: Maximum total history rows expanded in one loop.

    Returns:
        CPU chunk boundaries and lengths in request order. Every prefix is read
        exactly once; a finished request contributes no further rows.

    When the equal share ``capacity // batch_size`` is below 1024 rows, the
    plan falls back to ``chunking`` with that share as the chunk width: FP8
    attention rounding depends on the partition size, and pooling such short
    chunks into larger ones can amplify its error. Otherwise each loop hands
    the whole capacity to the requests that still have prefix rows, smallest
    remainder first: a request that fits within an equal share of what is
    left takes exactly its remainder, and the rest of the budget is split
    evenly (remainder rows go to the first requests) among the others.
    Example, ``prefix_lens = [900, 300, 9000]``, ``capacity = 6144``
    (share ``2048 >= 1024``, so the capacity is shared)::

        loop 0: active (by remainder) req1 300, req0 900, req2 9000
                6144 / 3 = 2048 >= 300  -> req1 takes 300, budget 5844
                5844 / 2 = 2922 >= 900  -> req0 takes 900, budget 4944
                4944 / 1 = 4944 <  9000 -> req2 takes 4944
        loop 1: active req2 (4056 left) -> takes 4056

                      req0  req1  req2
        starts[0]        0     0     0     ends[0]   900  300  4944
        starts[1]      900   300  4944     ends[1]   900  300  9000
        len_in_chunk   [[900, 300, 4944], [0, 0, 4056]]
        boundaries     [[0, 0, 0], [900, 300, 4944], [900, 300, 9000]]

    Equal-width chunks of ``6144 // 3 = 2048`` rows would need five loops
    for the same batch, with two requests idle in the last four.
    """
    # 译：在仍有剩余历史的请求之间共享一个有界的展开缓冲区。
    #
    #     参数：
    #         prefix_lens_cpu：每个请求一个非负的已缓存前缀长度，位于 CPU。
    #         capacity：一个循环中展开的历史行总数上限。
    #
    #     返回：
    #         按请求顺序排列的 CPU 端块边界和长度。每个前缀恰好被读一次；已读完的请求不再贡献行。
    #
    #     当平均份额 ``capacity // batch_size`` 小于 1024 行时，规划退回到以该份额为块宽的 ``chunking``：
    #     FP8 注意力的舍入取决于分区大小，把这样的短块合并成更大的块可能放大其误差。否则，每个循环把整个容量
    #     交给仍有前缀行的请求，剩余量最小的优先：剩余量不超过“剩下容量的平均份额”的请求恰好拿走它的剩余量，
    #     其余预算在其他请求之间平均分配（除不尽的行分给排在前面的请求）。
    #     例：``prefix_lens = [900, 300, 9000]``，``capacity = 6144``（份额 ``2048 >= 1024``，所以共享容量）：
    #
    #         loop 0：活跃请求（按剩余量）req1 300、req0 900、req2 9000
    #                 6144 / 3 = 2048 >= 300  -> req1 拿 300，预算剩 5844
    #                 5844 / 2 = 2922 >= 900  -> req0 拿 900，预算剩 4944
    #                 4944 / 1 = 4944 <  9000 -> req2 拿 4944
    #         loop 1：活跃 req2（还剩 4056）-> 拿 4056
    #
    #                       req0  req1  req2
    #         starts[0]        0     0     0     ends[0]   900  300  4944
    #         starts[1]      900   300  4944     ends[1]   900  300  9000
    #         len_in_chunk   [[900, 300, 4944], [0, 0, 4056]]
    #         boundaries     [[0, 0, 0], [900, 300, 4944], [900, 300, 9000]]
    #
    #     对同一批次，用 ``6144 // 3 = 2048`` 行的等宽块需要五个循环，而且后四个循环中有两个请求空闲。
    if prefix_lens_cpu.device.type != "cpu" or prefix_lens_cpu.ndim != 1:
        raise ValueError("prefix lengths must be a one-dimensional CPU tensor")
    batch_size = prefix_lens_cpu.numel()
    if capacity <= 0 or capacity < batch_size:
        raise ValueError("history capacity must be positive and cover the batch")
    lengths = prefix_lens_cpu.tolist()
    if any(length < 0 for length in lengths):
        raise ValueError("prefix lengths must be nonnegative")
    if not lengths:
        empty = torch.empty((0, 0), dtype=torch.int32)
        return Chunks(empty, empty, empty)
    share = capacity // batch_size
    # Keep small partitions stable: FP8 attention rounding depends on the
    # partition size, and pooling these short chunks can amplify its error.
    # 译：让小分区保持稳定：FP8 注意力的舍入取决于分区大小，合并这些短块可能放大其误差。
    if share < 1024:
        return chunking(
            prefix_lens_cpu, (max(lengths) + share - 1) // share, batch_size, share
        )
    consumed = [0] * batch_size
    starts, ends = [], []
    while consumed != lengths:
        remaining = [length - start for length, start in zip(lengths, consumed)]
        active = sorted(
            (i for i, length in enumerate(remaining) if length),
            key=remaining.__getitem__,
        )
        allocated = [0] * batch_size
        budget = capacity
        for offset, request in enumerate(active):
            share, remainder = divmod(budget, len(active) - offset)
            if remaining[request] <= share:
                allocated[request] = remaining[request]
                budget -= remaining[request]
            else:
                for position, rest in enumerate(active[offset:]):
                    allocated[rest] = share + (position < remainder)
                break
        starts.append(consumed)
        consumed = [start + size for start, size in zip(consumed, allocated)]
        ends.append(consumed)
    start_tensor = torch.tensor(starts, dtype=torch.int32).reshape(-1, batch_size)
    end_tensor = torch.tensor(ends, dtype=torch.int32).reshape(-1, batch_size)
    return Chunks(start_tensor, end_tensor, end_tensor - start_tensor)


def get_chunks_paged(prefix_lens, prefix_lens_cpu, page_table, page_size):
    """Chunk the cached prefixes and resolve each chunk's KV slots through
    the batch-ordered ``[bs, max_pages]`` kernel page table."""
    # 译：对已缓存前缀切块，并通过按批次排序的 ``[bs, max_pages]`` kernel 页表解析每个块的 KV 槽位。
    device: torch.device = prefix_lens.device
    batch_size = len(prefix_lens_cpu)
    chunks_cpu = plan_prefix_chunks(prefix_lens_cpu, get_max_chunk_capacity())
    num_chunks = chunks_cpu.starts.shape[0]
    # Stage the host plan in pinned memory so the device copy is asynchronous:
    # prefill planning must not synchronize (a copy from pageable memory
    # would), and fresh pinned storage cannot be refilled under an in-flight
    # copy.
    # 译：把 Host 端规划暂存在锁页内存中，使拷到设备是异步的：prefill 规划不能同步（从可分页内存拷贝会同步），
    #     而且新的锁页存储不能在一次进行中的拷贝下被重新填充。
    host_arrays = torch.empty(
        (3, num_chunks, batch_size), dtype=torch.int32, pin_memory=True
    )
    torch.stack(
        (chunks_cpu.starts, chunks_cpu.ends, chunks_cpu.len_in_chunk),
        out=host_arrays,
    )
    arrays = host_arrays.to(device=device, non_blocking=True)
    chunks = Chunks(*arrays.unbind(0))

    num_tokens_per_forward = chunks_cpu.len_in_chunk.sum(dim=1).tolist()

    chunk_kv_indices_list = []
    for idx in range(num_chunks):
        chunk_kv_indices = torch.empty(
            num_tokens_per_forward[idx], dtype=torch.int32, device=device
        )
        create_chunked_cache_kv_indices_paged[(batch_size,)](
            page_table,
            chunks.starts[idx],
            chunks.len_in_chunk[idx],
            chunks.cum_seq_lens[idx],
            chunk_kv_indices,
            page_table.stride(0),
            page_size,
        )
        chunk_kv_indices_list.append(chunk_kv_indices)

    return chunks, chunk_kv_indices_list, chunks_cpu


def build_chunked_prefill_metadata_arrays(
    extend_prefix_lens,
    extend_prefix_lens_cpu,
    page_table,
    page_size,
):
    """Build the per-prefix-loop arrays for chunked-prefill MLA.

    Run once per chunked-prefill iteration in the backend's
    ``_init_prefill_metadata``; ``page_table`` is the batch-ordered
    ``[num_extends, max_pages]`` kernel page table. Returns:

    - ``chunked_loop_num``: number of prefix loop iterations
    - ``chunk_kv_indices_list``: List[Tensor], paged KV indices per loop_idx
    - ``chunked_seq_len``: (chunked_loop_num, num_extends) int32 GPU — per-seq
      KV length within each loop_idx (zero for seqs whose prefix doesn't
      reach this chunk).
    - ``cu_chunked_seq_len``: (chunked_loop_num, num_extends+1) int32 GPU —
      cumsum along the seq dim, fed to the chunker as ``cum_seq_lens_kv``.
    - ``max_chunk_len_per_loop``: List[int], CPU max-seq-len per loop_idx,
      fed to the chunker as ``max_kv_len``.
    - ``chunk_bounds_cpu``: (chunked_loop_num+1, num_extends) CPU positions,
      shared with the distributed history gather.

    The q-side cumsum / max do not appear here: callers alias them to the
    causal pass's ``cum_extend_seq_lens`` / ``max_extend_seq_len``, since
    every prefix-chunk forward sees the same ``q_lens == extend_seq_lens``.
    """
    # 译：为分块 prefill 的 MLA 构建每个前缀循环的数组。
    #
    #     在后端的 ``_init_prefill_metadata`` 中每次分块 prefill 迭代运行一次；``page_table`` 是按批次排序的
    #     ``[num_extends, max_pages]`` kernel 页表。返回：
    #
    #     - ``chunked_loop_num``：前缀循环的迭代次数
    #     - ``chunk_kv_indices_list``：List[Tensor]，每个 loop_idx 的分页 KV 下标
    #     - ``chunked_seq_len``：(chunked_loop_num, num_extends) int32 GPU——每个 loop_idx 内每个序列的 KV 长度
    #       （前缀够不到这个块的序列为零）。
    #     - ``cu_chunked_seq_len``：(chunked_loop_num, num_extends+1) int32 GPU——沿序列维的前缀和，
    #       作为 ``cum_seq_lens_kv`` 传给分块计算。
    #     - ``max_chunk_len_per_loop``：List[int]，每个 loop_idx 在 CPU 上的最大序列长度，作为 ``max_kv_len`` 传给分块计算。
    #     - ``chunk_bounds_cpu``：(chunked_loop_num+1, num_extends) CPU 位置，与分布式历史收集共享。
    #
    #     q 侧的前缀和 / 最大值不在这里：调用方把它们直接设为因果那一遍的 ``cum_extend_seq_lens`` / ``max_extend_seq_len``，
    #     因为每次前缀块前向看到的都是同样的 ``q_lens == extend_seq_lens``。
    chunks, chunk_kv_indices_list, chunks_cpu = get_chunks_paged(
        extend_prefix_lens,
        extend_prefix_lens_cpu,
        page_table,
        page_size,
    )
    chunked_loop_num = chunks.starts.shape[0]
    max_chunk_len_per_loop = [
        chunks_cpu.len_in_chunk[i].max().item() for i in range(chunked_loop_num)
    ]
    return (
        chunked_loop_num,
        chunk_kv_indices_list,
        chunks.len_in_chunk,
        chunks.cum_seq_lens,
        max_chunk_len_per_loop,
        chunks_cpu.boundaries,
    )
