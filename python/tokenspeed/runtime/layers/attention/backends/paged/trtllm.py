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

"""
MHA attention leaf for TokenSpeed scheduling.
Uses fused kernels optimized for SM100 (Blackwell).
Supports sliding window, attention sinks, and FP8 KV cache.
"""

# 译：面向 TokenSpeed 调度的 MHA 注意力叶子。
#     使用为 SM100（Blackwell）优化的融合 kernel。
#     支持滑动窗口、attention sink 和 FP8 KV cache。

# TRTLLMMHAAttnBackend：--attention-backend trtllm 时的 MHA 叶子，调用 TRT-LLM 的 trtllm-gen kernel（经 FlashInfer
# 封装，由 tokenspeed_kernel 引入）。接口与 mha.py 的 MHAAttnBackend 相同，读法可对照 mha.py。
#
# 与 mha.py 的差异：
#   1. kernel 页大小固定为 64（default_kernel_page_size = TRTLLM_MHA_PAGE_SIZE）。若 cache group 的 block 粒度
#      更大（例如 128），router 会把每个 block 展开成多个 kernel 页（ratio = 2）；mha.py 则直接用 block 粒度
#   2. KV cache 布局按 kernel 要求看成 HND：[页数, KV 头数, 页大小, head_dim]（view + permute，不拷贝）
#   3. extend 总是走 trtllm_batch_context_with_kv_cache（从 cache 读 K/V），没有“全新 prefill 直接用 q/k/v”的分支
#   4. 两个元数据槽：forward_prefill_metadata（extend / mixed，以及投机验证）与 forward_decode_metadata（单 token decode）
#      投机验证虽然是 DECODE 模式，却读 prefill 槽：每个请求 N 个 query，cu_seqlens_q 是步长为 N 的等差数列
#   5. 需要一块共享 workspace，以及 multi-CTA KV 模式的计数器缓冲区（见 __init__）
#
# Qwen3-8B 单卡：q [T, 32, 128]，k_cache [NP, 8, 64, 128]（permute 后），输出 [T, 4096]。
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.mha.flashinfer import (
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
    trtllm_batch_context_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
)
from tokenspeed_kernel.ops.attention.tree import tree_window_attention

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.execution.workspace import workspace_pool
from tokenspeed.runtime.layers.attention.backends.base import reject_query_shard
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeCascadeRows,
    TreeDraftInputs,
)
from tokenspeed.runtime.layers.attention.backends.support import TreeSupport
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    TRTLLM_MHA_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.common import fp8_cast_contiguous
from tokenspeed.runtime.utils.env import envs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:
    """Adjust degenerate strides for a tensor, make it canonical.

    When a dimension has size=1, PyTorch may use the same stride as the next dim.
    This causes TMA desc validation failures in the TRT-LLM MHA kernels.
    See: https://github.com/flashinfer-ai/flashinfer/issues/2232
    """
    # 译：调整张量的退化步长，使其规范化。
    #
    #     当某个维度 size=1 时，PyTorch 可能让它使用与下一维相同的步长。这会导致 TRT-LLM MHA kernel 的 TMA 描述符校验失败。
    #     见：https://github.com/flashinfer-ai/flashinfer/issues/2232
    # 例：KV 头数为 1 时，permute 后的 [NP, 1, 64, 128] 第 1 维 size=1，其步长可能等于第 2 维的步长；
    # 这里按行主序重新计算步长（size=1 维的步长不影响寻址，所以 as_strided 不改变数据）。
    sizes = tensor.size()
    strides = tensor.stride()
    ndim = tensor.dim()

    need_fix = any(
        sizes[i] == 1 and strides[i] == strides[i + 1] for i in range(ndim - 1)
    )

    if not need_fix:
        return tensor

    new_strides = [0] * ndim
    new_strides[-1] = 1
    for i in range(ndim - 2, -1, -1):
        new_strides[i] = new_strides[i + 1] * sizes[i + 1]

    return tensor.as_strided(sizes, new_strides)


# 字段与 mha.py 的 MHAExtendMetadata 对应：cache_seqlens_int32 = 每个请求的 KV 总长度，
# cu_seqlens_q / cu_seqlens_k = query / KV 长度的前缀和，max_seq_len_q = 单请求最大 query 行数。
@dataclass
class TRTLLMMHAMetadata:
    cache_seqlens_int32: torch.Tensor = None
    max_seq_len_q: int = 1
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None


class TRTLLMMHAAttnBackend(PagedAttentionBackend):
    """The ``trtllm`` MHA leaf: TRT-LLM fused kernels for SM100 (Blackwell)."""

    # 译：``trtllm`` MHA 叶子：用于 SM100（Blackwell）的 TRT-LLM 融合 kernel。

    default_kernel_page_size = TRTLLM_MHA_PAGE_SIZE
    # Both kernel call sites forward layer.sliding_window_size.
    # 译：两个 kernel 调用点都传入 layer.sliding_window_size。
    supports_layer_sliding_window: bool = True

    def __init__(self, config: AttnConfig, spec: MHAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)

        self.tp_q_head_num = max(spec.num_attention_heads // spec.attn_tp_size, 1)
        self.tp_kv_head_num = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.kv_cache_dtype = config.kv_cache_dtype

        self._workspace_pool = workspace_pool(config.device)
        self._workspace_nbytes = envs.TOKENSPEED_WORKSPACE_TRTLLM_MHA_MB.get() * (
            1 << 20
        )
        # Warm the shared block to this backend's peak now: graph capture runs
        # the forward with the pool frozen, and under --disable-autotune no
        # earlier forward will have grown the block by then.
        # 译：现在就把共享的块预热到本后端的峰值：graph 捕获在 pool 冻结的状态下运行前向，
        #     而在 --disable-autotune 下，到那时还没有更早的前向把块扩大过。
        self._workspace_pool.allocate(((self._workspace_nbytes,), torch.uint8))

        # DFLASH draft: the drafter predicts a whole block of spec_num_tokens
        # per decode forward and needs non-causal (block-diffusion) attention.
        # Instead of a non-causal mask, materialize one single-query decode
        # metadata entry per block position (block_decode_expansion), all
        # sharing the SAME block-end seq_len, so each block position attends
        # over the whole block; target verify and plain decode are untouched.
        # 译：DFLASH 草稿：草稿模型每次 decode 前向预测一整块 spec_num_tokens 个 token，需要非因果（块扩散）注意力。
        #     不使用非因果掩码，而是为每个块内位置物化一个单 query 的 decode 元数据条目（block_decode_expansion），
        #     它们共享“同一个”块末尾 seq_len，所以每个块内位置都关注整个块；目标模型验证和普通 decode 不受影响。
        self.draft_block_decode = bool(config.draft_block_decode)

        # trtllm-gen's multi-CTA KV mode keeps one completion counter per
        # (request, q head). Without a counter buffer FlashInfer zero-fills a
        # fresh one on every call, one extra kernel per layer. The kernel
        # resets its counters at the end of each launch, so a buffer zeroed
        # once here serves every call; a batch it cannot hold falls back to
        # FlashInfer's own allocation. It is pool-independent, so a rebind
        # keeps it.
        # 译：trtllm-gen 的 multi-CTA KV 模式为每个（请求，q 头）保留一个完成计数器。没有计数器缓冲区时，FlashInfer 每次调用
        #     都会新建并清零一个，每层多一个 kernel。kernel 在每次启动结束时会重置计数器，所以这里清零一次的缓冲区可以服务
        #     每次调用；装不下的批次回退到 FlashInfer 自己的分配。它与 pool 无关，所以重新绑定时保留它。
        self._kv_counter_buffer = None
        self._max_bs = config.max_bs
        if torch.device(config.device).type == "cuda":
            self._sm_count = torch.cuda.get_device_properties(
                config.device
            ).multi_processor_count
            self._kv_counter_buffer = torch.zeros(
                get_trtllm_gen_multi_ctas_kv_counter_bytes(
                    config.max_bs * self.block_decode_expansion,
                    self.tp_q_head_num,
                    self._sm_count,
                ),
                dtype=torch.uint8,
                device=config.device,
            )

        # Separate slots for prefill-kernel vs decode-kernel forward paths:
        # forward_extend reads prefill; forward_decode picks by q_len (target
        # verify is DECODE mode but rides the prefill slot's uniform stride).
        # 译：prefill kernel 与 decode kernel 两条前向路径各用一个槽：forward_extend 读 prefill 槽；forward_decode 按 q_len 选择
        #     （目标模型验证是 DECODE 模式，却借用 prefill 槽的统一步长）。
        self.forward_prefill_metadata: TRTLLMMHAMetadata | None = None
        self.forward_decode_metadata: TRTLLMMHAMetadata | None = None

        # KV seqlens clamped to >= spec_num_tokens for the MTP verify path.
        # Padded decode requests have seq_len=1; with q_len=spec_num_tokens
        # they'd hit an empty causal span and the kernel returns NaN.
        # 译：为 MTP 验证路径截断到 >= spec_num_tokens 的 KV 序列长度。padding 的 decode 请求 seq_len=1；
        #     当 q_len=spec_num_tokens 时，它们会遇到空的因果区间，kernel 返回 NaN。
        self.spec_cache_seqlens_buf: torch.Tensor | None = None
        # Pure aranges per bs: pool-independent, so a rebind keeps them.
        # 译：每个 bs 的纯 arange：与 pool 无关，所以重新绑定时保留它们。
        self._cu_seqlens_by_bs: dict[int, torch.Tensor] = {}
        self._verify_views_by_bs: dict[int, TRTLLMMHAMetadata] = {}

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self.forward_prefill_metadata = None
        self.forward_decode_metadata = None
        self.spec_cache_seqlens_buf = None
        self._verify_views_by_bs = {}

    def supports_narrowed_draft_decode(self, forward_mode: ForwardMode) -> bool:
        del forward_mode
        return True

    # ------------------------------------------------------------------
    # Metadata initialisation
    # ------------------------------------------------------------------
    # 译：元数据初始化

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        query_shard: QueryShardPlan | None,
        page_table_cpu: torch.Tensor | None,
        **kwargs,
    ):
        reject_query_shard(query_shard, "TRTLLMMHAAttnBackend")
        del page_table_cpu
        if not forward_mode.is_extend_or_mixed():
            raise RuntimeError(
                "trtllm decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend ({forward_mode})"
            )
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        # Draft decode metadata is refreshed separately, so its prefill slot
        # only describes the leading extend requests. Target MHA still sends
        # the full packed batch through forward_extend.
        # 译：草稿的 decode 元数据单独刷新，所以它的 prefill 槽只描述排在前面的 extend 请求。
        #     目标 MHA 仍把完整的打包批次送进 forward_extend。
        prefill_bs = num_extends if self.is_draft else bs
        cache_seqlens_int32 = seq_lens[:prefill_bs]
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seq_lens[:prefill_bs], dim=0, dtype=torch.int32), (1, 0)
        )

        # Read the max from the pinned-CPU mirror — avoids a per-iter
        # GPU->CPU sync that would block the host on the previous step's
        # forward and erase prefill/decode overlap. Both branches want
        # max(new tokens per request); for a no-prefix extend that's
        # seq_lens, for a prefix-cached extend it's seq_lens-prefix_lens —
        # extend_seq_lens_cpu holds those new-token counts in either case.
        # 译：从锁页的 CPU 镜像读取最大值——避免每轮一次 GPU->CPU 同步，那会让 Host 阻塞等待上一步的前向，抹掉 prefill/decode 的重叠。
        #     两个分支要的都是 max(每个请求的新 token 数)；无前缀 extend 时就是 seq_lens，带前缀缓存的 extend 时是
        #     seq_lens-prefix_lens——两种情况下 extend_seq_lens_cpu 存的都是这些新 token 数。
        max_seq_len_q = int(extend_seq_lens_cpu[:prefill_bs].max().item())

        if prefill_bs > num_extends:
            # A target MIXED round: each decode request after the extend
            # requests queries only its spec_num_tokens new rows (1 without
            # speculation) at the end of its cache, as in a decode step.
            # 译：目标模型的 MIXED 轮：排在 extend 请求之后的每个 decode 请求，只在其 cache 末尾查询自己的 spec_num_tokens 个新行
            #     （无投机时为 1），和 decode 步一样。
            # 例：MIXED 批 A extend 300、B decode（N=1）：q_lens=[300, 1] -> cu_seqlens_q=[0, 300, 301]，
            # cu_seqlens_k=[0, 300, 1301]，max_seq_len_q=300。
            q_lens = seq_lens.new_full((prefill_bs,), self.spec_num_tokens)
            q_lens[:num_extends] = extend_seq_lens[:num_extends]
            cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(q_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            max_seq_len_q = max(max_seq_len_q, self.spec_num_tokens)
        elif extend_with_prefix and bool(extend_prefix_lens_cpu[:prefill_bs].any()):
            extend_lens = seq_lens[:prefill_bs] - extend_prefix_lens[:prefill_bs]
            cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(extend_lens, dim=0, dtype=torch.int32), (1, 0)
            )
        # 无前缀的纯 extend：每个请求的 query 长度等于 KV 长度，cu_seqlens_q 直接复用 cu_seqlens_k。
        else:
            cu_seqlens_q = cu_seqlens_k

        self.forward_prefill_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seq_len_q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table[:prefill_bs],
        )

    # ------------------------------------------------------------------
    # CUDA graph / decode refresh
    # ------------------------------------------------------------------
    # 译：CUDA graph / decode 刷新

    def init_cuda_graph_state(self, max_bs: int) -> None:
        super().init_cuda_graph_state(max_bs)
        self.spec_cache_seqlens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self._cu_seqlens_by_bs = {}
        self._verify_views_by_bs = {}

    def _uniform_cu_seqlens(self, bs: int, stride: int) -> torch.Tensor:
        """Cached ``arange(0, bs*stride+1, stride)`` (pointer-stable per bs)."""
        # 译：缓存的 ``arange(0, bs*stride+1, stride)``（每个 bs 指针稳定）。
        # 例：bs=3、stride=4（N=4 的投机验证）-> [0, 4, 8, 12]；stride=1（普通 decode）-> [0, 1, 2, 3]。
        key = bs * 100000 + stride
        cu = self._cu_seqlens_by_bs.get(key)
        if cu is None:
            cu = torch.arange(
                0, bs * stride + 1, stride, dtype=torch.int32, device=self.device
            )
            self._cu_seqlens_by_bs[key] = cu
        return cu

    def _decode_views(self, bs: int) -> TRTLLMMHAMetadata:
        """Single-token decode slot views (block decode: one per block
        position)."""
        # 译：单 token decode 槽的视图（块 decode 时每个块内位置一个）。
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is None:
            if self.block_decode_active:
                expanded_bs = bs * self.block_decode_expansion
                metadata = TRTLLMMHAMetadata(
                    cache_seqlens_int32=self.seq_lens_buf[:expanded_bs],
                    max_seq_len_q=1,
                    cu_seqlens_q=self._uniform_cu_seqlens(expanded_bs, 1),
                    page_table=self.page_table_buf[:expanded_bs],
                )
            else:
                metadata = TRTLLMMHAMetadata(
                    cache_seqlens_int32=self.seq_lens_buf[:bs],
                    max_seq_len_q=1,
                    cu_seqlens_q=self._uniform_cu_seqlens(bs, 1),
                    page_table=self.page_table_buf[:bs],
                )
            self._decode_views_by_bs[bs] = metadata
        return metadata

    def _verify_views(self, bs: int) -> TRTLLMMHAMetadata:
        """Target-verify slot views: uniform ``spec_num_tokens`` stride over
        the clamped seqlens buffer."""
        # 译：目标模型验证槽的视图：在截断后的 seqlens 缓冲区上使用统一的 ``spec_num_tokens`` 步长。
        metadata = self._verify_views_by_bs.get(bs)
        if metadata is None:
            spec = self.spec_num_tokens
            metadata = TRTLLMMHAMetadata(
                cache_seqlens_int32=self.spec_cache_seqlens_buf[:bs],
                max_seq_len_q=spec,
                cu_seqlens_q=self._uniform_cu_seqlens(bs, spec),
                page_table=self.page_table_buf[:bs],
            )
            self._verify_views_by_bs[bs] = metadata
        return metadata

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_extends: int = 0,
        for_graph_replay: bool = False,
    ) -> None:
        if self.block_decode_active:
            # DFLASH draft block: replicate the page table to each request's
            # block positions. Under replay the seq_lens are re-derived
            # in-graph; eager (and the capture seeding) fills them here.
            # 译：DFLASH 草稿块：把页表复制到每个请求的块内位置。回放时 seq_lens 在 graph 内重新推导；eager（以及捕获时的初始填充）在这里填充。
            spec = self.spec_num_tokens
            self.page_table_buf[: bs * spec].view(bs, spec, self.max_num_pages).copy_(
                page_table[:bs, None, :]
            )
            if not for_graph_replay or actual_bs == 0:
                self.fill_block_decode_seq_lens(bs, seq_lens)
            self.forward_decode_metadata = self._decode_views(bs)
            return

        self.seq_lens_buf[:bs].copy_(seq_lens[:bs])
        self.page_table_buf[:bs].copy_(page_table[:bs])
        self.publish_decode_metadata(bs, seq_lens, num_extends=num_extends)

    def decode_buffer_targets(
        self, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.block_decode_active:
            return None
        return self.seq_lens_buf[:bs], self.page_table_buf[:bs]

    def publish_decode_metadata(
        self, bs: int, seq_lens: torch.Tensor, *, num_extends: int
    ) -> None:
        self.forward_decode_metadata = self._decode_views(bs)
        # Pure verify (and the draft's multi-token step 1) reads the prefill
        # slot. An extend/mixed draft refresh only seeds later decode steps and
        # must preserve the ragged prefill metadata built immediately before it.
        # 译：纯验证（以及草稿的多 token 第 1 步）读 prefill 槽。extend / mixed 的草稿刷新只是为之后的 decode 步做准备，
        #     必须保留紧挨在它之前构建的 ragged prefill 元数据。
        if self.spec_num_tokens > 1 and num_extends == 0:
            torch.clamp_min(
                seq_lens[:bs],
                self.spec_num_tokens,
                out=self.spec_cache_seqlens_buf[:bs],
            )
            self._refresh_tree_prefix(bs, self.spec_cache_seqlens_buf)
            self.forward_prefill_metadata = self._verify_views(bs)

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------
    # 译：KV cache 辅助函数

    @property
    def workspace_buffer(self) -> torch.Tensor:
        """Per-use view of the shared scratch block (contract: workspace.py).

        Fetched on every use rather than held: the content (softmax stats plus
        multi-CTA KV scratch) is consumed within each attention op, so sharing
        the block is safe, and a held view would go stale if the block grew.
        """
        # 译：共享暂存块的按次使用视图（契约见 workspace.py）。
        #
        #     每次使用时都重新获取而不是持有：其内容（softmax 统计量加上 multi-CTA KV 暂存）在每个注意力算子内部就消费完了，
        #     所以共享这个块是安全的，而持有的视图在块变大后会失效。
        (buf,) = self._workspace_pool.allocate(((self._workspace_nbytes,), torch.uint8))
        return buf

    def bind_tree_draft(self, inputs: TreeDraftInputs) -> None:
        super().bind_tree_draft(inputs)
        if self._kv_counter_buffer is None:
            return
        # A sliding-window draft decodes each lane row as its own request.
        # 译：滑动窗口草稿把每个通道行当作单独的请求来 decode。
        needed = get_trtllm_gen_multi_ctas_kv_counter_bytes(
            self._max_bs * inputs.topk, self.tp_q_head_num, self._sm_count
        )
        if needed > self._kv_counter_buffer.numel():
            self._kv_counter_buffer = torch.zeros(
                needed, dtype=torch.uint8, device=self._kv_counter_buffer.device
            )

    def _kv_counter_buffer_for(
        self, batch_size: int, num_qo_heads: int
    ) -> torch.Tensor | None:
        """The persistent multi-CTA KV counter buffer if it holds this batch."""
        # 译：持久的 multi-CTA KV 计数器缓冲区（若它装得下这个批次）。
        buf = self._kv_counter_buffer
        if buf is None or buf.numel() < get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch_size, num_qo_heads, self._sm_count
        ):
            return None
        return buf

    def _get_kv_cache_permuted(self, layer: PagedAttention, token_to_kv_pool):
        """Get KV cache in [num_pages, num_kv_heads, page_size, head_dim] layout."""
        # 译：以 [num_pages, num_kv_heads, page_size, head_dim] 布局获取 KV cache。
        # 例：k_buffer [NP*64, 8, 128] -> view [NP, 64, 8, 128] -> permute [NP, 8, 64, 128]。
        # 注意：permute 只改步长，物理存储仍是 [页, 页内偏移, 头, dim]（与 prologue 写入的布局一致），kernel 按步长读取。
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(
            -1, self.kernel_page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.kernel_page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)

        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)

        return k_cache, v_cache

    def _prepare_q(self, q: torch.Tensor, layer: PagedAttention) -> torch.Tensor:
        if self.kv_cache_dtype == torch.float8_e4m3fn:
            q = fp8_cast_contiguous(q)
        else:
            q = q.contiguous()

        return q.view(-1, layer.tp_q_head_num, layer.head_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    # 译：前向

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        **kwargs,
    ) -> torch.Tensor:
        lanes = self._tree_lane_rows(bs)
        if lanes is not None:
            return self._tree_cascade(
                q, layer, token_to_kv_pool, lanes, sinks=kwargs.get("sinks")
            )
        if self.block_decode_active:
            # DFLASH draft block: metadata is expanded to bs*spec_num_tokens
            # single-query entries, so use the decode slot directly. Inferring
            # q_len_per_req from q.shape[0]//bs would be spec_num_tokens and
            # wrongly pick the verify slot.
            # 译：DFLASH 草稿块：元数据被展开为 bs*spec_num_tokens 个单 query 条目，所以直接用 decode 槽。
            #     若用 q.shape[0]//bs 推断 q_len_per_req，会得到 spec_num_tokens，从而错选验证槽。
            metadata = self.forward_decode_metadata
        else:
            # Multi-token decode (q_len > 1) reads the verify slot's
            # uniform-stride metadata; plain decode reads the single-token slot.
            # 译：多 token decode（q_len > 1）读验证槽统一步长的元数据；普通 decode 读单 token 槽。
            q_len_per_req = q.shape[0] // bs if bs > 0 else 1
            metadata = (
                self.forward_prefill_metadata
                if q_len_per_req > 1
                else self.forward_decode_metadata
            )

        tree = self._tree_verify_rows(
            bs,
            metadata.max_seq_len_q,
            metadata.page_table,
            metadata.cache_seqlens_int32,
        )
        if tree is not None:
            return self._tree_cascade(
                q, layer, token_to_kv_pool, tree, sinks=kwargs.get("sinks")
            )
        # 例：B 一个请求普通 decode：q [1, 32, 128]，q_len_per_req=1，读 forward_decode_metadata；
        # N=4 投机验证且 bs=2：q [8, 32, 128]，q_len_per_req=4，读 prefill 槽的验证视图（cu_seqlens_q=[0, 4, 8]）。
        # FP8 KV cache 时 _prepare_q 把 q 也转成 FP8。
        q = self._prepare_q(q, layer)
        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)

        attention_sink = kwargs.get("sinks", None)
        if attention_sink is not None:
            attention_sink = attention_sink.float()

        o = trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=metadata.page_table,
            seq_lens=metadata.cache_seqlens_int32,
            max_seq_len=self.max_context_len,
            bmm1_scale=layer.scaling,
            bmm2_scale=1.0,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            out_dtype=self.dtype,
            q_len_per_req=metadata.max_seq_len_q,
            multi_ctas_kv_counter_buffer=self._kv_counter_buffer_for(
                q.shape[0] // metadata.max_seq_len_q, q.shape[1]
            ),
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def tree_support(self) -> TreeSupport:
        if self.kv_cache_dtype in (self.dtype, torch.float8_e4m3fn):
            return TreeSupport(verify_blocker=None, draft_blocker=None)
        return TreeSupport(
            verify_blocker=f"{type(self).__name__} reads the verify window K/V in the "
            f"model dtype or FP8 E4M3; kv_cache_dtype {self.kv_cache_dtype} is not supported",
            draft_blocker=f"{type(self).__name__} reads the lane window K/V in the "
            f"model dtype or FP8 E4M3; kv_cache_dtype {self.kv_cache_dtype} is not supported",
        )

    # draft tree 投机解码（第 11 阶段），初读可跳过。
    def _tree_cascade(
        self,
        q: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        tree: TreeCascadeRows,
        *,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Draft-tree attention (verify nodes or drafting lanes): trtllm-gen's causal
        ``rows``-token decode over the committed prefix, then the tree window
        and the prefix tail merged into it (``tree_window_attention``).

        A sliding-window draft layer sees the window frozen at the round's
        prefix end: each lane row decodes the prefix alone (its query placed at
        the prefix's last key), so no prefix tail remains for the window kernel.

        ``q`` is unquantized: trtllm-gen takes it cast like any decode query,
        and the window kernel takes it as is, widening an FP8 cache's K/V."""
        # 译：draft tree 注意力（验证节点或起草通道）：先由 trtllm-gen 在已提交的前缀上做 ``rows`` 个 token 的因果 decode，
        #     再把树窗口和前缀尾部合并进去（``tree_window_attention``）。
        #
        #     滑动窗口的草稿层看到的窗口冻结在本轮前缀的末尾：每个通道行单独 decode 前缀（其 query 放在前缀最后一个 key 处），
        #     所以不会剩下前缀尾部给窗口 kernel。
        #
        #     ``q`` 不做量化：trtllm-gen 像对待任何 decode query 一样转换它，而窗口 kernel 原样接收它，并把 FP8 cache 的 K/V 加宽。
        causal_prefix = layer.sliding_window_size < 0
        # Only lane rows (their own tables) decode a sliding window.
        # 译：只有通道行（各自有自己的表）才 decode 滑动窗口。
        if sinks is not None or (not causal_prefix and tree.row_page_table is None):
            raise NotImplementedError(
                "draft trees have no sliding-window or attention-sink path yet"
            )
        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)
        # trtllm-gen also loads the key right after the prefix: the window's first slot, written this forward.
        # 译：trtllm-gen 也会加载紧跟在前缀之后的那个 key：窗口的第一个槽位，在本次前向中写入。
        q_len = tree.rows if causal_prefix else 1
        prefix_out, prefix_lse = trtllm_batch_decode_with_kv_cache(
            query=self._prepare_q(q, layer),
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=tree.page_table if causal_prefix else tree.row_page_table,
            seq_lens=tree.prefix_lens if causal_prefix else tree.row_prefix_lens,
            max_seq_len=self.max_context_len,
            bmm1_scale=layer.scaling,
            bmm2_scale=1.0,
            out_dtype=self.dtype,
            q_len_per_req=q_len,
            window_left=layer.sliding_window_size,
            return_lse=True,
            multi_ctas_kv_counter_buffer=self._kv_counter_buffer_for(
                q.shape[0] // q_len, layer.tp_q_head_num
            ),
        )
        kv_heads, dim = layer.tp_k_head_num, layer.head_dim
        k_rows, v_rows = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        out = tree_window_attention(
            q.contiguous().view(-1, layer.tp_q_head_num, dim),
            k_rows.view(-1, kv_heads, dim),
            v_rows.view(-1, kv_heads, dim),
            tree.page_table,
            tree.seq_lens,
            tree.mask,
            prefix_out,
            prefix_lse,
            rows_per_req=tree.rows,
            window=tree.window,
            causal_prefix=causal_prefix,
            page_size=self.kernel_page_size,
            sm_scale=layer.scaling,
        )
        return out.view(-1, layer.tp_q_head_num * dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        **kwargs,
    ) -> torch.Tensor:
        # extend / mixed（以及 MIXED 批中的 decode 行）统一走 context kernel，K/V 全部从分页 cache 读取
        # （prologue 已写入本次的新 token）。batch_size = cu_seqlens_q 长度 - 1。
        metadata = self.forward_prefill_metadata
        q = self._prepare_q(q, layer)
        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)

        attention_sink = kwargs.get("sinks", None)
        if attention_sink is not None:
            attention_sink = attention_sink.float()

        o = trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=metadata.page_table,
            seq_lens=metadata.cache_seqlens_int32,
            max_q_len=metadata.max_seq_len_q,
            max_kv_len=self.max_context_len,
            bmm1_scale=layer.scaling,
            bmm2_scale=1.0,
            batch_size=metadata.cu_seqlens_q.shape[0] - 1,
            cum_seq_lens_q=metadata.cu_seqlens_q,
            cum_seq_lens_kv=metadata.cu_seqlens_k,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            out_dtype=self.dtype,
            multi_ctas_kv_counter_buffer=self._kv_counter_buffer_for(
                metadata.cu_seqlens_q.shape[0] - 1, q.shape[1]
            ),
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


register_backend("trtllm", {AttentionArch.MHA}, TRTLLMMHAAttnBackend)
