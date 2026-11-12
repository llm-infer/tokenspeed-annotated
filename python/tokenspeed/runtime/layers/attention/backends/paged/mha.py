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

# MHAAttnBackend：标准 MHA / GQA 的分页注意力叶子，Qwen3-8B 默认走这里（--attention-backend 为 mha / fa3 / fa4 /
# triton / flashinfer 时都是本类，只是 kernel_solution 不同）。
#
# 你在这里：CacheGroupRouter（router.py）为每个 cache group 建一个本类实例；router 传来已换算好的 kernel 页表，
# 本类据此建元数据，再在每层调 tokenspeed_kernel 的注意力 API。本类不写 KV（prologue 已经写好）。
#
# 两种元数据：
#   MHAExtendMetadata：extend / mixed 前向，每轮新建（ragged：按 cu_seqlens 拼接的变长序列）
#   MHADecodeMetadata：decode 前向，是持久缓冲区（page_table_buf / seq_lens_buf）上的视图，原地刷新
#
# 三种 kernel 调用：
#   mha_prefill               全新 prefill（无前缀、非 mixed）：直接对本次的 q/k/v 做因果注意力，不读 cache
#   mha_extend_with_kvcache   有前缀的 extend 或 mixed 批：q 是新 token，K/V 从分页 cache 读（含刚写入的）
#   mha_decode_with_kvcache   decode / 投机验证：每个请求 1 个（或 N 个）query，K/V 从分页 cache 读
#
# 形状（Qwen3-8B 单卡，P=64）：q [T, 32, 128]，k/v [T, 8, 128]，k_cache/v_cache [NP, 64, 8, 128]，输出 [T, 4096]
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_schedule,
    mha_decode_takes_schedule,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_plan,
    mha_prefill,
)
from tokenspeed_kernel.ops.attention.tree import tree_window_attention
from tokenspeed_kernel.ops.quantization import quantize_mxfp8
from tokenspeed_kernel.ops.tuning import is_warming_up

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import slice_to_real_tokens
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.backends.base import (
    DecodeKVView,
    reject_query_shard,
)
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeCascadeRows,
)
from tokenspeed.runtime.layers.attention.backends.support import TreeSupport
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


# --attention-backend 的取值 -> tokenspeed_kernel 的 solution 名。"mha" 对应 None：由 kernel registry 按平台自动选择。
_KERNEL_SOLUTION_BY_BACKEND = {
    "mha": None,
    "fa3": "fa3",
    "fa4": "fa4",
    "triton": "triton",
    "flashinfer": "flashinfer",
}


def _fp8_scale_kwargs(layer: PagedAttention) -> dict:
    """The per-tensor scales an FP8 cache was written with, for the kernel's read."""
    # 译：FP8 cache 写入时使用的逐张量 scale，供 kernel 读取时使用。
    scales = layer.fp8_kv_scales
    return {} if scales is None else {"fp8_kv_scales": (scales.k, scales.v)}


def _slice_extend_inputs(metadata, *tensors):
    """Remove prefill-graph padding rows before calling an attention kernel.

    The live cu-seqlens describe only real tokens. Some kernels tolerate extra
    zero rows, but others still derive work from the tensor shape. Use the
    pinned CPU mirror (sync-free) so every solution receives the same exact-row
    contract. No-op on normal unpadded forwards.
    """
    # 译：调用注意力 kernel 之前去掉 prefill graph 的 padding 行。
    #
    #     实际的 cu-seqlens 只描述真实 token。有些 kernel 能容忍多出来的零行，但另一些仍然根据张量形状推导工作量。
    #     使用锁页的 CPU 镜像（无需同步），让每种实现都收到同样的“精确行数”契约。普通的未 padding 前向上是空操作。
    return slice_to_real_tokens(metadata.cu_extend_seq_lens_cpu[-1], *tensors)


# extend 元数据。贯穿例子（MIXED 批：A extend 300 个新 token、无前缀；B decode，seq_len=1001，N=1）：
#   B 被当作 query 长度为 N=1 的 extend 请求，前缀 = 1000
#   seq_lens               = [300, 1001]
#   extend_seq_lens        = [300, 1]          （每个请求的 query 行数）
#   cu_extend_seq_lens     = [0, 300, 301]     （q 中第 i 个请求的行区间）
#   cu_seqlens_kv          = [0, 300, 1301]
#   extend_prefix_lens     = [0, 1000]
#   max_extend_seq_len     = 300
#   max_seqlen_k           = context_len（mixed 时 decode 长度只在 GPU 上，取上限，避免读回 CPU）
#   is_mixed=True，max_extend_prefix_len = 0（只统计 extend 请求）
@dataclass(kw_only=True)
class MHAExtendMetadata:
    # Device-side metadata:
    # - seq_lens: total length after this step
    # - extend_seq_lens: length of new tokens
    #   cu_extend_seq_lens: the cumsum version of extend_seq_lens
    #   cu_seqlens_kv: the cumsum version of seq_lens
    # - extend_prefix_lens: length of the cached prefix tokens
    # seq_lens[i] = extend_prefix_lens[i] + extend_seq_lens[i]
    # 译：设备端元数据：
    #     - seq_lens：本步之后的总长度
    #     - extend_seq_lens：新 token 的长度
    #       cu_extend_seq_lens：extend_seq_lens 的前缀和版本
    #       cu_seqlens_kv：seq_lens 的前缀和版本
    #     - extend_prefix_lens：已缓存前缀 token 的长度
    #     seq_lens[i] = extend_prefix_lens[i] + extend_seq_lens[i]
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cu_extend_seq_lens: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens_cpu: list[int]
    cu_extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    # Bound on seq_lens for the cached-prefix kernel, which may size its work
    # (e.g. split counts) from it: the longest request when every length is
    # known on the host, else the context limit.
    # 译：带缓存前缀的 kernel 所用的 seq_lens 上界，kernel 可能据此决定工作量（例如拆分数）：
    #     每个长度在 Host 上都已知时取最长请求的长度，否则取上下文上限。
    max_seqlen_k: int
    # Mixed target queries are cache-backed even when the leading prefills
    # have no prefix; the fresh-prefill path must not discard decode history.
    # 译：即使前面的 prefill 请求没有前缀，mixed 的目标模型 query 也是基于 cache 的；全新 prefill 路径不能丢掉 decode 的历史。
    is_mixed: bool
    max_extend_prefix_len: int = 0


@dataclass(kw_only=True)
class MHADecodeMetadata:
    page_table: torch.Tensor
    seq_lens: torch.Tensor


class MHAAttnBackend(PagedAttentionBackend):
    """Standard MHA leaf routed through tokenspeed_kernel attention APIs.

    Full-attention decode calls whose shape takes a split schedule
    (``mha_decode_takes_schedule``, e.g. speculative verify on gfx950) read
    one schedule per forward from ``decode_schedule_buf``. Like the layers'
    shared sparse top-k, the first such layer computes it in place from the
    step's lengths (``mha_decode_schedule``), inside the captured graph
    under replay, and the rest reuse it; each decode metadata build marks it
    stale.
    """

    # 译：通过 tokenspeed_kernel 注意力 API 路由的标准 MHA 叶子。
    #
    #     形状需要拆分调度的全注意力 decode 调用（``mha_decode_takes_schedule``，例如 gfx950 上的投机验证）每次前向
    #     从 ``decode_schedule_buf`` 读一份调度。与层间共享的稀疏 top-k 一样，第一个这样的层根据本步的长度
    #     原地计算它（``mha_decode_schedule``），回放时在被捕获的 graph 内计算，其余层复用；每次构建 decode 元数据都把它标记为过期。

    # Every kernel call site forwards layer.sliding_window_size.
    # 译：每个 kernel 调用点都传入 layer.sliding_window_size。
    supports_layer_sliding_window: bool = True
    # Set from the resolved kernel solution in __init__: a token sink reaches
    # only the Triton decode leaf, so a backend pinned to another solution
    # cannot serve one and says so before a layer declares a sink.
    # 译：在 __init__ 中根据解析出的 kernel 实现设置：token sink 只有 Triton decode 叶子支持，所以固定到别的实现的后端
    #     无法提供它，并在某层声明 sink 之前就说明这一点。
    supports_layer_window_sink: bool = False

    def __init__(self, config: AttnConfig, spec: MHAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        # Map the selected backend to the corresponding kernel solution string.
        # 译：把选中的后端映射到对应的 kernel 实现字符串。
        backend_name = spec.backend_name or "mha"
        self.kernel_solution = _KERNEL_SOLUTION_BY_BACKEND[backend_name]
        # ``None`` leaves selection free, so the sink-capable leaf is reachable;
        # a pinned solution must be the one that declares the trait.
        # 译：``None`` 让选择保持自由，因而能选到支持 sink 的实现；固定的实现必须是声明了该特性的那个。
        self.supports_layer_window_sink = self.kernel_solution in (None, "triton")

        self.skip_softmax_threshold = spec.skip_softmax_threshold

        self.tp_q_head_num = max(spec.num_attention_heads // spec.attn_tp_size, 1)
        self.tp_kv_head_num = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.qkv_dtype = config.dtype
        self.kv_cache_dtype = config.kv_cache_dtype
        self.is_mxfp8 = bool(config.kv_cache_mxfp8)
        # mxfp8 shares the fp8 storage dtype but uses block scales; keep it off
        # the per-tensor casts
        # 译：mxfp8 与 fp8 共用存储 dtype，但使用块级 scale；不让它走逐张量的类型转换
        self.is_fp8 = (
            self.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not self.is_mxfp8
        )
        # plan：预先绑定 dtype / head_dim / 实现，前向时再按层的窗口、logit cap、sink 得到 kernel 方案（含 extend_mode）。
        # Qwen3-8B bf16：dtype=bf16、head_dim=128。
        self.plan = partial(
            mha_plan,
            dtype=(
                torch.float8_e4m3fn
                if self.is_mxfp8
                else (self.kv_cache_dtype if self.is_fp8 else self.qkv_dtype)
            ),
            head_dim=self.head_dim,
            return_lse=False,
            solution=self.kernel_solution,
        )
        # DFLASH draft: the whole block in one decode forward, with one
        # decode metadata entry per block position and uniform non-causal
        # seq_lens (block_decode_expansion).
        # 译：DFLASH 草稿：整个块在一次 decode 前向中完成，每个块内位置一个 decode 元数据条目，
        #     seq_lens 统一且非因果（block_decode_expansion）。
        self.draft_block_decode = bool(config.draft_block_decode)

        self.forward_decode_metadata: MHADecodeMetadata | None = None
        self.forward_extend_metadata: MHAExtendMetadata | None = None
        # The per-forward split schedule (see the class docstring), whether
        # this forward's first schedule-reading layer has yet to compute it,
        # and which (layer, query rows per request) calls read it.
        # 译：每次前向的拆分调度（见类 docstring）、本次前向中第一个读调度的层是否还没算它，
        #     以及哪些（层，每请求 query 行数）调用会读它。
        self.decode_schedule_buf: torch.Tensor | None = None
        self._schedule_stale = True
        self._layer_reads_schedule: dict[tuple[int, int], bool] = {}

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self.forward_decode_metadata = None
        self.forward_extend_metadata = None
        self.decode_schedule_buf = None

    def init_cuda_graph_state(self, max_bs: int) -> None:
        super().init_cuda_graph_state(max_bs)
        # One header row and one row per request (mha_decode_schedule).
        # 译：一个表头行，加上每个请求一行（mha_decode_schedule）。
        self.decode_schedule_buf = torch.zeros(
            (self.seq_lens_buf.shape[0] + 1, 2), dtype=torch.int32, device=self.device
        )

    def supports_narrowed_draft_decode(self, forward_mode: ForwardMode) -> bool:
        # Decode metadata exists only in decode rounds.
        # 译：decode 元数据只在 decode 轮中存在。
        return forward_mode.is_decode()

    def reads_fp8_kv_scales(self, layer: PagedAttention) -> bool:
        # Both reads hand them to the kernel (_fp8_scale_kwargs).
        # 译：两种读取都把它们交给 kernel（_fp8_scale_kwargs）。
        return True

    def decode_kv_view(
        self, layer: PagedAttention, token_to_kv_pool
    ) -> DecodeKVView | None:
        # A sliding window reads the cache through the page table alone; full
        # layers' verify decode takes a split schedule, and scaled caches more.
        # A bound draft tree attends through its mask (forward_decode's tree
        # cascade), which the view does not carry.
        # 译：滑动窗口只通过页表读 cache；全注意力层的验证 decode 需要拆分调度，带 scale 的 cache 需要的更多。
        #     绑定的 draft tree 通过它的掩码做注意力（forward_decode 的树级联），而这个视图不携带掩码。
        metadata = self.forward_decode_metadata
        if (
            metadata is None
            or not self.is_fp8
            or layer.fp8_kv_scales is not None
            or layer.sliding_window_size < 0
            or self.draft_block_decode
            or self.tree_verify is not None
            or self.tree_lane_step_active
        ):
            return None
        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        return DecodeKVView(
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            seq_lens=metadata.seq_lens,
            max_seqlen_k=self.max_context_len,
        )

    def query_dtype(
        self, layer: PagedAttention, forward_mode: ForwardMode, activation: torch.dtype
    ) -> torch.dtype:
        # Decode reads an FP8 cache with FP8 queries. A fresh prefill attends
        # the activation-dtype K/V rows instead, so extend converts its own.
        # 译：decode 用 FP8 query 读 FP8 cache。全新 prefill 关注的是激活 dtype 的 K/V 行，所以 extend 自己做转换。
        if self.is_fp8 and forward_mode.is_decode():
            return self.kv_cache_dtype
        return activation

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    # 译：元数据

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
    ) -> None:
        # extend / mixed 元数据构建。page_table 来自 router（[bs, max_num_pages]，A、B 两行）。
        # 例子见上面 MHAExtendMetadata 的注释。只有 query_lens_cpu、max_extend_seq_len 等 Host 端值需要 CPU 数据，
        # 它们来自锁页的 Host 镜像（extend_seq_lens_cpu），不读回 GPU。
        reject_query_shard(query_shard, "MHAAttnBackend")
        del page_table_cpu
        if not forward_mode.is_extend_or_mixed():
            raise RuntimeError(
                "MHA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        is_mixed = forward_mode.is_mixed()
        if is_mixed and self.is_draft and not self.draft_block_decode:
            raise RuntimeError("MHA mixed draft metadata requires block decoding")
        # Block drafters share the target's mixed metadata initialization.
        # Before their model forward, they refresh decode metadata and publish
        # a uniform block window starting at each request's accepted prefix.
        # This does not enable a mixed forward for ordinary token drafters.
        # 译：块草稿模型共用目标模型的 mixed 元数据初始化。在它们的模型前向之前，它们会刷新 decode 元数据，
        #     并发布一个从每个请求已接受前缀开始的统一块窗口。这不会让普通的逐 token 草稿模型启用 mixed 前向。
        if is_mixed and not 0 < num_extends < bs:
            raise ValueError(
                "MHA mixed metadata requires both prefill and decode requests"
            )

        seq_lens = seq_lens[:bs]
        prefill_prefix_lens = extend_prefix_lens[:num_extends]
        query_lens_cpu = [int(x) for x in extend_seq_lens_cpu[:num_extends].tolist()]
        if is_mixed:
            # The runner supplies extend lengths only for the leading prefill
            # requests. Decode/verify queries follow them in request order and
            # attend a causal suffix of each request's already-written cache.
            # 译：runner 只为排在前面的 prefill 请求提供 extend 长度。decode / 验证的 query 按请求顺序跟在它们之后，
            #     关注各自请求已写入 cache 的一段因果后缀。
            query_lens = seq_lens.new_full((bs,), self.spec_num_tokens)
            query_lens[:num_extends] = extend_seq_lens[:num_extends]
            query_lens_cpu.extend([self.spec_num_tokens] * (bs - num_extends))
            prefix_lens = seq_lens - query_lens
        else:
            query_lens = extend_seq_lens[:bs]
            prefix_lens = prefill_prefix_lens
        cu_extend_seq_lens = torch.nn.functional.pad(
            torch.cumsum(query_lens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        cu_extend_seq_lens_cpu = [0]
        for length in query_lens_cpu:
            cu_extend_seq_lens_cpu.append(cu_extend_seq_lens_cpu[-1] + length)
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        # Mixed batches keep the context limit: their decode lengths live only
        # on the device.
        # 译：mixed 批保留上下文上限：它们的 decode 长度只存在于设备上。
        max_seqlen_k = self.max_context_len
        if not is_mixed:
            max_seqlen_k = int(
                (
                    extend_prefix_lens_cpu[:num_extends]
                    + extend_seq_lens_cpu[:num_extends]
                )
                .max()
                .item()
            )
        self.forward_extend_metadata = MHAExtendMetadata(
            page_table=page_table[:bs],
            seq_lens=seq_lens,
            extend_seq_lens=query_lens,
            cu_extend_seq_lens=cu_extend_seq_lens,
            cu_seqlens_kv=cu_seqlens_kv,
            extend_prefix_lens=prefix_lens,
            extend_seq_lens_cpu=query_lens_cpu,
            cu_extend_seq_lens_cpu=cu_extend_seq_lens_cpu,
            max_extend_seq_len=max(query_lens_cpu),
            max_seqlen_k=max_seqlen_k,
            is_mixed=is_mixed,
            # Only prefills have a host prefix mirror. Mixed batches always
            # use cached attention, without reading decode lengths back to CPU.
            # 译：只有 prefill 有 Host 端的前缀镜像。mixed 批总是使用基于 cache 的注意力，不需要把 decode 长度读回 CPU。
            max_extend_prefix_len=int(
                extend_prefix_lens_cpu[:num_extends].max().item()
            ),
        )

    def _decode_views(self, bs: int) -> MHADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers: one
        builder for capture and refresh, cached per bs — pointer-stable."""
        # 译：持久缓冲区之上的按 bs decode 元数据视图：捕获和刷新共用一个构建函数，按 bs 缓存——指针稳定。
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is None:
            expanded_bs = bs * self.block_decode_expansion
            metadata = MHADecodeMetadata(
                page_table=self.page_table_buf[:expanded_bs],
                seq_lens=self.seq_lens_buf[:expanded_bs],
            )
            self._decode_views_by_bs[bs] = metadata
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
        # 例：纯 decode 一步，B 一个请求、seq_len=1001，graph 桶 bs=1：
        #   seq_lens_buf[0] = max(1001, verify_floor) = 1001；page_table_buf[0] = B 的 640 列 kernel 页表
        #   forward_decode_metadata = 这两个缓冲区前 1 行的视图（同一个 bs 总是同一个视图对象，graph 记录的地址不变）
        # router 走快路径时（decode_buffer_targets 非 None），这两次拷贝由 router 批量完成，只调用 publish_decode_metadata。
        del num_extends
        if self.block_decode_active:
            # DFLASH draft: replicate each request's page table to its
            # spec_num_tokens block positions. Under replay the block-end
            # seq_lens are re-derived inside the captured graph
            # (fill_block_decode_seq_lens); eager has no in-graph writer, and
            # the capture seeding needs the same safe baseline.
            # 译：DFLASH 草稿：把每个请求的页表复制到它的 spec_num_tokens 个块内位置。回放时块末尾的 seq_lens 在被捕获的
            #     graph 内重新推导（fill_block_decode_seq_lens）；eager 没有 graph 内的写入者，捕获时的初始填充也需要同样安全的基线。
            spec = self.spec_num_tokens
            self.page_table_buf[: bs * spec].view(bs, spec, self.max_num_pages).copy_(
                page_table[:bs, None, :]
            )
            if not for_graph_replay or actual_bs == 0:
                self.fill_block_decode_seq_lens(bs, seq_lens)
        else:
            # Clamp short requests (padded requests replay at seq_len 1) to
            # the verify floor: verify derives per-token lengths as
            # seq - N + t + 1, which must stay positive. Plain decode and
            # drafts have floor 1 (identity).
            # 译：把短请求（padding 请求以 seq_len 1 回放）截断到验证下限：验证按 seq - N + t + 1 推导逐 token 长度，
            #     它必须保持为正。普通 decode 和草稿的下限为 1（恒等）。
            torch.clamp_min(
                seq_lens[:bs], self.verify_floor, out=self.seq_lens_buf[:bs]
            )
            self._refresh_tree_prefix(bs, self.seq_lens_buf)
            self.page_table_buf[:bs].copy_(page_table[:bs])
        self.forward_decode_metadata = self._decode_views(bs)
        self._schedule_stale = True

    def decode_buffer_targets(
        self, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.block_decode_active:
            return None
        return self.seq_lens_buf[:bs], self.page_table_buf[:bs]

    @property
    def decode_seq_lens_floor(self) -> int:
        return self.verify_floor

    def publish_decode_metadata(
        self, bs: int, seq_lens: torch.Tensor, *, num_extends: int
    ) -> None:
        del seq_lens, num_extends
        # The router filled seq_lens_buf (floored) through decode_buffer_targets.
        # 译：router 已经通过 decode_buffer_targets 填好了 seq_lens_buf（已截断下限）。
        self._refresh_tree_prefix(bs, self.seq_lens_buf)
        self.forward_decode_metadata = self._decode_views(bs)
        self._schedule_stale = True

    def _decode_schedule(
        self,
        layer: PagedAttention,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_seqlen_q: int,
        sinks: torch.Tensor | None,
        scale_kwargs: dict,
    ) -> torch.Tensor | None:
        """The split schedule this layer's decode call reads, or None for
        the kernel's default split. Only the target's full-attention layers
        of the leaf's own shape read it; drafts keep the default, as their
        in-round length updates rebuild no metadata."""
        # 译：本层 decode 调用读取的拆分调度；返回 None 表示用 kernel 默认的拆分。只有目标模型中与叶子自身形状一致的
        #     全注意力层会读它；草稿保持默认，因为它们在一轮内的长度更新不会重建元数据。
        if self.is_draft or layer.sliding_window_size >= 0:
            return None
        key = (layer.layer_id, max_seqlen_q)
        reads = self._layer_reads_schedule.get(key)
        if reads is None:
            reads = (
                q.shape[1] == self.tp_q_head_num
                and k_cache.shape[2] == self.tp_kv_head_num
                and max_seqlen_q == self.verify_floor
                and mha_decode_takes_schedule(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    max_seqlen_q=max_seqlen_q,
                    logit_cap=layer.logit_cap,
                    sinks=sinks,
                    solution=self.kernel_solution,
                    **scale_kwargs,
                )
            )
            self._layer_reads_schedule[key] = reads
        if not reads:
            return None
        if self._schedule_stale:
            computed = mha_decode_schedule(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                max_seqlen_q=max_seqlen_q,
                logit_cap=layer.logit_cap,
                sinks=sinks,
                out=self.decode_schedule_buf,
                solution=self.kernel_solution,
                **scale_kwargs,
            )
            if computed is None:
                # No schedule kernel serves these calls. Its selection can
                # differ from the decode kernel's, for example under an
                # override. These calls keep the default split rather than
                # read an unset schedule.
                # 译：没有调度 kernel 服务这些调用。它的选择可能与 decode kernel 的选择不同，例如在覆盖设置下。
                #     这些调用保持默认拆分，而不是读一份未设置的调度。
                self._layer_reads_schedule[key] = False
                return None
            self._schedule_stale = False
        return self.decode_schedule_buf

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    # 译：前向

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        **kwargs,
    ) -> torch.Tensor:
        # decode 注意力。例：B 一个请求，N=1 -> q [1, 32, 128]，max_seqlen_q = 1；
        # kernel 按 page_table 行找到 B 的 16 个页，读前 1001 个位置的 K/V（含 prologue 刚写入的位置 1000），
        # GQA：32 个 Q 头分 8 组，每组共享一个 KV 头。输出 reshape 为 [1, 32 * 128] = [1, 4096]。
        # 投机验证时每个请求 N 个 query：q [bs * N, 32, 128]，max_seqlen_q = N，kernel 内部按因果关系处理窗口。
        assert layer.qk_head_dim == layer.v_head_dim

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        metadata = self.forward_decode_metadata
        max_seqlen_q = q.shape[0] // metadata.seq_lens.shape[0]

        tree = self._tree_lane_rows(bs) or self._tree_verify_rows(
            bs, max_seqlen_q, metadata.page_table, metadata.seq_lens
        )
        if tree is not None:
            return self._tree_cascade(
                q, layer, token_to_kv_pool, tree, sinks=kwargs.get("sinks")
            )

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8:
            # A no-op on the prologue's decode query (see query_dtype).
            # 译：对 prologue 产生的 decode query 来说这是空操作（见 query_dtype）。
            q = q.to(self.kv_cache_dtype)
            scale_kwargs = _fp8_scale_kwargs(layer)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        sinks = kwargs.get("sinks")
        schedule = self._decode_schedule(
            layer,
            q,
            k_cache,
            v_cache,
            metadata.seq_lens,
            max_seqlen_q,
            sinks,
            scale_kwargs,
        )
        output = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lens,
            window_left=layer.sliding_window_size,
            window_sink=layer.sliding_window_sink,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            max_seqlen_k=self.max_context_len,
            max_seqlen_q=max_seqlen_q,
            schedule=schedule,
            solution=self.kernel_solution,
            **scale_kwargs,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def tree_support(self) -> TreeSupport:
        if self.kv_cache_dtype == self.qkv_dtype:
            return TreeSupport(verify_blocker=None, draft_blocker=None)
        name = type(self).__name__
        return TreeSupport(
            verify_blocker=f"{name} reads the verify window K/V in the model dtype; "
            f"kv_cache_dtype {self.kv_cache_dtype} is not supported",
            draft_blocker=f"{name} reads the lane window K/V in the model dtype; "
            f"kv_cache_dtype {self.kv_cache_dtype} is not supported",
        )

    # 以下 _tree_cascade、_lane_prefix_decode 属于 draft tree 投机解码（第 11 阶段），初读可跳过。
    def _tree_cascade(
        self,
        q: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        tree: TreeCascadeRows,
        *,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Draft-tree attention (verify nodes or drafting lanes): a causal
        ``rows``-token pass over the committed prefix with its log-sum-exp,
        then the tree window and the prefix tail merged into it
        (``tree_window_attention``); the trtllm leaf's cascade with a portable
        prefix half.

        A sliding-window draft layer sees the window frozen at the round's
        prefix end: each lane decodes the prefix alone (its query placed at the
        prefix's last key), so no prefix tail remains for the window kernel.
        """
        # 译：draft tree 注意力（验证节点或起草通道）：先在已提交的前缀上做一次 ``rows`` 个 token 的因果计算并得到 log-sum-exp，
        #     再把树窗口和前缀尾部合并进去（``tree_window_attention``）；即 trtllm 叶子的级联，只是前缀那一半用可移植实现。
        #
        #     滑动窗口的草稿层看到的窗口冻结在本轮前缀的末尾：每个通道单独 decode 前缀（其 query 放在前缀最后一个 key 处），
        #     所以不会剩下前缀尾部给窗口 kernel。
        causal_prefix = layer.sliding_window_size < 0
        # Only lane rows (their own tables) decode a sliding window.
        # 译：只有通道行（各自有自己的表）才 decode 滑动窗口。
        if sinks is not None or (not causal_prefix and tree.row_page_table is None):
            raise NotImplementedError(
                "draft trees have no sliding-window or attention-sink path yet"
            )
        if layer.logit_cap:
            raise NotImplementedError("draft trees have no logit-cap path yet")
        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        if causal_prefix:
            # Pinned to Triton, bypassing selection: its causal mask keeps row r
            # at key P - R + r even when the prefix P is shorter than the R rows
            # (the partial then covers fewer keys, or none), which the window
            # kernel's merge assumes. Its decode folds a request's rows into one
            # pass over the prefix, which every row shares.
            # 译：固定使用 Triton，绕过选择：它的因果掩码让第 r 行对应 key P - R + r，即使前缀 P 比 R 行还短
            #     （此时部分结果覆盖的 key 更少，或者没有），而窗口 kernel 的合并依赖这一点。它的 decode 把一个请求的
            #     各行合并成对前缀的一次遍历，各行共享这次遍历。
            prefix_out, prefix_lse = mha_decode_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=tree.page_table,
                cache_seqlens=tree.prefix_lens,
                max_seqlen_k=self.max_context_len,
                max_seqlen_q=tree.rows,
                window_left=-1,
                logit_cap=0.0,
                sinks=None,
                return_lse=True,
                softmax_scale=layer.scaling,
                solution="triton",
            )
        else:
            prefix_out, prefix_lse = self._lane_prefix_decode(
                q, layer, k_cache, v_cache, tree
            )
        dim = layer.qk_head_dim
        out = tree_window_attention(
            q,
            k_cache.view(-1, layer.tp_k_head_num, dim),
            v_cache.view(-1, layer.tp_v_head_num, dim),
            tree.page_table,
            tree.seq_lens,
            tree.mask,
            prefix_out,
            # The window kernel merges in base 2.
            # 译：窗口 kernel 以 2 为底做合并。
            prefix_lse * 1.4426950408889634,
            rows_per_req=tree.rows,
            window=tree.window,
            causal_prefix=causal_prefix,
            page_size=self.kernel_page_size,
            sm_scale=layer.scaling,
        )
        return out.view(-1, layer.tp_q_head_num * dim)

    def _lane_prefix_decode(
        self,
        q: torch.Tensor,
        layer: PagedAttention,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        tree: TreeCascadeRows,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The drafting lanes' sliding-window prefix partial. Every lane of a
        request queries the same keys (the window ending at the request's
        prefix end), so the lanes ride as extra query heads of one decode per
        request: head ``h * rows + r`` is lane ``r``'s head ``h``, a grouped
        query of KV head ``h``, and each request's keys are read once instead
        of once per lane."""
        # 译：起草通道的滑动窗口前缀部分结果。一个请求的每个通道查询的是同一组 key（以该请求前缀末尾结束的窗口），
        #     所以这些通道作为每请求一次 decode 中额外的 query 头参与：第 ``h * rows + r`` 个头是通道 ``r`` 的第 ``h`` 个头，
        #     是 KV 头 ``h`` 的一个分组 query，这样每个请求的 key 只读一次，而不是每个通道读一次。
        rows = tree.rows
        num_reqs = q.shape[0] // rows
        heads, dim = q.shape[1], q.shape[2]
        folded = q.view(num_reqs, rows, heads, dim).transpose(1, 2)
        # Pinned to Triton, bypassing selection: its folded decode returns the
        # log-sum-exp the window kernel merges with, and one program per KV
        # head and key split serves every lane's grouped heads.
        # 译：固定使用 Triton，绕过选择：它合并后的 decode 返回窗口 kernel 要合并的 log-sum-exp，
        #     而且每个 KV 头和 key 拆分一个 program，就能服务所有通道的分组头。
        out, lse = mha_decode_with_kvcache(
            q=folded.reshape(num_reqs, heads * rows, dim),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=tree.page_table,
            cache_seqlens=tree.prefix_lens,
            max_seqlen_k=self.max_context_len,
            max_seqlen_q=1,
            window_left=layer.sliding_window_size,
            return_lse=True,
            softmax_scale=layer.scaling,
            solution="triton",
        )
        out = out.view(num_reqs, heads, rows, -1).transpose(1, 2)
        lse = lse.view(num_reqs, heads, rows).transpose(1, 2)
        return out.reshape(q.shape[0], heads, -1), lse.reshape(q.shape[0], heads)

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
        # extend 注意力的路径选择：
        #   非 mixed、所有请求都无前缀、kernel 方案为 postwrite -> _forward_prefill：mha_prefill 直接用本次的 q/k/v
        #     （例：只有 A 的 300 个新 token，cu_seqlens=[0, 300]，标准的变长因果 prefill）
        #   其他情况（有前缀命中、分块 prefill 的后续块、mixed 批）-> _forward_extend：mha_extend_with_kvcache
        #     q 只有新 token，K/V 从分页 cache 按页表读（前缀 + prologue 刚写入的新 token），因果掩码按位置对齐
        #     （例：MIXED 批 A 300 行 + B 1 行，cu_seqlens_q=[0, 300, 301]，cu_seqlens_kv=[0, 300, 1301]）
        # extend_mode 由 mha_plan 按所选 kernel 实现给出，默认 "prewrite"（总是从 cache 读）。
        assert layer.qk_head_dim == layer.v_head_dim
        assert k is not None and v is not None

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        metadata = self.forward_extend_metadata
        sinks = kwargs.get("sinks")
        plan = self.plan(
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
        )
        extend_mode = plan.get("extend_mode", "prewrite")
        if (
            not metadata.is_mixed
            and metadata.max_extend_prefix_len == 0
            and extend_mode == "postwrite"
        ):
            if is_warming_up():
                # Startup warms kernels with a fresh prefill, which skips the
                # cached-prefix kernel a later prefix-cache hit runs. Run that
                # too, on the same rows, so it compiles its variants now; its
                # output is not used.
                # 译：启动时用一次全新 prefill 预热 kernel，这会跳过之后命中前缀缓存时才运行的带缓存前缀 kernel。
                #     在同样的行上把它也跑一遍，让它现在就编译各个变体；其输出不使用。
                self._forward_extend(q, layer, token_to_kv_pool, metadata, sinks)
            return self._forward_prefill(q, k, v, layer, metadata, sinks)
        return self._forward_extend(q, layer, token_to_kv_pool, metadata, sinks)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        metadata: MHAExtendMetadata,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        q, k, v = _slice_extend_inputs(metadata, q, k, v)
        output = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=metadata.cu_extend_seq_lens,
            cu_seqlens_cpu=metadata.cu_extend_seq_lens_cpu,
            max_seqlen=metadata.max_extend_seq_len,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            skip_softmax_threshold=self.skip_softmax_threshold,
            solution=self.kernel_solution,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_extend(
        self,
        q: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        (q,) = _slice_extend_inputs(metadata, q)

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8:
            q = q.to(self.kv_cache_dtype)
            scale_kwargs = _fp8_scale_kwargs(layer)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        output = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=metadata.cu_extend_seq_lens,
            cu_seqlens_kv=metadata.cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lens,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_k=metadata.max_seqlen_k,
            is_causal=True,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
            **scale_kwargs,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # MXFP8 helpers
    # ------------------------------------------------------------------
    # 译：MXFP8 辅助函数

    def _quantize_mxfp8_tokens(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token MXFP8: (fp8-e4m3 [T, H, D], UE8M0 scales [T, H, D // 32]).

        Accepts query rows as [T, H, D] or [T, H * D].
        """
        # 译：逐 token 的 MXFP8：（fp8-e4m3 [T, H, D]，UE8M0 scale [T, H, D // 32]）。
        #
        #     接受形状为 [T, H, D] 或 [T, H * D] 的 query 行。
        t, d = x.shape[0], self.head_dim
        h = x.numel() // (t * d)
        # (A PDL triton variant measured 0.07 ms slower e2e at decode Q shapes; flashinfer stays)
        # 译：（一个 PDL 的 triton 变体在 decode 的 Q 形状下端到端测得慢 0.07 ms；保留 flashinfer）
        data, sf = quantize_mxfp8(x.reshape(t * h, d))
        return (
            data.view(t, h, d),
            sf.view(torch.float8_e8m0fnu).view(t, h, d // 32),
        )

    def _get_kv_cache(self, layer: PagedAttention, token_to_kv_pool):
        # Page ids are in this leaf's kernel-page units (the router expanded
        # them), so view the cache at that size.
        # 译：页 id 以本叶子的 kernel 页为单位（router 已经展开过），所以按这个大小来看 cache。
        # 例：Qwen3-8B 第 i 层 k_buffer [NP*64, 8, 128]（slot 为下标）-> view 成 [NP, 64, 8, 128]（页号, 页内偏移）；
        # slot 1387 就是第 21 页第 43 行。只是换个视角看同一块显存，没有拷贝。
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            self.kernel_page_size,
            layer.tp_k_head_num,
            layer.qk_head_dim,
        )
        v_cache = token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1,
            self.kernel_page_size,
            layer.tp_v_head_num,
            layer.v_head_dim,
        )
        return k_cache, v_cache


# 把本类注册到 5 个 --attention-backend 名字下（MHA 架构），见 layers/attention/registry.py。
for _backend_name in _KERNEL_SOLUTION_BY_BACKEND:
    register_backend(_backend_name, {AttentionArch.MHA}, MHAAttnBackend)
