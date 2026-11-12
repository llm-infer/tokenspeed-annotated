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

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
    is_block_drafter,
    model_wide_kwargs,
    resolve_cache_layer_types,
    resolve_dtype,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


# 标准多头 / GQA 注意力（Qwen3、Llama 等）的配置组件。
# generate：从启动参数和 HF 配置读出头数、KV 头数、head_dim、注意力 TP 宽度、每层 cache 标签、滑动窗口，
#   再加上模型范围的参数，生成完整的 AttnConfig。
# cache_cell_size：每个 token 在一层中占用的 KV 字节数 = 本 rank 的 KV 头数 × head_dim × 2（K 和 V）× 每个元素的字节数
#   Qwen3-8B、TP=1、bf16：8 × 128 × 2 × 2 = 4096 字节；36 层合计每个 token 144 KiB
#   （所以 1 GiB 显存大约能放 7281 个 token 的 KV）
@dataclass(kw_only=True)
class MHAConfig(SoftmaxAttnConfig):
    # Mixed full+sliding models are ONE component carrying the inherited
    # per-layer cache_layer_types/window vectors; the per-layer kv-head counts
    # live on the cache-pool spec (``CachePoolSpec.layer_kv_head_counts``).
    # 译：全注意力 + 滑动窗口混合的模型是一个组件，携带继承来的按层 cache_layer_types/窗口向量；按层的 KV 头数
    #     放在 cache 池的 spec 上（``CachePoolSpec.layer_kv_head_counts``）。

    # BLASST skip-softmax sparsity, gluon MHA prefill only (gfx950); see
    # ServerArgs.skip_softmax_threshold.
    # 译：BLASST 跳过 softmax 的稀疏化，只用于 gluon 的 MHA prefill（gfx950）；见 ServerArgs.skip_softmax_threshold。
    skip_softmax_threshold: float = 0.0

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ) -> AttnConfig:
        kv_cache_dtype = server_args.kv_cache_dtype
        draft_block_decode = is_block_drafter(
            server_args.speculative_algorithm, is_draft
        )
        # 块草稿器（DFlash / DSpark）不用 TRT-LLM 后端时，KV cache 固定为 bf16（第 11 阶段）
        if draft_block_decode and server_args.drafter_attention_backend != "trtllm":
            kv_cache_dtype = "bfloat16"

        hf_config = model_config.hf_config
        cache_layer_types = resolve_cache_layer_types(
            hf_config,
            num_layers=model_config.num_attention_layers,
            is_draft=is_draft,
            draft_block_decode=draft_block_decode,
        )
        # The retention window belongs to the storage labels: a block drafter
        # has none of its own (its window is a compute mask on its layers).
        # 译：保留窗口属于存储标签：块草稿器自己没有保留窗口（它的窗口是作用在其层上的计算掩码）。
        sliding_window_tokens = (
            None if draft_block_decode else getattr(hf_config, "sliding_window", None)
        )
        spec = cls(
            # 计算后端：目标模型取 --attention-backend，草稿模型取 --drafter-attention-backend；为 None 时由注册表选默认后端
            backend_name=(
                server_args.attention_backend
                if not is_draft
                else server_args.drafter_attention_backend
            ),
            num_attention_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            attn_tp_size=server_args.attn_tp_size or server_args.mapping.attn.tp_size,
            cache_layer_types=cache_layer_types,
            sliding_window_tokens=sliding_window_tokens,
            skip_softmax_threshold=server_args.skip_softmax_threshold,
        )
        return AttnConfig(
            components=(spec,),
            **model_wide_kwargs(
                server_args,
                model_config,
                is_draft,
                kv_cache_dtype=resolve_dtype(kv_cache_dtype),
                kv_cache_mxfp8=kv_cache_dtype == "mxfp8",
                draft_block_decode=draft_block_decode,
            ),
        )

    def cache_cell_size(self, config: AttnConfig) -> int:
        cell = (
            max(self.num_kv_heads // self.attn_tp_size, 1)
            * self.head_dim
            * 2
            * torch._utils._element_size(config.kv_cache_dtype)
        )
        if config.kv_cache_mxfp8:
            # One UE8M0 byte per 32 fp8 data bytes.
            # 译：每 32 个 fp8 数据字节配一个 UE8M0 字节。
            cell += cell // 32
        return cell
