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

"""Inference-only dense Llama model compatible with HuggingFace weights.

Covers Llama-2 / Llama-3 / Llama-3.1 / Llama-3.2 dense checkpoints whose
``config.architectures`` is ``["LlamaForCausalLM"]``. MoE and Eagle3 draft
variants have their own modules (``longcat_large.py``, ``llama_eagle3.py``).
"""

# 译：仅用于推理的稠密 Llama 模型，兼容 HuggingFace 权重。
#
#     覆盖 ``config.architectures`` 为 ``["LlamaForCausalLM"]`` 的 Llama-2 / Llama-3 / Llama-3.1 / Llama-3.2
#     稠密 checkpoint。MoE 和 Eagle3 草稿变体有各自的模块（``longcat_large.py``、``llama_eagle3.py``）。

from __future__ import annotations

import torch
from torch import nn
from transformers import LlamaConfig

from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.linear import QKVParallelLinear, RowParallelLinear
from tokenspeed.runtime.layers.mlp import GatedMLP, gated_activation
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseDecoderLayer,
    BaseTransformerModel,
)
from tokenspeed.runtime.models.utils import (
    validate_attention_partition,
)
from tokenspeed.runtime.utils import add_prefix

# 稠密 Llama 模型（Llama-2 / 3 / 3.1 / 3.2）。结构与 models/qwen3.py 基本相同，
# 读完 Qwen3 后可以对照着看，差异只有这几处：
#   1. 没有 QK-Norm：传给 PagedAttention 的 qk_norm=None
#   2. 全部是全注意力层（sliding_window_size=-1）
#   3. 注意力模块对空 batch 直接返回空张量（Qwen3 依赖基类在 IDLE 时跳过注意力）
#   4. LlamaAttention 把“PagedAttention 调用”和“RoPE 模块的创建”拆成可覆盖的方法
#      （_attn、resolve_rotary_emb），供 EAGLE3 草稿头（models/llama_eagle3.py）复用
#   5. 权重名映射的片段不带点号（"qkv_proj" 而不是 ".qkv_proj."）：纯文本 Llama 不含
#      自带合并 qkv_proj 权重的子模块（如视觉编码器），所以不会出现 Qwen3 注释里提到的
#      “v_proj 误匹配 qkv_proj”问题
#   6. supports_aux_capture=True：可以作为 EAGLE3 / DFlash 的目标模型（第 11 阶段）
# 其余（QKV 合并的列并行投影、行并行 o_proj 不做 all-reduce、GatedMLP、权重加载流程）
# 与 Qwen3 完全一致。


# Llama 一层的注意力：QKV 投影 -> PagedAttention（RoPE + 写 KV + 核心注意力）-> o_proj。
# qkv_input_size：QKV 投影的输入宽度，默认等于 hidden_size；
# EAGLE3 草稿头的输入是拼接后的更宽向量，会传入不同的值。
class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        qkv_input_size: int | None = None,
        *,
        sliding_window_size: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.attn_tp_size = mapping.attn.tp_size
        self.attn_tp_rank = mapping.attn.tp_rank
        attn_tp_group = mapping.attn.tp_group

        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        validate_attention_partition(
            self.total_num_heads,
            self.total_num_kv_heads,
            self.attn_tp_size,
        )
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Dense Llama is consistently bias-free (`attention_bias=False` in every
        # upstream release). Still read it off the config so forks that flip
        # the flag load without surprises.
        # 译：稠密 Llama 一直不带 bias（每个上游版本中都是 `attention_bias=False`）。仍然从配置中读取它，
        #     这样翻转了该标志的派生版本也能正常加载，不会出意外。
        attention_bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            qkv_input_size or hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=attn_tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=attn_tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = self.resolve_rotary_emb(config)
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            rotary_emb=self.rotary_emb,
            qk_norm=None,
        )

    # 创建 RoPE 模块；子类可以覆盖（例如返回 None 表示不用 RoPE）。
    # Llama-3.1 起 rope_scaling 为 llama3 类型，get_rope 据此选择对应的缩放变体。
    def resolve_rotary_emb(self, config: LlamaConfig) -> nn.Module | None:
        """The rotary embedding of q and k, or None for attention without one."""
        # 译：q 和 k 的旋转位置编码；对不使用它的注意力返回 None。
        return get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=get_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

    # 与 Qwen3Attention.forward 相同：qkv_proj -> split -> 注意力 -> o_proj，形状见 models/qwen3.py。
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        # Skip the QKV projection, RoPE, attention, and o_proj kernels when
        # the batch row is empty (e.g. idle ranks under DP attention). Matches
        # the short-circuit ``GatedMLP.forward`` already has.
        # 译：当 batch 没有行时（例如 DP attention 下的空闲 rank），跳过 QKV 投影、RoPE、注意力和 o_proj 的 kernel。
        #     与 ``GatedMLP.forward`` 中已有的提前返回保持一致。
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros(
                (0, self.hidden_size), dtype=hidden_states.dtype
            )
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self._attn(positions, q, k, v, ctx)
        output, _ = self.o_proj(attn_output)
        return output

    def _attn(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Attention before ``o_proj``; the Eagle3 draft head overrides it."""
        # 译：``o_proj`` 之前的注意力部分；Eagle3 草稿头会覆盖它。
        return self.attn(q, k, v, positions, ctx)


# Llama 解码层：只提供注意力和 MLP 两个零件，流程见 models/base/decoder_layer.py。
class LlamaDecoderLayer(BaseDecoderLayer):

    def resolve_attn(self, prefix: str) -> nn.Module:
        return LlamaAttention(
            config=self.config,
            mapping=self.mapping,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            layer_id=self.layer_id,
            quant_config=self.quant_config,
            prefix=add_prefix("self_attn", prefix),
            sliding_window_size=-1,
        )

    def resolve_mlp(self, prefix: str) -> nn.Module:
        return GatedMLP.tensor_parallel(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            activation=gated_activation(self.config.hidden_act, swiglu_limit=None),
            tp_rank=self.mapping.dense.tp_rank,
            tp_size=self.mapping.dense.tp_size,
            tp_group=self.mapping.dense.tp_group,
            reduce_results=False,
            fusions=(),
            quant_config=self.quant_config,
            prefix=add_prefix("mlp", prefix),
        )


# 主干：embedding -> L 个 LlamaDecoderLayer -> final norm，全部继承自基类。
class LlamaModel(BaseTransformerModel):
    layer_cls = LlamaDecoderLayer


# 完整模型：主干 + lm_head + LogitsProcessor，并提供 q/k/v、gate/up 的权重名映射。
class LlamaForCausalLM(BaseCausalLM):
    model_cls = LlamaModel
    supports_aux_capture = True

    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:
        return [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]


EntryClass = LlamaForCausalLM
