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

"""Inference-only Qwen2 model compatible with HuggingFace weights."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import QKVParallelLinear, RowParallelLinear
from tokenspeed.runtime.layers.mlp import GatedMLP, gated_activation
from tokenspeed.runtime.layers.paged_attention import (
    PagedAttention,
    hf_sliding_window_to_window_left,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseDecoderLayer,
    BaseTransformerModel,
)
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.utils import add_prefix


# Qwen3稠密模型
#
# 执行主线
#   Rust控制面产生调度计划 -> HostDataPlane.execute -> DeviceHandle.execute
#   -> ModelExecutor.execute_forward_op -> ModelRunner.forward
#   -> 【Qwen3ForCasualLM.forward】 -> 采样
#
# shape记号
#   T：本次forward需要计算的token数
#       - prefill为所有req拼接后的长度
#       - decode为batch大小*每请求token数（普通解码每请求1token，投机解码每请求多个token）
#   B：本次forward的请求数（batch）
#   E：本次forward做extend(prefill)的请求数，其余B-E个请求做decode
#   H：hidden_size
#   Hq/Hkv：当前rank负责计算的Q/KV头数（已除rank）
#   D：head_dim
#   P：page_size，一个KV page存多少个token
#   NP：KV页总数
#   L：层数
#   V：词表大小
#
# 需要注意的地方
#   1. forward显示传ForwardContext（前向模式、计数、注意力后端、kv池），不存全局上下文变量
#   2. QK-Norm和RoPE交给PagedAttention处理，在投影之后、注意力计算之前一次性完成QK-Norm+RoPE+写KV cache
#   3. 注意力和MLP用两套并行参数，分别是mapping.attn和mapping.dense。单卡时两者tp=1，多卡时tp可以不同
#   4. o_proj和down_proj都不reduce，而是由每层的CommManager统一处理，尽量和写一个RMSNorm融合成一个kernel
#
# 模型的数据流
#   input_ids [T] -> embed_tokens [T, H]
#        │
#        ▼ x L 层 Qwen3DecoderLayer
#   ┌──────────────────────────────────────────────────────────┐
#   │ input_layernorm（与上一层的残差相加融合）                    │
#   │ Qwen3Attention：qkv_proj -> PagedAttention -> o_proj     │
#   │ post_attention_layernorm（与残差相加融合）                  │
#   │ GatedMLP：gate_up_proj -> SiluAndMul -> down_proj        │
#   └──────────────────────────────────────────────────────────┘
#        │
#        ▼
#   final_norm -> hidden [T, H]
#        │ LogitsProcessor：只取每个请求需要采样的那一行（prefill采最后一个token的logit，decode采唯一token的logit）
#        ▼
#   logits [B, V] -> 交给采样


# Qwen3一层的注意力模块：qkv投影+PagedAttention+o投影
# 构造时做两件事：按注意力TP切分头数，并把QK-Norm、RoPE交给PagedAttention
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        max_position_embeddings: int = 32768,
        quant_config: QuantizationConfig | None = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.hidden_size = hidden_size
        # 注意力部分的张量并行策略，提前生成好并通过mapping传递进来
        # 注意力部分和MLP部分用不同的并行策略
        self.tp_rank = self.mapping.attn.tp_rank
        self.tp_size = self.mapping.attn.tp_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        # 检查注意力头能否按TP切分
        # 情况一：KV 头数 ≥ tp_size，KV 头要被平分
        # 情况二：KV 头数 < tp_size，KV 头要被复制，保证每张卡分到的东西一样多、一样整齐
        #        每个 KV 头复制到 tp / num_kv 张卡上，每张卡正好拿 1 个 KV 头
        validate_attention_partition(
            self.total_num_heads,
            self.total_num_kv_heads,
            self.tp_size,
        )
        # 本rank负责的Q/KV头数
        # KV头不够分时，例如TP=16，8个kv头，8//16=0
        # 取 max(1, 0) = 1，每个KV头复制到 16 / 8 = 2 个 rank 上
        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # qkv拆成q、k、v时用
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # softmax前的缩放系数
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # 对每个头的q、k各做一次RMSNorm
        # 注意是每个头做归一化，维度是D，不是H
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Q、K、V三个投影合并成一个线性层，按头切分到注意力TP的各个rank
        # 本地权重[(Hq+2*Hkv)*D, H]，输出按[q|k|v]排列
        # 注意传入的是总头数，内部按照tp_size完成切分
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        # 输出投影：行并行，各rank的结果是部分和
        # 不做all-reduce，交给本层的CommManager
        # 它在tp>1时把all-reduce、残差相加、post_attn_layernorm融合成一个kernel
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            reduce_results=False,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

        # RoPE模块，含cos/sin缓存表，由PagedAttention模块调用
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # Visibility only; the cache plan decides storage (bound at startup).
        # 注意力能看到多少历史，-1表示能看到全部（全注意力）
        sliding_window_size = -1
        if config.layer_types[layer_id] == "sliding_attention":
            if config.sliding_window is None:
                raise ValueError(
                    "Qwen3 sliding_attention layers require config.sliding_window."
                )
            # HF 的 sliding_window 把当前 token 也算在内；kernel 要的是
            # “当前 token 之前还能看到几个 token”（window_left），所以减 1。
            sliding_window_size = hf_sliding_window_to_window_left(
                config.sliding_window
            )
        # 本层的注意力计算。接收投影后的q/k/v，自己完成：
        #   prologue：QK-Norm、RoPE、把k/v写入kv cache
        #   核心注意力：调用attn backend算softmax(QK^T)V
        # layer_id用于从KV池查找本层的kv cache
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            rotary_emb=self.rotary_emb,
            qk_norm=(self.q_norm, self.k_norm),
        )

    # 一层注意力的前向
    # positions [T]: 每个token在自己序列中的位置
    # hidden_states [T, H]：已经过input_layernorm的输入
    # ctx：上下文，透传给PagedAttention
    #
    # hidden_states [T, H]
    #        │ qkv_proj
    #        ▼
    # qkv [T, q_size + kv_size * 2]
    #        │ split
    #        ├── q [T, q_size]
    #        ├── k [T, kv_size]
    #        └── v [T, kv_size]
    #        ▼
    # self.attn(PagedAttention): QK-Norm+RoPE+写KV cache+核心注意力
    #        │ prefill：只往前看（因果）；decode：读已缓存的KV
    #        ▼
    # attn_output [T, H]
    #        │ o_proj（行并行，部分和）
    #        ▼
    # output [T, H]
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        # split得到的是视图，不是拷贝
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, positions, ctx)
        # 注意力后端可能返回 [T, Hq, D], o_proj需要[T, Hq*D]
        if len(attn_output.size()) == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output


# Qwen3的解码层：norm -> attn -> norm -> MLP
# 流程和norm模块全部由BaseDecoderLayer固定好，子类只需要提供attn、MLP模块
class Qwen3DecoderLayer(BaseDecoderLayer[Qwen3Config]):
    def resolve_attn(self, prefix: str) -> nn.Module:
        return Qwen3Attention(
            config=self.config,
            mapping=self.mapping,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            layer_id=self.layer_id,
            rope_theta=get_rope_theta(self.config, 1000000),
            rope_scaling=getattr(self.config, "rope_scaling", None),
            head_dim=getattr(self.config, "head_dim", None),
            max_position_embeddings=getattr(
                self.config, "max_position_embeddings", 32768
            ),
            quant_config=self.quant_config,
            rms_norm_eps=self.config.rms_norm_eps,
            attention_bias=self.config.attention_bias,
            # 权重名前缀，例如“model.layers.3.self_attn”，加载权重时按它匹配checkpoint
            prefix=add_prefix("self_attn", prefix),
        )

    # MLP模块
    #   gate_up_proj -> SiluAndMul -> down_proj
    #   Qwen3-8B，TP=1：[T, 4096] -> [T, 2 * 12288] -> [T, 12288] -> [T, 4096]
    # 并行参数去mapping.dense（MLP和attn的TP配置可以不同）
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


# Qwen3的主干：embedding -> L个Qwen3DecoderLayer -> final norm
# 流程全部继承BaseTransformerModel，这里只指定层类型
class Qwen3Model(BaseTransformerModel):
    layer_cls = Qwen3DecoderLayer

    # token id -> embedding。
    # scale_emb 是部分兼容 Qwen3 结构的 checkpoint 才带的配置项（embedding 乘一个常数）；
    # 标准 Qwen3 配置没有它，直接查表。
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.embed_tokens(input_ids) * self.config.scale_emb
        return self.embed_tokens(input_ids)


# 完整的 Qwen3 因果语言模型：主干 + lm_head + LogitsProcessor。
# 前向流程：主干算hidden [T, H] -> LogitsProcessor 去采样的行 [B, H] -> 乘 lm_head得到logits [B, V]
#
# 文件末尾的 EntryClass 把本类登记到模型注册表：HF 配置里
# architectures = ["Qwen3ForCausalLM"] 的 checkpoint 会用本类构建
class Qwen3ForCausalLM(BaseCausalLM):
    model_cls = Qwen3Model

    def __init__(
        self,
        config: Qwen3Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    # checkpoint权重名 -> 合并后的参数名，供BaseCasualLM.load_weights使用
    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:
        # Dots keep an encoder's fused qkv_proj from matching v_proj.
        return [
            (".qkv_proj.", ".q_proj.", "q"),
            (".qkv_proj.", ".k_proj.", "k"),
            (".qkv_proj.", ".v_proj.", "v"),
            (".gate_up_proj.", ".gate_proj.", 0),
            (".gate_up_proj.", ".up_proj.", 1),
        ]

    # 加载权重。通用流程在BaseCausalLM.load_weights：这里只处理一个特例
    # Qwen3-Embedding 的 checkpoint 权重名没有 "model." 前缀
    # （是 "layers.0..." 而不是 "model.layers.0..."），先补上前缀再交给通用流程。
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
    ) -> set[str]:
        # Qwen3-Embedding checkpoints omit the backbone's model prefix.
        if "Embedding" in self.config.name_or_path:
            weights = ((add_prefix(name, "model"), w) for name, w in weights)
        return super().load_weights(weights, **kwargs)

    # 下面两个方法用于投机解码：草稿模型和目标模型共用 embedding 和 lm_head 权重，不必各存一份
    # get_embed_and_head：作为目标模型时，交出自己的两份权重
    # set_embed_and_head：作为草稿模型时，删掉自己的两份权重，换成目标模型的
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        # 先删除再赋值：nn.Module 不允许直接把普通张量赋给已注册的 Parameter 名字
        # 相当于把 weight 从参数登记表里删掉，之后赋什么都不会被拦
        # 同时删掉 draft 自己原来那两份权重的引用。只要没有别的地方再引用它们，这两块显存就能被回收。
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        # 释放原来两份权重占用的显存（V*H，Qwen3-8B 约 1.2 GB / 份，bf16）
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = Qwen3ForCausalLM
