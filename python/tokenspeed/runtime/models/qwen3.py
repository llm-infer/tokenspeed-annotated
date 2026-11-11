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

# 译：仅用于推理的 Qwen2 模型，兼容 HuggingFace 权重。
# 注意：原文写作 Qwen2，本文件实际实现的是 Qwen3。

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

# Qwen3 稠密模型（Qwen3-0.6B ~ 32B 这些非 MoE 的版本），执行面注释的起点。
#
# 你在这里（执行面主线，详见 ANNOTATION_PLAN.md 第 2.1 节）
#   Rust 控制面产出一轮的计划 -> HostDataPlane.execute -> DeviceHandle.execute
#   -> ModelExecutor.execute_forward_op -> ModelRunner.forward
#   -> 【Qwen3ForCausalLM.forward】 -> 采样
#
# Shape记号（全项目统一，后续文件直接引用，不再重复定义）
#   T：本次前向的 token 总数（prefill 为所有请求拼接后的长度；decode 为 batch 大小 × 每请求 token 数）
#   B：本次前向的请求数（bs）
#   E：本次前向中做 prefill/extend 的请求数（num_extends），其余 B-E 个请求做 decode
#   H：hidden_size
#   Hq / Hkv：当前 rank 上 Q / KV 的头数（已除以 TP）
#   D：head_dim
#   P：page_size，一个 KV 页存多少个 token
#   NP：KV 页总数
#   L：层数
#   V：词表大小
#
# 贯穿全文的例子（之后所有走查都用这组数字）
#   模型：Qwen3-8B，H=4096，L=36，Q 头 32，KV 头 8（每 4 个 Q 头共享 1 个 KV 头），D=128，
#         intermediate=12288，V=151936
#   单卡（TP=1），P=64（示例取值，实际由启动时的 cache 规划决定）
#   一轮混合批：请求 A 新到达，prompt 300 个 token；请求 B 已在 decode，当前长度 1000
#     -> B=2，E=1，T=300+1=301（A 的 300 行在前，B 的 1 行在后）
#
# 本文件只是“拼零件”：结构上就是熟悉的 Qwen3（pre-norm、QK-Norm、GQA、SwiGLU），
# 值得注意的是这几处和直观写法不同的地方：
#   1. forward 显式接收 ctx（ForwardContext，前向上下文），本次前向的模式、计数、
#      注意力后端、KV 池都从它拿；不存在全局的上下文变量。
#   2. QK-Norm 和 RoPE 不在本文件里调用：q_norm / k_norm / rotary_emb 只是交给
#      PagedAttention，由它的 prologue（投影之后、核心注意力之前的统一入口）
#      一次完成 QK-Norm + RoPE + 写 KV cache。见 layers/paged_attention.py（第 3 阶段）。
#   3. 注意力和 MLP 用两套并行参数：注意力用 mapping.attn，MLP 用 mapping.dense。
#      单卡时两者都是 tp_size=1；多卡时它们可以不同（第 9 阶段）。
#   4. o_proj 和 down_proj 都设了 reduce_results=False：行并行之后的 all-reduce
#      不在线性层里做，而是由每层的 CommManager 统一处理，并尽量和下一个 RMSNorm
#      融合成一个 kernel。见 models/base/decoder_layer.py。
#
# 整个模型的数据流（单卡，T=301）
#   input_ids [T] ──embed_tokens──> hidden [T, H] = [301, 4096]
#        │
#        ▼  × L=36 层（Qwen3DecoderLayer，结构见 models/base/decoder_layer.py）
#   ┌──────────────────────────────────────────────────────────┐
#   │ input_layernorm（与上一层的残差相加融合）               │
#   │ Qwen3Attention：qkv_proj -> PagedAttention -> o_proj      │
#   │ post_attention_layernorm（与残差相加融合）              │
#   │ GatedMLP：gate_up_proj -> SiluAndMul -> down_proj         │
#   └──────────────────────────────────────────────────────────┘
#        │
#        ▼
#   final norm -> hidden [T, H]
#        │  LogitsProcessor：只取每个请求要采样的那一行（A 的第 300 行、B 的唯一一行）
#        ▼
#   logits [2, V] -> 交给采样


# Qwen3 一层的注意力模块：投影 + PagedAttention + 输出投影。
# 构造时做两件事：按注意力 TP 切分头数，并把 QK-Norm、RoPE 交给 PagedAttention。
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
        # 注意力部分的张量并行：本 rank 在注意力 TP 组中的编号和组大小。
        # 注意力可以和 MLP 用不同的 TP 宽度（mapping.attn vs mapping.dense），单卡时都是 1。
        self.tp_rank = self.mapping.attn.tp_rank
        self.tp_size = self.mapping.attn.tp_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        # 检查头数能否按 TP 切分：Q 头数必须被 tp_size 整除；
        # KV 头数 >= tp_size 时要被整除，KV 头数 < tp_size 时反过来要求 tp_size 被 KV 头数整除。
        validate_attention_partition(
            self.total_num_heads,
            self.total_num_kv_heads,
            self.tp_size,
        )
        # 本 rank 的头数（记号 Hq / Hkv）。
        #   Qwen3-8B，TP=1：Hq=32，Hkv=8；TP=2：Hq=16，Hkv=4
        #   KV 头不够分时（例如 TP=16，只有 8 个 KV 头）：8 // 16 = 0，
        #   取 max(1, 0) = 1，即每个 rank 仍持有 1 个 KV 头，每个 KV 头复制到 16 / 8 = 2 个 rank 上
        #   （复制由 QKVParallelLinear 在加载权重时完成，见 layers/linear.py）。
        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        # head_dim 以配置为准：Qwen3 的 head_dim 不一定等于 hidden_size / 头数
        # （Qwen3-0.6B：hidden=1024、16 个头，但 head_dim=128，Q 的总宽度 2048 > hidden）。
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 拆分 qkv 时用到的宽度：q_size = Hq*D，kv_size = Hkv*D
        #   Qwen3-8B，TP=1：q_size = 32*128 = 4096，kv_size = 8*128 = 1024
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # softmax 前的缩放系数 1/sqrt(D)
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # QK-Norm：对每个头的 q、k 各做一次 RMSNorm（归一化维度是 D，不是 H）。
        # 这里只创建模块，不在 forward 里调用：它们会交给下面的 PagedAttention。
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Q、K、V 三个投影合并成一个列并行线性层，按头切分到注意力 TP 组的各个 rank。
        # 本地权重形状 [(Hq + 2*Hkv) * D, H]，输出按 [q | k | v] 排列。
        # 注意传入的是总头数，切分由 QKVParallelLinear 内部按 tp_size 完成。
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
        # 输出投影：行并行，各 rank 的结果是部分和。
        # reduce_results=False：这里不做 all-reduce。求和交给本层的
        # CommManager.post_attn_reduce_norm，它在 TP>1 且条件允许时把 all-reduce、
        # 残差相加和 post_attention_layernorm 融合成一个 kernel；TP=1 时没有通信。
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

        # RoPE 模块（含 cos/sin 缓存表），根据 rope_scaling 选择具体变体（YaRN 等）。
        # 和 q_norm / k_norm 一样，只交给 PagedAttention，本文件不直接调用。
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # Visibility only; the cache plan decides storage (bound at startup).
        # 译：仅影响可见范围；存储由 cache 计划决定（在启动时绑定）。
        # 滑动窗口：-1 表示全注意力（能看到全部历史）。
        # 这个值只决定注意力“能看到”多少历史；KV 实际保留多少由启动时的 cache 规划决定
        # （滑动窗口层会被分到单独的 sliding_window cache group，第 14 阶段）。
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
        # 本层的注意力计算。它接收投影后的 q/k/v，自己完成：
        #   prologue：q/k 做 QK-Norm -> q/k 做 RoPE -> 把 k/v 写入 KV cache
        #   核心注意力：调用注意力后端（ctx.attn_backend）算 softmax(QK^T)V
        # num_kv_heads 传的是本 rank 的 KV 头数，GQA 的分组关系（Hq / Hkv = 4）由后端处理。
        # layer_id 用来在 KV 池里找到本层的 cache。
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

    # 一层注意力的前向（Qwen3-8B、TP=1、T=301）
    #   positions [T]：每个 token 在自己序列中的位置。
    #     A 的 300 行是 0..299；B 的 1 行是 1000（B 已有 1000 个 token，新 token 的位置是 1000）。
    #     由 InputBuffers 在 GPU 上算好传进来（第 4 阶段）。
    #   hidden_states [T, H]：已经过 input_layernorm 的输入。
    #   ctx：本次前向的上下文，这里只是透传给 PagedAttention。
    #
    #   hidden_states [T, H] = [301, 4096]
    #        │ qkv_proj：一次 GEMM 同时算出 Q、K、V
    #        ▼
    #   qkv [301, 4096 + 1024 + 1024]
    #        │ split
    #        ├── q [301, 4096] = 32 个 Q 头 × 128
    #        ├── k [301, 1024] =  8 个 KV 头 × 128
    #        └── v [301, 1024]
    #        ▼
    #   self.attn（PagedAttention）：QK-Norm + RoPE + 写 KV cache + 核心注意力
    #        │ A 的 300 行：各自只看 A 自己的前缀（因果）
    #        │ B 的 1 行：通过页表读出 B 已缓存的 1000 个 token 的 KV，再加上它自己
    #        ▼
    #   attn_output [301, 4096]（部分后端或计算路径返回 [301, 32, 128]，下面统一 reshape 成二维）
    #        │ o_proj（行并行，部分和；TP=1 时就是完整结果）
    #        ▼
    #   output [301, 4096]
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        # q_size = Hq*D = 4096，kv_size = Hkv*D = 1024；split 得到的是视图，不拷贝
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, positions, ctx)
        # 注意力后端可能返回 [T, Hq, D] 的三维结果，o_proj 需要 [T, Hq*D]
        if len(attn_output.size()) == 3:
            attn_output = attn_output.reshape(attn_output.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output


# Qwen3 的解码层。层内的流程（norm -> 注意力 -> norm -> MLP，以及残差和通信）
# 全部由 BaseDecoderLayer 固定好，子类只需要提供两个零件：注意力模块和 MLP 模块。
class Qwen3DecoderLayer(BaseDecoderLayer[Qwen3Config]):
    # 零件 1：本层的注意力模块。参数全部从 HF 配置读取；
    # getattr 带默认值是为了兼容缺少这些字段的旧配置。
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
            # 权重名前缀，例如 "model.layers.3.self_attn"，加载权重时按它匹配 checkpoint
            prefix=add_prefix("self_attn", prefix),
        )

    # 零件 2：本层的 MLP（SwiGLU）。
    #   gate_up_proj（列并行，gate 与 up 合并）-> SiluAndMul -> down_proj（行并行）
    #   Qwen3-8B，TP=1：[T, 4096] -> [T, 2*12288] -> [T, 12288] -> [T, 4096]
    # 并行参数取 mapping.dense（稠密 MLP 的 TP 组），可以和注意力的 TP 组不同。
    # reduce_results=False：down_proj 之后的求和同样交给 CommManager（post_mlp_fused）。
    # TP>1 且条件允许时（开启了 all-reduce 融合、token 数不超过阈值），它把这次 all-reduce
    # 推迟到下一层开头，和下一层的 input_layernorm 融合成一个 kernel。
    # fusions=()：不启用 GatedMLP 的可选融合（第 2 阶段 layers/mlp.py）。
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


# Qwen3 的主干：embedding -> L 个 Qwen3DecoderLayer -> final norm。
# 流程全部继承自 BaseTransformerModel，这里只指定层的类型。
class Qwen3Model(BaseTransformerModel):
    layer_cls = Qwen3DecoderLayer

    # token id -> embedding。
    # scale_emb 是部分兼容 Qwen3 结构的 checkpoint 才带的配置项（embedding 乘一个常数）；
    # 标准 Qwen3 配置没有它，直接查表。
    # 这是 HF 风格的接口，留给需要单独取 embedding 的调用方；
    # 仓库内的普通前向不经过它，而是在 BaseTransformerModel.forward 里直接调用 embed_tokens。
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.embed_tokens(input_ids) * self.config.scale_emb
        return self.embed_tokens(input_ids)


# 完整的 Qwen3 因果语言模型：主干 + lm_head + LogitsProcessor。
# 前向流程（BaseCausalLM.forward）：主干算出 hidden -> LogitsProcessor 只取要采样的行
# -> 乘 lm_head 得到 logits。本类只补充权重加载相关的细节。
#
# 文件末尾的 EntryClass 把本类登记到模型注册表：HF 配置里
# architectures = ["Qwen3ForCausalLM"] 的 checkpoint 会用本类构建（见 models/registry.py）。
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

    # checkpoint 权重名 -> 合并后参数名 的映射，供 BaseCausalLM.load_weights 使用。
    # 每项是 (合并参数名片段, checkpoint 名片段, shard_id)：
    #   checkpoint 里的 "model.layers.0.self_attn.k_proj.weight"
    #   -> 改名为 "model.layers.0.self_attn.qkv_proj.weight"，shard_id="k"
    #   -> 调用该参数的 weight_loader(param, tensor, "k")，把 k 写到 [q | k | v] 中 k 的那一段
    # gate/up 用整数 shard_id（0、1）表示在 [gate | up] 中的第几段。
    #
    # 片段两边带点号是有原因的：不带点号时 "v_proj" 会匹配到 "qkv_proj" 的子串，
    # 视觉编码器等已经自带合并 qkv_proj 的 checkpoint 就会被错误改名。
    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:
        # Dots keep an encoder's fused qkv_proj from matching v_proj.
        # 译：两边的点号防止编码器自带的合并 qkv_proj 匹配到 v_proj。
        return [
            (".qkv_proj.", ".q_proj.", "q"),
            (".qkv_proj.", ".k_proj.", "k"),
            (".qkv_proj.", ".v_proj.", "v"),
            (".gate_up_proj.", ".gate_proj.", 0),
            (".gate_up_proj.", ".up_proj.", 1),
        ]

    # 加载权重。通用流程在 BaseCausalLM.load_weights；这里只处理一个特例：
    # Qwen3-Embedding 的 checkpoint 权重名没有 "model." 前缀
    # （是 "layers.0..." 而不是 "model.layers.0..."），先补上前缀再交给通用流程。
    # weights 是 (名字, 张量) 的迭代器，这里用生成器表达式惰性改名，不会一次读入全部权重。
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
    ) -> set[str]:
        # Qwen3-Embedding checkpoints omit the backbone's model prefix.
        # 译：Qwen3-Embedding 的 checkpoint 省略了主干的 model 前缀。
        if "Embedding" in self.config.name_or_path:
            weights = ((add_prefix(name, "model"), w) for name, w in weights)
        return super().load_weights(weights, **kwargs)

    # 下面两个方法用于投机解码：草稿模型和目标模型共用 embedding 和 lm_head 权重，
    # 不必各存一份（第 11 阶段）。
    #   get_embed_and_head：作为目标模型时，交出自己的两份权重
    #   set_embed_and_head：作为草稿模型时，删掉自己的两份权重，换成目标模型的
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        # 先删除再赋值：nn.Module 不允许直接把普通张量赋给已注册的 Parameter 名字
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        # 释放原来两份权重占用的显存（V*H，Qwen3-8B 约 1.2 GB / 份，bf16）
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = Qwen3ForCausalLM
