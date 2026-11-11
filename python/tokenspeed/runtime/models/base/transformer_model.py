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

"""Base transformer model: embed -> layers -> norm."""

# 译：Transformer 主干基类：embed -> 各层 -> norm。

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext, TargetCaptureSink
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.base.decoder_layer import BaseDecoderLayer
from tokenspeed.runtime.utils import add_prefix, make_layers

# 所有 Transformer 主干的公共基类：embedding -> L 个解码层 -> final norm。
# 具体模型（如 Qwen3Model）只指定 layer_cls，必要时覆盖 resolve_embed / resolve_layers。
#
# 你在这里：BaseCausalLM.forward -> 【BaseTransformerModel.forward】 -> 每个解码层
# 形状记号见 models/qwen3.py 文件头。
#
# 基础概念：层与层之间“残差延迟相加”的约定
#   每个解码层返回一对 (hidden_states, residual)，它们还没有相加：
#     本层真正的输出 = hidden_states + residual
#   相加推迟到下一层开头，和下一层的 input_layernorm 在同一个 kernel 里完成
#   （“add residual + RMSNorm”融合）。最后一层之后没有下一层了，
#   所以循环结束后由 final_norm 完成最后一次“相加 + 归一化”。
#
#   第 i 层开头（CommManager.input_reduce_norm）：
#     residual'      = hidden_states + residual      ← 补上第 i-1 层欠下的残差相加
#     hidden_states' = RMSNorm(residual')            ← 本层注意力的输入
#   第 0 层开头 residual 为 None，此时不相加：residual' = 输入的 embedding。
#
#   多卡时还有一层含义：hidden_states 可能只是行并行的“部分和”，all-reduce 也被推迟到
#   下一层开头，和相加、归一化融合成一个 kernel（条件允许时，见下面 forward 里的说明）。
#
# 数据流（单卡，Qwen3-8B，T=301）
#   input_ids [301] ──embed_tokens──> hidden [301, 4096]，residual = None
#        │
#        ▼ 第 0 层 ... 第 35 层：(hidden, residual) = layer(positions, hidden, ctx, residual)
#        │
#        ▼ final_norm：hidden = RMSNorm(hidden + residual)
#   hidden [301, 4096] ──> 返回给 BaseCausalLM，交给 LogitsProcessor


# 把刚捕获的一个层输入交给本次前向的接收方（投机解码用，第 11 阶段回访）。
#
# 背景：EAGLE3 / DFlash 这类草稿模型需要目标模型若干中间层的 hidden（“层输入捕获”）。
# 默认情况下捕获结果只是收集在 aux_hidden_states 列表里，前向结束后一起返回；
# 如果草稿器在 ctx 上挂了 target_capture_sink，就每捕获一个立即交给它。
#   capture_idx_map：层号 -> 该层在草稿模型输入里的位置（第几个捕获）
#   只有当列表长度正好等于 capture_idx + 1（即这一层刚刚追加了一项）时才交出去；
#   空跑前向（idle）会跳过注意力部分、不追加，这时就什么也不做。
def notify_target_capture(
    sink: TargetCaptureSink,
    capture_idx_map: dict[int, int],
    layer_idx: int,
    aux_hidden_states: list[torch.Tensor],
) -> None:
    """Hand the tap capture layer ``layer_idx`` just appended to the
    forward's sink.

    Skipped when the capture layer appended no tap: an idle forward skips
    BaseDecoderLayer's attention block, so nothing was appended. Layers that
    append an empty tap on an idle forward still hand over the zero-row tensor.

    Args:
        sink: The forward's ``ctx.target_capture_sink``.
        capture_idx_map: Capture layer index to the draft's positional tap
            index; layers absent from it are not handed over.
        layer_idx: The capture layer that just ran.
        aux_hidden_states: The taps collected so far in this forward.
    """
    # 译：把捕获层 ``layer_idx`` 刚追加的捕获结果交给本次前向的接收方（sink）。
    #
    #     如果该捕获层没有追加结果就跳过：空跑前向会跳过 BaseDecoderLayer 的注意力部分，
    #     因此什么也没追加。在空跑前向中追加了空结果的层，仍会交出这个零行张量。
    #
    #     参数：
    #         sink：本次前向的 ``ctx.target_capture_sink``。
    #         capture_idx_map：捕获层号 -> 草稿模型输入中的位置序号；不在其中的层不交出。
    #         layer_idx：刚运行完的捕获层。
    #         aux_hidden_states：本次前向到目前为止收集到的捕获结果。
    capture_idx = capture_idx_map.get(layer_idx)
    if capture_idx is None or len(aux_hidden_states) != capture_idx + 1:
        return
    sink.on_target_capture(capture_idx, aux_hidden_states[capture_idx])


# Transformer 主干。成员：
#   embed_tokens：词表并行的 embedding（按注意力 TP 组切分词表）
#   layers：L 个解码层（nn.ModuleList），类型由子类的 layer_cls 指定
#   norm：最后一层之后的 RMSNorm
#   layers_to_capture：需要捕获输入 hidden 的层号（投机解码用；默认为空，不捕获）
class BaseTransformerModel(nn.Module):

    # 子类覆盖为自己的解码层类型，例如 Qwen3Model.layer_cls = Qwen3DecoderLayer
    layer_cls: type[BaseDecoderLayer] = BaseDecoderLayer

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:

        super().__init__()

        self.config = config
        self.quant_config = quant_config
        self.mapping = mapping
        self.padding_idx: int | None = getattr(config, "pad_token_id", None)
        self.vocab_size: int = config.vocab_size

        self.embed_tokens = self.resolve_embed(config, prefix)
        self.layers = self.resolve_layers(config, quant_config, prefix)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 要捕获“输入 hidden”的层号。层号 i 捕获的是第 i 层的输入（= 第 i-1 层的输出），
        # 特殊值 L+1 表示捕获 final norm 处的残差（= 最后一层的输出）。
        # 由 BaseCausalLM.set_eagle3_layers_to_capture / set_dflash_layers_to_capture 设置。
        self.layers_to_capture: list[int] = []
        # DFLASH: each capture layer's positional tap index (the draft
        # concatenates taps in this order); set_dflash_layers_to_capture().
        # 译：DFLASH：每个捕获层在草稿输入中的位置序号（草稿模型按这个顺序拼接各个捕获结果）；
        #     由 set_dflash_layers_to_capture() 设置。
        self._dflash_capture_idx_map: dict[int, int] = {}

    # embedding 的切分跟随注意力 TP 组（mapping.attn），因为 embedding 的输出
    # 直接进入第 0 层的注意力部分，二者的行布局要一致。
    def resolve_embed(self, config: PretrainedConfig, prefix: str) -> nn.Module:
        return VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )

    # 创建 L 个解码层。make_layers 按全局层号逐个调用工厂函数，
    # 权重名前缀依次为 "model.layers.0"、"model.layers.1"……
    # 这里没有传流水线并行的层范围，所以本 rank 持有全部层；
    # 支持流水线并行的模型会覆盖本方法，把不归本 stage 的层换成占位的 PPMissingLayer
    # （第 9 阶段）。
    def resolve_layers(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.ModuleList:
        return make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: self.layer_cls(
                config=config,
                layer_id=idx,
                mapping=self.mapping,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )

    # 主干的前向。
    #   input_ids [T]：本次前向所有 token 的 id，所有请求首尾拼接（A 的 300 个在前，B 的 1 个在后）
    #   positions [T]：每个 token 在自己序列中的位置（A: 0..299，B: 1000）
    #   ctx：前向上下文，逐层透传
    #   input_embeds [T, H]：可选，调用方已经算好的 embedding（例如拼入了图像特征的多模态输入），
    #     给了就跳过 embed_tokens
    # 返回：(hidden [T, H]，捕获的中间 hidden 列表或 None)
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:

        hidden_states = input_embeds
        residual = None

        if input_embeds is None:
            # When TP > 1 and fused allreduce+norm is available, skip the
            # NCCL allreduce in the embedding and let the first decoder layer
            # fuse it with the input layernorm via the fused all-reduce kernel.
            # 译：当 TP > 1 且可以使用 all-reduce + norm 融合时，跳过 embedding 中的 NCCL all-reduce，
            #     让第一个解码层通过融合的 all-reduce kernel 把它和 input layernorm 一起完成。
            # 词表并行的 embedding：每个 rank 只查自己那段词表，其余位置为 0，
            # 本来要 all-reduce 才得到完整的 embedding。
            # TP>1 且第 0 层能用“all-reduce + 相加 + RMSNorm”融合 kernel 时
            # （should_fuse 还会检查本次 token 数是否超过融合阈值），就先不 reduce：
            #   - embed_tokens(reduce_results=False) 返回本 rank 的部分结果
            #   - residual 置为全 0
            #   - 第 0 层开头的融合 kernel 计算 all_reduce(hidden) + 0，再做 RMSNorm
            # 效果和“先 all-reduce，再 residual=None 走普通路径”相同，但少一次单独的通信 kernel。
            # 单卡（tp_size=1）时 fuse_embed_reduce 恒为 False，走普通路径。
            first_layer = self.layers[0]
            fuse_embed_reduce = (
                self.embed_tokens.tp_size > 1
                and first_layer.comm_manager.should_fuse(input_ids.shape[0])
            )
            hidden_states = self.embed_tokens(
                input_ids, reduce_results=not fuse_embed_reduce
            )
            if fuse_embed_reduce:
                residual = torch.zeros_like(hidden_states)

        # 本次前向捕获到的中间 hidden（只有 layers_to_capture 非空时才会追加）
        aux_hidden_states: list[torch.Tensor] = []

        for i, layer in enumerate(self.layers):

            capture = i in self.layers_to_capture

            # 每层接收上一层的 (hidden, residual)，返回本层的 (hidden, residual)，
            # 二者的相加留给下一层（见文件头“残差延迟相加”）。
            # 只有要捕获的层才传入列表，层内会把自己的输入追加进去。
            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                residual,
                aux_hidden_states=aux_hidden_states if capture else None,
            )

            if capture and ctx.target_capture_sink is not None:
                notify_target_capture(
                    ctx.target_capture_sink,
                    self._dflash_capture_idx_map,
                    i,
                    aux_hidden_states,
                )

        # 空跑前向（IDLE：attention DP 下本 rank 没有请求，只为配合其他 rank 的集合通信）
        # 没有任何行，不需要 final norm。
        if not ctx.forward_mode.is_idle():

            if residual is None:
                raise RuntimeError("residual is required for non-idle forward mode.")

            # 最后一次“相加 + 归一化”：hidden = RMSNorm(hidden + residual)。
            # 借用最后一层的 CommManager 来做，因为只有它知道最后一层输出的通信状态：
            # 是否还欠一次 all-reduce（可融合进这次 norm），
            # 或者各 rank 只持有部分行、需要先 all-gather 回完整行（RSAG 布局，第 9 阶段）。
            # final_residual 是相加之后、归一化之前的值，即最后一层真正的输出。
            hidden_states, final_residual = layer.comm_manager.final_norm(
                hidden_states, residual, ctx, self.norm
            )

            # Capture index num_layers + 1 (EAGLE3's id num_layers, DFLASH's
            # last layer id) selects the final norm's output residual as an
            # aux state, gathered across attn-TP as the in-loop captures are.
            # 译：捕获层号 num_layers + 1（即 EAGLE3 的 num_layers 号、DFLASH 的最后一层号）选中 final norm
            #     输出的残差作为辅助状态，并像循环中的捕获一样在注意力 TP 组内收集成完整行。
            # 层号 L+1 表示捕获最后一层的输出。各 rank 可能只持有部分行，
            # gather_residual 先收集成完整行；clone 是因为后续计算可能原地改写它。
            if (
                aux_hidden_states is not None
                and final_residual is not None
                and len(self.layers) + 1 in self.layers_to_capture
            ):
                final_residual = layer.comm_manager.gather_residual(final_residual, ctx)
                aux_hidden_states.append(final_residual.clone())
                if ctx.target_capture_sink is not None:
                    notify_target_capture(
                        ctx.target_capture_sink,
                        self._dflash_capture_idx_map,
                        len(self.layers) + 1,
                        aux_hidden_states,
                    )

        # 没有捕获时返回 None 而不是空列表，调用方据此判断是否有中间 hidden
        return hidden_states, aux_hidden_states or None
