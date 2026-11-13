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


# 把一个层的输入交给本次前向的接收方（投机解码用）
# eagle3、dflash这类草稿模型需要模型中间层的hidden
# 默认情况下，hidden收集在aux_hidden_states列表里，前向结束后一起返回
# 如果草稿器挂了target_capture_sink，每捕获一个hidden就立即交给它
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
    capture_idx = capture_idx_map.get(layer_idx)
    if capture_idx is None or len(aux_hidden_states) != capture_idx + 1:
        return
    sink.on_target_capture(capture_idx, aux_hidden_states[capture_idx])


# 所有Transformer主干的公共基类：embedding -> L * Decoder -> final norm
#   embed_tokens: 词表并行的embedding
#   layers：L个解码层
#   norm：最后一层之后的norm
#   layers_to_capture：需要捕获输入hidden的层号（投机解码用） 
#
# 残差延迟相加：每个解码层返回(hidden_states, residual)，相加推迟到下一层开头，
#   和下一层的input_layernorm、add-reduce（tp并行时需要合并个rank的输出）融合
class BaseTransformerModel(nn.Module):

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
        # 要捕获输入hidden的层号，层号i捕获第i层的输入（第i-1层的输出）
        # L+1表示捕获final norm处的残差（最后一层的输出）
        # 由 BaseCausalLM.set_eagle3_layers_to_capture / set_dflash_layers_to_capture 设置
        self.layers_to_capture: list[int] = []
        # DFLASH: each capture layer's positional tap index (the draft
        # concatenates taps in this order); set_dflash_layers_to_capture().
        # DFLASH每个捕获层在草稿输入中的位置序号
        self._dflash_capture_idx_map: dict[int, int] = {}

    # embedding的切分跟随注意力TP
    # 因为embedding的输出直接进入第0层的注意力部分，两者的行布局要一致
    # 才能保证能直接对接
    #
    # 为什么embedding层要切分？embedding权重比较大，为了降低显存开销
    def resolve_embed(self, config: PretrainedConfig, prefix: str) -> nn.Module:
        return VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )

    # 创建L个解码层
    # 流水线并行的模型会覆盖本方法，把不属于本stage的层换成占位的PPMissingLayer
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

    # 主干前向
    #   input_ids [T]:本次前向所有token
    #   positions [T]：每个token在自己序列中的位置
    #   ctx：上下文，逐层透传
    #   input_embeds [T, H]：可选。调用方已经算好的embedding（例如拼入了图像特征的多模态输入）
    # 返回：hidden [T, H], 捕获的中间层hidden
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
            # 
            # 取出第一层，决定是否可以融合embedding和norm
            #
            # 为什么考虑融合embedding和norm？节省中间数据搬运和kernel launch
            #
            # 传统执行方式：one-by-one
            #   embedding局部输出
            #         ↓
            #   kernel1：allreduce
            #         ↓
            #   完整的embedding
            #         ↓
            #   kernel2：RMSNorm
            #         ↓
            #   归一化后的embedding
            #
            # 融合执行：
            #   embedding局部输出
            #         ↓
            #   fused kernel：Allreduce->RMS->归一化
            #         ↓
            #   归一化后的embedding
            first_layer = self.layers[0]
            # 判断是否可以融合
            #   条件1：TP>1，说明embedding输出需要allreduce
            #   条件2：第一层的comm_manager支持当前 token 数量下的allreduce+norm融合
            fuse_embed_reduce = (
                self.embed_tokens.tp_size > 1
                and first_layer.comm_manager.should_fuse(input_ids.shape[0])
            )
            # token ID -> embedding
            # 如果fuse_embed_reduce，则先不reduce，交给后续融合执行
            hidden_states = self.embed_tokens(
                input_ids, reduce_results=not fuse_embed_reduce
            )
            # 如果embedding跳过allreduce
            # 则需要初始化redisual张量
            if fuse_embed_reduce:
                residual = torch.zeros_like(hidden_states)
        # 保存投机解码所需的中间层 hidden
        # 默认：等待前向结束后，通过返回值返给调用方
        # sink：捕捉到的hidden立刻交给sink
        aux_hidden_states: list[torch.Tensor] = []

        # 串行执行decoder layers
        for i, layer in enumerate(self.layers):
            # 判断当前层是否需要捕获输入 hidden。
            capture = i in self.layers_to_capture
            # 执行当前 Decoder Layer。
            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                residual,
                aux_hidden_states=aux_hidden_states if capture else None,
            )

            # 将捕捉到的hidden实时交给dflash
            if capture and ctx.target_capture_sink is not None:
                notify_target_capture(
                    ctx.target_capture_sink,
                    self._dflash_capture_idx_map,
                    i,
                    aux_hidden_states,
                )

        # 执行最终 RMSNorm
        # idle 通常表示当前前向没有正常的有效计算任务，
        # 例如某些调度场景下需要维持执行流程但不处理有效 token。
        if not ctx.forward_mode.is_idle():

            # 最后一层的residual留到这里统一处理
            if residual is None:
                raise RuntimeError("residual is required for non-idle forward mode.")

            # 使用最后一层的通信管理器执行最终 Norm。
            # 为什么通过layer.comm_manager？
            # 最终norm可能需要与attn tp通信、residual合并等操作配合执行
            #   TP>1时，必须按照分片布局处理，不能直接add或norm
            hidden_states, final_residual = layer.comm_manager.final_norm(
                hidden_states, residual, ctx, self.norm
            )

            # Capture index num_layers + 1 (EAGLE3's id num_layers, DFLASH's
            # last layer id) selects the final norm's output residual as an
            # aux state, gathered across attn-TP as the in-loop captures are.
            
            # 捕获最终 Norm 对应的 residual 用于投机解码
            # 为什么单独捕获？
            #   EAGLE3 / DFLASH 等投机解码流程，
            #   可以根据配置将最终 residual 也作为草稿模型的输入特征。
            if (
                aux_hidden_states is not None
                and final_residual is not None
                and len(self.layers) + 1 in self.layers_to_capture
            ):
                # 将residual拼成attn tp对应的布局
                # 为什么residual需要gather？
                # residual可能是部分数据，需要gather拼装为投机解码需要的格式
                final_residual = layer.comm_manager.gather_residual(final_residual, ctx)
                aux_hidden_states.append(final_residual.clone())
                # 如果存在实时接收方，则立即通知它。
                if ctx.target_capture_sink is not None:
                    notify_target_capture(
                        ctx.target_capture_sink,
                        self._dflash_capture_idx_map,
                        len(self.layers) + 1,
                        aux_hidden_states,
                    )

        # 返回主干的输出
        # Hidden state：正常前向时经过final norm的最终hidden
        # aux_hidden_states：为投机解码捕获的中间层hidden
        return hidden_states, aux_hidden_states or None
