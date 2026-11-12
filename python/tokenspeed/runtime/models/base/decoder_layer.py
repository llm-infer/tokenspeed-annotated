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

"""Base decoder layer classes.

``BaseDecoderLayer`` uses CommManager for communication.
"""

# 译：解码层基类。
#
#     ``BaseDecoderLayer`` 使用 CommManager 处理通信。

from __future__ import annotations

from typing import Generic, TypeVar

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.quantization import QuantizationConfig as Q

# 所有解码层的公共基类。它把“一层”的流程固定下来，子类只提供零件：
#   resolve_attn()：注意力模块（必须覆盖）
#   resolve_mlp()：MLP 或 MoE 模块（必须覆盖）
#   resolve_norm()：两个 norm 的类型（默认 RMSNorm，可覆盖）
#
# 你在这里：BaseTransformerModel.forward 的层循环 -> 【BaseDecoderLayer.forward】
#           -> 注意力模块（如 Qwen3Attention）/ MLP 模块（如 GatedMLP）
# 形状记号见 models/qwen3.py 文件头。
#
# 基础概念：层内通信由 CommManager（层内通信管理）集中负责
#   注意力的 o_proj 和 MLP 的 down_proj 都是行并行、且都设置了 reduce_results=False，
#   线性层自己不做 all-reduce。层与层之间所有的通信（all-reduce、reduce-scatter、
#   all-gather）以及它们和 RMSNorm 的融合，都集中在每层一个的 CommManager 里。
#   这样多卡时可以灵活选择通信方式和融合时机，而模型代码不用跟着改。
#   单卡（TP=1）时，所有通信都是空操作，CommManager 退化为“相加残差 + RMSNorm”。
#
# 一层的流程（单卡，Qwen3-8B，T=301）。残差约定见 models/base/transformer_model.py 文件头：
# 输入的 (hidden, residual) 还没相加，本层开头补上。
#
#   (hidden [301,4096], residual [301,4096] 或 None)    ← 上一层的输出
#        │
#        │ forward_attn
#        │ ① input_reduce_norm：residual = hidden + residual；hidden = input_layernorm(residual)
#        │ ② capture_layer_input：如果本层要捕获，保存 residual（= 本层的输入）
#        │ ③ pre_attn_comm：单卡为空操作
#        │ ④ self_attn：注意力模块，[301,4096] -> [301,4096]
#        │ ⑤ post_attn_reduce_norm：residual = hidden + residual；hidden = post_attention_layernorm(residual)
#        ▼
#        │ forward_mlp
#        │ ⑥ pre_mlp_comm：单卡为空操作
#        │ ⑦ mlp：[301,4096] -> [301,4096]
#        │ ⑧ post_mlp_fused：单卡为空操作
#        ▼
#   (hidden = MLP 输出, residual = 注意力之后的残差流)    ← 留给下一层的 ① 相加
#
# 多卡（TP>1）时这几个点分别插入的通信（详见 distributed/comm_manager.py，第 2、9 阶段）：
#   ⑤ o_proj 的部分和在这里 all-reduce，条件允许时与相加、norm 融合成一个 kernel
#   ⑧ down_proj 的部分和在这里 all-reduce；条件允许时跳过，推迟到下一层的 ① 融合进去
#   ③ ⑥ 在 RSAG 布局下（层输出 reduce-scatter、各 rank 只持有部分行）负责 all-gather 回完整行
#
# 图示一：all-reduce 布局（注意力 TP = MLP TP = 2，2 张卡，T=301）
#   两个 rank 都持有全部 301 行，只是各算一半的头 / 一半的 intermediate。
#
#             rank 0                                     rank 1
#   (hidden 部分和 [301,4096], residual [301,4096])   （同左）
#        │ ① all-reduce(hidden) + residual，再 input_layernorm   ← 条件允许时三步融合成一个 kernel
#        │    ◄════════════════ all-reduce ════════════════►
#        ▼                                                  ▼
#   ④ 注意力：q 头 0~15、KV 头 0~3                  ④ 注意力：q 头 16~31、KV 头 4~7
#      o_proj 输出部分和 [301,4096]                    o_proj 输出部分和 [301,4096]
#        │ ⑤ all-reduce + 相加残差 + post_attention_layernorm（条件允许时融合）
#        │    ◄════════════════ all-reduce ════════════════►
#        ▼                                                  ▼
#   ⑦ MLP：intermediate 的前一半（6144）           ⑦ MLP：intermediate 的后一半（6144）
#      down_proj 输出部分和 [301,4096]                 down_proj 输出部分和 [301,4096]
#        │ ⑧ 融合时：什么也不做，带着部分和进入下一层的 ①
#        │   不融合时：在这里 all-reduce
#        ▼                                                  ▼
#   ③ ⑥ 在这种布局下都是空操作（每个 rank 已有全部行）
#
# 图示二：RSAG 布局（4 张卡：注意力 TP=2 × DP=2，MLP TP=4）
#   注意力 TP 宽度 2 ≠ MLP TP 宽度 4，所以用 reduce-scatter + all-gather。
#   attention DP：rank 0、1 一组处理批次 X（301 行），rank 2、3 一组处理批次 Y（200 行）。
#   层出口 reduce-scatter 后，每个 rank 只持有本组的一部分行（前面的 rank 多分一行）：
#     rank 0：X 的第 0~150 行（151 行）    rank 1：X 的第 151~300 行（150 行）
#     rank 2：Y 的第 0~99 行（100 行）     rank 3：Y 的第 100~199 行（100 行）
#
#        rank 0        rank 1              rank 2        rank 3
#   ① 各自对自己的那份行做 相加残差 + input_layernorm（行数 151 / 150 / 100 / 100）
#        │              │                    │              │
#   ③ 注意力 TP 组内 all-gather，拼回本组完整的行
#        ◄══ all-gather ══►                  ◄══ all-gather ══►
#     [301,4096]     [301,4096]          [200,4096]     [200,4096]
#   ④ 注意力（组内各算一半的头），o_proj 输出部分和
#   ⑤ 注意力 TP 组内 reduce-scatter，每个 rank 拿回自己那份行的完整和，再 相加残差 + norm
#        ◄═ reduce-scatter ═►                ◄═ reduce-scatter ═►
#     [151,4096]     [150,4096]          [100,4096]     [100,4096]
#   ⑥ MLP TP 组（4 个 rank）内 all-gather，每个 rank 都拿到全部 501 行
#        ◄═══════════════════════ all-gather ═══════════════════════►
#     [501,4096]     [501,4096]          [501,4096]     [501,4096]
#   ⑦ MLP（各算 1/4 的 intermediate，即 3072），down_proj 输出部分和 [501,4096]
#   ⑧ MLP TP 组内 reduce-scatter，每个 rank 拿回自己那份行
#        ◄═════════════════════ reduce-scatter ═════════════════════►
#     [151,4096]     [150,4096]          [100,4096]     [100,4096]    ← 进入下一层的 ①
#
#   第 0 层略有不同：embedding 输出的是本组的完整行，所以 ③ 不需要 all-gather，
#   ⑤ 的 reduce-scatter 之后，残差直接从完整行中切出本 rank 的那份。


# 如果本层需要捕获（投机解码的草稿模型要用目标模型的中间 hidden，第 11 阶段），
# 把本层的输入追加到 aux_hidden_states；否则什么也不做。
#   residual：① 之后的残差，即上一层输出与残差相加后的结果 = 本层真正的输入。
#     即使上一层的 all-reduce 被推迟、融合进了 ①，此时它也已经是完整求和后的值。
def capture_layer_input(
    comm_manager: CommManager,
    residual: torch.Tensor,
    ctx: ForwardContext,
    aux_hidden_states: list[torch.Tensor] | None,
) -> None:
    """Append a capture layer's input to ``aux_hidden_states`` as full rows.

    Args:
        comm_manager: The layer's communication policy; gathers RSAG rows.
        residual: The residual right after the layer's opening (all-reduce +)
            add, i.e. the reduced layer input even when the previous layer's
            MLP all-reduce was fused into that add.
        ctx: The forward context.
        aux_hidden_states: The taps collected so far; ``None`` when this layer
            is not captured.
    """
    # 译：把捕获层的输入以完整行的形式追加到 ``aux_hidden_states``。
    #
    #     参数：
    #         comm_manager：本层的通信策略；负责收集 RSAG 布局下的行。
    #         residual：本层开头（all-reduce +）相加之后的残差，即求和后的本层输入，
    #             即使上一层 MLP 的 all-reduce 被融合进了这次相加也是如此。
    #         ctx：前向上下文。
    #         aux_hidden_states：到目前为止收集到的捕获结果；本层不需要捕获时为 ``None``。
    if aux_hidden_states is None:
        return
    # Under RSAG the residual is reduce-scattered across the attn TP group;
    # aux consumers (e.g. the EAGLE3 drafter) expect full rows. Clone because
    # later norms update the residual in place.
    # 译：在 RSAG 布局下，残差在注意力 TP 组内做了 reduce-scatter（各 rank 只持有部分行）；
    #     辅助状态的使用方（例如 EAGLE3 草稿器）需要完整的行。要 clone，是因为后面的 norm
    #     会原地更新残差。
    # RSAG 布局下各 rank 只持有部分行，草稿模型需要完整的行，所以先 all-gather；
    # 单卡时 gather_residual 原样返回。clone 是因为后面的 fused add+norm 会原地改写 residual。
    aux_hidden_states.append(comm_manager.gather_residual(residual, ctx).clone())


# 解码层的配置类型参数：子类写 BaseDecoderLayer[Qwen3Config]，self.config 就有准确的类型提示
_C = TypeVar("_C", bound=PretrainedConfig)


# 解码层基类。成员：
#   input_layernorm / post_attention_layernorm：注意力前、MLP 前的两个 norm
#   self_attn / mlp：子类通过 resolve_attn / resolve_mlp 提供
#   comm_manager：本层的通信管理；它持有两个 norm 的引用，这样才能把通信和 norm 融合
class BaseDecoderLayer(nn.Module, Generic[_C]):
    """Default decoder layer using CommManager for communication.

    Subclasses override ``resolve_attn()`` and ``resolve_mlp()``.
    """

    # 译：默认的解码层，使用 CommManager 处理通信。
    #
    #     子类覆盖 ``resolve_attn()`` 和 ``resolve_mlp()``。

    def __init__(
        self,
        config: _C,
        layer_id: int,
        mapping: Mapping,
        quant_config: Q | None = None,
        prefix: str = "",
    ) -> None:

        super().__init__()

        self.config = config
        self.quant_config = quant_config
        # 全局层号（流水线并行时也是全局编号），注意力模块用它在 KV 池里找到本层的 cache
        self.layer_id = layer_id
        self.total_layers = config.num_hidden_layers
        self.mapping = mapping

        self.input_layernorm = self.resolve_norm()
        self.post_attention_layernorm = self.resolve_norm()

        # 子类在 resolve_* 里可以读取 self.config、self.layer_id 等，
        # 所以这两行必须在上面的字段都赋值之后
        self.self_attn = self.resolve_attn(prefix)
        self.mlp = self.resolve_mlp(prefix)

        # 本层的通信管理。它需要知道：
        #   is_moe / prev_is_moe：本层、上一层的 MLP 是否为 MoE。MoE 与稠密 MLP 的并行组可能不同，
        #     决定本层出口用 all-reduce 还是 reduce-scatter，以及入口是否要 all-gather
        #   两个 norm：用于把通信和 norm 融合
        #   query_sharded=False：本层不按查询上下文并行切分 extend 行（QCP，第 12 阶段）；
        #     CommManager 会检查它与 mapping 是否一致
        self.comm_manager = CommManager(
            mapping=self.mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=self.is_prev_moe_layer,
            dense_batch_invariant=self.dense_batch_invariant,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
            query_sharded=False,
        )

    # 本层的 MLP 是否为 MoE。稠密模型恒为 False；MoE 层用 BaseMoEDecoderLayer（文件末尾）。
    @property
    def is_moe_layer(self) -> bool:

        return False

    # 上一层是否为 MoE。默认假设所有层同构（与本层相同）；
    # 稠密层与 MoE 层交替的模型需要覆盖它。
    @property
    def is_prev_moe_layer(self) -> bool:

        return self.is_moe_layer

    # 是否启用“批大小无关”的稠密 MLP 收尾方式（--tp-batch-invariant，第 9 阶段）。
    # 默认关闭。
    @property
    def dense_batch_invariant(self) -> bool:
        """Whether the dense MLP tail transposes rows instead of
        reduce-scattering head/channel partials (``--tp-batch-invariant``)."""
        # 译：稠密 MLP 的收尾是否改为转置行，而不是对按头/通道切分的部分和做 reduce-scatter
        #     （``--tp-batch-invariant``）。

        return False

    def resolve_norm(self) -> nn.Module:

        return RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def resolve_attn(self, prefix: str) -> nn.Module:

        raise NotImplementedError

    def resolve_mlp(self, prefix: str) -> nn.Module:

        raise NotImplementedError

    # 层的前半段：norm -> 注意力 -> norm。步骤编号对应文件头的流程图。
    #   输入：上一层的 (hidden_states, residual)，二者尚未相加
    #   输出：(MLP 的输入, 注意力之后的残差流)
    def forward_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ① 补上上一层欠下的残差相加，再做 input_layernorm（多卡时可能融合 all-reduce）
        #   第 0 层 residual 为 None：residual = embedding，hidden = norm(embedding)
        hidden_states, residual = self.comm_manager.input_reduce_norm(
            hidden_states, residual
        )

        # ② 此时 residual 就是本层的输入，需要捕获的层在这里保存
        capture_layer_input(self.comm_manager, residual, ctx, aux_hidden_states)

        # ③ RSAG 布局下把上一层留下的部分行 all-gather 回完整行；其余情况原样返回
        hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)

        # ④ 注意力模块：投影、prologue（QK-Norm、RoPE、写 KV）、核心注意力、o_proj
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
        )

        # ⑤ o_proj 的部分和在这里求和（多卡），然后相加残差、做 post_attention_layernorm
        #   residual = attn_out + residual；hidden = post_attention_layernorm(residual)
        hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
            hidden_states, residual, ctx
        )

        return hidden_states, residual

    # 空跑前向（IDLE）的注意力部分：本 rank 没有任何行，默认什么也不做。
    # attention DP 下其他 rank 有请求时，本 rank 仍要跑一遍，以便参与 MLP / MoE 中的集合通信；
    # 如果某个模型的注意力部分自己也有集合通信，就覆盖这个方法，在这里参与。
    def forward_idle_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        aux_hidden_states: list | None,
    ) -> None:
        """Run the attention half of an idle forward, which has no rows.

        Nothing by default; a layer whose attention joins collectives, or
        whose tap must stay aligned across ranks, runs them here.
        """
        # 译：运行空跑前向（没有任何行）的注意力部分。
        #
        #     默认什么也不做；如果某个层的注意力要参与集合通信，或者它的捕获结果必须在各 rank
        #     之间保持对齐，就在这里执行这些操作。

    # 层的后半段：MLP（或 MoE）。
    #   num_global_tokens / max_num_tokens_per_gpu：只有 MoE 用到，
    #   用于确定跨 rank 分发 token 时的缓冲区大小（第 10 阶段）
    # 返回 MLP 的输出；残差不在这里相加，留给下一层开头（或 final norm）。
    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:

        # ⑥ MLP 的并行组和注意力不同时（或 RSAG 布局下），先 all-gather 出 MLP 需要的行
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)

        # ⑦ MLP / MoE 本身
        if self.is_moe_layer:

            hidden_states = self.mlp(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu
            )

        else:

            hidden_states = self.mlp(hidden_states)

        # ⑧ down_proj 的部分和在这里求和（all-reduce 或 reduce-scatter）。
        #   条件允许融合时这里什么也不做，求和推迟到下一层的 ①，和相加、norm 一起完成。
        #   返回的 residual 未被修改（下面也没有使用它）。
        hidden_states, residual = self.comm_manager.post_mlp_fused(
            hidden_states, residual, ctx
        )

        return hidden_states

    # 一层的完整前向。
    #   positions [T]、hidden_states [T, H]、residual [T, H] 或 None、ctx：见 BaseTransformerModel.forward
    #   aux_hidden_states：本层需要捕获时传入收集列表，否则为 None
    # 返回 (hidden_states, residual)，二者尚未相加。
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 本次前向在 MLP 所在通信组里的 token 总数，以及单个 rank 上最多的 token 数。
        # 单卡时两者都等于 T（301）；只有 MoE 层会用到。
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if not ctx.forward_mode.is_idle():

            hidden_states, residual = self.forward_attn(
                positions,
                hidden_states,
                ctx,
                residual,
                aux_hidden_states,
            )

        else:

            # 空跑前向跳过注意力（没有行可算），但 MLP 部分照常执行：
            # MoE 的 token 分发、通信要求所有 rank 一起参与，缺一个就会卡住
            self.forward_idle_attn(positions, hidden_states, ctx, aux_hidden_states)

        hidden_states = self.forward_mlp(
            hidden_states,
            residual,
            ctx,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )

        return hidden_states, residual


# MoE 解码层的基类：只把 is_moe_layer 改为 True，CommManager 和 forward_mlp 据此
# 选择 MoE 的通信组和调用方式。MoE 模型的解码层继承它（第 10 阶段）。
class BaseMoEDecoderLayer(BaseDecoderLayer):

    @property
    def is_moe_layer(self) -> bool:

        return True
