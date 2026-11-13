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


# 所有解码层的基类：基类把一层的流程（骨架）固定下来，子类负责提供零件（模块）
#   resolve_attn：注意力模块（必须）
#   resolve_mlp：MLP或MoE（必须）
#   resolve_norm：两个norm（默认RMSNorm）
# 
# 在Transformer中的位置：
# BaseTransformerModel.forward -> BaseDecoderLayer.forward 
#   -> attn / MLP
#
# 基础概念：层内通信由CommManager集中负责
#   注意力的 o_proj 和 MLP 的 down_proj 都是行并行
#   线性层自己不做 all-reduce。层与层之间所有的通信（all-reduce、reduce-scatter、
#   all-gather）以及它们和 RMSNorm 的融合，都集中在每层一个的 CommManager 里
#   单卡（TP=1）时，所有通信都是空操作，CommManager 退化为“相加残差 + RMSNorm”。
#
# 一层forward流程：
#   hidden [T, H], redidual [T, H] 或 None
#        │
#        │ forward_attn
#        │ 1. input_reduce_norm：residual+=hidden；hidden=input_layernorm(residual)
#        │ 2. capture_layer_input：保存residual
#        │ 3. pre_attn_comm：单卡为空操作
#        │ 4. self_attn：[T, H] -> [T, H]
#        │ 5. post_attn_reduce_norm: residual+=hidden；hidden=post_layernorm(residual)
#        ▼
#        │ forward_mlp
#        │ 6. pre_mlp_comm: 单卡为空操作
#        │ 7. mlp: [T, H] -> [T, H]
#        │ 8. post_mlp_fused: 单卡为空操作
#        ▼
#   hidden = MLP输出，residual = 注意力之后的残差流  ← 留给下一层的 ① 相加


# 在某些指定层的开头，把“这一层真正的输入”完整地存下来，留给 EAGLE3 草稿模型用
# - 为什么要捕获：EAGLE3 要中间层特征
#   EAGLE3 的草稿器不只看目标模型最后一层的输出。它会从目标模型的若干层取 hidden，通常是低、中、高三层
# - 为什么取的是 residual，而不是 hidden_states
#   residual是“上一层输出 + 残差”的完整结果，正是要捕获的东西
# - 为什么要 gather_residual
#   RSAG 布局下，residual 按 token 行切在各 rank 上，草稿器却要每个 token 的完整行
#   以 TP=2、4 个 token 为例：
#
#   普通 all-reduce：每个 rank 都有全部行
#     rank0: [t0 t1 t2 t3]        rank1: [t0 t1 t2 t3]
#
#   RSAG（只做 reduce-scatter）：每个 rank 只有 1/TP 的行
#     rank0: [t0 t1]              rank1: [t2 t3]
#                  \              /
#                   all-gather（gather_residual）
#                  /              \
#     rank0: [t0 t1 t2 t3]        rank1: [t0 t1 t2 t3]   ← 草稿器要的是这个
#
#   非 RSAG / 单卡时 residual 本来就完整，gather_residual 原样返回
# - 为什么要 clone
#   后面的 fused add+norm 会原地改写 residual（residual += attn_out），
#   只存引用的话，草稿器读到的就是被改过的值
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
    if aux_hidden_states is None:
        return
    # Under RSAG the residual is reduce-scattered across the attn TP group;
    # aux consumers (e.g. the EAGLE3 drafter) expect full rows. Clone because
    # later norms update the residual in place.
    aux_hidden_states.append(comm_manager.gather_residual(residual, ctx).clone())


_C = TypeVar("_C", bound=PretrainedConfig)


# 解码层基类
#   - input_layernorm/post_attention_layernorm：注意力前、MLP前的两个norm
#   - self_attn/mlp：子类提供
#   - comm_manager：本层的通信管理，他持有两个norm引用，这样才能把通信和norm融合
class BaseDecoderLayer(nn.Module, Generic[_C]):
    """Default decoder layer using CommManager for communication.

    Subclasses override ``resolve_attn()`` and ``resolve_mlp()``.
    """

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
        # 本层的层号，注意力模块通过layer_id在KV池找到本层的kv cache
        self.layer_id = layer_id
        self.total_layers = config.num_hidden_layers
        self.mapping = mapping

        self.input_layernorm = self.resolve_norm()
        self.post_attention_layernorm = self.resolve_norm()

        self.self_attn = self.resolve_attn(prefix)
        self.mlp = self.resolve_mlp(prefix)

        # 本层的通信管理
        # is_moe/pre_is_moe：本层/上一层的MLP模块是不是moe。MoE与dense MLP并行组可能不同
        #   决定本层出口用all-reduce还是reduce-scatter，以及入口是否要all-gather
        # 两个norm：把通信和norm融合
        # query_sharded=False：本层不按查询上下文并行切分 extend 行（QCP）
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

    @property
    def is_prev_moe_layer(self) -> bool:

        return self.is_moe_layer

    # 是否从「切分输入通道、计算部分和、最后相加」
    # 切换成「完整输入通道、切分输出列、最后交换数据」
    #
    # 假设 TP=2，计算 A @ W
    #
    # 原始矩阵：
    # A = [[1, 2, 3, 4],
    #      [5, 6, 7, 8]]
    #
    # W = [[1, 0],
    #      [0, 1],
    #      [1, 1],
    #      [2, 0]]
    #
    # =========================================================
    # 方案一：Reduce-Scatter（先拆分，再相加）
    # =========================================================
    #
    # 1. 按输入通道切分：
    #
    # GPU0:                         GPU1:
    # A0 = [[1, 2],                 A1 = [[3, 4],
    #       [5, 6]]                       [7, 8]]
    #
    # W0 = [[1, 0],                 W1 = [[1, 1],
    #       [0, 1]]                       [2, 0]]
    #
    # 2. 分别计算部分结果：
    #
    # GPU0: P0 = A0 @ W0            GPU1: P1 = A1 @ W1
    #       = [[1, 2],                    = [[11, 3],
    #          [5, 6]]                       [23, 7]]
    #
    # 3. Reduce：两个 GPU 的部分结果相加
    # Y = P0 + P1 = [[12, 5],
    #                [28, 13]]
    #
    # 4. Scatter：按 Token 行分发
    # GPU0: [[12, 5]]               GPU1: [[28, 13]]
    #       Token 0                      Token 1
    #
    #
    # =========================================================
    # 方案二：Transpose Rows（完整输入，切分输出列）
    # =========================================================
    #
    # 1. All-Gather：两张 GPU 获取完整 A
    #
    # GPU0:                         GPU1:
    # A = [[1, 2, 3, 4],            A = [[1, 2, 3, 4],
    #      [5, 6, 7, 8]]                 [5, 6, 7, 8]]
    #
    # 2. 按输出列切分 W：
    #
    # GPU0:                         GPU1:
    # W0 = [[1],                    W1 = [[0],
    #       [0],                          [1],
    #       [1],                          [1],
    #       [2]]                          [0]]
    #
    # 3. 分别计算完整输入对应的输出列：
    #
    # GPU0: Y0 = A @ W0             GPU1: Y1 = A @ W1
    #       = [[12],                      = [[5],
    #          [28]]                         [13]]
    #
    # 4. All-to-All：交换数据，重新按 Token 行排列
    #
    # GPU0:                         GPU1:
    # [[12], [5]]                   [[28], [13]]
    #      ↓                             ↓
    # [[12, 5]]                     [[28, 13]]
    #    Token 0                       Token 1
    #
    # 最终结果：
    # Y = [[12,  5],
    #      [28, 13]]
    #
    # 核心区别：
    # - Reduce-Scatter：GPU0、GPU1 各算一部分，最后相加。
    # - Transpose Rows：GPU0、GPU1 各算不同输出列，最后交换数据。
    @property
    def dense_batch_invariant(self) -> bool:
        """Whether the dense MLP tail transposes rows instead of
        reduce-scattering head/channel partials (``--tp-batch-invariant``)."""

        return False

    # 提供默认的RMSNorm，子类可覆盖
    def resolve_norm(self) -> nn.Module:

        return RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def resolve_attn(self, prefix: str) -> nn.Module:

        raise NotImplementedError

    def resolve_mlp(self, prefix: str) -> nn.Module:

        raise NotImplementedError

    # Decoder前半段：norm->注意力->norm
    #   输入：上一层的hidden_states, residual, 两者尚未相加
    #   输出：MLP的输入，注意力之后的残差流
    def forward_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 把上次段计算的MLP输出接回残差主干，再做input_layernorm
        #   hidden_states：上一段MLP的输出，形状通常为[T, H]
        #   residual：残差主干，形状为[T, H]，两者此时还没相加
        #
        # 先做残差相加，再做RMSNorm（融合）
        #   residual += hidden_states
        #   hidden_states = input_layernorm(residual)
        # 
        # 为什么多卡时，需要通信？
        #   行并行Linear会让不同的rank计算最终结果的”部分和“
        #   必须先跨rank做规约，才能得到完整的hidden_states
        #      GPU0 部分结果 [1, 2]  +  GPU1 部分结果 [3, 4]
        #                └──────── All-Reduce 求和 ────────┘
        #                         完整结果 [4, 6]
        #      residual [10, 20] + 完整结果 [4, 6] = [14, 26]
        #      hidden_states = RMSNorm([14, 26])；residual 保留 [14, 26]
        hidden_states, residual = self.comm_manager.input_reduce_norm(
            hidden_states, residual
        )

        # 在attn前捕获本层输入，留给草稿模型使用
        # 上一步不是已经规约为完整结果了吗？为什么还要all-reduce？
        # 上一步的完整可能只表示每个本地token的H维度完整
        #   - All-Reduce：每个rank已有全部T行
        #   - RSAG：每个rank只有自己的token行，eagle3需要完整的行，所以需要沿token维度all-gather  
        capture_layer_input(self.comm_manager, residual, ctx, aux_hidden_states)

        # 把hidden_states的token重新收集完整
        #   普通TP attn要求每个rank都有本轮完整的token行（T行）
        #   但如果上一层走Reduce-Scatter，每个rank只有自己负责的那部分token行
        #   因此进入attn前需要沿tp做token all-gather
        #       gather 前： GPU0 [t0, t1]       GPU1 [t2, t3]
        #       gather 后： GPU0 [t0,t1,t2,t3]  GPU1 [t0,t1,t2,t3]
        #
        # 是否真的通信由 comm_manager.needs_pre_attn_all_gather判断，通常要满足
        #   1. 不是query-sharded模式
        #   2. 不是第0层，第0层没有上一层的RSAG输出需要收集
        #   3. Attn TP已启用
        #   4. 上一层的Dense/MoE通信组与attn TP组大小不同，上一层采用RSAG输出
        # 如果上一层已经all-reduce、当前布局本来就有完整行，或tp=1/QCP，则无需通信
        hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)

        # TP时，各rank只负责一部分heads，产出的是最终hidden_states的部分贡献
        # 需要comm_manager做规约计算最终结果
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
        )

        # 处理attn输出，形成MLP输入
        #   attn的o_proj可能是行并行，不同的rank算出部分贡献
        #   需要按当前并行布局做必要的规约
        hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
            hidden_states, residual, ctx
        )

        return hidden_states, residual

    # 让空闲rank仍参与集合通信，否则其他rank可能等待
    # 如果还需要对齐某些属性，也可在此处执行
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

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:

        # 把hidden_states调整为MLP所需的并行布局
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)

        if self.is_moe_layer:
            # MoE：Token需要经过Router分配给不同Expert,计算后还要combile回来
            # num_global_tokens：本次计算的token数量；
            #   例：GPU0 当前有 3 个 Token，GPU1 有 2 个，则该组总数是 5。
            # max_num_tokens_per_gpu：单卡处理/接收Token上限, 做容量规划
            # 这两个值是实现路由所需的信息；Dense MLP 不需要这些参数。
            hidden_states = self.mlp(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu
            )

        else:
            # DenseMLP：所有token走同一套前馈网络
            # 典型流程是：linear -> 激活 -> Linear
            hidden_states = self.mlp(hidden_states)

        # 处理MLP的输出布局
        #   - Dense MLP：输出通常是行并行，各rank产生的是最终结果的部分和
        #   - MoE：需要combile各个expert的输出
        hidden_states, residual = self.comm_manager.post_mlp_fused(
            hidden_states, residual, ctx
        )

        return hidden_states

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # 从context读取本轮token统计
        # tp、dp时，rank看到的本地数据无法代表整个通信布局
        # 上游调度会把token统计信息写入ctx，交给CommManager统一算
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if not ctx.forward_mode.is_idle():
            # 普通模式：完整执行attn前处理，attn和attn后处理
            hidden_states, residual = self.forward_attn(
                positions,
                hidden_states,
                ctx,
                residual,
                aux_hidden_states,
            )

        else:
            # idle：不执行attn计算，调用hook只是为了对齐通信/状态
            self.forward_idle_attn(positions, hidden_states, ctx, aux_hidden_states)

        # 执行mlp
        # idle模式下，输入可能是空张量
        hidden_states = self.forward_mlp(
            hidden_states,
            residual,
            ctx,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )

        # hidden_states：MLP输出，作为下一层input_reduce_norm的输入
        # residual：残差主干，在下一层做残差相加，是为了在下一层融合add/norm/comm操作
        return hidden_states, residual


class BaseMoEDecoderLayer(BaseDecoderLayer):
    # MoE Decoder复用基类的模块和调度骨架
    # 只是告诉基类本层MLP是MoE，让CommManager按MoE的并行策略配置通信
    @property
    def is_moe_layer(self) -> bool:

        return True
