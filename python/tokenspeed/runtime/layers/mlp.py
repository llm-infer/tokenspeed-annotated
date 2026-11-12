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

"""Gated MLP shared by dense layers and shared experts."""

# 译：稠密层和共享专家共用的门控 MLP。

# 门控 MLP（SwiGLU MLP）：down_proj(act(gate_up_proj(x)))，gate 和 up 两个投影合并成一个。
# 结构本身是熟悉的：gate_up_proj（列并行）-> 门控激活 -> down_proj（行并行）。
#
# 你在这里：BaseDecoderLayer.forward_mlp 的 ⑦（models/base/decoder_layer.py）；
#           MoE 模型的共享专家（shared expert）也复用本类（第 10 阶段）。
#
# 本文件新增的内容是“融合 kernel”（GatedMLPFusion）：权重为 FP4 量化格式时，
# 可以在 gate_up 的 GEMM kernel 里直接算完激活（甚至顺带把结果量化好给 down_proj），
# 省掉中间结果写回显存再读出的开销。是否启用由模型显式传入的 fusions 列表决定。
#
# Qwen3-8B、bf16、TP=1：x [T, 4096] -> gate_up [T, 24576] -> act [T, 12288] -> out [T, 4096]
from __future__ import annotations

import enum
from collections.abc import Sequence

import torch
from tokenspeed_kernel.ops.gemm._cute_dsl.nvfp4_gemm_swiglu import (
    nvfp4_gemm_swiglu_nvfp4_quant,
)
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.distributed.mapping import Group
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.dense.mxfp4 import Mxfp4LinearMethod
from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod
from tokenspeed.runtime.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import envs

# 是否为 Blackwell 架构 GPU（sm100 系列）。NVFP4 融合 kernel 只在 Blackwell 上可用
_is_blackwell = current_platform().is_blackwell


# 可选的融合 kernel：
#   NONE：不融合，gate_up GEMM 和激活分开执行
#   NVFP4_GEMM_SWIGLU_QUANT：NVFP4 权重的 gate_up GEMM，在同一个 kernel 的收尾（epilogue）中
#     计算 SwiGLU，并把结果量化成 NVFP4，直接作为 down_proj 的输入
#   MXFP4_GEMM_SWIGLU：MXFP4 权重的 gate_up GEMM，收尾时计算 SwiGLU
class GatedMLPFusion(enum.Enum):
    """A kernel that applies SwiGLU inside the gate/up GEMM, or ``NONE``."""

    # 译：在 gate/up GEMM 内部完成 SwiGLU 的 kernel，或者 ``NONE``。

    NONE = "none"
    # NVFP4 GEMM whose epilogue also quantizes for an NVFP4 down_proj.
    # 译：NVFP4 GEMM，其收尾阶段还会为 NVFP4 的 down_proj 量化输出。
    NVFP4_GEMM_SWIGLU_QUANT = "nvfp4_gemm_swiglu_quant"
    MXFP4_GEMM_SWIGLU = "mxfp4_gemm_swiglu"

    @property
    def is_nvfp4_gemm_swiglu_quant(self) -> bool:
        return self is GatedMLPFusion.NVFP4_GEMM_SWIGLU_QUANT

    @property
    def is_mxfp4_gemm_swiglu(self) -> bool:
        return self is GatedMLPFusion.MXFP4_GEMM_SWIGLU


# 检查两个投影的量化方式是否允许使用某个融合 kernel：
#   NVFP4 融合：环境变量开关打开、Blackwell GPU、两个投影都是 NVFP4 量化
#   MXFP4 融合：两个投影都是 MXFP4 量化
def _supports(
    fusion: GatedMLPFusion, gate_up_proj: LinearBase, down_proj: LinearBase
) -> bool:
    """Whether both projections' quantization can run ``fusion``."""
    # 译：两个投影的量化方式是否都能运行 ``fusion``。
    gate_up_method = gate_up_proj.quant_method
    down_method = down_proj.quant_method
    if fusion.is_nvfp4_gemm_swiglu_quant:
        return (
            envs.TOKENSPEED_NVFP4_GEMM_SWIGLU_NVFP4_QUANT.get()
            and _is_blackwell
            and isinstance(gate_up_method, Nvfp4LinearMethod)
            and isinstance(down_method, Nvfp4LinearMethod)
        )
    if fusion.is_mxfp4_gemm_swiglu:
        return isinstance(gate_up_method, Mxfp4LinearMethod) and isinstance(
            down_method, Mxfp4LinearMethod
        )
    raise ValueError(f"Not a fused gated MLP kernel: {fusion}")


# 根据 HF 配置的 hidden_act 构造门控激活。目前只支持 "silu"（即 SwiGLU）；
# swiglu_limit 为截断阈值，None 表示不截断。
def gated_activation(hidden_act: str, swiglu_limit: float | None) -> nn.Module:
    """The gated activation for a checkpoint's ``hidden_act`` and SwiGLU clamp."""
    # 译：为 checkpoint 的 ``hidden_act`` 和 SwiGLU 截断值构造对应的门控激活。
    if hidden_act != "silu":
        raise ValueError(f"Unsupported gated MLP activation: {hidden_act}")
    return SiluAndMul(swiglu_limit)


# 门控 MLP。成员：
#   gate_up_proj：gate 与 up 合并的投影（通常是列并行，由调用方构造）
#   down_proj：输出投影（通常是行并行）
#   act_fn：门控激活
#   fusion：构造时选定的融合 kernel（NONE 表示不融合）
class GatedMLP(nn.Module):
    """``down_proj(activation(gate_up_proj(x)))`` with gate and up stacked.

    The caller builds the projections, which fixes their parallel layout and
    quantization; ``tensor_parallel`` builds the usual pair. Construction
    picks the first of ``fusions`` that both projections support, else runs
    the gate/up GEMM and then ``down_proj.forward_with_activation``, whose
    quantization method may fuse the activation into its input quantization.
    ``fusions`` lists the fused kernels validated for the model, since each
    rounds differently from the separate kernels.
    """

    # 译：``down_proj(activation(gate_up_proj(x)))``，其中 gate 和 up 堆叠在一起。
    #
    #     两个投影由调用方构造，因此它们的并行布局和量化方式由调用方决定；``tensor_parallel`` 构造常用的
    #     那一对。构造时从 ``fusions`` 中选出两个投影都支持的第一个融合 kernel；都不支持时，先执行 gate/up
    #     GEMM，再调用 ``down_proj.forward_with_activation``，后者的量化方法可能把激活融合进它的输入量化。
    #     ``fusions`` 列出对该模型验证过的融合 kernel，因为每个融合 kernel 的舍入方式都与分开执行的 kernel 不同。

    def __init__(
        self,
        gate_up_proj: LinearBase,
        down_proj: LinearBase,
        activation: nn.Module,
        fusions: Sequence[GatedMLPFusion],
    ) -> None:
        super().__init__()
        # 融合 kernel 内部实现的是不带截断的标准 SwiGLU，所以只要传了 fusions，激活就必须是
        # 不截断的 SiluAndMul，否则报错（避免悄悄算成另一个函数）
        if fusions and not (
            isinstance(activation, SiluAndMul) and activation.swiglu_limit is None
        ):
            raise ValueError(
                f"Fused gated MLP kernels compute SwiGLU, not {activation}"
            )
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.act_fn = activation
        # 按列表顺序选第一个两个投影都支持的融合；都不支持则为 NONE
        self.fusion = next(
            (f for f in fusions if _supports(f, gate_up_proj, down_proj)),
            GatedMLPFusion.NONE,
        )
        # Both fused GEMMs need their weight layout chosen before loading.
        # 译：两种融合 GEMM 都需要在加载权重之前选定权重的布局。
        # NVFP4 融合要求 gate_up 的权重按 gate、up 逐行交错存放，这样 kernel 收尾时每对相邻输出
        # 正好是同一个位置的 gate 和 up；MXFP4 融合则让量化方法自己准备对应布局。
        # 这两个设置会影响 weight_loader 如何摆放权重，所以必须在加载权重之前完成。
        gate_up_proj.interleave_linear_and_gate = self.fusion.is_nvfp4_gemm_swiglu_quant
        if self.fusion.is_mxfp4_gemm_swiglu:
            gate_up_proj.quant_method.enable_swiglu(gate_up_proj)

    # 构造常规的张量并行 MLP：gate_up_proj 为合并的列并行（输出宽度 [intermediate, intermediate]，
    # 按 TP 切分后每个 rank 各持有 gate、up 的 1/tp），down_proj 为行并行。
    # reduce_results：down_proj 是否自己做 all-reduce。解码层里传 False，求和交给 CommManager
    # （见 models/base/decoder_layer.py）。
    @classmethod
    def tensor_parallel(
        cls,
        hidden_size: int,
        intermediate_size: int,
        activation: nn.Module,
        tp_rank: int,
        tp_size: int,
        tp_group: Group | None,
        reduce_results: bool,
        fusions: Sequence[GatedMLPFusion],
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> GatedMLP:
        """Column-parallel gate/up and row-parallel down over one TP group."""
        # 译：在一个 TP 组上构造列并行的 gate/up 和行并行的 down。
        gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("down_proj", prefix),
        )
        return cls(gate_up_proj, down_proj, activation, fusions)

    # NVFP4 量化时 gate_up_proj 量化输入所用的全局 scale；不是 NVFP4 量化时返回 None
    def nvfp4_global_scale(self) -> torch.Tensor | None:
        """The global scale gate_up_proj quantizes its input to NVFP4 with, if it does."""
        # 译：gate_up_proj 把输入量化为 NVFP4 时使用的全局 scale（如果它做这种量化的话）。
        return self.gate_up_proj.quant_method.nvfp4_global_scale(self.gate_up_proj)

    # 前向。三条路径：
    #   1. NVFP4 融合：输入量化为 NVFP4（如果上游已经量化好，直接用 x_nvfp4）
    #      -> 一个 kernel 完成 gate_up GEMM + SwiGLU + 量化 -> down_proj
    #   2. MXFP4 融合：一个 kernel 完成 gate_up GEMM + SwiGLU -> down_proj
    #   3. 不融合（bf16 的 Qwen3 走这里）：gate_up_proj -> down_proj.forward_with_activation
    #      forward_with_activation 默认就是先算激活再做 down_proj 的 GEMM；
    #      个别量化方法会把激活和 down_proj 的输入量化融合在一起。
    # x_nvfp4：上游（例如前面的 norm）已经把 x 量化成 NVFP4 时传入的 (数据, scale)，可省一次量化。
    def forward(
        self,
        x: torch.Tensor,
        x_nvfp4: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the MLP; ``x_nvfp4`` is gate_up_proj's input when its producer quantized ``x``."""
        # 译：执行 MLP；当上游已经量化了 ``x`` 时，``x_nvfp4`` 就是 gate_up_proj 的输入。
        # 没有任何行时（例如 attention DP 下的空跑前向）直接返回，不启动任何 kernel
        if x.shape[0] == 0:
            return x
        if self.fusion.is_nvfp4_gemm_swiglu_quant:
            if x_nvfp4 is None:
                x_nvfp4 = fp4_quantize(x, self.gate_up_proj.input_scale_inv)
            activated = nvfp4_gemm_swiglu_nvfp4_quant(
                *x_nvfp4,
                self.gate_up_proj.weight_swiglu_interleaved,
                self.gate_up_proj.weight_scale_swiglu_interleaved,
                self.gate_up_proj.alpha,
                self.down_proj.input_scale_inv,
            )
            output, _ = self.down_proj(activated)
        elif self.fusion.is_mxfp4_gemm_swiglu:
            activated = self.gate_up_proj.quant_method.apply_swiglu(
                self.gate_up_proj, x
            )
            output, _ = self.down_proj(activated)
        else:
            gate_up, _ = self.gate_up_proj(x if x_nvfp4 is None else x_nvfp4)
            output, _ = self.down_proj.forward_with_activation(gate_up, self.act_fn)
        return output
