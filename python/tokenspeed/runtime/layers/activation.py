# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

"""Fused operators for activation layers."""

# 译：激活层的融合算子。

# 门控激活函数（gated activation）：MLP 中 gate_up_proj 的输出 [..., gate | up] 进来，
# 输出 act(gate) * up，宽度减半。
#
# 你在这里：GatedMLP.forward 中 gate_up_proj 之后、down_proj 之前（layers/mlp.py）；
#           MoE 专家内部也会用到（第 10 阶段）。
#
# 本文件的几种激活：
#   SiluAndMul：标准 SwiGLU，silu(gate) * up，可选截断（swiglu_limit）。Qwen3、Llama 等使用
#   SituAndMul：SiTU，Kimi 系列（如 Kimi-K3）使用
#   SwigluOAI：gate * sigmoid(alpha * gate) * (up + 1)，带截断。MiniMax-M3 使用
#   文件末尾的 Triton 函数：SwigluOAI 写成 MoE 融合 GEMM 的后处理（epilogue）形式
#
# kernel 都来自 tokenspeed_kernel（kernel 包，执行面通过它按硬件选择实现）；
# 非 GPU 设备走纯 PyTorch 的参考实现（forward_native）。
#
# Qwen3-8B 的例子：gate_up_proj 输出 [T, 2*12288] -> SiluAndMul -> [T, 12288]
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.activation import silu_and_mul

# SwiGLU-OAI has only the portable Triton solution.
# 译：SwiGLU-OAI 只有可移植的 Triton 实现。
from tokenspeed_kernel.ops.activation.triton import swiglu_oai
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.utils import (
    get_colorful_logger,
)
from tokenspeed.runtime.utils.triton import tl, triton

# 是否为 AMD GPU。AMD 上不支持 SiluAndMul 的 FP8 输出
_is_amd = current_platform().is_amd

logger = get_colorful_logger(__name__)


# SwiGLU：out = silu(gate) * up，输入 [..., 2*d] 的前一半是 gate、后一半是 up，输出 [..., d]。
# swiglu_limit：可选的截断阈值（部分模型训练时就带截断）。设置后先截断再计算：
#   gate = min(gate, limit)，up = clamp(up, -limit, limit)
#   不设置、或设置为 <= 0 的值时表示不截断（存成 None）。
class SiluAndMul(torch.nn.Module):
    def __init__(self, swiglu_limit: float | None = None) -> None:
        super().__init__()
        self.swiglu_limit = (
            float(swiglu_limit)
            if swiglu_limit is not None and swiglu_limit > 0
            else None
        )

    # fp8_out=True：输出直接量化成 FP8，同时返回量化 scale，供下一个 FP8 GEMM（down_proj）直接使用，
    #   省掉一次“写出 bf16 再量化”的显存往返。只在 NVIDIA GPU 上、且不截断时支持。
    # 返回：fp8_out=False 时返回激活结果；fp8_out=True 时返回 (FP8 结果, scale)。
    def forward(self, x: torch.Tensor, fp8_out: bool = False) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"SwiGLU expects an even [gate, up] width, got {x.shape[-1]}"
            )
        # 非 CUDA 设备：NPU 且不截断时用 kernel 包的实现，其余（CPU 等）用纯 PyTorch 参考实现
        if not x.is_cuda:
            if fp8_out:
                raise NotImplementedError("CPU fp8_out silu_and_mul is not implemented")
            if x.device.type == "npu" and self.swiglu_limit is None:
                return silu_and_mul(x)
            return self.forward_native(x)

        # NVIDIA GPU 路径
        if not _is_amd:

            # 按“每个 token、每 128 列一个 scale”的分块量化方式，为 FP8 输出分配 scale 张量。
            #   逻辑形状 [行数, 列数/128]，但内存按列优先（同一个块列的各行连续存放），
            #   并且行数向上对齐到 4 的倍数（4 个 fp32 = 16 字节，满足 TMA 拷贝的对齐要求）。
            #   做法：先按 [列数/128, 对齐后的行数] 分配，permute 成 [对齐后的行数, 列数/128]，再切掉多出来的行。
            #   例：x [301, 12288] -> 分配 [96, 304]，返回的视图形状 [301, 96]。
            def get_tma_aligned_scale(x):
                aligned_size = (x.shape[-2] + 3) // 4 * 4
                x_s = torch.empty(
                    x.shape[:-2] + (x.shape[-1] // 128, aligned_size),
                    device=x.device,
                    dtype=torch.float32,
                ).permute(-1, -2)[: x.shape[-2], :]
                return x_s

            d = x.shape[-1] // 2
            output_shape = x.shape[:-1] + (d,)
            if fp8_out:
                if self.swiglu_limit is not None:
                    raise NotImplementedError(
                        "clamped fp8_out silu_and_mul is not implemented"
                    )
                out = torch.empty(
                    output_shape, dtype=torch.float8_e4m3fn, device=x.device
                )
                scale = get_tma_aligned_scale(out)
                # 延迟导入：只有用到 FP8 输出时才导入这个 CUDA 专用的融合 kernel
                from tokenspeed_kernel.ops.activation.cuda import (
                    silu_and_mul_fuse_block_quant,
                )

                out, scale = silu_and_mul_fuse_block_quant(x, scale, out)
                return out, scale
            # 普通路径：分配输出，调用 kernel 包的 silu_and_mul（截断阈值作为参数传入）
            out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            return silu_and_mul(
                x,
                out,
                limit=self.swiglu_limit,
            )

        # AMD GPU 路径：与上面相同，只是不支持 FP8 输出
        if fp8_out:
            raise NotImplementedError("AMD fp8_out silu_and_mul is not implemented")
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        return silu_and_mul(
            x,
            out,
            limit=self.swiglu_limit,
        )

    # 纯 PyTorch 参考实现：在 fp32 中计算，再转回输入的 dtype
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].float()
        up = x[..., d:].float()
        if self.swiglu_limit is not None:
            gate = gate.clamp_max(self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        return (torch.nn.functional.silu(gate) * up).to(x.dtype)


# SiTU 门控激活（Kimi 系列）。公式见下面的 docstring：gate 分支是 beta * tanh(gate / beta) * sigmoid(gate)，
# 即先用 tanh 把 gate 平滑地限制在 [-beta, beta] 内，再乘 sigmoid(gate)；
# 设置了 linear_beta 时，up 分支也用 tanh 平滑截断到 [-linear_beta, linear_beta]。
class SituAndMul(torch.nn.Module):
    """SiTU / SituGLU gated activation used by Kimi models (e.g. Kimi-K3).

    Over the fused ``[..., gate | up]`` input, computes::

        gate = beta * tanh(gate / beta) * sigmoid(gate)
        up   = linear_beta * tanh(up / linear_beta)   # only if linear_beta set
        out  = gate * up

    Args:
        beta: Softplus-like temperature applied to the gate branch.
        linear_beta: If set, softly clips the up branch; disabled when ``None``.

    CUDA inputs run the fused Triton kernel (one launch instead of the ~10
    sliced elementwise kernels of the native path, ULP-identical fp32 math);
    other devices keep the native reference implementation.
    """

    # 译：Kimi 模型（例如 Kimi-K3）使用的 SiTU / SituGLU 门控激活。
    #
    #     在合并的 ``[..., gate | up]`` 输入上计算::
    #
    #         gate = beta * tanh(gate / beta) * sigmoid(gate)
    #         up   = linear_beta * tanh(up / linear_beta)   # 仅当设置了 linear_beta 时
    #         out  = gate * up
    #
    #     参数：
    #         beta：作用在 gate 分支上、类似 softplus 温度的系数。
    #         linear_beta：设置时对 up 分支做软截断；为 ``None`` 时不启用。
    #
    #     CUDA 输入运行融合的 Triton kernel（一次启动，代替参考实现中约 10 个切片后的逐元素 kernel，
    #     fp32 计算结果在 ULP 级别完全一致）；其他设备保留参考实现。

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None) -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"SiTU beta must be positive, got {beta}")
        if linear_beta is not None and linear_beta <= 0.0:
            raise ValueError(f"SiTU linear_beta must be positive, got {linear_beta}")
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"SiTU expects an even [gate, up] width, got {x.shape[-1]}"
            )
        if x.is_cuda:
            from tokenspeed_kernel.ops.activation import situ_and_mul

            return situ_and_mul(
                x,
                beta=self.beta,
                linear_beta=self.linear_beta,
            )
        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].float()
        up = x[..., d:].float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(x.dtype)


# SwiGLU-OAI（MiniMax-M3 使用）：out = gate * sigmoid(alpha * gate) * (up + 1)。
# 与标准 SwiGLU 的区别：sigmoid 里多一个系数 alpha，up 要先加 1；并且总是带截断。
# 只有 Triton 实现（可在 NVIDIA 和 AMD 上运行）。
class SwigluOAI(torch.nn.Module):
    """SwiGLU-OAI gated activation: ``gate * sigmoid(alpha * gate) * (up + 1)``.

    The gate is upper-clamped to ``limit`` and up is clamped to
    ``[-limit, limit]``.
    """

    # 译：SwiGLU-OAI 门控激活：``gate * sigmoid(alpha * gate) * (up + 1)``。
    #
    #     gate 的上限被截断到 ``limit``，up 被截断到 ``[-limit, limit]``。

    def __init__(self, alpha: float, limit: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu_oai(x, alpha=self.alpha, limit=self.limit)


# 下面三个 Triton 函数把 SwiGLU-OAI 写成“MoE 融合 GEMM 的后处理（epilogue）”形式：
# MoE 的第一个 GEMM 算出结果后，在同一个 kernel 里直接做激活，不再单独启动激活 kernel。
# 注意：执行面中目前没有调用它们的地方，实际使用的是 AMD kernel 包
# （tokenspeed-kernel-amd 的 moe/_common.py 等）里的同名副本。
#
# clip：把 x 的上限截断到 limit；clip_lower 为真时下限也截断到 -limit。
# clip_lower 是编译期常量（tl.constexpr），两种取值各编译一份。
@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


# SwiGLU-OAI 的核心计算（参数名 gelu 指 gate，linear 指 up）：
#   先乘 scale（量化时的反缩放系数；不需要时传 1.0），再按 limit 截断，
#   s = gate * sigmoid(alpha * gate)，结果 = s * (up + 1)
@triton.jit
def compute_swiglu(gelu, linear, scale, alpha, limit):
    gelu = gelu.to(tl.float32) * scale
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32) * scale
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)

    # gate / (1 + exp(-alpha * gate)) 就是 gate * sigmoid(alpha * gate)
    s = gelu / (1 + tl.exp(-alpha * gelu))

    # fma(s, linear, s) = s * linear + s，一条融合乘加指令算出 s * (up + 1)
    return tl.fma(s, linear, s)  # (s * (linear + 1))
    # 译：即 (s * (linear + 1))


# MoE GEMM 后处理的入口。repr 指定编译出的函数名为 "_swiglu"。
#   exclusive_sum：每个本地专家 token 数的前缀和；只取第一个到最后一个本地专家对应的行。
#   注意输入布局与 SiluAndMul 不同：这里 gate 和 up 是逐列交错排列的
#   [g0, u0, g1, u1, ...]，reshape 成 (行, 列/2, 2) 后 split 拆出 gate 和 up。
@triton.jit(repr=lambda _: "_swiglu")
def swiglu_fn(input, alpha, limit, exclusive_sum, local_num_experts):
    begin = exclusive_sum[0]
    end = exclusive_sum[local_num_experts]
    input = input[begin:end]

    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    return compute_swiglu(gelu, linear, 1.0, alpha, limit)


# SwiGLU 的两个参数，供 MoE 专家层（layers/moe/expert.py）传给融合 kernel。
#   alpha：sigmoid 中的系数；limit：截断阈值
@dataclass
class SwigluArg:
    alpha: float
    limit: float
