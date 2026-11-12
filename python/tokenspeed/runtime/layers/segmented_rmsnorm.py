"""Engine-local segmented RMSNorm used by the K3 DSpark draft."""

# 译：引擎内部使用的分段 RMSNorm，供 K3 DSpark 草稿模型使用。

from __future__ import annotations

import torch

from tokenspeed.runtime.utils.triton import tl, triton


# 分段 RMSNorm：把每一行的最后一维看成若干段，每段用自己的权重各做一次 RMSNorm。
#
# 用途：Kimi K3 的 DSpark 草稿模型（models/kimi_k3_dspark.py，第 11 阶段）把目标模型若干层的
# hidden 拼在一起作为输入，每一层的 hidden 要用各自的 norm 权重归一化。分段做可以用一个 kernel
# 完成，不必拆开张量逐段调用 RMSNorm。
#
# 形状：x [..., S, Hs]（S 段，每段 Hs 个数），weight [S, Hs]，输出与 x 同形状。
#   out[r, s, :] = x[r, s, :] / sqrt(mean(x[r, s, :]^2) + eps) * weight[s, :]
#
# Triton kernel 的划分：grid = (行数, 段数)，每个 program 处理一行中的一段（Hs 个数）。
#   BLOCK = 大于等于 Hs 的最小 2 的幂，超出 Hs 的位置用 mask 屏蔽。
#   计算在 fp32 中进行，结果按 out 的 dtype 写回。
@triton.jit
def _segmented_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_segments: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    segment = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < hidden_size

    # 第 row 行、第 segment 段在 x 中的起始位置 = row * (S*Hs) + segment * Hs
    row_width = n_segments * hidden_size
    input_offsets = row * row_width + segment * hidden_size + offsets
    weight_offsets = segment * hidden_size + offsets

    x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    # 方差：这一段所有元素平方的平均值（被 mask 的位置读成 0，不影响求和）
    variance = tl.sum(x * x, axis=0) / hidden_size
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + input_offsets, x * tl.rsqrt(variance + eps) * weight, mask=mask)


# 包装函数：检查输入、分配输出、启动 kernel。
# 要求 x、weight 都在 GPU 上且内存连续，weight 的形状必须等于 x 最后两维。
def segmented_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply independent RMSNorm to the final ``[segments, hidden]`` axes."""
    # 译：对最后两个 ``[segments, hidden]`` 轴分别独立地做 RMSNorm。
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("segmented_rmsnorm requires CUDA tensors")
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {x.ndim}")
    if weight.ndim != 2 or tuple(x.shape[-2:]) != tuple(weight.shape):
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match "
            f"input segments {tuple(x.shape[-2:])}"
        )
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")

    out = torch.empty_like(x) if out is None else out
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("out must match x shape, dtype, and device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if x.numel() == 0:
        return out

    # 除最后两维外的所有维度合并成“行”。
    # num_warps：hidden 很大（BLOCK >= 4096）时用 8 个 warp，否则 4 个。
    num_segments, hidden_size = weight.shape
    n_rows = x.numel() // (num_segments * hidden_size)
    block = triton.next_power_of_2(hidden_size)
    num_warps = 8 if block >= 4096 else 4
    _segmented_rmsnorm_kernel[(n_rows, num_segments)](
        x,
        weight,
        out,
        num_segments,
        hidden_size,
        eps,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out
