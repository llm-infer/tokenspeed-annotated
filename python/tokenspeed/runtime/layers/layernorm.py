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

"""Fused operators for normalization layers."""

# 译：归一化层的融合算子。

# 归一化层：LayerNorm、RMSNorm、GemmaRMSNorm、FusedRMSNorm，以及 sandwich norm 辅助函数。
#
# 你在这里：解码层的 input_layernorm / post_attention_layernorm（经 CommManager 调用）、
#           主干末尾的 final norm、Qwen3 注意力里的 q_norm / k_norm（经 PagedAttention 调用）。
#
# RMSNorm 本身和“相加残差 + RMSNorm”融合都是熟悉的内容。本文件的重点是：
#   1. 带 residual 调用时是原地计算（NVIDIA）：residual = x + residual；x = RMSNorm(residual)，
#      返回的就是传入的两个张量。所以别处需要保留旧值时必须先 clone。
#   2. 和通信融合：TP>1 时，行并行输出的 all-reduce（或 reduce-scatter、all-gather）可以和
#      相加、RMSNorm 合成一个 kernel（forward_with_allreduce_fusion 等），由 CommManager 决定何时使用。
#   3. 顺带量化：归一化的结果可以在同一个 kernel 里额外写出一份 FP8 / NVFP4 副本，
#      直接作为下一个量化 GEMM 的输入（GemmaRMSNorm.add_norm_with_fp8 / add_norm_with_fp4）。
#
# kernel 来自 tokenspeed_kernel，按平台选择：NVIDIA 用 TRT-LLM / FlashInfer 等实现，
# AMD 用 Iris / Triton 实现，其他设备用纯 PyTorch 组合。
from collections.abc import Callable
from typing import ClassVar

import torch
import torch.nn as nn
from tokenspeed_kernel.ops.communication import (
    allreduce_residual_rmsnorm as iris_allreduce_residual_rmsnorm,
)
from tokenspeed_kernel.ops.communication import (
    allreduce_sandwich_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allgather_dual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    allreduce_residual_rmsnorm as trtllm_allreduce_residual_rmsnorm,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    reducescatter_residual_rmsnorm,
)
from tokenspeed_kernel.ops.layernorm import (
    add_rmsnorm,
    gemma_rmsnorm,
    nvfp4_copy_supported,
    rmsnorm,
)
from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_rmsnorm
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.utils import (
    get_colorful_logger,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

_platform = current_platform()
_is_amd = _platform.is_amd


# 通用的“all-reduce + 相加残差 + RMSNorm”：先用 torch.distributed 做 all-reduce，再调用普通的
# 融合 add + RMSNorm。非 NVIDIA、非 AMD 平台使用这个分开执行的版本。返回值形式与融合版本一致
# （第三项为 None）。
def _torch_allreduce_residual_rmsnorm(
    *,
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    group,
    eps: float,
    **_,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    torch.distributed.all_reduce(input_tensor, group=group)
    output, updated_residual = rmsnorm(
        input_tensor,
        weight,
        eps,
        residual=residual,
    )
    return output, updated_residual, None


# 按平台选择 all-reduce + 残差 + RMSNorm 的融合实现：
#   AMD：Iris 实现；NVIDIA：TRT-LLM 实现；其他：上面的分开执行版本。
# 同时导入 FusedRMSNorm 用的“两个张量并行归一化”的 kernel，以及 NVIDIA 上 LayerNorm 用的 kernel。
if _is_amd:
    from tokenspeed_kernel.ops.layernorm.triton import (
        rmsnorm_fused_parallel as triton_rmsnorm_fused_parallel,
    )

    _allreduce_residual_rmsnorm = iris_allreduce_residual_rmsnorm
elif _platform.is_nvidia:
    from tokenspeed_kernel.ops.layernorm.cuda import rmsnorm_fused_parallel
    from tokenspeed_kernel.ops.layernorm.flashinfer import layernorm

    _allreduce_residual_rmsnorm = trtllm_allreduce_residual_rmsnorm
else:
    _allreduce_residual_rmsnorm = _torch_allreduce_residual_rmsnorm


logger = get_colorful_logger(__name__)


# 按 rank 元组取得对应的设备通信组（融合 kernel 需要 torch 的 ProcessGroup 对象）
def _get_process_group(group: tuple[int, ...]):
    return pg_manager.get_device_process_group(group)


# 标准 LayerNorm（带 bias），参数用 fp32 存储。Transformer 主干一般用 RMSNorm，
# LayerNorm 主要用在视觉编码器等模块。NVIDIA 上调用 FlashInfer 的 kernel，其他平台在 fp32 中计算。
class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # There might be no tokens here (e.g. idle/padded graph rows).
        # 译：这里可能没有任何 token（例如空闲 rank 或 CUDA Graph 中的 padding 行）。
        if x.shape[0] == 0:
            return x
        if _platform.is_nvidia:
            return layernorm(x, self.weight, self.bias, self.variance_epsilon)
        return nn.functional.layer_norm(
            x.float(),
            (x.shape[-1],),
            self.weight,
            self.bias,
            self.variance_epsilon,
        ).to(x.dtype)


# RMSNorm。成员：
#   weight [H]：缩放系数（初始化为 1，由 checkpoint 覆盖）
#   variance_epsilon：eps
#   kernel：实际执行的函数，默认是 kernel 包的 rmsnorm（按平台分派）；
#     少数模型会传入特定 kernel，此时不能使用下面的通信融合
# weight_offset=0 表示缩放系数就是 weight 本身（对比 GemmaRMSNorm 的 1 + weight）。
class RMSNorm(torch.nn.Module):
    weight_offset: ClassVar[float] = 0.0

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        kernel: Callable | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.kernel = rmsnorm if kernel is None else kernel

    # 两种调用方式：
    #   norm(x)：返回 RMSNorm(x)，形状 [T, H]
    #   norm(x, residual)：融合 add + RMSNorm，返回 (RMSNorm(x + residual), x + residual)，
    #     NVIDIA 上原地写回 x 和 residual
    #   out：可选的输出缓冲区（只能用于不带 residual 的调用）
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # There might be no tokens here
        # 译：这里可能没有任何 token
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            else:
                return x

        return self.kernel(
            x,
            self.weight.data,
            self.variance_epsilon,
            residual=residual,
            out=out,
        )

    # “all-reduce + 相加残差 + RMSNorm”融合，由 CommManager 在 TP>1 且条件允许时调用。
    #   x：本 rank 行并行输出的部分和；residual：残差流
    #   rank / group：本 rank 在通信组中的编号、组内所有 rank 的全局编号
    #   返回三元组 (RMSNorm(sum(x) + residual), sum(x) + residual, 额外输出)：
    #     第三项是可选的 FP8 分块量化 scale 等，不需要时为 None
    #   fuse_block_quant_fp8 等参数：在同一个 kernel 中顺带完成的附加工作（第 9 阶段）
    #
    # 注意回退路径：组里只有 1 个 rank、没有 residual，或者融合 kernel 返回 None 时，
    # 直接执行 self.forward(x, residual)，即只做相加和 RMSNorm、不做 all-reduce。
    # 组大小大于 1 时，NVIDIA 的 TRT-LLM 实现要么返回结果要么报错；AMD 的 Iris 实现在输入不满足
    # 条件（非 bf16、非连续、token 数超过上限等）时会返回 None，此时这里也不会做 all-reduce。
    def forward_with_allreduce_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        fuse_block_quant_fp8: bool = False,
        residual_reduce_scattered: bool = False,
        max_sm_to_use: int | None = None,
        trigger_completion_at_end: bool = False,
        has_partial_norm_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method with allreduce fusion, prioritizing flashinfer fused operations
        """
        # 译：带 all-reduce 融合的前向方法，优先使用 flashinfer 的融合算子
        if self.kernel is not rmsnorm:
            raise ValueError("A selected RMSNorm kernel cannot use all-reduce fusion")

        if residual is not None:

            if len(group) > 1:
                fused_result = _allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.weight,
                    rank=rank,
                    group=_get_process_group(group),
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                    block_quant_fp8=fuse_block_quant_fp8,
                    residual_reduce_scattered=residual_reduce_scattered,
                    max_sm_to_use=max_sm_to_use,
                    trigger_completion_at_end=trigger_completion_at_end,
                    has_partial_norm_out=has_partial_norm_out,
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None

    # 与上面相同，只是把 all-reduce 换成 reduce-scatter（RSAG 布局，见第 9 阶段）：
    # 求和后每个 rank 只保留自己那部分 token 行，再做相加和 RMSNorm。
    # add_in：可选的额外加数，一并在 kernel 中加上。
    def forward_with_reducescatter_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        fuse_block_quant_fp8: bool = False,
        add_in: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method with reducescatter fusion, prioritizing flashinfer fused operations
        """
        # 译：带 reduce-scatter 融合的前向方法，优先使用 flashinfer 的融合算子
        if self.kernel is not rmsnorm:
            raise ValueError(
                "A selected RMSNorm kernel cannot use reduce-scatter fusion"
            )

        if residual is not None:

            if len(group) > 1:
                fused_result = reducescatter_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.weight,
                    rank=rank,
                    group=_get_process_group(group),
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                    use_oneshot=True,
                    block_quant_fp8=fuse_block_quant_fp8,
                    add_in=add_in,
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None


# Gemma 风格的 RMSNorm：缩放系数是 (1 + weight)，weight 初始化为 0。
# 为了让只接受“乘 weight”的通用融合 kernel 也能用，预先算好 gemma_weight = weight + 1.0；
# 加载权重时（_weight_loader）同步更新。
class GemmaRMSNorm(torch.nn.Module):
    # The multiplier is 1 + weight.
    # 译：缩放系数是 1 + weight。
    weight_offset: ClassVar[float] = 1.0

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.register_buffer("gemma_weight", self.weight.data + 1.0, persistent=False)
        # (Chen-0210) Gemma weight = standard_weight + 1. Precompute once.
        # 译：(Chen-0210) Gemma 的权重 = 标准权重 + 1。预先计算一次。
        # 把 weight 参数的加载函数换成下面的 _weight_loader，加载时同时刷新 gemma_weight。
        # _nvfp4_copy_supported：该 hidden_size 下，融合 kernel 输出的 NVFP4 副本能否与单独量化的结果完全一致。
        self.weight.weight_loader = self._weight_loader
        self._nvfp4_copy_supported = nvfp4_copy_supported(hidden_size)

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if param.size() != loaded_weight.size():
            raise ValueError(
                f"Shape mismatch: {param.size()} != {loaded_weight.size()}."
            )
        param.data.copy_(loaded_weight)
        self.gemma_weight = param.data + 1.0

    # forward 直接用 gemma_rmsnorm kernel（kernel 内部处理 1 + weight），不需要 gemma_weight
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] == 0:
            if residual is not None:
                return x, residual
            else:
                return x

        return gemma_rmsnorm(x, self.weight.data, self.variance_epsilon, residual)

    # 下面两个方法在 add + RMSNorm 的同时，额外输出一份量化副本，供下一个量化 GEMM 直接使用：
    #   add_norm_with_fp8：静态 scale 的 FP8 副本
    #   add_norm_with_fp4：NVFP4 副本 (数值, scale)
    # 返回 (归一化结果, 量化副本或 None, 更新后的残差)。
    def add_norm_with_fp8(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        fp8_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """``forward(x, residual)``, plus the normed rows in static FP8 when ``fp8_scale`` is given.

        Returns the normed rows, their FP8 copy or ``None`` where it is not
        made (AMD), and the updated residual.
        """
        # 译：``forward(x, residual)``，并在给出 ``fp8_scale`` 时额外输出归一化行的静态 FP8 副本。
        #
        #     返回归一化后的行、它们的 FP8 副本（不生成时为 ``None``，如 AMD），以及更新后的残差。
        x, x_fp8, _, residual = self._add_norm(
            x, residual, fp8_scale=fp8_scale, fp4_scale=None
        )
        return x, x_fp8, residual

    def add_norm_with_fp4(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        fp4_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
        """``forward(x, residual)``, plus ``fp4_quantize(rows, fp4_scale)`` when ``fp4_scale`` is given.

        Returns the normed rows, their NVFP4 ``(values, scales)`` or ``None``
        where the copy cannot equal ``fp4_quantize``'s (see
        ``nvfp4_copy_supported``) or rows are not BF16, and the updated residual.
        """
        # 译：``forward(x, residual)``，并在给出 ``fp4_scale`` 时额外输出 ``fp4_quantize(rows, fp4_scale)``。
        #
        #     返回归一化后的行；它们的 NVFP4 ``(values, scales)``，如果这份副本无法与 ``fp4_quantize`` 的结果
        #     完全一致（见 ``nvfp4_copy_supported``）或行不是 BF16，则为 ``None``；以及更新后的残差。
        x, _, x_fp4, residual = self._add_norm(
            x, residual, fp8_scale=None, fp4_scale=fp4_scale
        )
        return x, x_fp4, residual

    # 两个公开方法的共同实现。AMD 上只做普通的 add + RMSNorm，不生成量化副本。
    # NVFP4 副本的存储：数值 [行, 列/2]（两个 4 bit 数打包进一个 uint8），
    # scale [行数向上对齐到 128, 列/16]（每 16 个数一个 scale，行数按 kernel 要求对齐）。
    def _add_norm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        *,
        fp8_scale: torch.Tensor | None,
        fp4_scale: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor,
    ]:
        if _is_amd:
            x, residual = self(x, residual)
            return x, None, None, residual
        # The NVFP4 linear's prequantized path returns BF16 whatever the model dtype.
        # 译：NVFP4 线性层的预量化路径不论模型 dtype 如何都返回 BF16。
        if not (self._nvfp4_copy_supported and x.dtype == torch.bfloat16):
            fp4_scale = None
        rows, cols = x.shape
        x_fp8 = (
            None
            if fp8_scale is None
            else torch.empty_like(x, dtype=torch.float8_e4m3fn)
        )
        x_fp4 = None
        if fp4_scale is not None:
            x_fp4 = (
                torch.empty(rows, cols // 2, dtype=torch.uint8, device=x.device),
                torch.empty(
                    (rows + 127) // 128 * 128,
                    cols // 16,
                    dtype=torch.uint8,
                    device=x.device,
                ),
            )
        add_rmsnorm(
            x,
            residual,
            self.weight.data,
            self.variance_epsilon,
            x2=None,
            out=x,
            out_fp8=x_fp8,
            fp8_scale=fp8_scale,
            out_fp4=x_fp4,
            fp4_scale=fp4_scale,
            gemma=True,
        )
        return x, x_fp8, x_fp4, residual

    # 与 RMSNorm.forward_with_allreduce_fusion 相同，只是把 gemma_weight（= weight + 1）当作缩放系数传入，
    # 这样通用的融合 kernel 算出的就是 x * (1 + weight)。回退路径同样不做 all-reduce。
    def forward_with_allreduce_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        fuse_block_quant_fp8: bool = False,
        residual_reduce_scattered: bool = False,
        max_sm_to_use: int | None = None,
        trigger_completion_at_end: bool = False,
        has_partial_norm_out: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Forward method with allreduce fusion for GemmaRMSNorm.
        Uses gemma_weight (= weight + 1.0) as gamma so that the standard
        fused kernel computes x * (1 + weight) matching GemmaRMSNorm semantics.
        """
        # 译：GemmaRMSNorm 的带 all-reduce 融合的前向方法。
        #     用 gemma_weight（= weight + 1.0）作为 gamma，使标准的融合 kernel 计算 x * (1 + weight)，
        #     与 GemmaRMSNorm 的语义一致。

        if residual is not None:

            if len(group) > 1:
                fused_result = _allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.gemma_weight,
                    rank=rank,
                    group=_get_process_group(group),
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                    block_quant_fp8=fuse_block_quant_fp8,
                    residual_reduce_scattered=residual_reduce_scattered,
                    max_sm_to_use=max_sm_to_use,
                    trigger_completion_at_end=trigger_completion_at_end,
                    has_partial_norm_out=has_partial_norm_out,
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None

    def forward_with_reducescatter_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        fuse_block_quant_fp8: bool = False,
        add_in: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method with reducescatter fusion for GemmaRMSNorm.
        Uses gemma_weight (= weight + 1.0) as gamma so that the standard
        fused kernel computes x * (1 + weight) matching GemmaRMSNorm semantics.
        """
        # 译：GemmaRMSNorm 的带 reduce-scatter 融合的前向方法。
        #     用 gemma_weight（= weight + 1.0）作为 gamma，使标准的融合 kernel 计算 x * (1 + weight)，
        #     与 GemmaRMSNorm 的语义一致。

        if residual is not None:

            if len(group) > 1:
                fused_result = reducescatter_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.gemma_weight,
                    rank=rank,
                    group=_get_process_group(group),
                    eps=self.variance_epsilon,
                    max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                    use_oneshot=True,
                    block_quant_fp8=fuse_block_quant_fp8,
                    add_in=add_in,
                )
                if fused_result[0] is not None:
                    return fused_result

        result = self.forward(x, residual)
        if isinstance(result, tuple):
            return result[0], result[1], None
        return result, None, None


# 同时归一化两个张量的 RMSNorm：一个 kernel 同时完成两个各自独立的 RMSNorm，少启动一次 kernel。
# 两个 norm 的权重仍由原来的两个 RMSNorm 模块持有。
# 最初为 MLA（DeepSeek-V3、GLM-5 等，第 12 阶段）设计：q 的低秩投影输出（q_a）和 kv 的低秩投影输出
# （kv_a）各有一个 RMSNorm，所以参数名叫 q_a / kv_a。EAGLE3 草稿模型也借用它同时归一化两个输入
# （此时参数名与 MLA 无关）。
class FusedRMSNorm(nn.Module):
    """Fused RMSNorm layer for normalizing two tensors simultaneously.

    This layer wraps two independent RMSNorm layers (q_a and kv_a) and performs
    fused normalization during forward pass. The RMSNorm layers are passed in as
    parameters, allowing reuse of existing normalization layers.
    """

    # 译：同时归一化两个张量的融合 RMSNorm 层。
    #
    #     这个层包装了两个独立的 RMSNorm 层（q_a 和 kv_a），在前向时做融合的归一化。
    #     RMSNorm 层以参数形式传入，因此可以复用已有的归一化层。

    def __init__(
        self,
        q_a_norm: RMSNorm,
        kv_a_norm: RMSNorm,
    ) -> None:
        super().__init__()
        self.q_a_norm = q_a_norm
        self.kv_a_norm = kv_a_norm

    @property
    def weight_q_a(self) -> nn.Parameter:
        """Expose weight_q_a from q_a_norm for backward compatibility."""
        # 译：暴露 q_a_norm 的 weight，用于向后兼容。
        return self.q_a_norm.weight

    @property
    def weight_kv_a(self) -> nn.Parameter:
        """Expose weight_kv_a from kv_a_norm for backward compatibility."""
        # 译：暴露 kv_a_norm 的 weight，用于向后兼容。
        return self.kv_a_norm.weight

    def supports(self, input_q_a: torch.Tensor) -> bool:
        """Return whether the fused implementation supports this input."""
        # 译：返回融合实现是否支持这个输入。
        return input_q_a.device.type != "cpu"

    # 未指定输出缓冲区时，结果原地写回输入张量。注意返回的总是两个输入张量本身
    # （传了 output_q_a 时，q 的结果在 output_q_a 里，而不在返回值里）。
    def forward(
        self,
        input_q_a: torch.Tensor,
        input_kv_a: torch.Tensor,
        output_q_a: torch.Tensor | None = None,
        output_kv_a: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize two tensors in parallel using fused computation.

        Args:
            input_q_a: Q tensor to normalize
            input_kv_a: KV tensor to normalize

        Returns:
            Tuple of (normalized_q_a, normalized_kv_a)
        """
        # 译：用融合计算并行地归一化两个张量。
        #
        #     参数：
        #         input_q_a：要归一化的 Q 张量
        #         input_kv_a：要归一化的 KV 张量
        #
        #     返回：
        #         元组 (归一化后的 q_a, 归一化后的 kv_a)
        if _platform.is_npu:
            # Ascend uses the registered RMSNorm leaves. Preserve the dual
            # norm contract: omitted outputs overwrite their input buffers.
            # 译：Ascend 使用注册的 RMSNorm 叶子算子。保持双 norm 的约定：省略输出时覆盖它们的输入缓冲区。
            self.q_a_norm(
                input_q_a, out=input_q_a if output_q_a is None else output_q_a
            )
            self.kv_a_norm(
                input_kv_a, out=input_kv_a if output_kv_a is None else output_kv_a
            )
        elif _is_amd:
            triton_rmsnorm_fused_parallel(
                input1=input_q_a,
                weight1=self.weight_q_a,
                output1=output_q_a if output_q_a is not None else input_q_a,
                input2=input_kv_a,
                weight2=self.weight_kv_a,
                output2=output_kv_a if output_kv_a is not None else input_kv_a,
                eps=self.q_a_norm.variance_epsilon,
            )
        else:
            rmsnorm_fused_parallel(
                input1=input_q_a,
                weight1=self.weight_q_a,
                output1=output_q_a if output_q_a is not None else input_q_a,
                input2=input_kv_a,
                weight2=self.weight_kv_a,
                output2=output_kv_a if output_kv_a is not None else input_kv_a,
                eps=self.q_a_norm.variance_epsilon,
            )
        return input_q_a, input_kv_a

    # MLA 在 TP 下的融合：先 all-gather 各 rank 的 [q_a | kv_a | k_rope] 行，再同时归一化 q_a 和 kv_a，
    # 并可选地把 q_a 的结果做 FP8 分块量化，全部在一个 kernel 里完成（第 12 阶段）。
    # 回退路径（单 rank 或融合 kernel 不可用）：不做 all-gather，直接对本地行做两个归一化。
    def forward_with_allgather_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        qkv: torch.Tensor,
        total_num_tokens: int,
        fuse_block_quant_fp8: bool = False,
        trigger_completion_at_end: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Forward method with allgather fusion, performing allgather + dual RMSNorm + optional FP8 block quantization.

        This method fuses allgather communication with dual RMSNorm computation
        and optional FP8 block-wise quantization in a single kernel launch.

        Args:
            qkv: Input tensor to allgather, shape [num_token_current_rank, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
            fuse_block_quant_fp8: Whether to perform FP8 block-wise quantization on the first norm output
            trigger_completion_at_end: Whether to trigger completion event at the end of kernel

        Returns:
            Tuple of (allgather_out, quant_out, k_nope, block_scale):
                - allgather_out: Gathered tensor, shape [num_token_all_group, hidden_dim]
                - quant_out: FP8 quantized first norm output (q_contiguous), None if fuse_block_quant_fp8=False
                - k_nope: Second norm output
                - block_scale: Quantization scales, None if fuse_block_quant_fp8=False
        """
        # 译：带 all-gather 融合的前向方法，执行 all-gather + 双 RMSNorm + 可选的 FP8 分块量化。
        #
        #     该方法把 all-gather 通信与双 RMSNorm 计算、以及可选的 FP8 分块量化融合在一次 kernel 启动中。
        #
        #     参数：
        #         qkv：要 all-gather 的输入张量，形状 [num_token_current_rank, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
        #         fuse_block_quant_fp8：是否对第一个 norm 的输出做 FP8 分块量化
        #         trigger_completion_at_end：是否在 kernel 结束时触发完成事件
        #
        #     返回：
        #         元组 (allgather_out, quant_out, k_nope, block_scale)：
        #             - allgather_out：收集后的张量，形状 [num_token_all_group, hidden_dim]
        #             - quant_out：第一个 norm 输出的 FP8 量化结果（q_contiguous），fuse_block_quant_fp8=False 时为 None
        #             - k_nope：第二个 norm 的输出
        #             - block_scale：量化 scale，fuse_block_quant_fp8=False 时为 None

        if len(group) > 1:
            fused_result = allgather_dual_rmsnorm(
                qkv=qkv,
                total_num_tokens=total_num_tokens,
                rank=rank,
                group=_get_process_group(group),
                weight_q_a=self.weight_q_a,
                weight_kv_a=self.weight_kv_a,
                eps_q=self.q_a_norm.variance_epsilon,
                eps_kv=self.kv_a_norm.variance_epsilon,
                max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
                block_quant_fp8=fuse_block_quant_fp8,
                trigger_completion_at_end=trigger_completion_at_end,
                fp32_acc=False,
            )
            if fused_result[0] is not None:
                return fused_result

        q_lora_rank = self.weight_q_a.shape[0]
        kv_lora_rank = self.weight_kv_a.shape[0]
        q = qkv[..., :q_lora_rank]
        k_nope = qkv[..., q_lora_rank : q_lora_rank + kv_lora_rank]
        q_contiguous = torch.empty_like(q)
        if q.shape[0] > 0:
            self.forward(input_q_a=q, input_kv_a=k_nope, output_q_a=q_contiguous)

        return qkv, q_contiguous, k_nope, None


# 下面是 sandwich norm（“三明治”式归一化）：子层输出先经过自己的 post norm，再与残差相加，
# 然后经过下一层的 pre norm。与普通 pre-norm 相比，多了一个作用在子层输出上的 norm：
#   a = post_norm(x)；residual_out = a + residual；out = norm(residual_out)
# 部分模型（如 Beam，第 18 阶段）使用这种结构。
#
# _check_plain_gamma：sandwich 边界的 kernel 只支持“乘 weight”的 norm，遇到 GemmaRMSNorm 这类
# （乘 weight + offset）的 norm 直接报错。
def _check_plain_gamma(*norms: RMSNorm) -> None:
    """The boundary scales by ``weight`` itself: refuse a norm whose
    multiplier is ``weight + weight_offset`` (``GemmaRMSNorm``)."""
    # 译：边界按 ``weight`` 本身缩放：拒绝缩放系数为 ``weight + weight_offset`` 的 norm（``GemmaRMSNorm``）。
    for norm in norms:
        offset = getattr(norm, "weight_offset", 0.0)
        if offset != 0.0:
            raise ValueError(
                f"{type(norm).__name__} scales by weight + {offset}; the "
                "sandwich norm boundary takes norms that scale by weight"
            )


# 不含通信的 sandwich 边界：两次 Triton rmsnorm。第二次带 residual，并要求相加的结果按 BF16 舍入，
# 以保证与下面融合 kernel 的数值完全一致（每一步的舍入点都相同）。
def sandwich_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_norm: RMSNorm,
    norm: RMSNorm,
    *,
    x_scale: float | None = None,
    residual_scale: float | None = None,
    token_fp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A pre/post ("sandwich") norm boundary on a reduced sublayer output.

    ``a = post_norm(x)``, then ``residual_out = RN(RN(x_scale * a) +
    RN(residual_scale * residual))`` (each RN a BF16 rounding; without a scale
    pair the plain BF16 sum) and ``out = norm(residual_out)``. Both norms run
    ``ops/layernorm/triton.py``'s ``rmsnorm`` with the modules' weights and
    epsilons: the arithmetic the fused all-reduce epilogue
    (``allreduce_sandwich_rmsnorm``) reproduces.

    Args:
        x: BF16 ``[T, H]`` all-reduced sublayer output.
        residual: BF16 ``[T, H]`` residual stream.
        post_norm: The norm of the sublayer output (applied before the add).
        norm: The norm of the new residual stream (the next pre-norm).
        x_scale, residual_scale: Optional multipliers of the two addends
            (a missing one is 1.0; NVIDIA only, as ``rmsnorm``'s).
        token_fp8_out: Optional ``(codes, scales)`` buffers (contiguous FP8
            E4M3 ``[T, H]``, FP32 ``[T, 1]``): the second norm also writes
            ``out`` quantized per token, as ``rmsnorm``'s ``token_fp8_out``.

    Returns:
        ``(out, residual_out)``.

    Raises:
        ValueError: A norm scales by ``weight + weight_offset`` with a nonzero
            offset (``GemmaRMSNorm``); both norms must scale by ``weight``.
    """
    # 译：作用在已求和的子层输出上的 pre/post（“三明治”）norm 边界。
    #
    #     ``a = post_norm(x)``，然后 ``residual_out = RN(RN(x_scale * a) + RN(residual_scale * residual))``
    #     （每个 RN 是一次 BF16 舍入；没有 scale 时就是普通的 BF16 求和），``out = norm(residual_out)``。
    #     两个 norm 都用模块自己的权重和 epsilon 运行 ``ops/layernorm/triton.py`` 的 ``rmsnorm``：
    #     这正是融合的 all-reduce 收尾（``allreduce_sandwich_rmsnorm``）所复现的运算。
    #
    #     参数：
    #         x：BF16 ``[T, H]``，已 all-reduce 的子层输出。
    #         residual：BF16 ``[T, H]``，残差流。
    #         post_norm：子层输出的 norm（在相加之前作用）。
    #         norm：新残差流的 norm（即下一个 pre-norm）。
    #         x_scale, residual_scale：两个加数的可选乘数（缺省为 1.0；与 ``rmsnorm`` 的一样，仅限 NVIDIA）。
    #         token_fp8_out：可选的 ``(codes, scales)`` 缓冲区（连续的 FP8 E4M3 ``[T, H]``、FP32 ``[T, 1]``）：
    #             第二个 norm 还会把 ``out`` 按 token 量化后写入，与 ``rmsnorm`` 的 ``token_fp8_out`` 相同。
    #
    #     返回：
    #         ``(out, residual_out)``。
    #
    #     异常：
    #         ValueError：某个 norm 按 ``weight + weight_offset``（offset 非零）缩放（``GemmaRMSNorm``）；
    #             两个 norm 都必须按 ``weight`` 缩放。
    _check_plain_gamma(post_norm, norm)
    a = triton_rmsnorm(
        x, post_norm.weight.data, post_norm.variance_epsilon, enable_pdl=None
    )
    return triton_rmsnorm(
        a,
        norm.weight.data,
        norm.variance_epsilon,
        residual=residual,
        enable_pdl=None,
        round_residual_sum_bf16=True,
        x_scale=x_scale,
        residual_scale=residual_scale,
        **({} if token_fp8_out is None else {"token_fp8_out": token_fp8_out}),
    )


def _sandwich_fusion_applies(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_norm: RMSNorm,
    norm: RMSNorm,
    group: tuple[int, ...],
) -> bool:
    """Whether to try the fused launch: ``--enable-allreduce-fusion``, a
    group of two or more ranks, ``1 <= T <= --comm-fusion-max-num-tokens``,
    BF16 tensors and weights and one epsilon for both norms (the fused
    kernel's contract)."""
    # 译：是否尝试融合启动：开启了 ``--enable-allreduce-fusion``、组内有两个及以上 rank、
    #     ``1 <= T <= --comm-fusion-max-num-tokens``、张量和权重都是 BF16、两个 norm 的 epsilon 相同
    #     （融合 kernel 的约定）。
    return (
        global_server_args_dict.get("enable_allreduce_fusion", False)
        and len(group) > 1
        and post_norm.variance_epsilon == norm.variance_epsilon
        and 1 <= x.shape[0] <= global_server_args_dict["comm_fusion_max_num_tokens"]
        and all(
            t.dtype == torch.bfloat16
            for t in (x, residual, post_norm.weight, norm.weight)
        )
    )


# 带通信的 sandwich 边界：模型在“行并行子层输出遇到它的 post norm、残差相加和下一个 pre norm”处调用。
# 满足条件时用一个融合 kernel 完成 all-reduce + 整个边界；否则（或融合 kernel 返回 None 时）
# 先单独 all-reduce，再调用上面的 sandwich_rmsnorm。与 RMSNorm.forward_with_allreduce_fusion 不同，
# 这里的回退路径包含 all-reduce。
def sandwich_rmsnorm_with_allreduce(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_norm: RMSNorm,
    norm: RMSNorm,
    *,
    rank: int,
    group: tuple[int, ...],
    x_scale: float | None = None,
    residual_scale: float | None = None,
    token_fp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce a sublayer partial and run the sandwich boundary on the sum.

    The helper a model calls where a tensor-parallel sublayer output meets
    its own output norm, the residual add and the next pre-norm. Under
    ``--enable-allreduce-fusion`` a call of at most
    ``--comm-fusion-max-num-tokens`` rows runs as one MNNVL launch when the
    group's workspace serves it (``allreduce_sandwich_rmsnorm``); every other
    call all-reduces ``x`` through the runtime backend and runs
    :func:`sandwich_rmsnorm`. The two paths perform the same operations with
    the same BF16 rounding points. The fused launch sums the partials as the
    plain MNNVL all-reduce does; where the runtime backend takes that path
    too, only the order of the sum-of-squares reductions differs, and where
    it takes another (IPC, NCCL) the order of the sum may differ as well.
    Every rank of ``group`` takes the same path: the choice depends only on
    the launch options, the shapes and the group's shared workspace.

    Args:
        x: This rank's BF16 ``[T, H]`` sublayer partial (not yet reduced).
        residual: BF16 ``[T, H]`` residual stream, replicated across ``group``.
        post_norm: The norm of the reduced sublayer output.
        norm: The norm of the new residual stream.
        rank: This rank's index in ``group``.
        group: Global ranks of the all-reduce.
        x_scale, residual_scale: Optional multipliers of the two addends.
        token_fp8_out: Optional ``(codes, scales)`` buffers (contiguous FP8
            E4M3 ``[T, H]``, FP32 ``[T, 1]``) that both paths fill with
            ``out`` quantized per token (``fp8_quantize_per_token``'s codes
            and scales of ``out``): the fused launch in its epilogue, the
            unfused chain in its second norm.

    Returns:
        ``(out, residual_out)`` as :func:`sandwich_rmsnorm` of the reduced
        ``x``.

    Raises:
        ValueError: As :func:`sandwich_rmsnorm`, before any collective.
    """
    # 译：对子层的部分和做 all-reduce，并在求和结果上执行 sandwich 边界。
    #
    #     这是模型在张量并行子层的输出遇到它自己的输出 norm、残差相加和下一个 pre-norm 时调用的辅助函数。
    #     在 ``--enable-allreduce-fusion`` 下，当组的工作区能满足时，不超过 ``--comm-fusion-max-num-tokens``
    #     行的调用作为一次 MNNVL 启动运行（``allreduce_sandwich_rmsnorm``）；其他所有调用都通过运行时的
    #     通信后端对 ``x`` 做 all-reduce，再运行 :func:`sandwich_rmsnorm`。两条路径执行相同的运算，BF16 舍入点
    #     也相同。融合启动按普通 MNNVL all-reduce 的方式对部分和求和；当运行时后端也走那条路径时，只有平方和
    #     归约的顺序不同；当它走其他路径（IPC、NCCL）时，求和的顺序也可能不同。组内每个 rank 都走同一条路径：
    #     选择只取决于启动选项、形状和组的共享工作区。
    #
    #     参数：
    #         x：本 rank 的 BF16 ``[T, H]`` 子层部分和（尚未求和）。
    #         residual：BF16 ``[T, H]`` 残差流，在 ``group`` 内各 rank 上相同。
    #         post_norm：已求和的子层输出的 norm。
    #         norm：新残差流的 norm。
    #         rank：本 rank 在 ``group`` 中的序号。
    #         group：参与 all-reduce 的全局 rank。
    #         x_scale, residual_scale：两个加数的可选乘数。
    #         token_fp8_out：可选的 ``(codes, scales)`` 缓冲区（连续的 FP8 E4M3 ``[T, H]``、FP32 ``[T, 1]``），
    #             两条路径都会把 ``out`` 按 token 量化的结果（``fp8_quantize_per_token`` 对 ``out`` 的 codes 和
    #             scales）写入其中：融合启动在其收尾阶段写入，非融合的链条在其第二个 norm 中写入。
    #
    #     返回：
    #         ``(out, residual_out)``，与对求和后的 ``x`` 调用 :func:`sandwich_rmsnorm` 的结果相同。
    #
    #     异常：
    #         ValueError：与 :func:`sandwich_rmsnorm` 相同，在任何集合通信之前抛出。
    _check_plain_gamma(post_norm, norm)
    fp8_kwargs = {} if token_fp8_out is None else {"token_fp8_out": token_fp8_out}
    if _sandwich_fusion_applies(x, residual, post_norm, norm, group):
        fused = allreduce_sandwich_rmsnorm(
            x.contiguous(),
            residual.contiguous(),
            post_norm.weight.data,
            norm.weight.data,
            rank,
            _get_process_group(group),
            eps=norm.variance_epsilon,
            x_scale=x_scale,
            residual_scale=residual_scale,
            max_token_num=global_server_args_dict["comm_fusion_max_num_tokens"],
            **fp8_kwargs,
        )
        if fused is not None:
            return fused
    if len(group) > 1:
        x = all_reduce(x, group)
    return sandwich_rmsnorm(
        x,
        residual,
        post_norm,
        norm,
        x_scale=x_scale,
        residual_scale=residual_scale,
        **fp8_kwargs,
    )
