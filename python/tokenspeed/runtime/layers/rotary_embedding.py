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


"""Rotary Positional Embeddings."""

# 译：旋转位置编码（RoPE）。

# 旋转位置编码（RoPE）及其各种长上下文变体。
#
# 你在这里：Qwen3Attention 构造时 get_rope(...) 创建本文件的对象，交给 PagedAttention；
#           前向时 PagedAttention 调用 rotary_emb.as_rotary(positions)，由注意力 prologue 在同一个 kernel 里
#           完成 QK-Norm、RoPE 和写 KV（layers/paged_attention.py，第 3 阶段）。
#
# RoPE 原理（q、k 每两个数配成一对，按位置 p 旋转 p × 频率 的角度）和 cos/sin 缓存表都是熟悉的内容，简述如下：
#   inv_freq[i] = base^(-2i / rotary_dim)，i = 0 .. rotary_dim/2 - 1
#   cos_sin_cache[p] = [cos(p * inv_freq) | sin(p * inv_freq)]，形状 [最大位置数, rotary_dim]
# Qwen3-8B：rotary_dim = head_dim = 128，base = rope_theta = 1e6，max_position = 40960
#   -> cos_sin_cache [40960, 128]，前 64 列是 cos、后 64 列是 sin
#   例：B 的新 token 位置 p = 1000：第 0 对旋转 1000 × 1 = 1000 弧度；
#       第 63 对旋转 1000 × 1e6^(-126/128) ≈ 0.00124 弧度（低频，几乎不转）
#
# 本文件需要注意的几点：
#   1. 两种配对方式（RopeStyle）
#      neox：第 i 个数与第 i + d/2 个数配对（前一半与后一半配对）。Qwen3、Llama 都用这种
#      gptj：第 2i 个数与第 2i+1 个数配对（相邻两个配对）
#   2. 在执行面中 RoPE 通常不单独调用：as_rotary 返回一个描述对象 Rotary(cos_sin_cache, positions, 风格, MRoPE 信息)，
#      由注意力 prologue 在融合 kernel 里使用。forward 只在少数需要单独做 RoPE 的地方使用。
#   3. 长上下文变体（rope_scaling）：都通过改写 inv_freq、或者给 cos/sin 乘一个系数来实现，
#      各自的公式写在对应类上。get_rope 按 rope_type 选择：
#        default（无缩放）、linear、dynamic（NTK）、yarn、deepseek_yarn、llama3、longrope（Phi-3）、
#        default 且带 mrope_section 时为多模态 MRoPE（第 16 阶段）
#   4. get_rope 有全局缓存：参数相同的 RoPE 对象只创建一次，36 层共用一个 cos/sin 表。
import dataclasses
import itertools
import logging
import math
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn as nn
from tokenspeed_kernel.ops.attention.prologue import MRope, RopeStyle, Rotary
from tokenspeed_kernel.ops.embedding import apply_rope
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)


# 两种配对方式下的“旋转一半”：x 旋转后 = x * cos + rotate(x) * sin。
#   neox：[x1, x2] -> [-x2, x1]（x1、x2 是前后两半）
#   gptj：相邻两个 (a, b) -> (-b, a)
def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


# Copied from transformers
# 译：从 transformers 复制而来
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 译：把输入的一半 hidden 维旋转。
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 纯 PyTorch 的 RoPE 参考实现（HF 风格，neox 配对）：cos、sin 已经按位置取好，在 fp32 中计算再转回原 dtype。
# 供少数不经过融合 kernel 的模型（如部分视觉编码器）使用。
def apply_rotary_pos_emb_native(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()

    # embedding is performed in float
    # 译：在 float 中计算位置编码
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)

    return q_embed, k_embed


# 标准 RoPE（无缩放），也是所有变体的基类。成员：
#   head_size：头的维度；rotary_dim：参与旋转的维度（可以小于 head_size，即只旋转前一部分）
#   max_position_embeddings：cos/sin 表覆盖的最大位置数
#   base：频率的底数（rope_theta）
#   is_neox_style：配对方式
#   cos_sin_cache：[最大位置数, rotary_dim] 的 fp32 缓存，注册为不保存到 checkpoint 的 buffer
#     （随模块一起移动到 GPU，但不会出现在 state_dict 里）
# 子类通过覆盖 _compute_inv_freq 或 _compute_cos_sin_cache 实现各种缩放。
class RotaryEmbedding(torch.nn.Module):
    """Original rotary positional embedding."""

    # 译：原始的旋转位置编码。

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: int | float) -> torch.Tensor:
        """Compute the inverse frequency."""
        # 译：计算频率的倒数（inv_freq）。
        #  To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        # 译：为了与 HF 的实现完全一致，需要在 CPU 上计算缓存再移到 GPU。不过我们在 GPU 上创建缓存以加快
        #     初始化。这可能导致我们的实现与 HF 的实现之间有细微的数值差异。
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        # 译：计算 cos 和 sin 缓存。
        # freqs[p][i] = p * inv_freq[i]（einsum 就是外积），形状 [最大位置数, rotary_dim/2]；
        # 拼接 cos、sin 得到 [最大位置数, rotary_dim]
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    # 单独执行 RoPE：用 kernel 包的 apply_rope 按 positions 查表并旋转 q、k。
    # 主线（Qwen3）不走这里，而是走 as_rotary + 注意力 prologue。offsets 只有 DeepSeek 变体支持。
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if offsets is not None:
            raise ValueError("embedding.rope does not support offsets")
        return apply_rope(
            positions=positions,
            q=query,
            k=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )

    # 把本对象打包成注意力 prologue 的“旋转步骤”描述：cos/sin 表 + 本次前向每个 token 的位置 + 配对方式。
    # 例：T=301，positions = [0, 1, ..., 299, 1000]（A 的 300 个 token、B 的 1 个 token）
    def as_rotary(self, positions: torch.Tensor) -> Rotary:
        """This embedding as the attention prologue's rotary step."""
        # 译：把本编码作为注意力 prologue 的旋转步骤。
        style = RopeStyle.NEOX if self.is_neox_style else RopeStyle.GPTJ
        return Rotary(self.cos_sin_cache, positions, style, None)

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


# 线性缩放（位置插值）：位置除以 factor 再查表，等于把所有频率除以 factor，使 factor 倍长的上下文
# 落在训练时见过的角度范围内。cos/sin 表覆盖 原长度 × factor 个位置。
# 支持多个 factor（为多个 LoRA 适配器准备），各自的表首尾拼接，用 offset 区分。
class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    It supports multiple scaling factors. Since multiple LoRA adapters may have
    different scaling factors, we need multiple cos/sin caches. In this way,
    instead of running rotary embedding kernel per lora, we can run multiple
    lora in a batched way.

    In addition to that, we also keep the cos/sin cache for the scaling factor
    of 1 (default) at all times.

    Exemplary for two scaling factors x=1, y and z with embeddings
    [[x11, x12, ... x1m], ..., [xn1, xn2, ..., xnm]] and
    [[y11, y12, ... y1o], ..., [yn1, yn2, ..., yno]], and
    [[z11, z12, ... z1p], ..., [zn1, zn2, ..., znp]],

    we construct the cos/sin cache as follows:
    [[x11, x12, ... x1m, y11, y12, ... y1o, z11, z12, ... z1p],
        ...
     [xn1, xn2, ... xnm, yn1, yn2, ... yno, zn1, zn2, ... znp]]

    We then use offsets to index into the cos/sin cache for
    the respective scaling factors.

    The offset to cache can be accessed via `scaling_factor_to_offset` API.

    Credits to the Reddit user /u/kaiokendev
    """

    # 译：扩展了线性缩放的 RotaryEmbedding。
    #
    #     它支持多个缩放因子。由于多个 LoRA 适配器可能有不同的缩放因子，我们需要多个 cos/sin 缓存。这样，就不必为每个
    #     lora 分别运行旋转编码 kernel，而可以批量地运行多个 lora。
    #
    #     此外，我们也始终保留缩放因子为 1（默认）的 cos/sin 缓存。
    #
    #     以三个缩放因子 x=1、y 和 z 为例，它们的编码分别为
    #     [[x11, x12, ... x1m], ..., [xn1, xn2, ..., xnm]]、
    #     [[y11, y12, ... y1o], ..., [yn1, yn2, ..., yno]] 和
    #     [[z11, z12, ... z1p], ..., [zn1, zn2, ..., znp]]，
    #
    #     我们按如下方式构造 cos/sin 缓存：
    #     [[x11, x12, ... x1m, y11, y12, ... y1o, z11, z12, ... z1p],
    #         ...
    #      [xn1, xn2, ... xnm, yn1, yn2, ... yno, zn1, zn2, ... znp]]
    #
    #     然后用偏移量索引到对应缩放因子的 cos/sin 缓存。
    #
    #     缓存的偏移量可以通过 `scaling_factor_to_offset` 接口获取。
    #
    #     感谢 Reddit 用户 /u/kaiokendev

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: list[float] | float,
        dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: list[float] = scaling_factors
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        # Lazy initialized.
        # 译：延迟初始化。
        self._scaling_factor_to_offset: dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: list[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        # 译：到张量中下一个缓存的偏移量。
        #     每个偏移量与 scaling_factors 中相同下标的因子对应。
        offsets: list[int] = []
        for scaling_factor in self.scaling_factors:
            #  self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            # 译：self.max_position_embeddings 是应用 rope 缩放之前的原始最大长度。
            #     因此应用 rope 缩放之后的最大长度是 self.max_position_embeddings * self.scaling_factor。
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        if len(self.scaling_factors) != len(offsets):
            raise RuntimeError("scaling factor offsets were not initialized correctly.")
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> dict[float, int]:
        return self._scaling_factor_to_offset


# 动态 NTK 缩放：不改位置，而是增大 base（等价于主要降低低频维度的频率），
#   base' = base × (s × 新长度 / 原长度 - (s - 1))^(d / (d - 2))，s 为 factor，d 为 rotary_dim
# 这里新长度固定为 原长度 × s（即按最大长度一次算好，而不是随当前序列长度动态变化）。
class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    # 译：扩展了动态 NTK 缩放的 RotaryEmbedding。
    #
    #     感谢 Reddit 用户 /u/bloc97 和 /u/emozilla

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        #  self.max_position_embeddings is the original
        # maximum length before applying the rope scaling.
        # Thus, the maximum length after applying the rope scaling is
        # self.max_position_embeddings * self.scaling_factor.
        # 译：self.max_position_embeddings 是应用 rope 缩放之前的原始最大长度。
        #     因此应用 rope 缩放之后的最大长度是 self.max_position_embeddings * self.scaling_factor。
        max_len = self.max_position_embeddings * self.scaling_factor
        base = self.base * (
            (self.scaling_factor * max_len / self.max_position_embeddings)
            - (self.scaling_factor - 1)
        ) ** (self.rotary_dim / (self.rotary_dim - 2))
        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(max_len, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


# YaRN 的辅助函数。核心想法：按维度区别对待——
#   高频维度（在原长度内转很多圈）：位置信息靠局部相对位置，保持原频率（外推）
#   低频维度（在原长度内转不满一圈）：按 factor 插值（频率除以 factor）
#   中间的维度：线性过渡
# _yarn_find_correction_dim：反解“在原长度内转 num_rotations 圈的是第几个维度对”：
#   dim × ln(L / (2π × r)) / (2 × ln(base))
# Inverse dim formula to find dim based on number of rotations
# 译：根据旋转圈数反求维度的公式
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    """The YaRN correction range: rotary-dimension bounds for the ramp mask.

    Inverts the RoPE frequency formula to find, for each rotation count, the
    rotary dimension whose period over ``max_position_embeddings`` positions
    completes that many rotations. Dimensions below the low bound rotate more
    than ``low_rot`` times (high frequency, left unscaled by YaRN); dimensions
    above the high bound rotate fewer than ``high_rot`` times (low frequency,
    fully interpolated); the ramp from ``yarn_linear_ramp_mask`` blends between
    the two across ``[low, high]``.

    Args:
        low_rot: Rotation count marking the end of the unscaled region
            (YaRN's ``beta_fast``).
        high_rot: Rotation count marking the start of the fully interpolated
            region (YaRN's ``beta_slow``).
        dim: Rotary dimension of the embedding.
        base: RoPE frequency base.
        max_position_embeddings: Original (pre-scaling) maximum position.

    Returns:
        ``(low, high)`` rotary-dimension indices, floor/ceil-rounded and
        clamped to ``[0, dim - 1]``.
    """
    # 译：YaRN 的修正范围：斜坡掩码在旋转维度上的上下界。
    #
    #     反解 RoPE 的频率公式，对每个旋转圈数找出在 ``max_position_embeddings`` 个位置内恰好转那么多圈的旋转维度。
    #     低于下界的维度转的圈数多于 ``low_rot``（高频，YaRN 不缩放）；高于上界的维度转的圈数少于 ``high_rot``
    #     （低频，完全插值）；``yarn_linear_ramp_mask`` 生成的斜坡在 ``[low, high]`` 之间把两者混合。
    #
    #     参数：
    #         low_rot：标记不缩放区域结束的旋转圈数（YaRN 的 ``beta_fast``）。
    #         high_rot：标记完全插值区域开始的旋转圈数（YaRN 的 ``beta_slow``）。
    #         dim：编码的旋转维度。
    #         base：RoPE 的频率底数。
    #         max_position_embeddings：原始（缩放前的）最大位置。
    #
    #     返回：
    #         ``(low, high)`` 旋转维度下标，分别向下 / 向上取整，并截断到 ``[0, dim - 1]``。
    # 例（Qwen3 官方建议的 YaRN 设置：factor=4，原长度 32768，base=1e6，dim=128，beta_fast=32，beta_slow=1）：
    #   low = floor(23.6) = 23，high = ceil(39.7) = 40
    #   -> 第 0~22 对保持原频率，第 40~63 对频率除以 4，中间线性过渡
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case
    # 译：以防万一，截断取值


def yarn_linear_ramp_mask(
    low: float,
    high: float,
    dim: int,
    dtype: torch.dtype,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    """The YaRN linear ramp ``clamp((i - low) / (high - low), 0, 1)``.

    Args:
        low: Ramp start (inclusive) in rotary-dimension index units.
        high: Ramp end in the same units; a degenerate ``low == high`` is
            widened by 0.001.
        dim: Number of ramp entries (half the rotary dimension).
        dtype: Output dtype.
        device: Device the arange and division run on (CPU and CUDA round the
            division differently at ulp level); None for the default.

    Returns:
        ``[dim]`` ramp on ``device``.
    """
    # 译：YaRN 的线性斜坡 ``clamp((i - low) / (high - low), 0, 1)``。
    #
    #     参数：
    #         low：斜坡起点（包含），单位是旋转维度下标。
    #         high：斜坡终点，单位相同；退化的 ``low == high`` 会被加宽 0.001。
    #         dim：斜坡的项数（旋转维度的一半）。
    #         dtype：输出的 dtype。
    #         device：执行 arange 和除法的设备（CPU 和 CUDA 在 ulp 级别上对除法的舍入不同）；None 表示默认设备。
    #
    #     返回：
    #         ``device`` 上的 ``[dim]`` 斜坡。
    if low == high:
        high += 0.001  # Prevent singularity
        # 译：防止出现奇点（除以 0）

    linear_func = (torch.arange(dim, dtype=dtype, device=device) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    """The YaRN attention magnitude correction ``0.1 * mscale * ln(scale) + 1``.

    YaRN scales attention logits up slightly as the context is extended to
    compensate for the entropy drop of interpolated rotary embeddings; the
    cos/sin cache is multiplied by this factor.

    Args:
        scale: Context extension factor (``rope_scaling["factor"]``); no
            correction is applied at or below 1.
        mscale: Model-specific multiplier on the log term (DeepSeek's
            ``mscale`` / ``mscale_all_dim``); 1 gives the original YaRN
            formula.

    Returns:
        The magnitude correction factor, 1.0 when ``scale <= 1``.
    """
    # 译：YaRN 的注意力幅度修正 ``0.1 * mscale * ln(scale) + 1``。
    #
    #     随着上下文被扩展，YaRN 会略微放大注意力 logits，以补偿插值后的旋转编码带来的熵下降；cos/sin 缓存会乘以这个系数。
    #
    #     参数：
    #         scale：上下文扩展倍数（``rope_scaling["factor"]``）；小于等于 1 时不做修正。
    #         mscale：对数项上与模型相关的乘数（DeepSeek 的 ``mscale`` / ``mscale_all_dim``）；为 1 时就是原始的 YaRN 公式。
    #
    #     返回：
    #         幅度修正系数，``scale <= 1`` 时为 1.0。
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# YaRN：
#   inv_freq = 插值频率 × (1 - mask) + 原频率 × mask，mask 由上面的斜坡决定（高频维度 mask=1）
#   cos/sin 都乘以 mscale：q、k 都被放大 mscale 倍，所以注意力 logits 被放大 mscale² 倍
#     （factor=4 时 mscale = 0.1 × ln4 + 1 ≈ 1.139）
#   cos/sin 表覆盖 原长度 × factor 个位置
# 注意构造参数 max_position_embeddings 传入的是原始长度（见 _build_yarn）。
class YaRNScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    # 译：扩展了 YaRN 方法的 RotaryEmbedding。
    #
    #     感谢 Peng 等人 github.com/jquesnelle/yarn

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        # 译：得到针对插值修正过的 n 维幅度缩放
        self.mscale = float(yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        # 译：得到针对外推修正过的 n 维旋转缩放
        inv_freq_mask = (
            1
            - yarn_linear_ramp_mask(
                low, high, self.rotary_dim // 2, dtype=torch.float, device=None
            )
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor, dtype=torch.float32
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache


# Phi-3 的 LongRoPE：每个维度对有自己的缩放系数（short_factor / long_factor 两套），
# cos/sin 表由两段拼接：前一段用 short_factor、覆盖原长度，后一段用 long_factor、覆盖扩展后的长度。
# as_rotary：只要本次前向中有任何一个位置超过原长度，整个 batch 的所有 token 都改用后一段（long）。
class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    """Phi3 family of models scaled rotary embedding."""

    # 译：Phi3 系列模型的缩放旋转编码。

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        short_factor: list[float],
        long_factor: list[float],
        short_mscale: float | None = None,
        long_mscale: float | None = None,
    ):
        super().__init__()

        if is_neox_style is False:
            raise ValueError(
                "`Phi3LongRoPEScaledRotaryEmbedding` only supports neox_style."
            )

        self.rotary_dim = rotary_dim
        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.short_factor = short_factor
        self.long_factor = long_factor

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) / math.log(self.original_max_position_embeddings)
            )
        if short_mscale is None:
            short_mscale = scaling_factor
        if long_mscale is None:
            long_mscale = scaling_factor

        self.short_mscale = short_mscale
        self.long_mscale = long_mscale

        # Short-context rows, then long-context rows: as_rotary offsets into the second half.
        # 译：先是短上下文的行，再是长上下文的行：as_rotary 偏移到后一半。
        rope_cache = torch.cat(
            [
                self._compute_cos_sin_cache(
                    original_max_position_embeddings, short_factor, short_mscale
                ),
                self._compute_cos_sin_cache(
                    max_position_embeddings, long_factor, long_mscale
                ),
            ]
        )
        self.register_buffer("cos_sin_cache", rope_cache, persistent=False)

    def _compute_inv_freq(self, rescale_factors: list[float]) -> torch.Tensor:
        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32)
        inv_freq = 1.0 / (
            rescale_factors
            * (
                self.base
                ** (
                    torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                    / self.rotary_dim
                )
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(
        self,
        max_position_embeddings: int,
        rescale_factors: list[float],
        mscale: float,
    ) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(rescale_factors)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def as_rotary(self, positions: torch.Tensor) -> Rotary:
        """Once any position passes the original context, every token reads
        the long-context rows."""
        # 译：一旦有任何位置超过原始上下文长度，每个 token 都读取长上下文的行。
        original = self.original_max_position_embeddings
        offset = torch.any(positions > original) * original
        return Rotary(self.cos_sin_cache, positions + offset, RopeStyle.NEOX, None)


# DeepSeek 风格的 YaRN：与上面的 YaRN 相同，区别在于：
#   mscale = yarn_get_mscale(s, mscale) / yarn_get_mscale(s, mscale_all_dim) × attn_factor
#     （DeepSeek 把一部分幅度修正放在 softmax 的缩放系数里，这里只保留剩余部分）
#   可以指定 cos/sin 表、斜坡、频率分别在哪个设备上计算：不同设备的浮点舍入略有差异，
#     为了与训练框架的数值逐位一致，由启动参数 --yarn-ramp-mask-device / --yarn-frequency-device 控制
#   forward 支持 offsets（位置偏移），走纯 PyTorch 路径
class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    # 译：扩展了 YaRN 方法的 RotaryEmbedding。
    #
    #     感谢 Peng 等人 github.com/jquesnelle/yarn

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        device: str | None = None,
        ramp_device: str,
        frequency_device: str,
    ) -> None:
        """
        Args:
            device: Device the cos/sin cache is built on and the inverse
                frequencies are moved to; None selects the platform default
                (``"npu"`` on NPU, ``"cuda"`` elsewhere).
            ramp_device: Device that computes the YaRN linear ramp mask.
            frequency_device: Device that computes the powers, divisions and
                interpolation. The ramp is transferred here before combining.
        """
        # 译：参数：
        #         device：构建 cos/sin 缓存、以及频率倒数被移动到的设备；None 选择平台默认值
        #             （NPU 上为 ``"npu"``，其他为 ``"cuda"``）。
        #         ramp_device：计算 YaRN 线性斜坡掩码的设备。
        #         frequency_device：计算幂、除法和插值的设备。斜坡在合并之前会被传输到这里。
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        # 译：得到针对插值修正过的 n 维幅度缩放。
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        self.device = (
            device
            if device is not None
            else ("npu" if current_platform().is_npu else "cuda")
        )
        self.ramp_device = ramp_device
        self.frequency_device = frequency_device
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        # Power/division and ramp construction can have different rounding
        # contracts. Move the ramp without recomputing its values.
        # 译：幂 / 除法和斜坡构造可能有不同的舍入约定。移动斜坡时不重新计算它的值。
        pos_freqs = self.base ** (
            torch.arange(
                0, self.rotary_dim, 2, dtype=torch.float, device=self.frequency_device
            )
            / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        # 译：得到针对外推修正过的 n 维旋转缩放
        inv_freq_mask = (
            1
            - yarn_linear_ramp_mask(
                low,
                high,
                self.rotary_dim // 2,
                dtype=torch.float,
                device=self.ramp_device,
            ).to(self.frequency_device)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq.to(self.device)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            device=self.device,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if offsets is None:
            return super().forward(positions=positions, query=query, key=key)

        dtype = query.dtype
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            #  Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            # 译：这里我们假定 positions 张量的形状是 [batch_size, seq_len]。
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query.to(dtype), key.to(dtype)


# Llama 3.1 的缩放：按波长（2π / 频率）把维度分成三段，原长度记为 L：
#   波长 < L / high_freq_factor（高频）：保持原频率
#   波长 > L / low_freq_factor（低频）：频率除以 factor
#   中间：在两者之间平滑过渡，smooth = (L / 波长 - low_freq_factor) / (high_freq_factor - low_freq_factor)
# 例（Llama 3.1：factor=8、low=1、high=4、L=8192）：波长 < 2048 不变，> 8192 频率除以 8
class Llama3RotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, base: int | float) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = 0
        new_freqs = torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor + smooth * inv_freqs,
            ),
        )
        return new_freqs


# NTK-alpha 动态缩放：base' = base × alpha^(d / (d - 2))，cos/sin 表仍只覆盖原长度。
class DynamicNTKAlphaRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    # 译：扩展了动态 NTK 缩放的 RotaryEmbedding。
    #
    #     感谢 Reddit 用户 /u/bloc97 和 /u/emozilla

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_alpha: float,
        dtype: torch.dtype,
    ) -> None:
        self.scaling_alpha = scaling_alpha
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        max_len = self.max_position_embeddings
        base = self.base * self.scaling_alpha ** (
            self.rotary_dim / (self.rotary_dim - 2)
        )

        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(max_len, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


# 多模态 RoPE（MRoPE，Qwen2-VL / Qwen3-VL 等，第 16 阶段回访）：每个 token 有 (时间, 高, 宽) 三个位置，
# rotary_dim / 2 个维度对按 mrope_section 分成三段，各段用对应的那个位置旋转。
#   mrope_section：三段各有多少个维度对，和必须等于 rotary_dim / 2（不等时按比例自动修正并打印警告）
#   mrope_interleaved：三段在维度上交错排列而不是连续排列
# 只能通过 as_rotary 在注意力 prologue 中使用，forward 直接报错。
# get_rope_index / get_rope_index_glm4v：根据 input_ids 中的图像 / 视频 token 和网格大小，算出每个 token 的三维位置
# 以及位置偏移量（文本 token 的三个位置相同，视觉 token 按 t、h、w 网格编号）。
class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    # 译：带多模态分段的旋转编码。

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: list[int],
        mrope_interleaved: bool,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        if self.mrope_section:
            expected_sum = rotary_dim // 2
            actual_sum = sum(self.mrope_section)
            if actual_sum != expected_sum:
                logger.warning(
                    f"MRoPE section sum mismatch: expected {expected_sum}, got {actual_sum}. "
                    f"Adjusting mrope_section to match rotary_dim // 2 = {expected_sum}"
                )
                # Auto-correct by scaling the mrope_section proportionally
                # 译：按比例缩放 mrope_section 来自动修正
                if actual_sum > 0:
                    scale_factor = expected_sum / actual_sum
                    self.mrope_section = [
                        max(1, int(section * scale_factor))
                        for section in self.mrope_section
                    ]
                    # Ensure the sum exactly matches by adjusting the last element
                    # 译：调整最后一个元素，确保总和恰好相等
                    current_sum = sum(self.mrope_section)
                    if current_sum != expected_sum:
                        self.mrope_section[-1] += expected_sum - current_sum
                else:
                    # If all sections are 0, create a default distribution
                    # 译：如果所有分段都是 0，就创建一个默认的分配
                    self.mrope_section = [
                        expected_sum // len(self.mrope_section)
                    ] * len(self.mrope_section)
                    # Handle remainder
                    # 译：处理余数
                    remainder = expected_sum % len(self.mrope_section)
                    for i in range(remainder):
                        self.mrope_section[i] += 1

                logger.warning(
                    f"Corrected mrope_section: {self.mrope_section} (sum={sum(self.mrope_section)})"
                )

    def as_rotary(self, positions: torch.Tensor) -> Rotary:
        """This embedding, with its multimodal sections, as the prologue's rotary step."""
        # 译：把本编码连同其多模态分段，作为 prologue 的旋转步骤。
        style = RopeStyle.NEOX if self.is_neox_style else RopeStyle.GPTJ
        return Rotary(
            self.cos_sin_cache,
            positions,
            style,
            MRope(tuple(self.mrope_section), self.mrope_interleaved),
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("M-RoPE rotates inside the attention prologue")

    @staticmethod
    def get_rope_index(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        model_type: str,
        tokens_per_second: int | None = None,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    if model_type == "qwen2_5_vl":
                        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                        expanded_range = range_tensor.expand(
                            -1, llm_grid_h * llm_grid_w
                        )

                        time_tensor = (
                            expanded_range * second_per_grid_t * tokens_per_second
                        )

                        time_tensor_long = time_tensor.long()
                        t_index = time_tensor_long.flatten()
                    elif model_type in (
                        "qwen2_vl",
                        "qwen3_vl",
                        "qwen3_vl_moe",
                        "qwen3_5",
                        "qwen3_5_moe",
                        "qwen4_exp",
                    ):
                        t_index = (
                            torch.arange(llm_grid_t)
                            .view(-1, 1)
                            .expand(-1, llm_grid_h * llm_grid_w)
                            .flatten()
                        )
                    else:
                        raise RuntimeError("Unimplemented")
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            s = input_ids.shape[1]
            position_ids = torch.arange(s)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - s
            return position_ids, mrope_position_deltas

    @staticmethod
    def get_rope_index_glm4v(
        input_ids: torch.Tensor,
        hf_config: Any,
        image_grid_thw: list[list[int]] | torch.Tensor,
        video_grid_thw: list[list[int]] | torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mrope input positions and delta value for GLM4V."""
        # 译：为 GLM4V 计算 mrope 的输入位置和偏移值。
        image_token_id = hf_config.image_token_id
        video_start_token_id = hf_config.video_start_token_id
        video_end_token_id = hf_config.video_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            video_group_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                input_tokens = input_ids.tolist()

                input_token_type = []
                video_check_flg = False
                for token in input_tokens:
                    if token == video_start_token_id:
                        video_check_flg = True
                    elif token == video_end_token_id:
                        video_check_flg = False

                    if token == image_token_id and not video_check_flg:
                        input_token_type.append("image")
                    elif token == image_token_id and video_check_flg:
                        input_token_type.append("video")
                    else:
                        input_token_type.append("text")

                input_type_group = []
                for key, group in itertools.groupby(
                    enumerate(input_token_type), lambda x: x[1]
                ):
                    group = list(group)
                    start_index = group[0][0]
                    end_index = group[-1][0] + 1
                    input_type_group.append((key, start_index, end_index))

                llm_pos_ids_list = []
                video_frame_num = 1
                for modality_type, start_idx, end_idx in input_type_group:
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )

                    if modality_type == "image":
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        t_index = (
                            torch.arange(llm_grid_t)
                            .view(-1, 1)
                            .expand(-1, llm_grid_h * llm_grid_w)
                            .flatten()
                        )
                        h_index = (
                            torch.arange(llm_grid_h)
                            .view(1, -1, 1)
                            .expand(llm_grid_t, -1, llm_grid_w)
                            .flatten()
                        )
                        w_index = (
                            torch.arange(llm_grid_w)
                            .view(1, 1, -1)
                            .expand(llm_grid_t, llm_grid_h, -1)
                            .flatten()
                        )
                        llm_pos_ids_list.append(
                            torch.stack([t_index, h_index, w_index]) + st_idx
                        )

                        image_index += 1
                        video_frame_num = 1

                    elif modality_type == "video":
                        t, h, w = (
                            video_frame_num,
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )

                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t,
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        for t_idx in range(llm_grid_t):
                            t_index = (
                                torch.tensor(t_idx)
                                .view(-1, 1)
                                .expand(-1, llm_grid_h * llm_grid_w)
                                .flatten()
                            )

                            h_index = (
                                torch.arange(llm_grid_h)
                                .view(1, -1, 1)
                                .expand(1, -1, llm_grid_w)
                                .flatten()
                            )
                            w_index = (
                                torch.arange(llm_grid_w)
                                .view(1, 1, -1)
                                .expand(1, llm_grid_h, -1)
                                .flatten()
                            )
                            llm_pos_ids_list.append(
                                torch.stack([t_index, h_index, w_index]) + st_idx
                            )

                        video_group_index += 1

                        if video_group_index >= video_grid_thw[video_index][0]:
                            video_index += 1
                            video_group_index = 0

                        video_frame_num += 1

                    else:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        video_frame_num = 1

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


# 下面是 get_rope 的构造流程：
#   RopeScaling：把配置中的 rope_scaling 字典解析成带类型的字段，并检查该类型必需的键
#   _build_*：每种 rope_type 一个构造函数，登记在 _ROPE_BUILDERS 表中
@dataclasses.dataclass(frozen=True)
class RopeScaling:
    """A ``rope_scaling`` config dict parsed once into typed fields.

    ``from_dict`` validates the type tag (``rope_type``, legacy ``type``) and
    the keys the selected scaling type requires, so a malformed checkpoint
    config fails here with the missing keys named instead of a ``KeyError``
    deep in ``get_rope``. Keys the selected type does not consume stay
    ignored, as before.
    """

    # 译：一次性解析为带类型字段的 ``rope_scaling`` 配置字典。
    #
    #     ``from_dict`` 校验类型标签（``rope_type``，旧版为 ``type``）以及所选缩放类型要求的键，所以格式错误的 checkpoint
    #     配置会在这里报错并指出缺失的键，而不是在 ``get_rope`` 深处抛出 ``KeyError``。所选类型不使用的键像以前一样被忽略。

    rope_type: str
    factor: float | None = None
    original_max_position_embeddings: int | None = None
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    alpha: float | None = None
    extrapolation_factor: float = 1
    attn_factor: float = 1
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1
    mscale_all_dim: float = 0
    short_factor: list[float] | None = None
    long_factor: list[float] | None = None
    short_mscale: float | None = None
    long_mscale: float | None = None
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False

    # Keys each scaling type must carry, beyond the type tag. "dynamic" needs
    # "alpha" for the NTK-alpha variant and "factor" otherwise.
    # 译：每种缩放类型除了类型标签之外必须携带的键。"dynamic" 在 NTK-alpha 变体中需要 "alpha"，否则需要 "factor"。
    _REQUIRED_KEYS: ClassVar[dict[str, tuple[str, ...]]] = {
        "llama3": (
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
        ),
        "linear": ("factor",),
        "yarn": ("factor", "original_max_position_embeddings"),
        "deepseek_yarn": ("factor", "original_max_position_embeddings"),
        "longrope": (
            "short_factor",
            "long_factor",
            "original_max_position_embeddings",
        ),
    }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RopeScaling":
        """Parse and validate a checkpoint's ``rope_scaling`` dict."""
        # 译：解析并校验 checkpoint 的 ``rope_scaling`` 字典。
        if "rope_type" in raw:
            rope_type = raw["rope_type"]
        elif "type" in raw:
            rope_type = raw["type"]
        else:
            raise ValueError("Unknown RoPE scaling type")

        known = {f.name for f in dataclasses.fields(cls)} - {"rope_type"}
        scaling = cls(
            rope_type=rope_type, **{k: v for k, v in raw.items() if k in known}
        )

        required = cls._REQUIRED_KEYS.get(rope_type, ())
        if rope_type == "dynamic":
            required = ("alpha",) if "alpha" in raw else ("factor",)
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(
                f"rope_scaling type {rope_type!r} misses required "
                f"key(s): {', '.join(missing)}"
            )
        return scaling


@dataclasses.dataclass(frozen=True)
class _RopeCommonArgs:
    """Constructor arguments every rotary embedding shares."""

    # 译：每种旋转编码共有的构造参数。

    head_size: int
    rotary_dim: int
    max_position: int
    base: int
    is_neox_style: bool
    dtype: torch.dtype


@dataclasses.dataclass(frozen=True)
class _YarnDevices:
    """Construction devices deepseek_yarn rounding depends on."""

    # 译：deepseek_yarn 的舍入所依赖的构造设备。

    ramp_device: str
    frequency_device: str


def _build_llama3(
    args: _RopeCommonArgs, scaling: RopeScaling, yarn: _YarnDevices
) -> RotaryEmbedding:
    return Llama3RotaryEmbedding(
        args.head_size,
        args.rotary_dim,
        args.max_position,
        args.base,
        args.is_neox_style,
        args.dtype,
        scaling.factor,
        scaling.low_freq_factor,
        scaling.high_freq_factor,
        scaling.original_max_position_embeddings,
    )


def _build_default(
    args: _RopeCommonArgs, scaling: RopeScaling, yarn: _YarnDevices
) -> RotaryEmbedding:
    if scaling.mrope_section is not None:
        return MRotaryEmbedding(
            args.head_size,
            args.rotary_dim,
            args.max_position,
            args.base,
            args.is_neox_style,
            args.dtype,
            mrope_section=scaling.mrope_section,
            mrope_interleaved=scaling.mrope_interleaved,
        )
    return RotaryEmbedding(
        args.head_size,
        args.rotary_dim,
        args.max_position,
        args.base,
        args.is_neox_style,
        args.dtype,
    )


def _build_linear(
    args: _RopeCommonArgs, scaling: RopeScaling, yarn: _YarnDevices
) -> RotaryEmbedding:
    return LinearScalingRotaryEmbedding(
        args.head_size,
        args.rotary_dim,
        args.max_position,
        args.base,
        args.is_neox_style,
        scaling.factor,
        args.dtype,
    )


def _build_dynamic(
    args: _RopeCommonArgs, scaling: RopeScaling, yarn: _YarnDevices
) -> RotaryEmbedding:
    if scaling.alpha is not None:
        return DynamicNTKAlphaRotaryEmbedding(
            args.head_size,
            args.rotary_dim,
            args.max_position,
            args.base,
            args.is_neox_style,
            scaling.alpha,
            args.dtype,
        )
    return DynamicNTKScalingRotaryEmbedding(
        args.head_size,
        args.rotary_dim,
        args.max_position,
        args.base,
        args.is_neox_style,
        scaling.factor,
        args.dtype,
    )


def _build_yarn(
    args: _RopeCommonArgs, scaling: RopeScaling, yarn: _YarnDevices
) -> RotaryEmbedding:
    return YaRNScalingRotaryEmbedding(
        args.head_size,
        args.rotary_dim,
        scaling.original_max_position_embeddings,
        args.base,
        args.is_neox_style,
        scaling.factor,
        args.dtype,
        extrapolation_factor=scaling.extrapolation_factor,
        attn_factor=scaling.attn_factor,
        beta_fast=scaling.beta_fast,
        beta_slow=scaling.beta_slow,
    )


def _build_deepseek_yarn(
    args: _RopeCommonArgs, scaling: RopeScaling, yarn: _YarnDevices
) -> RotaryEmbedding:
    return DeepseekScalingRotaryEmbedding(
        args.head_size,
        args.rotary_dim,
        scaling.original_max_position_embeddings,
        args.base,
        args.is_neox_style,
        scaling.factor,
        args.dtype,
        ramp_device=yarn.ramp_device,
        frequency_device=yarn.frequency_device,
        extrapolation_factor=scaling.extrapolation_factor,
        attn_factor=scaling.attn_factor,
        beta_fast=scaling.beta_fast,
        beta_slow=scaling.beta_slow,
        mscale=scaling.mscale,
        mscale_all_dim=scaling.mscale_all_dim,
    )


def _build_longrope(
    args: _RopeCommonArgs, scaling: RopeScaling, yarn: _YarnDevices
) -> RotaryEmbedding:
    return Phi3LongRoPEScaledRotaryEmbedding(
        args.head_size,
        args.rotary_dim,
        args.max_position,
        scaling.original_max_position_embeddings,
        args.base,
        args.is_neox_style,
        scaling.short_factor,
        scaling.long_factor,
        short_mscale=scaling.short_mscale,
        long_mscale=scaling.long_mscale,
    )


# One builder per rope_scaling type; get_rope only resolves the table.
# 译：每种 rope_scaling 类型一个构造函数；get_rope 只负责查表。
_ROPE_BUILDERS: dict[
    str, Callable[[_RopeCommonArgs, RopeScaling, _YarnDevices], RotaryEmbedding]
] = {
    "llama3": _build_llama3,
    "default": _build_default,
    "linear": _build_linear,
    "dynamic": _build_dynamic,
    "yarn": _build_yarn,
    "deepseek_yarn": _build_deepseek_yarn,
    "longrope": _build_longrope,
}


# 全局缓存：参数完全相同的 RoPE 对象只创建一次。
# Qwen3-8B 的 36 层都调用 get_rope(128, 128, 40960, 1e6, ...)，得到的是同一个对象、同一张 cos/sin 表。
_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}


# 创建（或从缓存中取出）RoPE 对象。
#   head_size / rotary_dim：头的维度、参与旋转的维度（partial_rotary_factor < 1 时只旋转前一部分）
#   max_position：cos/sin 表覆盖的最大位置（有缩放时由各变体自己决定实际覆盖范围）
#   base：rope_theta；is_neox_style：配对方式（默认 neox）
#   rope_scaling：HF 配置中的缩放字典，None 表示不缩放
# Qwen3-8B：rope_scaling 为 None -> 返回普通的 RotaryEmbedding
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        # 译：把每个列表类型的值转换为元组，以便用作缓存的键
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None

    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)

    # Both construction devices affect deepseek_yarn rounding, so a cached
    # table must match the launch's ramp and frequency choices.
    # 译：两个构造设备都会影响 deepseek_yarn 的舍入，所以缓存的表必须与本次启动的斜坡和频率设备选择一致。
    yarn_ramp_mask_device = global_server_args_dict["yarn_ramp_mask_device"]
    yarn_frequency_device = global_server_args_dict["yarn_frequency_device"]
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
        yarn_ramp_mask_device,
        yarn_frequency_device,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        scaling = RopeScaling.from_dict(rope_scaling)
        builder = _ROPE_BUILDERS.get(scaling.rope_type)
        if builder is None:
            raise ValueError(f"Unknown RoPE scaling type {scaling.rope_type}")
        rotary_emb = builder(
            _RopeCommonArgs(
                head_size, rotary_dim, max_position, base, is_neox_style, dtype
            ),
            scaling,
            _YarnDevices(yarn_ramp_mask_device, yarn_frequency_device),
        )
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
