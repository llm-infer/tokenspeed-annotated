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

"""Parameter helpers for sharded and packed layer weights."""

# 译：用于切分（sharded）和打包（packed）层权重的参数辅助类。

# “知道怎么加载自己”的参数类型。
#
# 背景：每个线性层的参数都要从 checkpoint 中的完整张量里切出本 rank 的那一份，再写到正确位置。
# 线性层（layers/linear.py）有两套加载方式：
#   v1（weight_loader）：参数是普通的 nn.Parameter，切分逻辑全部写在线性层的 weight_loader 里。
#     未量化（bf16）以及大多数量化方法走这条路，Qwen3-8B bf16 的主线不经过本文件。
#   v2（weight_loader_v2）：参数是本文件中的子类，切分逻辑放在参数自己的 load_* 方法里，
#     线性层只负责算出偏移和大小。目前 FP8（Fp8LinearMethod）和 compressed-tensors 量化方法使用 v2
#     （见 linear.py 中的 WEIGHT_LOADER_V2_METHODS）。
#
# 为什么要按参数类型区分：量化后一个线性层不只有 weight，还有各种 scale，它们的切分方式各不相同。
#   权重本身：列并行按输出维切、行并行按输入维切
#   per-channel scale（每个输出通道一个）：只按输出维切
#   per-tensor scale（每个逻辑矩阵一个）：QKV 合并层有 3 个，按 q/k/v 放进对应位置，不切
#   分组 / 分块量化的 scale：和权重一样两个方向都可能切
#   打包存储的权重（例如 8 个 int4 打包进一个 int32）：偏移和大小都要除以打包因子
# 每种情况对应一个参数类，线性层调用统一的 load_column_parallel_weight / load_row_parallel_weight /
# load_merged_column_weight / load_qkv_weight，具体怎么切由参数类自己决定。
from collections.abc import Callable
from fractions import Fraction

import torch
from torch.nn import Parameter

from tokenspeed.runtime.utils import get_colorful_logger

__all__ = [
    "BaseWeightParameter",
    "PackedWeightParameter",
    "PerTensorScaleParameter",
    "ModelWeightParameter",
    "ChannelQuantScaleParameter",
    "GroupQuantScaleParameter",
    "PackedColumnParameter",
    "RowParallelWeightParameter",
]

logger = get_colorful_logger(__name__)


# 形状不一致时报错：权重切错了位置往往表现为形状对不上，早报错比静默写错好
def _check_shape_match(actual: torch.Size, expected: torch.Size) -> None:
    if actual != expected:
        raise ValueError(f"Shape mismatch: {actual} != {expected}.")


# 所有参数类的基类：不切分，四种加载方法都要求 checkpoint 张量与参数形状完全相同，直接拷贝。
# 推理不需要梯度，所以 requires_grad=False。
# weight_loader：创建参数时由线性层传入的加载回调（通常是线性层的 weight_loader_v2）。
class BaseWeightParameter(Parameter):
    """Base parameter for TokenSpeed linear layers with custom weight loading."""

    # 译：TokenSpeed 线性层使用的参数基类，支持自定义的权重加载。

    def __new__(cls, data: torch.Tensor, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, weight_loader: Callable):
        """Initialize the parameter wrapper with a weight-loader callback."""
        # 译：用权重加载回调初始化这个参数包装。
        self._weight_loader = weight_loader

    @property
    def weight_loader(self) -> Callable:
        return self._weight_loader

    def _assert_and_load(self, loaded_weight: torch.Tensor) -> None:
        _check_shape_match(self.data.shape, loaded_weight.shape)
        self.data.copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)


# 列并行参数的公共加载逻辑。output_dim：输出维是第几维（对 [out, in] 的权重是 0）。
# use_presharded_weights：checkpoint 已经按 rank 切好时为 True，此时不再切，直接拷贝。
class _ColumnParallelWeightParameter(BaseWeightParameter):
    """Shared column-parallel weight-loading helpers."""

    # 译：列并行权重加载的共用辅助方法。

    def __init__(self, output_dim: int, **kwargs):
        self._output_dim = output_dim
        super().__init__(**kwargs)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    # 普通列并行：在输出维上取第 tp_rank 段。
    #   例：输出维 8192、TP=2，本 rank 参数的输出维是 4096，rank 1 取 checkpoint 的 [4096, 8192)
    def load_column_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        if not use_presharded_weights:
            shard_size = self.data.shape[self.output_dim]
            loaded_weight = loaded_weight.narrow(
                self.output_dim, tp_rank * shard_size, shard_size
            )
        _check_shape_match(self.data.shape, loaded_weight.shape)
        self.data.copy_(loaded_weight)

    # 合并列并行（例如 gate_up_proj）：checkpoint 中 gate、up 是两个张量，参数是一个。
    #   shard_offset / shard_size：这一段（gate 或 up）在本 rank 参数中的偏移和大小，由线性层算好传入
    #   先在参数上取出 [shard_offset, shard_offset + shard_size) 这一段，
    #   再从 checkpoint 张量上取第 tp_rank 个 shard_size 大小的块写进去。
    #   例：intermediate=12288、TP=2，up（第 2 段）在本 rank 参数中的偏移是 6144、大小 6144，
    #       rank 1 从 checkpoint 的 up_proj 中取 [6144, 12288) 行。
    #   打包存储且打包维就是输出维时，偏移和大小先换算成打包后的单位。
    def load_merged_column_weight(
        self, loaded_weight: torch.Tensor, tp_rank: int, **kwargs
    ):

        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        use_presharded_weights = kwargs.get("use_presharded_weights")
        if (
            isinstance(self, (PackedColumnParameter, PackedWeightParameter))
            and self.packed_dim == self.output_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        if not use_presharded_weights:
            loaded_weight = loaded_weight.narrow(
                self.output_dim, tp_rank * shard_size, shard_size
            )
        _check_shape_match(param_data.shape, loaded_weight.shape)
        param_data.copy_(loaded_weight)

    # QKV 合并层：与上面类似，区别是 k、v 段在 checkpoint 中取第几块。
    #   q：取第 tp_rank 块
    #   k / v：取第 tp_rank // num_heads 块。这里的 num_heads 实际传入的是每个 KV 头的复制份数
    #     （QKVParallelLinear.num_kv_head_replicas）。KV 头不少于 TP 时它为 1，等价于第 tp_rank 块；
    #     KV 头少于 TP 时多个 rank 共用同一个 KV 头。
    #   例：8 个 KV 头、TP=16，复制份数为 2：rank 0、1 取第 0 个 KV 头，rank 2、3 取第 1 个……
    def load_qkv_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
        **kwargs,
    ):

        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        shard_id = kwargs.get("shard_id")
        num_heads = kwargs.get("num_heads")

        if (
            isinstance(self, (PackedColumnParameter, PackedWeightParameter))
            and self.output_dim == self.packed_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data
        # 注意 shard_id 在这里被复用：从 "q"/"k"/"v" 改成了要取 checkpoint 的第几块
        shard_id = tp_rank if shard_id == "q" else tp_rank // num_heads
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        if not use_presharded_weights:
            loaded_weight = loaded_weight.narrow(
                self.output_dim, shard_id * shard_size, shard_size
            )

        _check_shape_match(param_data.shape, loaded_weight.shape)
        param_data.copy_(loaded_weight)


# 行并行参数：在输入维上取第 tp_rank 段。input_dim：输入维是第几维（对 [out, in] 的权重是 1）。
# 标量（0 维）的 checkpoint 张量先变成 1 维，以便与形状为 [1] 的参数比较。
class RowParallelWeightParameter(BaseWeightParameter):
    """Parameter class with row-parallel weight-loading support."""

    # 译：支持行并行权重加载的参数类。

    def __init__(self, input_dim: int, **kwargs):
        self._input_dim = input_dim
        super().__init__(**kwargs)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def load_row_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ):
        if not use_presharded_weights:
            shard_size = self.data.shape[self.input_dim]
            loaded_weight = loaded_weight.narrow(
                self.input_dim, tp_rank * shard_size, shard_size
            )

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        _check_shape_match(self.data.shape, loaded_weight.shape)
        self.data.copy_(loaded_weight)


# 下面几个类只是组合上面的能力，按用途起名，便于量化方法在创建参数时表达意图：
#   ModelWeightParameter：线性层权重，列并行、行并行都支持（取决于所在的层）
class ModelWeightParameter(_ColumnParallelWeightParameter, RowParallelWeightParameter):
    """
    Parameter class for linear layer weights. Uses both column and
    row parallelism.
    """

    # 译：线性层权重使用的参数类。同时支持列并行和行并行。

    pass


# GroupQuantScaleParameter：分组量化（例如每 128 个输入一组）的 scale，两个方向都可能切
class GroupQuantScaleParameter(
    _ColumnParallelWeightParameter, RowParallelWeightParameter
):
    """
    Parameter class for weight scales loaded for weights with
    grouped quantization. Uses both column and row parallelism.
    """

    # 译：分组量化权重的 scale 所用的参数类。同时支持列并行和行并行。

    pass


# ChannelQuantScaleParameter：逐通道量化的 scale，每个输出通道一个，只随输出维切分
class ChannelQuantScaleParameter(_ColumnParallelWeightParameter):
    """
    Parameter class for weight scales loaded for weights with
    channel-wise quantization. Equivalent to _ColumnParallelWeightParameter.
    """

    # 译：逐通道量化权重的 scale 所用的参数类。等同于 _ColumnParallelWeightParameter。

    pass


# PerTensorScaleParameter：每个逻辑矩阵一个 scale（per-tensor 量化）。
#   QKV 合并层有 3 个 scale（q、k、v 各一个），gate_up 合并层有 2 个；参数形状为 [逻辑矩阵数]。
#   加载时不切分，只按 shard_id 写到对应下标：q/k/v -> 0/1/2，gate/up -> 0/1。
#   列并行、行并行的加载方法都退化为整体拷贝（scale 不随 TP 切分）。
class PerTensorScaleParameter(BaseWeightParameter):
    """
    Parameter class for scales where the number of scales is
    equivalent to the number of logical matrices in fused linear
    layers (e.g. for QKV, there are 3 scales loaded from disk).
    This is relevant to weights with per-tensor quantization.
    Adds functionality to map the scalers to a shard during
    weight loading.

    Note: additional parameter manipulation may be handled
    for each quantization config specifically, within
    process_weights_after_loading
    """

    # 译：用于 scale 的参数类，scale 的个数等于合并线性层中逻辑矩阵的个数（例如 QKV 会从磁盘加载 3 个 scale）。
    #     适用于 per-tensor 量化的权重。增加了在加载权重时把 scale 映射到对应分片的功能。
    #
    #     注意：各量化配置可能在 process_weights_after_loading 中对参数做额外处理。

    def __init__(self, **kwargs):
        self.qkv_idxs = {"q": 0, "k": 1, "v": 2}
        super().__init__(**kwargs)

    def _shard_id_as_int(self, shard_id: str | int) -> int:
        if isinstance(shard_id, int):
            return shard_id

        # if not int, assume shard_id for qkv
        # map to int and return
        # 译：不是 int 时，假定它是 qkv 的 shard_id，映射成 int 后返回
        if not isinstance(shard_id, str):
            raise TypeError(
                f"shard_id must be a str or int, got {type(shard_id).__name__}."
            )
        if shard_id not in self.qkv_idxs:
            raise ValueError(f"Invalid qkv shard_id: {shard_id}.")
        return self.qkv_idxs[shard_id]

    # For row parallel layers, no sharding needed
    # load weight into parameter as is
    # 译：行并行层不需要切分，按原样把权重加载进参数
    def load_row_parallel_weight(self, *args, **kwargs):
        kwargs.pop("tp_rank", None)
        kwargs.pop("use_presharded_weights", None)
        super().load_row_parallel_weight(*args, **kwargs)

    def load_merged_column_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    def load_qkv_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    # 注意：列并行也调用父类的 load_row_parallel_weight，即同样不切分、整体拷贝
    def load_column_parallel_weight(self, *args, **kwargs):
        kwargs.pop("tp_rank", None)
        kwargs.pop("use_presharded_weights", None)
        super().load_row_parallel_weight(*args, **kwargs)

    def _load_into_shard_id(
        self, loaded_weight: torch.Tensor, shard_id: str | int, **kwargs
    ):
        """
        Slice the parameter data based on the shard id for
        loading.
        """
        # 译：根据 shard id 切出参数数据中的对应位置，用于加载。

        param_data = self.data
        shard_id = self._shard_id_as_int(shard_id)

        # AutoFP8 scales do not have a shape
        # compressed-tensors scales do have a shape
        # 译：AutoFP8 的 scale 没有形状（0 维）
        #     compressed-tensors 的 scale 有形状
        if len(loaded_weight.shape) != 0:
            if loaded_weight.shape[0] != 1:
                raise ValueError(
                    f"Expected scale shard with first dimension 1, got {loaded_weight.shape}."
                )
            loaded_weight = loaded_weight[0]

        param_data = param_data[shard_id]
        _check_shape_match(param_data.shape, loaded_weight.shape)
        param_data.copy_(loaded_weight)


# 打包存储的参数：多个低比特数值打包进一个更宽的整数（例如 8 个 int4 打包进 int32，packed_factor=8），
# packed_dim 是被打包的那一维。按逻辑元素计算的偏移和大小要除以 packed_factor 才是存储上的偏移。
# marlin_tile_size：Marlin kernel 的权重以 tile 为单位重新排列，偏移和大小还要再乘以 tile 大小。
# PackedColumnParameter 只支持列并行；PackedWeightParameter（下面）两个方向都支持。
class PackedColumnParameter(_ColumnParallelWeightParameter):
    """
    Parameter for model parameters which are packed on disk
    and support column parallelism only. See PackedWeightParameter
    for more details on the packed properties.
    """

    # 译：用于在磁盘上打包存储、且只支持列并行的模型参数。打包相关的属性详见 PackedWeightParameter。

    def __init__(
        self,
        packed_factor: int | Fraction,
        packed_dim: int,
        marlin_tile_size: int | None = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


class PackedWeightParameter(ModelWeightParameter):
    """
    Parameter for model weights which are packed on disk.
    Example: GPTQ Marlin weights are int4 or int8, packed into int32.
    Extends the ModelWeightParameter to take in the
    packed factor, the packed dimension, and optionally, marlin
    tile size for marlin kernels. Adjusts the shard_size and
    shard_offset for fused linear layers model weight loading
    by accounting for packing and optionally, marlin tile size.
    """

    # 译：用于在磁盘上打包存储的模型权重。
    #     例如：GPTQ Marlin 权重是 int4 或 int8，打包进 int32。
    #     在 ModelWeightParameter 的基础上增加了打包因子、打包维度，以及可选的 Marlin tile 大小（供 Marlin kernel 使用）。
    #     在合并线性层加载模型权重时，会考虑打包以及（可选的）Marlin tile 大小来调整 shard_size 和 shard_offset。

    def __init__(
        self,
        packed_factor: int | Fraction,
        packed_dim: int,
        marlin_tile_size: int | None = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


# 原地调整参数的维度顺序，使 input_dim、output_dim（以及 packed_dim）位于指定位置。
# 量化 kernel 往往要求固定的权重布局，加载后用它统一成已知布局。
# 例：当前 [out, in]（output_dim=0, input_dim=1），要求 input_dim=0、output_dim=1 -> 转置成 [in, out]。
# 只做维度置换，不支持改变打包维（需要重新打包时报错）。
def permute_param_layout_(
    param: BaseWeightParameter, input_dim: int, output_dim: int, **kwargs
) -> BaseWeightParameter:
    """
    Permute a parameter's layout to the specified input and output dimensions,
    useful for forcing the parameter into a known layout, for example, if I need
    a packed (quantized) weight matrix to be in the layout
        {input_dim = 0, output_dim = 1, packed_dim = 0}
    then I can call:
        permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
    to ensure x is in the correct layout (permuting it to the correct layout if
    required, asserting if it cannot get it to the correct layout)
    """
    # 译：把参数的布局置换到指定的输入维和输出维，用于把参数强制变成已知的布局。例如，如果我需要一个
    #     打包（量化）的权重矩阵处于布局
    #         {input_dim = 0, output_dim = 1, packed_dim = 0}
    #     那么可以调用：
    #         permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
    #     来确保 x 处于正确的布局（需要时把它置换成正确的布局；无法得到正确布局时报错）。

    curr_input_dim = getattr(param, "input_dim", None)
    curr_output_dim = getattr(param, "output_dim", None)

    if curr_input_dim is None or curr_output_dim is None:
        if param.data.dim() != 2:
            raise ValueError(
                "permute_param_layout_ only supports 2D parameters when either "
                "input_dim or output_dim is not set"
            )

    # if one of the dimensions is not set, set it to the opposite of the other
    #  we can only do this since we asserted the parameter is 2D above
    # 译：如果有一个维度没有设置，就把它设成另一个维度的“另一维”
    #      之所以能这样做，是因为上面已经断言了参数是二维的
    if curr_input_dim is None:
        if curr_output_dim is None:
            raise ValueError("either input or output dim must be set")
        curr_input_dim = (curr_output_dim + 1) % 2
    if curr_output_dim is None:
        if curr_input_dim is None:
            raise ValueError("either input or output dim must be set")
        curr_output_dim = (curr_input_dim + 1) % 2

    # create permutation from the current layout to the layout with
    # self.input_dim at input_dim and self.output_dim at output_dim preserving
    # other dimensions
    # 译：构造置换：从当前布局变成 input_dim 位于 input_dim、output_dim 位于 output_dim 的布局，
    #     其他维度的相对顺序保持不变
    perm = [
        i for i in range(param.data.dim()) if i not in [curr_input_dim, curr_output_dim]
    ]
    perm.insert(input_dim, curr_input_dim)
    perm.insert(output_dim, curr_output_dim)

    if "packed_dim" in kwargs:
        if not (
            hasattr(param, "packed_dim")
            and param.packed_dim == perm[kwargs["packed_dim"]]
        ):
            raise ValueError(
                "permute_param_layout_ currently doesn't support repacking"
            )

    param.data = param.data.permute(*perm)
    if hasattr(param, "_input_dim"):
        param._input_dim = input_dim
    if hasattr(param, "_output_dim"):
        param._output_dim = output_dim
    if "packed_dim" in kwargs and hasattr(param, "_packed_dim"):
        param._packed_dim = kwargs["packed_dim"]

    return param


# 下面两个函数把“按逻辑元素计数”的偏移和大小换算成存储上的偏移和大小：
#   先除以打包因子；如果是 Marlin 布局，再乘以 tile 大小。
#   例：int4 打包进 int32（packed_factor=8），逻辑偏移 4096、大小 4096 -> 存储偏移 512、大小 512
def _adjust_shard_indexes_for_marlin(shard_size, shard_offset, marlin_tile_size):
    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def _adjust_shard_indexes_for_packing(
    shard_size, shard_offset, packed_factor, marlin_tile_size
):
    shard_size = shard_size // packed_factor
    shard_offset = shard_offset // packed_factor
    if marlin_tile_size is not None:
        return _adjust_shard_indexes_for_marlin(
            shard_size=shard_size,
            shard_offset=shard_offset,
            marlin_tile_size=marlin_tile_size,
        )
    return shard_size, shard_offset


# BlockQuantScaleParameter：分块量化（例如 128×128 的权重块共用一个 scale）的 scale，两个方向都可能切。
# DeepSeek 风格的 FP8 分块量化使用它。
class BlockQuantScaleParameter(
    _ColumnParallelWeightParameter, RowParallelWeightParameter
):
    """
    Parameter class for weight scales loaded for weights with
    block-wise quantization. Uses both column and row parallelism.
    """

    # 译：分块量化权重的 scale 所用的参数类。同时支持列并行和行并行。

    pass
