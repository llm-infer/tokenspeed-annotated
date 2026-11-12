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


# 线性层：并行切分方式 × 量化方式 × 权重加载。
#
# 你在这里：Qwen3Attention 的 qkv_proj / o_proj（models/qwen3.py）、GatedMLP 的 gate_up_proj /
#           down_proj（layers/mlp.py），以及几乎所有模型的投影层。
#
# 本文件的类：
#   LinearBase：基类，负责按量化配置选出 quant_method
#   ReplicatedLinear：不切分
#   ColumnParallelLinear：列并行（切输出维）
#     MergedColumnParallelLinear：多个输入相同的列并行层合并（gate_up_proj）
#     QKVParallelLinear：q、k、v 合并，按头切分（qkv_proj）
#   RowParallelLinear：行并行（切输入维，输出是部分和）
#   DPColumnParallelLinear / DPRowParallelLinear：attention DP 下的特殊投影（第 9 阶段）
# 列并行、行并行、QKV 与 gate/up 合并的切分方式都是熟悉的内容，下面一句带过。
#
# 本文件真正需要细看的是两件事：
#
# 1. 计算委托给 quant_method（量化方法）
#    线性层自己不写矩阵乘法。构造时 LinearBase 按量化配置选出一个“量化方法”对象，它负责：
#      create_weights：在层上注册权重参数（形状、dtype，以及量化需要的各种 scale）
#      apply：执行前向计算（例如 bf16 的 F.linear，或 FP8 GEMM）
#      process_weights_after_loading：加载完后把权重转换成 kernel 需要的布局
#    未量化（bf16，Qwen3-8B 主线）用 UnquantizedLinearMethod；各种量化方法见第 8 阶段。
#    所以同一个 QKVParallelLinear，换一个量化配置就变成 FP8 / NVFP4 的线性层，模型代码不用改。
#
# 2. 权重加载（weight_loader）
#    checkpoint 里的张量是完整的、而且常常是分开的（q_proj、k_proj、v_proj 是三个张量），
#    参数却是本 rank 的一份、而且是合并的（qkv_proj 一个）。每个线性层的 weight_loader 负责
#    “从完整张量中切出本 rank 的那份，写到合并参数中的正确位置”。需要处理的情况很多：
#      合并层的第几段（shard_id）、KV 头复制、量化打包（多个低比特数打包进一个整数）、
#      Marlin 布局、bitsandbytes、AQLM codebook、per-tensor scale、分块量化 scale、预先切好的 checkpoint……
#    有两套实现：weight_loader（v1，用于普通参数）和 weight_loader_v2（v2，用于 layers/parameter.py 中
#    “知道怎么加载自己”的参数类，目前 FP8 和 compressed-tensors 使用），见 WEIGHT_LOADER_V2_METHODS。
#
# 所有线性层的 forward 都返回 (output, output_bias)：skip_bias_add=True 时 bias 不加在输出上，
# 而是作为第二个返回值交给调用方（以便和后续的逐元素操作融合）；否则第二个值为 None。
#
# Qwen3-8B 的形状（bf16）：
#   qkv_proj：TP=1 时权重 [6144, 4096]（q 4096 + k 1024 + v 1024 行）；TP=2 时每个 rank [3072, 4096]
#   o_proj：TP=1 时权重 [4096, 4096]；TP=2 时每个 rank [4096, 2048]（切输入维）
from typing import TYPE_CHECKING

import torch
from torch.nn.parameter import Parameter

from tokenspeed.runtime.distributed.comm_backend import CommBackend, get_global_backend
from tokenspeed.runtime.distributed.comm_backend.projection import (
    ProjectionSpec,
    ProjectionWorkspace,
)
from tokenspeed.runtime.distributed.comm_ops import (
    acquire_projection_output,
    all_gather,
    all_reduce,
    prepare_projection_collectives,
    projection_all_gather,
    projection_all_to_all,
    projection_reduce_scatter,
    try_fused_dp_column_linear,
    try_fused_dp_row_linear,
)
from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
from tokenspeed.runtime.distributed.utils import divide, split_tensor_along_last_dim
from tokenspeed.runtime.execution.workspace import WorkspacePool
from tokenspeed.runtime.layers.dense import (
    Fp8LinearMethod,
    Mxfp4LinearMethod,
    Nvfp4LinearMethod,
    UnquantizedLinearMethod,
    W8A8Fp8LinearMethod,
)
from tokenspeed.runtime.layers.parameter import (
    BaseWeightParameter,
    BlockQuantScaleParameter,
    PackedColumnParameter,
    PackedWeightParameter,
    PerTensorScaleParameter,
    RowParallelWeightParameter,
)
from tokenspeed.runtime.layers.quantization import (
    Fp8Config,
    Fp8WeightOnlyConfig,
    ModelOptMixedConfig,
    Mxfp4Config,
    Nvfp4Config,
    W8A8Fp8Config,
)
from tokenspeed.runtime.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from tokenspeed.runtime.layers.quantization.utils import (
    should_exclude_quant_module,
    should_ignore_quant_layer,
)
from tokenspeed.runtime.utils import get_colorful_logger, set_weight_attrs

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext

# These methods create parameters implementing the V2 sharded-loading protocol.
# Subclasses must preserve that contract; other methods retain the legacy loader.
# 译：这些方法创建实现 V2 分片加载协议的参数。
#     子类必须保持这一约定；其他方法保留旧的加载器。
WEIGHT_LOADER_V2_METHODS = (Fp8LinearMethod, CompressedTensorsLinearMethod)


# 下面三个函数调整权重加载时的偏移和大小：
#   adjust_marlin_shard：Marlin kernel 的权重按 tile 重排，偏移和大小要乘以 tile 大小
#   adjust_bitsandbytes_4bit_shard：bitsandbytes 4bit 把整个矩阵量化后存成一维，按比例换算偏移和大小
#   adjust_scalar_to_fused_array：合并层的 per-tensor scale 参数是长度为 N 的数组（每个逻辑矩阵一个），
#     取出第 shard_id 个元素作为加载目标
def adjust_marlin_shard(param, shard_size, shard_offset):
    marlin_tile_size = getattr(param, "marlin_tile_size", None)
    if marlin_tile_size is None:
        return shard_size, shard_offset

    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def adjust_bitsandbytes_4bit_shard(
    param: Parameter, qkv_offsets: dict[str, tuple[int, int]], loaded_shard_id: str
) -> tuple[int, int]:
    """Adjust the quantization offsets and sizes for BitsAndBytes sharding."""
    # 译：为 BitsAndBytes 的切分调整量化后的偏移和大小。

    total, _ = qkv_offsets["total"]
    orig_offset, orig_size = qkv_offsets[loaded_shard_id]

    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total

    return quantized_size, quantized_offset


def adjust_scalar_to_fused_array(param, loaded_weight, shard_id):
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    """
    # 译：对合并的模块（QKV 和 MLP），我们有一个长度为 N 的数组，每个“逻辑”矩阵对应一个 scale。所以参数是
    #     一个长度为 N 的数组。loaded_weight 对应磁盘上的某一个分片。这里我们根据 shard_id 切出参数中的对应位置用于加载。
    qkv_idxs = {"q": 0, "k": 1, "v": 2}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    # 译：AutoFP8 的 scale 没有形状
    #     compressed-tensors 的 scale 有形状
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    return param[shard_id], loaded_weight


# 所有线性层的基类。主要工作是在构造时选出 quant_method（量化方法）：
#   没有量化配置、或本层在量化配置的忽略列表中 -> UnquantizedLinearMethod（bf16 GEMM）
#   否则按配置类型选择对应的量化方法；各配置还有自己的排除规则（排除的层退回未量化）
# 这里只选方法，不创建权重；权重由子类在构造末尾调用 quant_method.create_weights 创建，
# 因为只有子类知道本 rank 的输入、输出大小。
class LinearBase(torch.nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        override_kernel_name: Optional kernel name passed down to the
            quant method's underlying ``tokenspeed_kernel.ops.gemm.mm`` dispatch
            (e.g. ``"cublaslt_mm_nvfp4"``). Lets the model force a
            specific kernel for a particular layer.
        interleave_linear_and_gate: If true, quantized post-load processing
            prepares a 64-row linear/gate interleaved weight view for
            fused GEMM+SwiGLU kernels.
    """

    # 译：线性层基类。
    #
    #     参数：
    #         input_size：线性层的输入维度。
    #         output_size：线性层的输出维度。
    #         bias：为真时加 bias。
    #         skip_bias_add：为真时不加 bias，而是把它返回。
    #         params_dtype：参数的数据类型。
    #         quant_config：量化配置。
    #         override_kernel_name：可选的 kernel 名字，传给量化方法底层的 ``tokenspeed_kernel.ops.gemm.mm`` 分派
    #             （例如 ``"cublaslt_mm_nvfp4"``）。让模型可以为某个特定的层强制使用某个 kernel。
    #         interleave_linear_and_gate：为真时，量化的加载后处理会为融合的 GEMM+SwiGLU kernel 准备一个
    #             linear/gate 按 64 行交错的权重视图。

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        override_kernel_name: str | None = None,
        interleave_linear_and_gate: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        # 译：保存输入的参数
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.prefix = prefix
        self.override_kernel_name = override_kernel_name
        self.interleave_linear_and_gate = interleave_linear_and_gate

        self.quant_config = quant_config
        if quant_config is None or should_ignore_quant_layer(
            prefix=prefix, ignored_layers=quant_config.ignored_layers
        ):
            self.quant_method: QuantizeMethodBase | None = UnquantizedLinearMethod()
        elif isinstance(quant_config, Nvfp4Config):
            # For NVFP4, excluded layers use unquantized (bf16)
            # 译：对 NVFP4，被排除的层使用未量化（bf16）
            if should_exclude_quant_module(prefix, quant_config.exclude_modules):
                self.quant_method = UnquantizedLinearMethod()
            else:
                self.quant_method = Nvfp4LinearMethod(quant_config)
        elif isinstance(quant_config, Mxfp4Config):
            fp8_route = quant_config.fp8_override_route(prefix)
            if fp8_route == "w8a8":
                self.quant_method = W8A8Fp8LinearMethod(quant_config.fp8_config)
            elif fp8_route == "dequant":
                # Raw-consumed weight, dequantized to BF16 during load.
                # 译：原样使用的权重，在加载时反量化为 BF16。
                self.quant_method = UnquantizedLinearMethod()
            elif quant_config.use_dynamic_mxfp4_activations:
                self.quant_method = Mxfp4LinearMethod(quant_config)
            else:
                # Existing MXFP4 support applies to MoE weights; dense weights
                # remain unquantized unless the checkpoint stores dense MXFP4.
                # 译：现有的 MXFP4 支持针对 MoE 权重；稠密权重保持未量化，除非 checkpoint 存储的是稠密 MXFP4。
                self.quant_method = UnquantizedLinearMethod()
        elif isinstance(quant_config, CompressedTensorsConfig):
            # get_quant_method resolves the scheme (setting layer.scheme),
            # honors the checkpoint's ignored layers, and refuses unsupported
            # combinations such as a quantized MLA kv_b_proj.
            # 译：get_quant_method 解析量化方案（设置 layer.scheme），遵循 checkpoint 中的忽略层，并拒绝不支持的组合，
            #     例如量化的 MLA kv_b_proj。
            self.quant_method = quant_config.get_quant_method(self, prefix)
        elif isinstance(quant_config, ModelOptMixedConfig):
            self.quant_method = quant_config.get_quant_method(self, prefix)
        elif isinstance(quant_config, Fp8WeightOnlyConfig):
            self.quant_method = quant_config.get_quant_method(self, prefix)
        else:
            if isinstance(quant_config, Fp8Config):
                if should_exclude_quant_module(prefix, quant_config.ignored_layers):
                    self.quant_method = UnquantizedLinearMethod()
                else:
                    self.quant_method = quant_config.get_quant_method(self, prefix)
            if isinstance(quant_config, W8A8Fp8Config):
                self.quant_method = W8A8Fp8LinearMethod(quant_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# 不切分的线性层：每个 rank 持有完整权重 [output_size, input_size]，输出就是完整结果，不需要通信。
# 用于不适合切分的小投影，以及 attention DP 下不切分的 lm_head（见 BaseCausalLM.resolve_lm_head）。
class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    # 译：复制的线性层。
    #
    #     参数：
    #         input_size：线性层的输入维度。
    #         output_size：线性层的输出维度。
    #         bias：为真时加 bias。
    #         skip_bias_add：为真时不加 bias，而是把它返回。
    #         params_dtype：参数的数据类型。
    #         quant_config：量化配置。
    #         prefix：该层在 state dict 中的名字，包括所有父模块（例如 model.layers.0.qkv_proj）

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
        )

        # All the linear layer supports quant method.
        # 译：所有线性层都支持量化方法。
        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    # 不切分时的加载：整体拷贝。另外支持两种“只写入一段”的方式（供把多个 checkpoint 张量合并到
    # 一个复制参数里的模型使用）：begin_size 指定起始行；shard_id 按等长分段写入第 shard_id 段。
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        shard_id=None,
        begin_size=None,
    ):
        # If the weight on disk does not have a shape, give it one
        # (such scales for AutoFp8).
        # 译：如果磁盘上的权重没有形状，就给它一个形状（例如 AutoFp8 的 scale）。
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        if begin_size is not None:
            shard_size = loaded_weight.shape[0]
            param[begin_size : begin_size + shard_size].data.copy_(loaded_weight)
        elif shard_id is not None:
            shard_size = loaded_weight.shape[0]
            param[shard_id * shard_size : (shard_id + 1) * shard_size].data.copy_(
                loaded_weight
            )
        else:
            assert param.size() == loaded_weight.size()
            param.data.copy_(loaded_weight)

    # block_scale / output_dtype：输入已经是 FP8 分块量化的数据时，由调用方传入其 scale 和期望的输出 dtype
    # （通信融合 kernel 顺带完成了量化，见 layernorm.py）。ctx 只是为了与 DP 线性层的调用接口一致。
    def forward(
        self,
        x: torch.Tensor,
        block_scale=None,
        output_dtype=None,
        *,
        ctx: "ForwardContext | None" = None,
    ) -> torch.Tensor:
        # ctx keeps the DP Linear call interface; replicated GEMM needs no owner counts.
        # 译：ctx 是为了保持 DP Linear 的调用接口；复制的 GEMM 不需要各 rank 的行数。
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        if block_scale is not None:
            # Note: block_scale is not None means flashinfer reduce-scatter fusion is used for fp8 block quant
            # in this case, the input_ is already quantized to a fp8 tensor
            # 译：注意：block_scale 不为 None 表示使用了 flashinfer 的 reduce-scatter 融合做 fp8 分块量化，
            #     此时 input_ 已经被量化为 fp8 张量
            output = self.quant_method.apply(self, x, bias, block_scale, output_dtype)
        else:
            output = self.quant_method.apply(self, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    # 先做激活、再做本线性层（GatedMLP 的 down_proj 用它）。量化方法实现了 apply_with_activation 时
    # 可以把激活与输入量化融合；否则就是 apply(activation(x))。
    def forward_with_activation(
        self, input_: torch.Tensor, activation: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply ``activation`` before this linear, fusing when prepared."""
        # 译：在本线性层之前应用 ``activation``，已准备好融合时就融合。
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        apply_with_activation = getattr(
            self.quant_method, "apply_with_activation", None
        )
        output = (
            apply_with_activation(self, input_, activation, bias=bias)
            if apply_with_activation is not None
            else self.quant_method.apply(self, activation(input_), bias=bias)
        )
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


# 列并行：权重 [output_size, input_size] 按输出维切成 tp_size 份，本 rank 持有
# [output_size / tp_size, input_size]，输出是完整输出的第 tp_rank 段，通常紧接一个行并行层，中间不通信。
# gather_output=True 时在末尾 all-gather 出完整输出（很少用）。
# output_sizes：合并层（QKV、gate_up）中每个逻辑矩阵的输出大小，每个都分别按 TP 切分。
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    # 译：列并行的线性层。
    #
    #     线性层定义为 Y = XA + b。A 沿第二维并行化为 A = [A_1, ..., A_p]。
    #
    #     参数：
    #         input_size：矩阵 A 的第一维。
    #         output_size：矩阵 A 的第二维。
    #         bias：为真时加 bias。
    #         gather_output：为真时对输出调用 all-gather，使 Y 在所有 GPU 上可用；否则每个 GPU 只有自己的输出 Y_i = XA_i
    #         skip_bias_add：为了实现把 bias 与其他逐元素操作融合的性能优化而加入。我们不加 bias，而是把它返回。
    #         params_dtype：参数的数据类型。
    #         quant_config：量化配置。
    #         output_sizes：打包进一个输出的各个输出大小的列表，例如对 QKV 这个列表的长度为 3。
    #         prefix：该层在 state dict 中的名字，包括所有父模块（例如 model.layers.0.qkv_proj）

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        output_sizes: list[int] | None = None,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
        use_presharded_weights: bool = False,
        override_kernel_name: str | None = None,
        interleave_linear_and_gate: bool = False,
    ):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            override_kernel_name,
            interleave_linear_and_gate,
        )
        self.gather_output = gather_output
        self.use_presharded_weights = use_presharded_weights
        assert self.quant_method is not None

        # 不传 TP 参数表示不切分；tp_size > 1 时必须给出通信组
        if tp_rank is None:
            assert tp_size is None
            assert tp_group is None
            tp_rank, tp_size = 0, 1
        assert 0 <= tp_rank < tp_size
        assert tp_size == 1 or tp_group is not None
        self.tp_rank, self.tp_size, self.tp_group = tp_rank, tp_size, tp_group

        self.output_size_per_partition = divide(self.output_size, self.tp_size)
        if output_sizes is None:
            self.output_sizes = [self.output_size]
            self.output_partition_sizes = [self.output_size_per_partition]
        else:
            self.output_sizes = output_sizes
            # If QKV or MergedColumn, use output size of each partition.
            # 译：如果是 QKV 或 MergedColumn，使用每个分区的输出大小。
            self.output_partition_sizes = [
                divide(output_size, self.tp_size) for output_size in self.output_sizes
            ]

        # 让量化方法创建权重：输入维不切（input_size_per_partition = input_size），输出维是各逻辑矩阵切分后的大小。
        # v2 量化方法创建的参数用 weight_loader_v2 加载，其余用 weight_loader。
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if isinstance(self.quant_method, WEIGHT_LOADER_V2_METHODS)
                else self.weight_loader
            ),
        )
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    # 列并行的 v1 加载：在输出维上取第 tp_rank 段。
    # 例：Qwen3-8B 的 o_proj 不是列并行；以 TP=2 时某个输出维 8192 的列并行层为例，rank 1 取 [4096, 8192) 行。
    # 没有 output_dim 属性的参数（如 per-tensor scale）不切分。
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)

        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        # 译：bitsandbytes 加载的就是特定的那一部分权重
        #     这里不需要再切
        if output_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            if not self.use_presharded_weights:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        # 译：从磁盘加载 scale 的特殊情况：它们往往没有形状（例如 AutoFP8 的情况）。
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: Parameter, loaded_weight: torch.Tensor):
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        # 译：从磁盘加载 scale 的特殊情况：它们往往没有形状（例如 AutoFP8 的情况）。
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        from tokenspeed.runtime.layers.parameter import _ColumnParallelWeightParameter

        if isinstance(param, _ColumnParallelWeightParameter):
            param.load_column_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:
            # Some DeepSeek V3 AWQ checkpoints still reach the generic column
            # loader path instead of the _ColumnParallelWeightParameter specialization.
            # 译：一些 DeepSeek V3 AWQ checkpoint 仍然会走到通用的列加载路径，而不是 _ColumnParallelWeightParameter 的特化版本。
            param.load_column_parallel_weight(loaded_weight)

    def forward(
        self,
        input_,
        block_scale=None,
        output_dtype=None,
        *,
        ctx: "ForwardContext | None" = None,
    ):
        # ctx keeps the DP Linear call interface; ordinary TP needs no owner counts.
        # 译：ctx 是为了保持 DP Linear 的调用接口；普通 TP 不需要各 rank 的行数。
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        # 译：矩阵乘法。
        assert self.quant_method is not None
        if block_scale is not None:
            # Note: block_scale is not None means flashinfer all-reduce fusion is used for fp8 block quant
            # in this case, the input_ is already quantized to a fp8 tensor
            # 译：注意：block_scale 不为 None 表示使用了 flashinfer 的 all-reduce 融合做 fp8 分块量化，
            #     此时 input_ 已经被量化为 fp8 张量
            output_parallel = self.quant_method.apply(
                self, input_, bias, block_scale, output_dtype
            )
        else:
            output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output and self.tp_size > 1:
            # All-gather across the partitions.
            # 译：在各分区之间做 all-gather。
            output = all_gather(output_parallel, self.tp_group, dim=-1)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


# 合并的列并行层（gate_up_proj）：多个输入相同的列并行层合并成一个，一次 GEMM 算完。
# 每个逻辑矩阵分别按 TP 切分后再在本地拼接：本 rank 的参数是 [gate 的第 tp_rank 段 | up 的第 tp_rank 段]。
# 例：Qwen3-8B，intermediate=12288
#   TP=1：参数 [24576, 4096] = [gate 12288 行 | up 12288 行]
#   TP=2：每个 rank [12288, 4096] = [gate 的 6144 行 | up 的 6144 行]
class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    # 译：带列并行的打包线性层。
    #
    #     与 ColumnParallelLinear 类似，但权重矩阵沿输出维拼接。加载权重矩阵时，各个分区分别切分。
    #
    #     参数：
    #         input_size：线性层的输入维度。
    #         output_sizes：线性层的各输出维度的列表。
    #         bias：为真时加 bias。
    #         gather_output：为真时对输出调用 all-gather，使输出在所有 GPU 上可用；否则每个 GPU 只有自己的输出。
    #         skip_bias_add：为了实现把 bias 与其他逐元素操作融合的性能优化而加入。我们不加 bias，而是把它返回。
    #         params_dtype：参数的数据类型。
    #         quant_config：量化配置。
    #         prefix：该层在 state dict 中的名字，包括所有父模块（例如 model.layers.0.qkv_proj）

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
        use_presharded_weights: bool = False,
        override_kernel_name: str | None = None,
        interleave_linear_and_gate: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            output_sizes=output_sizes,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            use_presharded_weights=use_presharded_weights,
            override_kernel_name=override_kernel_name,
            interleave_linear_and_gate=interleave_linear_and_gate,
        )

    # 合并列并行层的 v1 加载。loaded_shard_id 是这个 checkpoint 张量对应第几个逻辑矩阵（gate=0，up=1）。
    #   loaded_shard_id 为 None：checkpoint 里已经是合并好的张量（例如 Phi-3 的 gate_up_proj），
    #     先按 output_sizes 拆成各段，再对每段递归调用本函数
    #   正常情况：参数中这一段的偏移 = 前面各段大小之和 / tp_size，大小 = 本段大小 / tp_size；
    #     从 checkpoint 张量中取第 tp_rank 个这样大小的块写进去
    #   例（TP=2，up_proj，shard_id=1）：参数偏移 12288/2 = 6144、大小 6144；
    #     从 up_proj [12288, 4096] 中取 [6144, 12288) 行，写入参数的 [6144, 12288) 行
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        # 译：AQLM codebook 的特殊情况。
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        # 译：per-tensor scale 的特殊情况：把标量加载进合并的数组。
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            # 译：加载的权重在磁盘上已经是合并的（qkv/mlp）。
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                # 译：量化的特殊情况。
                #     如果是量化的，需要调整偏移和大小以考虑打包。
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    # 译：Marlin 的特殊情况。
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            # 译：量化的特殊情况。
            #     如果是量化的，需要调整偏移和大小以考虑打包。
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                # 译：Marlin 的特殊情况。
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = self.tp_rank * shard_size
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            # 译：bitsandbytes 加载的就是特定的那一部分权重
            #     这里不需要再切
            if not use_bitsandbytes_4bit and not self.use_presharded_weights:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        # Special case for AQLM codebooks.
        # 译：AQLM codebook 的特殊情况。
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            # 译：metadata 表示沿第 0 维拼接的固定大小
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # Special case for per-tensor scales in fused case.
        # 译：合并情况下 per-tensor scale 的特殊情况。
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(
        self, param: BaseWeightParameter, loaded_weight: torch.Tensor
    ):
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        # 译：处理 MLP 层在磁盘上已经合并的模型这一特殊情况。此时没有 shard id。本函数通过拆分这些层确定 shard id，
        #     然后用 shard id 调用权重加载器。
        #
        #     具有这种合并层的模型示例：
        #     https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

        current_shard_offset = 0
        shard_offsets: list[tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            # 译：量化的特殊情况。
            #     如果是量化的，需要调整偏移和大小以考虑打包。
            if (
                isinstance(param, (PackedColumnParameter, PackedWeightParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )
            # Special case for block-wise quantization scales.
            # The scale tensor is smaller than the weight tensor by a factor
            # of block_n, so we need to adjust offset and size accordingly.
            # 译：分块量化 scale 的特殊情况。
            #     scale 张量比权重张量小 block_n 倍，所以需要相应调整偏移和大小。
            elif isinstance(param, BlockQuantScaleParameter):
                weight_block_size = self.quant_method.quant_config.weight_block_size
                block_n = weight_block_size[0]
                shard_offset = (shard_offset + block_n - 1) // block_n
                shard_size = (shard_size + block_n - 1) // block_n

            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    # 合并列并行层的 v2 加载：算好这一段的偏移和大小，交给参数自己的 load_merged_column_weight。
    # 分块量化的 scale 每 block_n 行权重才有一行 scale，偏移和大小要先除以 block_n（向上取整）。
    def weight_loader_v2(
        self,
        param: BaseWeightParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ):
        if loaded_shard_id is None:
            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowParallelWeightParameter, BaseWeightParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id < len(self.output_sizes)

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = self.quant_method.quant_config.weight_block_size
            block_n, _ = weight_block_size[0], weight_block_size[1]
            shard_offset = (
                (sum(self.output_sizes[:loaded_shard_id]) + block_n - 1) // block_n
            ) // self.tp_size
            shard_size = (
                (self.output_sizes[loaded_shard_id] + block_n - 1)
                // block_n
                // self.tp_size
            )
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            tp_rank=self.tp_rank,
            shard_offset=shard_offset,
            shard_size=shard_size,
            use_presharded_weights=self.use_presharded_weights,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return None


# QKV 合并层：按注意力头切分。本 rank 持有 Hq 个 q 头和 Hkv 个 k、v 头（Hq = 总 Q 头 / TP）。
# KV 头不够分时（TP >= KV 头数）每个 rank 持有 1 个 KV 头，每个 KV 头复制到 num_kv_head_replicas 个 rank。
# 本 rank 参数的输出维布局：[q 头 | k 头 | v 头]。
# 例：Qwen3-8B（32 个 Q 头、8 个 KV 头、head_dim 128）
#   TP=1：[q 4096 | k 1024 | v 1024] = 6144 行
#   TP=2：[q 2048 | k 512 | v 512] = 3072 行（16 个 q 头、4 个 KV 头）
#   TP=16：[q 256 | k 128 | v 128]（2 个 q 头、1 个 KV 头，每个 KV 头在 2 个 rank 上各有一份）
class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    # 译：注意力中 QKV 变换的线性层。
    #
    #     用于注意力层中 query、key、value 向量线性变换的线性层。权重矩阵沿输出维拼接。该层沿头的维度并行化。
    #     当 key/value 头的数量少于 query 头的数量时（例如多查询 / 分组查询注意力），key/value 头可能被复制，
    #     而 query 头被切分。
    #
    #     参数：
    #         hidden_size：transformer 的输入 hidden state 大小。
    #         head_size：每个注意力头的大小。
    #         total_num_heads：注意力 query 头的总数。
    #         total_num_kv_heads：注意力 key/value 头的总数。为 None 时假定 total_num_kv_heads = total_num_heads。
    #         bias：为真时加 bias。
    #         skip_bias_add：为了实现把 bias 与其他逐元素操作融合的性能优化而加入。我们不加 bias，而是把它返回。
    #         params_dtype：参数的数据类型。
    #         quant_config：量化配置。
    #         prefix：该层在 state dict 中的名字，包括所有父模块（例如 model.layers.0.qkv_proj）

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
        load_presharded_attn: bool = False,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        tp_size = 1 if tp_size is None else tp_size
        self.num_heads = divide(self.total_num_heads, tp_size)

        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        # 注意 output_size、output_sizes 按“本 rank 头数 × tp_size”计算：KV 头被复制时，它们大于 checkpoint 中
        # k、v 的实际大小（例如 TP=16 时 k 段的总大小是 1*128*16 = 2048，而 checkpoint 的 k_proj 只有 1024 行）。
        # 父类再除以 tp_size 得到本 rank 的大小，正好是“1 个 KV 头”。
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        )
        output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            output_sizes=output_sizes,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            use_presharded_weights=load_presharded_attn,
        )

    # 本 rank 参数中 q、k、v 各段的偏移和大小（单位：行）。
    # 例（Qwen3-8B，TP=2）：q 偏移 0、大小 2048；k 偏移 2048、大小 512；v 偏移 2560、大小 512
    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + 2 * self.num_kv_heads) * self.head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(
        self, param: BaseWeightParameter, loaded_weight: torch.Tensor
    ):
        """
        Handle special case for models where QKV layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        # 译：处理 QKV 层在磁盘上已经合并的模型这一特殊情况。此时没有 shard id。本函数通过拆分这些层确定 shard id，
        #     然后用 shard id 调用权重加载器。
        #
        #     具有这种合并层的模型示例：
        #     https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            # 译：量化的特殊情况。
            #     如果是量化的，需要调整偏移和大小以考虑打包。
            if (
                isinstance(param, (PackedColumnParameter, PackedWeightParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            # 注意：use_presharded_weights 为 True 时下面的 if 不执行，loaded_weight_shard 没有被赋值，
            # 紧接着的调用会抛出 NameError；也就是说这条路径不支持预先切好的合并 QKV checkpoint。
            if not self.use_presharded_weights:
                loaded_weight_shard = loaded_weight.narrow(
                    param.output_dim, shard_offset, shard_size
                )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    # QKV 层的 v2 加载：算好 q/k/v 段的偏移和大小，交给参数自己的 load_qkv_weight；
    # num_heads 传入的是 KV 头的复制份数，参数据此决定 k、v 从 checkpoint 中取第几块（见 parameter.py）。
    def weight_loader_v2(
        self,
        param: BaseWeightParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        if loaded_shard_id is None:  # special case for certain models
            # 译：某些模型的特殊情况
            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowParallelWeightParameter, BaseWeightParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = self.quant_method.quant_config.weight_block_size
            block_n, _ = weight_block_size[0], weight_block_size[1]
            shard_offset = (shard_offset + block_n - 1) // block_n
            shard_size = (shard_size + block_n - 1) // block_n

        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
            use_presharded_weights=self.use_presharded_weights,
        )

    # QKV 层的 v1 加载（bf16 的 Qwen3 主线走这里）。loaded_shard_id 为 "q"/"k"/"v"：
    #   1. 取出参数中这一段：q 段 [0, Hq*D)，k 段 [Hq*D, (Hq+Hkv)*D)，v 段 [(Hq+Hkv)*D, (Hq+2*Hkv)*D)
    #   2. 决定从 checkpoint 张量中取第几块：q 取第 tp_rank 块；k、v 取第 tp_rank // num_kv_head_replicas 块
    #   3. 拷贝
    # 例（Qwen3-8B，TP=2，rank 1，k_proj [1024, 4096]）：参数的 k 段是 [2048, 2560) 行（512 行），
    #   复制份数为 1，取 k_proj 的第 1 块 [512, 1024) 行写进去。
    # loaded_shard_id 为 None 表示 checkpoint 里已经是合并的 qkv（如 Phi-3），先拆开再逐段调用本函数。
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        # 译：AQLM codebook 的特殊情况。
        is_metadata = getattr(param, "is_metadata", False)

        # Special case for per-tensor scales in fused case.
        # 译：合并情况下 per-tensor scale 的特殊情况。
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            # 译：加载的权重在磁盘上已经是合并的（qkv/mlp）。
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantized Weights.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                # 译：量化权重的特殊情况。
                #     如果是量化的，需要调整偏移和大小以考虑打包。
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor

                    # Special case for Marlin.
                    # 译：Marlin 的特殊情况。
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_heads * self.head_size),
                        "k": (
                            self.total_num_heads * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "v": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "total": (
                            (self.total_num_heads + 2 * self.total_num_kv_heads)
                            * self.head_size,
                            0,
                        ),
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id
                    )

                if not self.use_presharded_weights:
                    loaded_weight_shard = loaded_weight.narrow(
                        output_dim, shard_offset, shard_size
                    )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        # 译：如果定义了输出维，就使用默认的加载流程。
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            # Special case for Quantized Weights.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            # 译：量化权重的特殊情况。
            #     如果是量化的，需要调整偏移和大小以考虑打包。
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

                # Special case for Marlin.
                # 译：Marlin 的特殊情况。
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bitsandbytes_4bit:
                orig_qkv_offsets = {
                    "q": (0, self.num_heads * self.head_size),
                    "k": (
                        self.num_heads * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "v": (
                        (self.num_heads + self.num_kv_heads) * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "total": (
                        (self.num_heads + 2 * self.num_kv_heads) * self.head_size,
                        0,
                    ),
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id
                )

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_id = self.tp_rank
            else:
                shard_id = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size

            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            # 译：bitsandbytes 加载的就是特定的那一部分权重
            #     这里不需要再切
            if not use_bitsandbytes_4bit and not self.use_presharded_weights:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for for AQLM codebooks.
        # 译：AQLM codebook 的特殊情况。
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            # 译：metadata 表示沿第 0 维拼接的固定大小
            shard_size = loaded_weight.shape[0]
            shard_index = ["q", "k", "v"].index(loaded_shard_id)
            param_data = param_data.narrow(0, shard_index * shard_size, shard_size)
        # Special case for per-tensor scales in fused case.
        # 译：合并情况下 per-tensor scale 的特殊情况。
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return None


# 行并行：权重按输入维切分，本 rank 持有 [output_size, input_size / tp_size]，输入也是切分后的那一段
# （通常是前一个列并行层的输出），输出是 [T, output_size] 的部分和。
# reduce_results=True 时在本层末尾 all-reduce 得到完整结果；解码层里的 o_proj、down_proj 都传 False，
# 求和交给 CommManager（见 distributed/comm_manager.py）。
# bias 只在 rank 0 上加，否则 all-reduce 后会被加 tp_size 次。
class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.

    Attributes:
        supports_fp8_group128_input: Accepts prequantized E4M3 activations with
            per-128 FP32 scales and produces BF16 output.
    """

    # 译：行并行的线性层。
    #
    #     线性层定义为 Y = XA + b。A 沿第一维并行化，X 沿第二维并行化：
    #         A = [A_1; ...; A_p]（按行分块，见上方原文图示），X = [X_1, ..., X_p]
    #     参数：
    #         input_size：矩阵 A 的第一维。
    #         output_size：矩阵 A 的第二维。
    #         bias：为真时加 bias。注意 bias 不做并行切分。
    #         input_is_parallel：为真时，假定输入已经在各 GPU 之间切分好，不再切分。
    #         skip_bias_add：为了实现把 bias 与其他逐元素操作融合的性能优化而加入。我们不加 bias，而是把它返回。
    #         params_dtype：参数的数据类型。
    #         quant_config：量化配置。
    #
    #     属性：
    #         supports_fp8_group128_input：接受预先量化好的 E4M3 激活（每 128 个一个 FP32 scale），并产生 BF16 输出。

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
        use_presharded_weights: bool = False,
        override_kernel_name: str | None = None,
        interleave_linear_and_gate: bool = False,
    ):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            override_kernel_name,
            interleave_linear_and_gate,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        assert self.quant_method is not None
        self.use_presharded_weights = use_presharded_weights

        if tp_rank is None:
            assert tp_size is None
            assert tp_group is None
            tp_rank, tp_size = 0, 1
        assert 0 <= tp_rank < tp_size
        assert tp_size == 1 or tp_group is not None
        self.tp_rank, self.tp_size, self.tp_group = tp_rank, tp_size, tp_group

        # supports_fp8_group128_input：本层能否直接接收“每 128 个数一个 FP32 scale”的 FP8 输入
        # （上游 kernel 已经把激活量化好时，可以省掉一次量化）。条件：bf16 模型、FP8 分块量化且块宽 128。
        method = self.quant_method
        self.supports_fp8_group128_input = (
            self.params_dtype == torch.bfloat16
            and (self.input_is_parallel or self.tp_size == 1)
            and isinstance(method, Fp8LinearMethod)
            and method.block_quant
            and method.quant_config.weight_block_size[1] == 128
            and method.quant_config.weight_scale_dtype == torch.float32
        )

        self.input_size_per_partition = divide(input_size, self.tp_size)

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if isinstance(self.quant_method, WEIGHT_LOADER_V2_METHODS)
                else self.weight_loader
            ),
        )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    # 行并行的 v1 加载：在输入维上取第 tp_rank 段。
    # 例（Qwen3-8B，TP=2，rank 1，o_proj [4096, 4096]）：取 [:, 2048:4096] 写入参数 [4096, 2048]
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        # 译：bitsandbytes 加载的就是特定的那一部分权重
        #     这里不需要再切
        if (
            input_dim is not None
            and not use_bitsandbytes_4bit
            and not self.use_presharded_weights
        ):
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        # 译：从磁盘加载 scale 的特殊情况：它们往往没有形状（例如 AutoFP8 的情况）。
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=} {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BaseWeightParameter, loaded_weight: torch.Tensor):

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        # 译：从磁盘加载 scale 的特殊情况：它们往往没有形状（例如 AutoFP8 的情况）。
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        if isinstance(param, RowParallelWeightParameter):
            # This `BaseWeightParameter` is defined in tokenspeed/runtime/layers/parameter.py,
            # It supports additional parameters like tp_rank and use_presharded_weights.
            # 译：这个 `BaseWeightParameter` 定义在 tokenspeed/runtime/layers/parameter.py 中，
            #     它支持 tp_rank 和 use_presharded_weights 等额外参数。
            param.load_row_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:
            # Generic parameters do not support extra loading options.
            # 译：通用参数不支持额外的加载选项。
            param.load_row_parallel_weight(loaded_weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return None

    # 前向：输入 [T, input_size / tp_size]（input_is_parallel=False 时先切出本 rank 的一段）
    # -> 本地 GEMM 得到部分和 [T, output_size] -> reduce_results 时 all-reduce。
    #   scale：输入是预量化的 FP8 时它的 scale（输出固定为 bf16）
    #   out：可选的输出缓冲区，结果直接写进去（避免一次额外拷贝）
    def forward(
        self,
        input_,
        scale=None,
        *,
        ctx: "ForwardContext | None" = None,
        out: torch.Tensor | None = None,
    ):
        # ctx keeps the DP Linear call interface; ordinary TP needs no owner counts.
        # 译：ctx 是为了保持 DP Linear 的调用接口；普通 TP 不需要各 rank 的行数。
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        # 译：矩阵乘法。
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        # 译：只在 rank 0 上把 bias 加法融合进 GEMM（这保证在 TP>1 的情况下 bias 不会被加不止一次）
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        if out is not None:
            output_parallel = self.quant_method.apply_into(
                self,
                input_parallel,
                bias_,
                scale,
                torch.bfloat16 if scale is not None else input_parallel.dtype,
                out,
            )
        elif scale is not None:
            output_parallel = self.quant_method.apply(
                self, input_parallel, bias_, scale, torch.bfloat16
            )
        else:
            output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = all_reduce(output_parallel, self.tp_group)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

    def forward_with_activation(
        self, input_: torch.Tensor, activation: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply ``activation`` before this linear, fusing when prepared."""
        # 译：在本线性层之前应用 ``activation``，已准备好融合时就融合。
        if not self.input_is_parallel:
            return self.forward(activation(input_))
        input_parallel = input_

        assert self.quant_method is not None
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        apply_with_activation = getattr(
            self.quant_method, "apply_with_activation", None
        )
        output_parallel = (
            apply_with_activation(self, input_parallel, activation, bias=bias_)
            if apply_with_activation is not None
            else self.quant_method.apply(self, activation(input_parallel), bias=bias_)
        )
        if self.reduce_results and self.tp_size > 1:
            output = all_reduce(output_parallel, self.tp_group)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s


# 以下是 attention DP 下的“DP 线性层”（第 9 阶段回访）。
# 场景：注意力按请求分到不同 DP 组，每个 rank 只持有自己请求的 token 行（各 rank 行数不同）；
# 而某个投影希望按 TP 切分权重以节省显存。于是在投影前后做“按 token 分”与“按权重分”之间的交换：
#   DPColumnParallelLinear：all-gather 各 rank 的行 -> 本 rank 的列切分 GEMM -> all-to-all 把结果还给各行的主人
#   DPRowParallelLinear：all-to-all 把行按输入通道切分发出 -> 行切分 GEMM -> reduce-scatter 回各行的主人
# 能融合时调用融合 kernel（try_fused_*），否则分步执行。各 rank 的行数从 ctx 的全局 token 表读取。
#
# _projection_rows：读出组内各 rank 的行数，检查输入的形状、dtype 与预先准备的工作区一致，
# 返回 (组内最大行数, 各 rank 行数)。
def _projection_rows(
    inputs: torch.Tensor,
    ctx: "ForwardContext",
    parallel: DenseLayerMapping,
    workspace: ProjectionWorkspace | None,
    input_scales: torch.Tensor | None,
) -> tuple[int, tuple[int, ...]]:
    """Read explicit physical owner counts, including graph padding, by global rank.

    Projection TP/DP sizes describe weight sharding, not attention row ownership;
    they cannot supply missing counts or imply replicated inputs.

    Returns:
        Maximum physical rows in the subgroup and owner counts in subgroup order.
    """
    # 译：读取显式给出的各 rank 物理行数（包含 graph 的补齐），按全局 rank 编号。
    #
    #     投影的 TP/DP 大小描述的是权重的切分，而不是注意力行的归属；它们无法补出缺失的计数，也不意味着输入是复制的。
    #
    #     返回：
    #         子组中的最大物理行数，以及按子组顺序排列的各 rank 行数。
    if workspace is None:
        raise RuntimeError("DP linear communication must be prepared before forward")
    counts = (
        ctx.collective_global_num_tokens
        if ctx.collective_global_num_tokens is not None
        else ctx.global_num_tokens
    )
    if counts is None:
        raise ValueError(
            "DP linear requires explicit physical owner counts in "
            "ForwardContext.collective_global_num_tokens or global_num_tokens; "
            "counts cannot be inferred from the projection TP/DP mapping"
        )
    spec = workspace.spec
    if (
        len(counts) != parallel.world_size
        or any(count < 0 for count in counts)
        or inputs.ndim != 2
        or inputs.shape != (counts[parallel.rank], spec.input_size)
        or inputs.dtype
        != (torch.float8_e4m3fn if input_scales is not None else spec.dtype)
        or inputs.device != spec.device
    ):
        raise ValueError("DP linear requires matching physical per-rank token counts")
    if input_scales is not None and (
        spec.dtype != torch.bfloat16
        or input_scales.shape != (inputs.shape[0], spec.input_size // 128)
        or input_scales.dtype != torch.float32
        or input_scales.device != inputs.device
    ):
        raise ValueError("Prequantized DP linear requires FP32 per-128 input scales")
    owner_rows = tuple(counts[rank] for rank in parallel.tp_group)
    rows = max(owner_rows)
    if rows > spec.max_tokens:
        raise ValueError("DP linear exceeds its prepared per-rank row capacity")
    return rows, owner_rows


class DPColumnParallelLinear(ColumnParallelLinear):
    """Column-sharded weights with the replicated Linear's local-row contract.

    Inputs contain all channels of locally owned tokens; ctx supplies explicit
    physical per-rank row counts. Outputs preserve those tokens and their order.
    The parent owns weight creation, sharded loading and quantization. Stored
    output_size includes checkpoint padding; logical_output_size is returned.
    Forward composes AG -> GEMM -> inverse A2A through comm_ops, without knowing
    which communication kernel implements those operations. Bias is unsupported.
    """

    # 译：列切分的权重，配合复制的 Linear 那样“本地行”的约定。
    #
    #     输入包含本地拥有的 token 的所有通道；ctx 提供显式的各 rank 物理行数。输出保留这些 token 及其顺序。
    #     父类负责权重的创建、分片加载和量化。存储的 output_size 包含 checkpoint 的补齐；返回的是 logical_output_size。
    #     前向通过 comm_ops 组合 AG -> GEMM -> 逆 A2A，不需要知道由哪个通信 kernel 实现这些操作。不支持 bias。

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        padded_output_size: int,
        parallel: DenseLayerMapping,
        params_dtype: torch.dtype,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        if parallel.tp_size <= 1 or not 0 < output_size <= padded_output_size:
            raise ValueError("DP column linear requires TP > 1 and valid output widths")
        super().__init__(
            input_size=input_size,
            output_size=padded_output_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            output_sizes=None,
            prefix=prefix,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
            tp_group=parallel.tp_group,
            use_presharded_weights=False,
            override_kernel_name=None,
            interleave_linear_and_gate=False,
        )
        self.parallel = parallel
        self.logical_output_size = output_size
        self.projection_workspace: ProjectionWorkspace | None = None
        self.comm_backend: CommBackend | None = None

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        ctx: "ForwardContext",
    ) -> tuple[torch.Tensor, None]:
        rows, owner_rows = _projection_rows(
            inputs, ctx, self.parallel, self.projection_workspace, None
        )
        if rows == 0:
            return inputs.new_empty((0, self.logical_output_size)), None
        output = inputs.new_empty((rows, self.output_size))
        weight_scales = (
            self.weight_scale_inv
            if isinstance(self.quant_method, Fp8LinearMethod)
            and self.quant_method.block_quant
            and tuple(self.quant_method.quant_config.weight_block_size) == (128, 128)
            else None
        )
        if try_fused_dp_column_linear(
            inputs,
            self.weight,
            weight_scales,
            owner_rows,
            output,
            self.projection_workspace,
            self.comm_backend,
        ):
            return output[: inputs.shape[0], : self.logical_output_size], None
        values, _ = projection_all_gather(
            inputs, rows, False, self.projection_workspace, self.comm_backend
        )
        local, _ = super().forward(values, block_scale=None, output_dtype=None)
        output, _ = projection_all_to_all(
            local,
            rows,
            True,
            False,
            output,
            self.projection_workspace,
            self.comm_backend,
        )
        return output[: inputs.shape[0], : self.logical_output_size], None


class DPRowParallelLinear(RowParallelLinear):
    """Row-sharded weights with complete outputs for the original local tokens.

    Inputs contain all channels of locally owned tokens; ctx supplies explicit
    physical per-rank row counts. Outputs preserve those tokens and their order.
    A2A -> GEMM -> RS lives here; backends own padding and communication scratch.
    Parent reduction is disabled to avoid reducing GEMM partials twice. All
    subgroup ranks participate, including empty local owners. Bias is unsupported.
    """

    # 译：行切分的权重，为原来的本地 token 产生完整的输出。
    #
    #     输入包含本地拥有的 token 的所有通道；ctx 提供显式的各 rank 物理行数。输出保留这些 token 及其顺序。
    #     A2A -> GEMM -> RS 的过程在这里；后端负责补齐和通信用的临时空间。禁用父类的归约，以免对 GEMM 的部分和归约两次。
    #     子组的所有 rank 都参与，包括本地没有行的 rank。不支持 bias。

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel: DenseLayerMapping,
        params_dtype: torch.dtype,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        if parallel.tp_size <= 1 or input_size % parallel.tp_size:
            raise ValueError("DP row linear requires TP > 1 dividing input channels")
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=params_dtype,
            reduce_results=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
            tp_group=parallel.tp_group,
            use_presharded_weights=False,
            override_kernel_name=None,
            interleave_linear_and_gate=False,
        )
        self.supports_fp8_group128_input = (
            self.supports_fp8_group128_input
            and self.input_size_per_partition % 128 == 0
        )
        self.parallel = parallel
        self.projection_workspace: ProjectionWorkspace | None = None
        self.comm_backend: CommBackend | None = None

    def forward(
        self,
        inputs: torch.Tensor,
        scale: torch.Tensor | None = None,
        *,
        ctx: "ForwardContext",
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Project locally owned rows, optionally into a caller-owned out tensor.

        scale supplies FP32 per-128 scales for prequantized E4M3 inputs, which
        produce BF16 outputs. Omit scale for unquantized activation rows.
        """
        # 译：投影本地拥有的行，可选地写入调用方提供的 out 张量。
        #
        #     scale 为预先量化的 E4M3 输入提供每 128 个一个的 FP32 scale，此时产生 BF16 输出。
        #     未量化的激活行不要传 scale。
        if scale is not None and not self.supports_fp8_group128_input:
            raise ValueError("This DP row linear does not support prequantized inputs")
        rows, owner_rows = _projection_rows(
            inputs, ctx, self.parallel, self.projection_workspace, scale
        )
        output_dtype = self.projection_workspace.spec.dtype
        if out is not None and (
            out.shape != (inputs.shape[0], self.output_size)
            or out.dtype != output_dtype
            or out.device != inputs.device
            or not out.is_contiguous()
        ):
            raise ValueError("DP row linear requires a matching contiguous output")
        if rows == 0:
            return (
                out
                if out is not None
                else inputs.new_empty((0, self.output_size), dtype=output_dtype)
            ), None
        weight_scales = (
            self.weight_scale_inv
            if isinstance(self.quant_method, Fp8LinearMethod)
            and self.quant_method.block_quant
            and tuple(self.quant_method.quant_config.weight_block_size) == (128, 128)
            else None
        )
        fused_output = (
            out
            if out is not None
            else inputs.new_empty(
                (inputs.shape[0], self.output_size), dtype=output_dtype
            )
        )
        if try_fused_dp_row_linear(
            inputs,
            scale,
            self.weight,
            weight_scales,
            owner_rows,
            fused_output,
            self.projection_workspace,
            self.comm_backend,
        ):
            return fused_output, None
        values, scales = projection_all_to_all(
            inputs,
            rows,
            False,
            False,
            None,
            self.projection_workspace,
            self.comm_backend,
            input_scales=scale,
        )
        destination = acquire_projection_output(
            rows, self.projection_workspace, self.comm_backend
        )
        partial, _ = super().forward(values, scale=scales, out=destination)
        output = projection_reduce_scatter(
            partial, rows, self.projection_workspace, self.comm_backend
        )
        output = output[: inputs.shape[0]]
        if out is not None:
            out.copy_(output)
            output = out
        return output, None


# 启动时（在 KV cache 显存规划之前）为模型中所有 DP 线性层准备通信工作区和通信组；
# release_dp_linear_communication 在关闭时释放。没有 DP 线性层的模型（如 Qwen3）直接返回 False。
def prepare_dp_linear_communication(
    model: torch.nn.Module,
    max_tokens: int,
    model_activation_dtype: torch.dtype,
    backend: CommBackend | None,
) -> bool:
    """Bind model-owned communication before cache profiling.

    Models only select DP Linear modules. This generic preparation discovers the
    modules, binds their backend and creates their groups. Sequential projections
    share one fixed scratch pool per device; matching specs also share native
    resources. Concurrent models or streams need separate preparations.
    Repeated preparation with the same backend and capacity preserves captured
    pointers; replacing/growing an allocation requires destroying its graphs.
    """
    # 译：在 cache 显存测量之前绑定模型拥有的通信资源。
    #
    #     模型只负责选用 DP Linear 模块。这个通用的准备过程找出这些模块，绑定它们的后端并创建它们的组。顺序执行的投影
    #     在每个设备上共享一个固定的临时空间池；规格相同的还共享原生资源。并发的模型或 stream 需要分别准备。
    #     以相同的后端和容量重复准备会保留已捕获的指针；替换或扩大分配需要先销毁对应的 graph。
    linears = [
        module
        for module in model.modules()
        if isinstance(module, (DPColumnParallelLinear, DPRowParallelLinear))
    ]
    if not linears:
        return False
    if backend is None:
        backend = get_global_backend()
    prepared = {
        module.projection_workspace.spec: module.projection_workspace
        for module in linears
        if module.projection_workspace is not None
    }
    specs = [
        ProjectionSpec(
            group=module.tp_group,
            kind="column" if isinstance(module, DPColumnParallelLinear) else "row",
            input_size=module.input_size,
            output_size=module.output_size,
            max_tokens=max_tokens,
            dtype=model_activation_dtype,
            device=module.weight.device,
        )
        for module in linears
    ]
    pools = {}
    for module, spec in zip(linears, specs):
        if module.projection_workspace is not None and (
            module.projection_workspace.spec != spec
            or module.projection_workspace.backend is not backend
            or module.projection_workspace.closed
        ):
            raise RuntimeError("Cannot replace prepared DP linear communication")
        if module.projection_workspace is not None:
            pools[spec.device] = module.projection_workspace.scratch_pool
    # Size private pools before any workspace holds views. Reusing a prepared
    # pool keeps it frozen, so a larger request cannot invalidate graph pointers.
    # 译：在任何工作区持有视图之前确定私有池的大小。复用已准备好的池会保持它冻结，
    #     这样更大的请求就不会使 graph 中的指针失效。
    for spec in specs:
        if spec.device not in pools:
            pools[spec.device] = WorkspacePool(spec.device, initial_nbytes=0)
        pools[spec.device].allocate(*spec.scratch_specs)
    for pool in pools.values():
        if not pool.frozen:
            pool.freeze()
    for module, spec in zip(linears, specs):
        if spec not in prepared:
            prepared[spec] = prepare_projection_collectives(
                spec, backend, scratch_pool=pools[spec.device]
            )
        module.projection_workspace = prepared[spec]
        module.comm_backend = backend
    return True


def release_dp_linear_communication(model: torch.nn.Module) -> None:
    """Release a model's DP projection workspaces and allow preparation again.

    All subgroup ranks must call with the same module order, after forwards
    finish and their CUDA graphs are destroyed, before process groups close.
    Shared workspaces close once, in first-use module order. This releases only
    DP Linear resources, not other communication owned by the model. Repeated
    release is harmless; callers must also drop borrowed intermediate views.
    """
    # 译：释放模型的 DP 投影工作区，并允许再次准备。
    #
    #     子组的所有 rank 都必须以相同的模块顺序调用，时机是在前向结束、它们的 CUDA graph 被销毁之后，进程组关闭之前。
    #     共享的工作区按首次使用的模块顺序关闭一次。这只释放 DP Linear 的资源，不释放模型拥有的其他通信资源。
    #     重复释放无害；调用方还必须丢弃借用的中间视图。
    linears = [
        module
        for module in model.modules()
        if isinstance(module, (DPColumnParallelLinear, DPRowParallelLinear))
    ]
    workspaces = dict.fromkeys(
        module.projection_workspace
        for module in linears
        if module.projection_workspace is not None
    )
    for workspace in workspaces:
        workspace.close()
    for module in linears:
        module.projection_workspace = None
        module.comm_backend = None
