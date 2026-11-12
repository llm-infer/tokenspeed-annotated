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


# 词表并行的 Embedding 和 LM head。
#
# 你在这里：BaseTransformerModel.embed_tokens（token id -> 向量），以及 BaseCausalLM.lm_head
#           （它的权重被 LogitsProcessor 用来算 logits，第 5 阶段）。
#
# 按词表切分、每个 rank 只查自己那段、不属于本段的 token 输出 0、再 all-reduce 求和——这些是熟悉的内容。
# 本文件额外处理的细节：
#   1. 词表 padding：词表大小先向上对齐到 64 的倍数，再按 TP 均分，保证每个 rank 的分片大小相同。
#      Qwen3-8B：V=151936 正好是 64 的倍数，不需要 padding；TP=2 时每个 rank 75968 行。
#   2. “附加词表”（added vocab，LoRA 等方式新增的 token）：与原词表分别切分、分别 padding，
#      放在每个 rank 分片的末尾（布局见 VocabParallelEmbedding 的 docstring）。没有附加词表时这部分为空。
#   3. 融合的查表 kernel：一次 gather 同时完成“判断是否属于本 rank、换算本地下标、其他行置 0”。
#   4. 查询上下文并行（QCP，第 12 阶段）下的查表方式。
#   5. 批大小无关（batch-invariant）数值模式下，LM head 的矩阵乘使用固定的 kernel 计划。
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.gemm import plan_dense_gemm
from torch.nn.parameter import Parameter, UninitializedParameter

from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    token_all_gather_rows,
    token_reduce_scatter,
)
from tokenspeed.runtime.distributed.utils import divide
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.parameter import BaseWeightParameter
from tokenspeed.runtime.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from tokenspeed.runtime.utils import set_weight_attrs

# 词表 padding 的对齐单位
DEFAULT_VOCAB_PADDING_SIZE = 64
# 当前 torch 版本支持的所有 FP8 dtype（不同版本可用的名字不同，逐个探测）
_FLOAT8_DTYPES = tuple(
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, name, None)) is not None
)
# Weight dtypes the fused vocab-shard gather kernel is registered for.
# 译：融合的词表分片 gather kernel 所注册支持的权重 dtype。
_FUSED_GATHER_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


# 未量化的 embedding / LM head 的计算方式（与量化方法接口相同，见 layers/quantization/base_config.py）：
#   create_weights：创建 [本 rank 的词表行数, H] 的权重
#   embedding：按 token id 查表（Embedding 用）
#   apply：hidden @ weight^T 得到 logits（LM head 用）
# batch_invariant：数值模式要求“同一个请求的结果不随 batch 中其他请求变化”时为 True
#   （见 docs/design/numerics.md），此时 LM head 的矩阵乘使用预先选定、与 batch 大小无关的 kernel 计划。
class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    # 译：embedding 的未量化方法。

    def __init__(self):
        from tokenspeed.runtime.utils.env import global_server_args_dict

        self.batch_invariant = global_server_args_dict["numerics"] in BITWISE_ENVELOPES
        self.kernel_plan = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        # 译：为 embedding 层创建权重。
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        if self.batch_invariant:
            self.kernel_plan = plan_dense_gemm(
                mode="mm",
                input_size=input_size_per_partition,
                output_size=sum(output_partition_sizes),
                dtype=params_dtype,
                batch_size=None,
                batch_invariant=True,
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.batch_invariant:
            if self.kernel_plan is None:
                raise RuntimeError(
                    "Batch-invariant LM head requires a plan before execution"
                )
            return kernel_mm(x, layer.weight, bias=bias, plan=self.kernel_plan)
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


# 把词表大小向上对齐到 pad_to 的倍数。例：1010 -> 1024；151936 -> 151936
def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    # 译：把词表大小填充（pad）到给定值的倍数。
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


# 第 rank 个分片负责的词表区间 [start, end)：
#   per_partition：每个分片的大小已知时直接计算；global：先用全局大小除以分片数
#   offset：整体平移（附加词表的区间从原词表大小开始编号）
def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size, rank, offset=offset
    )


# 本 rank 分片的各个边界（全局 token id 编号）：
#   padded_org_* / padded_added_*：含 padding 的原词表段、附加词表段
#   org_* / added_*：去掉 padding 后真实存在的 token 区间
# 例（Qwen3-8B，TP=2，rank 1）：org [75968, 151936)，没有附加词表
@dataclass
class VocabParallelEmbeddingShardIndices:
    """Indices for a shard of a vocab parallel embedding."""

    # 译：词表并行 embedding 中一个分片的下标。

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self):
        # sanity checks
        # 译：合理性检查
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index

        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index

        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index

        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


# 把全局 token id 换算成本 rank 分片内的本地下标，并给出“不属于本 rank”的掩码：
#   属于原词表段：本地下标 = id - org_start
#   属于附加词表段：本地下标 = id - added_offset（附加段在分片中排在原词表段 + 其 padding 之后）
#   都不属于：本地下标置 0（随便查一行，结果稍后被掩码清零）
# 返回 (本地下标, 不属于本 rank 的掩码)。
def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


# 词表并行的 Embedding。成员中与切分有关的：
#   num_embeddings_padded：padding 后的总词表大小
#   num_embeddings_per_partition：每个 rank 的行数（= padding 后总大小 / tp_size）
#   shard_indices：本 rank 分片的各个边界（VocabParallelEmbeddingShardIndices）
#   quant_method：查表 / 矩阵乘的实现，默认未量化
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    The vocabulary size is padded so it is divisible by the number of model
    parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately (both padded),
    and place them in the same tensor.
    In this example, we will have the original vocab size = 1010,
    added vocab size = 16 and padding to 64. Therefore, the total
    vocab size with padding will be 1088 (because we first pad 1010 to
    1024, add 16, and then pad to 1088).
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|< -----LORA------ >|< -LORA PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  | 1010 | ... | 1015 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 | 1024 | ... | 1039 | 1040 | ... | 1087 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|< -----LORA------ >|< -LORA PADDING- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 1000 | ... | 1015 |  -1  | ... |  -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 527  |  520 | ... | 543 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|< -----------LORA PADDING----------- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |  -1  | ... |  -1  | -1  | ... |   -1 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 | 512  | ... | 519  | 520 | ... |  543 |

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
        quant_config: quant config for the layer
        prefix: full name of the layer in the state dict
    """  # noqa: E501

    # 译：在词表维度上并行的 Embedding。
    #
    #     词表大小会被填充，使其能被模型并行的 GPU 数整除。
    #
    #     为了支持多种加载方式，我们保证 LoRA 新增的 embedding 总是位于 TP 分片张量的末尾。换句话说，
    #     我们分别切分基础 embedding 和 LoRA embedding（两者都做填充），再把它们放进同一个张量。
    #     在这个例子中，原始词表大小 = 1010，新增词表大小 = 16，填充到 64 的倍数。因此填充后的总词表大小为
    #     1088（先把 1010 填充到 1024，加上 16，再填充到 1088）。
    #     因此张量格式如下：
    #     TP1，rank 0（不切分）：
    #         （见上方原文图示：BASE 为基础词表，BASE PADDING 为其填充，LORA 为新增词表，LORA PADDING 为其填充；
    #          corresponding token_id 为每个位置对应的 token id，-1 表示填充；index 为张量中的下标）
    #     TP2，rank 0：（同上）
    #     TP2，rank 1：（同上）
    #
    #     参数：
    #         num_embeddings：词表大小。
    #         embedding_dim：hidden state 的大小。
    #         params_dtype：参数的类型。
    #         org_num_embeddings：原始词表大小（不含 LoRA）。
    #         padding_size：词表的填充单位。
    #         quant_config：该层的量化配置
    #         prefix：该层在 state dict 中的完整名字

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
        use_presharded_weights: bool = False,
    ):
        super().__init__()
        self.quant_config = quant_config

        # 不传 TP 参数时表示不切分（单卡语义）；三个参数要么都传，要么都不传
        if tp_rank is None:
            assert tp_size is None
            assert tp_group is None
            self.tp_rank, self.tp_size, self.tp_group = 0, 1, None
        else:
            assert tp_size is not None
            assert tp_group is not None
            self.tp_rank, self.tp_size, self.tp_group = tp_rank, tp_size, tp_group

        # org_vocab_size：原词表大小；num_embeddings 比它大的部分就是附加词表。
        # use_presharded_weights：checkpoint 已经按 rank 切好（此时不支持附加词表）。
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.use_presharded_weights = use_presharded_weights
        if use_presharded_weights:
            assert (
                num_added_embeddings == 0
            ), "Lora is not supported with presharded weights."

        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, self.padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            self.tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        quant_method: QuantizeMethodBase = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        # 译：如果正在构造的是 embedding 层，那么量化方法必须实现 embedding 操作。
        #     如果是 ParallelLMHead 这类其他层，这一点就不重要。
        is_embedding_layer = type(self) is VocabParallelEmbedding

        # LM head（子类 ParallelLMHead）在 ModelOpt 混合精度量化下可以使用量化的线性方法；
        # 其他情况一律用未量化方法。
        if not is_embedding_layer and quant_config is not None:
            from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod
            from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
                ModelOptMixedConfig,
            )

            if isinstance(quant_config, ModelOptMixedConfig):
                method = quant_config.get_quant_method(self, prefix=prefix)
                if method is not None and not isinstance(
                    method, UnquantizedLinearMethod
                ):
                    quant_method = method

        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method)
        )
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        # Divide the weight matrix along the vocaburaly dimension.
        # 译：沿词表维度切分权重矩阵。
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(
            self.num_embeddings_padded, self.tp_size
        )
        assert (
            self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        )
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index
            - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index
            - self.shard_indices.added_vocab_start_index
        )

        # 这两个字段是线性层接口的一部分（量化方法创建权重时会读取），embedding 不使用，固定为默认值
        self.interleave_linear_and_gate = False
        self.override_kernel_name = None
        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    # 按 class docstring 中的布局，算出第 tp_rank 个分片的各个边界：
    #   原词表段和附加词表段各自按 TP 均分（含 padding）；再用真实大小截断，得到去掉 padding 的区间。
    @classmethod
    def _get_indices(
        cls,
        vocab_size_padded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """Get start and end indices for vocab parallel embedding, following the
        layout outlined in the class docstring, based on the given tp_rank and
        tp_size."""
        # 译：按照 class docstring 中描述的布局，根据给定的 tp_rank 和 tp_size，得到词表并行 embedding 的起止下标。
        num_added_embeddings_padded = vocab_size_padded - org_vocab_size_padded
        padded_org_vocab_start_index, padded_org_vocab_end_index = (
            vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank, tp_size)
        )
        padded_added_vocab_start_index, padded_added_vocab_end_index = (
            vocab_range_from_global_vocab_size(
                num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
            )
        )
        # remove padding
        # 译：去掉填充
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    # 供采样使用：各 rank 的 logits all-gather 拼起来后，第 i 列不一定对应 token i
    # （每个分片内部是“原词表段 + padding + 附加段 + padding”）。这里返回一个下标列表，
    # 用它重新索引拼接后的 logits，使下标与 token id 一一对应（真实 token 在前，所有 padding 排在最后）。
    # 单个 rank 时布局已经是一一对应的，返回 None。
    def get_sharded_to_full_mapping(self) -> list[int] | None:
        """Get a mapping that can be used to reindex the gathered
        logits for sampling.

        During sampling, we gather logits from all ranks. The relationship
        of index->token_id will follow the same format as outlined in the class
        docstring. However, after the gather, we want to reindex the final
        logits tensor to map index->token_id one-to-one (the index is always
        equal the token_id it corresponds to). The indices returned by this
        method allow us to do that.
        """
        # 译：得到一个映射，可用于为采样重新索引收集到的 logits。
        #
        #     采样时我们从所有 rank 收集 logits。下标 -> token_id 的关系遵循 class docstring 中描述的格式。
        #     但收集之后，我们希望重新索引最终的 logits 张量，使下标与 token_id 一一对应（下标总是等于它对应的
        #     token_id）。本方法返回的下标可以做到这一点。
        if self.tp_size < 2:
            return None

        base_embeddings: list[int] = []
        added_embeddings: list[int] = []
        padding: list[int] = []
        for tp_rank in range(self.tp_size):
            shard_indices = self._get_indices(
                self.num_embeddings_padded,
                self.org_vocab_size_padded,
                self.num_embeddings,
                self.org_vocab_size,
                tp_rank,
                self.tp_size,
            )
            range_start = self.num_embeddings_per_partition * tp_rank
            range_end = self.num_embeddings_per_partition * (tp_rank + 1)
            base_embeddings.extend(
                range(range_start, range_start + shard_indices.num_org_elements)
            )
            padding.extend(
                range(
                    range_start + shard_indices.num_org_elements,
                    range_start + shard_indices.num_org_elements_padded,
                )
            )
            added_embeddings.extend(
                range(
                    range_start + shard_indices.num_org_elements_padded,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                )
            )
            padding.extend(
                range(
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements,
                    range_start
                    + shard_indices.num_org_elements_padded
                    + shard_indices.num_added_elements_padded,
                )
            )
            assert (
                range_start
                + shard_indices.num_org_elements_padded
                + shard_indices.num_added_elements_padded
                == range_end
            )
        ret = base_embeddings + added_embeddings + padding
        assert len(ret) == self.num_embeddings_padded
        return ret

    # 加载 embedding / LM head 的权重：从完整词表中取出本 rank 的 [org_start, org_end) 行，
    # 写入参数的前面部分，分片中剩下的行（padding）清零。
    #   例（Qwen3-8B，TP=2，rank 1）：checkpoint [151936, 4096] -> 取 [75968, 151936) 行 -> 写入 [75968, 4096]
    # 没有 output_dim 属性的参数（例如 per-tensor scale）不切分，每个 rank 拷贝完整的一份。
    # UninitializedParameter：形状要等读到 checkpoint 才知道的参数，在这里按 1/tp 的大小实例化。
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)

        if isinstance(param, UninitializedParameter):
            shape = list(loaded_weight.shape)
            if output_dim is not None:
                shape[output_dim] = shape[output_dim] // self.tp_size
            param.materialize(tuple(shape), dtype=loaded_weight.dtype)

        # If parameter does not have output dim, then it should
        # be copied onto all gpus (e.g. g_idx for act_order gptq).
        # 译：如果参数没有输出维，就应该把它拷贝到所有 GPU 上（例如 act_order GPTQ 的 g_idx）。
        if output_dim is None:
            # A per-tensor scale may be stored as a scalar (an FP8 LM head's).
            # 译：per-tensor scale 可能以标量形式存储（FP8 LM head 的就是这样）。
            if (
                getattr(param, "needs_scalar_to_array", False)
                or loaded_weight.dim() == 0
            ):
                loaded_weight = loaded_weight.reshape(param.data.shape)
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return

        # Shard indexes for loading the weight
        # 译：加载权重时使用的分片下标
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        # If param packed on the same dim we are sharding on, then
        # need to adjust offsets of loaded weight by pack_factor.
        # 译：如果参数在我们切分的同一维上做了打包，就需要按打包因子调整加载权重的偏移。
        if packed_dim is not None and packed_dim == output_dim:
            packed_factor = (
                param.packed_factor
                if isinstance(param, BaseWeightParameter)
                else param.pack_factor
            )
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size // param.packed_factor
            )
            start_idx = start_idx // packed_factor
            shard_size = shard_size // packed_factor
        else:
            assert loaded_weight.shape[output_dim] == (
                self.org_vocab_size
                // (self.tp_size if self.use_presharded_weights else 1)
            ), f"{self.org_vocab_size=} {self.use_presharded_weights=} {loaded_weight.shape[output_dim]=}"

        # Copy the data.
        # 译：拷贝数据。
        if not self.use_presharded_weights:
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0] :].data.fill_(0)

    # 能否用融合的 gather kernel 一次完成查表：需要在 GPU 上、未量化、权重 dtype 受支持且最后一维连续。
    def _fused_shard_gather(self, input_: torch.Tensor) -> bool:
        """Whether the plain table lookup can run as one masked gather kernel.

        Quantized embedding methods and the FP8 PLE table keep the two-step
        mask-then-lookup path; their lookups are not a row gather of
        ``self.weight``.
        """
        # 译：普通的查表能否作为一个带掩码的 gather kernel 运行。
        #
        #     量化的 embedding 方法和 FP8 的 PLE 表保持“先掩码、再查表”的两步路径；它们的查表不是对
        #     ``self.weight`` 的按行 gather。
        return (
            input_.is_cuda
            and type(self.quant_method) is UnquantizedEmbeddingMethod
            # The gather kernel's dtypes (no FP8) and its unit-column-stride reads.
            # 译：gather kernel 支持的 dtype（不含 FP8），以及它要求列步长为 1 的读取方式。
            and self.weight.dtype in _FUSED_GATHER_DTYPES
            and self.weight.stride(-1) == 1
            and self.weight.is_cuda
        )

    # 查表前向：input_ [T] -> [T, H]。
    #   普通情况：本 rank 部分查表（不属于本 rank 的行为 0），TP>1 且 reduce_results 时 all-reduce 求和。
    #     reduce_results=False：由调用方推迟求和（见 BaseTransformerModel.forward 中与第 0 层 norm 的融合）。
    #   查询上下文并行（QCP）：每个 rank 只持有一部分 token 行，而每个词表分片都需要看到所有 token，
    #     所以先 all-gather token id，查表后用 reduce-scatter 把求和结果分回各 rank 的行。
    def forward(
        self,
        input_: torch.Tensor,
        reduce_results: bool = True,
        *,
        query_shard: QueryShardPlan | None = None,
    ) -> torch.Tensor:
        """Forward pass for the vocabulary parallel embedding.

        Args:
            input_: The input tensor.
            reduce_results: Whether to reduce the results across all the model parallel GPUs.
            query_shard: The forward's query shard when ``input_`` holds this
                rank's rows of a sharded extend (query context parallelism;
                ``ctx.query_shard``). ``None`` -- the default, every other
                caller -- means the ids are the same on every rank of the
                group. Under a shard every vocab shard must see every row's
                id, so the ids are all-gathered to the span, the partial
                lookup runs over the span and the sum is reduce-scattered
                back to the shard's rows (the gather and the reduce-scatter
                share the shard's gather order, so no permutation is
                needed): the same bytes as the replicated all-reduce, the
                rows this rank holds.

        Returns:
            The output tensor.

        """
        # 译：词表并行 embedding 的前向。
        #
        #     参数：
        #         input_：输入张量。
        #         reduce_results：是否在所有模型并行的 GPU 之间对结果求和。
        #         query_shard：当 ``input_`` 是一次分片 extend 中本 rank 的行时（查询上下文并行；``ctx.query_shard``），
        #             为本次前向的查询分片。``None``——默认值，也是其他所有调用方的情况——表示组内每个 rank 的 id 相同。
        #             在分片下，每个词表分片都必须看到每一行的 id，所以先把 id all-gather 到整个范围，在整个范围上做
        #             部分查表，再把和 reduce-scatter 回本分片的行（gather 和 reduce-scatter 使用分片相同的收集顺序，
        #             所以不需要置换）：字节与复制情况下的 all-reduce 相同，只是得到本 rank 持有的行。
        #
        #     返回：
        #         输出张量。
        sharded = query_shard is not None and query_shard.size > 1
        if sharded and self.tp_size > 1:
            if not reduce_results:
                raise ValueError("a query-sharded embedding lookup reduces its rows")
            if input_.dim() != 1 or input_.shape[0] != query_shard.local_rows:
                raise ValueError(
                    f"query shard rank {query_shard.rank} embeds "
                    f"{query_shard.local_rows} rows, got {tuple(input_.shape)}"
                )
            if len(self.tp_group) != query_shard.size:
                raise ValueError(
                    f"the embedding's TP group of {len(self.tp_group)} ranks is not "
                    f"the query shard group of {query_shard.size}"
                )
            counts = list(query_shard.row_counts)
            ids = token_all_gather_rows(
                input_.reshape(-1, 1), self.tp_group, counts
            ).reshape(-1)
            return token_reduce_scatter(
                self._partial_lookup(ids), self.tp_group, counts
            )
        output_parallel = self._partial_lookup(input_)
        if self.tp_size > 1 and reduce_results:
            return all_reduce(output_parallel, self.tp_group)
        return output_parallel

    # 本 rank 词表分片对每一行的贡献：属于本分片的 id 得到它的 embedding，其他 id 得到全 0 行。
    # 三条路径：
    #   1. 融合 gather kernel（GPU、未量化，最常见）
    #   2. 单卡：不需要掩码，但要把 id 截断到合法范围（CUDA Graph 捕获预热时可能喂入越界的 id）
    #   3. 通用路径：先换算本地下标并得到掩码，查表，再把不属于本 rank 的行清零
    def _partial_lookup(self, input_: torch.Tensor) -> torch.Tensor:
        """This rank's vocab shard's contribution to every row of ``input_``:
        the embedding of ids inside the shard, zero rows for the others. The
        sum over the TP group is the full embedding; a single rank's shard is
        the whole table."""
        # 译：本 rank 的词表分片对 ``input_`` 每一行的贡献：分片内的 id 得到其 embedding，其他 id 得到全零行。
        #     在 TP 组上求和就是完整的 embedding；单个 rank 时，它的分片就是整张表。
        if self._fused_shard_gather(input_):
            # One gather resolves the shard mask, the local index and the
            # zeroing of other ranks' rows.
            # 译：一次 gather 同时完成分片掩码、本地下标换算和其他 rank 的行清零。
            from tokenspeed_kernel.ops.embedding import vocab_shard_embedding

            return vocab_shard_embedding(
                self.weight,
                input_.contiguous(),
                (
                    self.shard_indices.org_vocab_start_index,
                    self.shard_indices.org_vocab_end_index,
                ),
                self.shard_indices.num_org_vocab_padding,
                (
                    self.shard_indices.added_vocab_start_index,
                    self.shard_indices.added_vocab_end_index,
                ),
            )
        if self.tp_size == 1:
            # No shard mask here: clamp the out-of-range ids CUDA-graph capture warmup can feed.
            # 译：这里没有分片掩码：把 CUDA Graph 捕获预热时可能喂入的越界 id 截断到合法范围。
            masked_input = input_.clamp(min=0, max=self.num_embeddings_padded - 1)
            return self.quant_method.embedding(self, masked_input)
        # Build the mask.
        # 译：构造掩码。
        masked_input, input_mask = get_masked_input_and_mask(
            input_,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index,
        )
        output_parallel = self.quant_method.embedding(self, masked_input)
        # Mask the output embedding.
        # 译：对输出的 embedding 做掩码。
        output_mask = input_mask.unsqueeze(-1)
        if output_parallel.dtype in _FLOAT8_DTYPES:
            # CUDA does not implement masked_fill_ for FP8. PLE keeps its
            # n-gram table in FP8 and dequantizes after this sharded
            # lookup, so preserve the FP8 payload while zeroing rows that
            # belong to another TP rank with the supported pointwise op.
            # 译：CUDA 没有为 FP8 实现 masked_fill_。PLE 把它的 n-gram 表存为 FP8，并在这次分片查表之后再反量化，
            #     所以用受支持的逐元素运算把属于其他 TP rank 的行置零，同时保留 FP8 数据。
            return torch.where(output_mask, 0.0, output_parallel)
        output_parallel.masked_fill_(output_mask, 0)
        return output_parallel

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", num_embeddings_padded={self.num_embeddings_padded}"
        s += f", tp_size={self.tp_size}"
        return s


# LM head：与 embedding 共用“按词表切分”的权重布局，但不用来查表。
# 它的 forward 直接报错——权重由 LogitsProcessor 取走做 hidden @ weight^T（第 5 阶段）。
# Qwen3-8B、TP=1：权重 [151936, 4096]。
class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    # 译：并行化的 LM head。
    #
    #     采样器中使用的输出 logits 权重矩阵。权重和 bias 张量会被填充，以保证能被模型并行的 GPU 数整除。
    #
    #     参数：
    #         num_embeddings：词表大小。
    #         embedding_dim：hidden state 的大小。
    #         bias：是否使用 bias。
    #         params_dtype：参数的类型。
    #         org_num_embeddings：原始词表大小（不含 LoRA）。
    #         padding_size：词表的填充单位。

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
        use_presharded_weights: bool = False,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            use_presharded_weights=use_presharded_weights,
        )
        self.quant_config = quant_config
        if bias:
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition, dtype=params_dtype)
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

    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        # 译：与词嵌入共享权重。
        self.weight = embed_tokens.weight
        return self

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
