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

"""Qwen3 model configuration definitions."""

# 译：Qwen3 模型配置的定义。

import logging

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

# Qwen3 的 HF 配置类：把 checkpoint 目录里 config.json 的字段读成 Python 属性，
# 模型代码（models/qwen3.py）从这里取 hidden_size、头数、head_dim 等超参数。
#
# 你在这里：启动时 utils/hf_transformers_utils.py 读取 config.json，按其中的
# model_type（Qwen3 是 "qwen3"）在 _CONFIG_REGISTRY 中找到本类并实例化（第 8 阶段）。
#
# 本文件与 HF transformers 的同名类基本一致；由于登记在 _CONFIG_REGISTRY 中，
# 解析 Qwen3 配置时总是用这里的定义。与推理相关、值得注意的只有 layer_types：每层是全注意力还是
# 滑动窗口注意力，Qwen3Attention 据此设置该层的窗口（见 __init__ 末尾的推导）。

# 每层注意力类型的合法取值
ALLOWED_LAYER_TYPES = (
    "full_attention",
    "sliding_attention",
    "chunked_attention",
)


def layer_type_validation(layer_types: list[str]):
    """Check that each entry in `layer_types` are allowed."""
    # 译：检查 `layer_types` 中的每一项都是允许的取值。
    if not all(layer_type in ALLOWED_LAYER_TYPES for layer_type in layer_types):
        raise ValueError(f"The `layer_types` entries must be in {ALLOWED_LAYER_TYPES}")


logger = logging.getLogger(__name__)


class Qwen3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3Model`]. It is used to instantiate a
    Qwen3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-8B [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.  if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen3Model, Qwen3Config

    >>> # Initializing a Qwen3 style configuration
    >>> configuration = Qwen3Config()

    >>> # Initializing a model from the Qwen3-8B style configuration
    >>> model = Qwen3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 译：这是用于存储 [`Qwen3Model`] 配置的配置类。它用于按指定的参数实例化一个 Qwen3 模型，定义模型的结构。
    #     用默认值实例化配置，会得到与 Qwen3-8B [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
    #     相近的配置。
    #
    #     配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请阅读 [`PretrainedConfig`] 的文档。
    #
    #     参数：
    #         vocab_size（`int`，可选，默认 151936）：
    #             Qwen3 模型的词表大小。决定调用 [`Qwen3Model`] 时传入的 `inputs_ids` 能表示多少种不同的 token。
    #         hidden_size（`int`，可选，默认 4096）：
    #             隐藏表示的维度。
    #         intermediate_size（`int`，可选，默认 22016）：
    #             MLP 表示的维度。
    #         num_hidden_layers（`int`，可选，默认 32）：
    #             Transformer 编码器中的隐藏层数。
    #         num_attention_heads（`int`，可选，默认 32）：
    #             Transformer 编码器中每个注意力层的注意力头数。
    #         num_key_value_heads（`int`，可选，默认 32）：
    #             用于实现分组查询注意力（GQA）的 key/value 头数。如果 `num_key_value_heads=num_attention_heads`，
    #             模型使用多头注意力（MHA）；如果 `num_key_value_heads=1`，模型使用多查询注意力（MQA）；
    #             否则使用 GQA。把多头 checkpoint 转换为 GQA checkpoint 时，每组的 key 和 value 头应由该组内
    #             所有原始头做平均池化得到。详见[这篇论文](https://arxiv.org/pdf/2305.13245.pdf)。
    #             未指定时默认为 `32`。
    #         head_dim（`int`，可选，默认 128）：
    #             注意力头的维度。
    #         hidden_act（`str` 或 `function`，可选，默认 `"silu"`）：
    #             解码器中的非线性激活函数（函数或字符串）。
    #         max_position_embeddings（`int`，可选，默认 32768）：
    #             该模型可能用到的最大序列长度。
    #         initializer_range（`float`，可选，默认 0.02）：
    #             初始化所有权重矩阵时，截断正态分布初始化器的标准差。
    #         rms_norm_eps（`float`，可选，默认 1e-06）：
    #             RMS 归一化层使用的 epsilon。
    #         use_cache（`bool`，可选，默认 `True`）：
    #             模型是否返回最后的 key/value 注意力（并非所有模型都使用）。只在 `config.is_decoder=True` 时有意义。
    #         tie_word_embeddings（`bool`，可选，默认 `False`）：
    #             模型的输入和输出词嵌入是否共享。
    #         rope_theta（`float`，可选，默认 10000.0）：
    #             RoPE 嵌入的基础周期。
    #         rope_scaling（`Dict`，可选）：
    #             包含 RoPE 嵌入缩放配置的字典。如果使用了新的 rope 类型并希望模型处理更长的
    #             `max_position_embeddings`，建议相应地更新这个值。
    #             预期内容：
    #                 `rope_type`（`str`）：
    #                     使用的 RoPE 子变体。可以是 ['default', 'linear', 'dynamic', 'yarn', 'longrope',
    #                     'llama3'] 之一，其中 'default' 是原始的 RoPE 实现。
    #                 `factor`（`float`，可选）：
    #                     除 'default' 外的所有 rope 类型都使用。施加在 RoPE 嵌入上的缩放因子。在大多数缩放类型中，
    #                     factor 为 x 时，模型能处理长度为“x × 原始预训练最大长度”的序列。
    #                 `original_max_position_embeddings`（`int`，可选）：
    #                     用于 'dynamic'、'longrope' 和 'llama3'。预训练时使用的原始最大位置数。
    #                 `attention_factor`（`float`，可选）：
    #                     用于 'yarn' 和 'longrope'。施加在注意力计算上的缩放因子。未指定时使用实现推荐的默认值，
    #                     由 `factor` 字段推导出建议值。
    #                 `beta_fast`（`float`，可选）：
    #                     仅用于 'yarn'。线性斜坡函数中设置外推（仅外推）边界的参数。未指定时默认为 32。
    #                 `beta_slow`（`float`，可选）：
    #                     仅用于 'yarn'。线性斜坡函数中设置内插边界的参数。未指定时默认为 1。
    #                 `short_factor`（`List[float]`，可选）：
    #                     仅用于 'longrope'。施加在短上下文（< `original_max_position_embeddings`）上的缩放因子。
    #                     必须是一个数字列表，长度等于 hidden size 除以注意力头数再除以 2。
    #                 `long_factor`（`List[float]`，可选）：
    #                     仅用于 'longrope'。施加在长上下文（< `original_max_position_embeddings`）上的缩放因子。
    #                     必须是一个数字列表，长度等于 hidden size 除以注意力头数再除以 2。
    #                 `low_freq_factor`（`float`，可选）：
    #                     仅用于 'llama3'。施加在 RoPE 低频分量上的缩放因子。
    #                 `high_freq_factor`（`float`，可选）：
    #                     仅用于 'llama3'。施加在 RoPE 高频分量上的缩放因子。
    #         attention_bias（`bool`，可选，默认 `False`）：
    #             自注意力中的 query、key、value 和输出投影层是否使用 bias。
    #         use_sliding_window（`bool`，可选，默认 `False`）：
    #             是否使用滑动窗口注意力。
    #         sliding_window（`int`，可选，默认 4096）：
    #             滑动窗口注意力（SWA）的窗口大小。未指定时默认为 `4096`。
    #         max_window_layers（`int`，可选，默认 28）：
    #             使用 SWA（滑动窗口注意力）的层数。底部的层使用 SWA，顶部的层使用全注意力。
    #         layer_types（`list`，可选）：
    #             每一层的注意力模式。
    #         attention_dropout（`float`，可选，默认 0.0）：
    #             注意力概率的 dropout 比例。
    #
    #     示例代码中的注释：
    #         # 初始化一个 Qwen3 风格的配置
    #         # 用 Qwen3-8B 风格的配置初始化一个模型
    #         # 访问模型的配置

    # config.json 里的 model_type 字段，配置注册表据此找到本类
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # 下面两个 plan 是 HF transformers 自己做张量并行 / 流水线并行时用的描述，
    # 本项目不读取它们（并行切分由各层的 Linear 和 Mapping 决定），保留只是为了与 HF 一致。
    # Default tensor parallel plan for base model `Qwen3`
    # 译：基础模型 `Qwen3` 的默认张量并行方案
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        # 没开启滑动窗口时把 sliding_window 置为 None，后面推导 layer_types 时据此判断
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        # 译：为了向后兼容
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # 译：校验旋转位置编码参数的正确性
        #     向后兼容：如果有 'type' 字段，把它移到 'rope_type'。
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # 每层的注意力类型。config.json 没有给出时按旧字段推导：
        #   开启滑动窗口（sliding_window 非 None）时，层号 >= max_window_layers 的层用滑动窗口，
        #   其余用全注意力；未开启时全部是全注意力。
        #   Qwen3-8B 发布的配置 use_sliding_window=false，所以 36 层全部是 "full_attention"。
        # 注意：类 docstring 中 max_window_layers 的说明是“底部的层用 SWA，顶部用全注意力”，
        #   但下面的代码是层号 >= max_window_layers 的层（靠上的层）用滑动窗口，以代码为准。
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                (
                    "sliding_attention"
                    if self.sliding_window is not None and i >= self.max_window_layers
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3Config"]
