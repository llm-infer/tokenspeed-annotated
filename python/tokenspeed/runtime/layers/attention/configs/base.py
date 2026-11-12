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

"""Composite attention configuration.

``AttnConfig`` carries the model-wide serving facts (device, cache precision,
paging, capacity, speculative shape) and a tuple of attention components.
Each component is an ``AttnComponentSpec`` subclass owning one mechanism's
kernel choice, head geometry, per-layer vectors, and parallel width;
``SoftmaxAttnConfig`` is the softmax-family base (MHA/MLA/DSA/MSA), the
counterpart of ``LinearAttnConfig``'s recurrent state.

The softmax component is unique per model (validated at construction) and
addressed by type: ``config.component(SoftmaxAttnConfig)``. Backends receive
their component explicitly: ``Backend(config, spec)``.
"""

# 译：组合式的注意力配置。
#
#     ``AttnConfig`` 承载模型范围的服务参数（设备、cache 精度、分页、容量、投机解码的形状）以及一组注意力组件。
#     每个组件是一个 ``AttnComponentSpec`` 子类，负责一种注意力机制的 kernel 选择、头的几何、按层的向量和并行宽度；
#     ``SoftmaxAttnConfig`` 是 softmax 系列（MHA/MLA/DSA/MSA）的基类，与 ``LinearAttnConfig`` 的递归状态相对。
#
#     softmax 组件在每个模型中是唯一的（构造时校验），并按类型访问：``config.component(SoftmaxAttnConfig)``。
#     后端显式接收它的组件：``Backend(config, spec)``。

# 注意力配置：启动时由 layers/attention/registry.py 根据模型配置和启动参数生成，交给注意力后端和 KV 池使用。
#
# 结构：AttnConfig（模型范围的参数）+ 若干组件（AttnComponentSpec）
#   模型范围的参数：设备、KV cache dtype、kernel 页大小、最大上下文长度、最大并发请求数、投机解码形状、
#     KVP / QCP 并行参数等
#   组件：每种注意力机制一个。普通模型只有一个 softmax 组件（Qwen3 是 MHAConfig）；
#     混合模型（如 Qwen3.5 的全注意力 + 线性注意力）再加一个 LinearAttnConfig（第 13 阶段）
#
# Qwen3-8B（单卡、bf16）对应的配置大致是：
#   AttnConfig(device="cuda", dtype=bf16, kv_cache_dtype=bf16, context_len=40960, ...,
#              components=(MHAConfig(num_attention_heads=32, num_kv_heads=8, head_dim=128, attn_tp_size=1),))
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeVar

import torch

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.utils.server_args import ServerArgs

ComponentT = TypeVar("ComponentT", bound="AttnComponentSpec")


# 投机解码相关的小函数（第 11 阶段回访），主线不使用
def resolve_speculative_num_tokens(
    server_args: ServerArgs, is_draft: bool = False
) -> int:
    """Return the query width seen by this attention backend.

    Target verification consumes the full candidate window. DSpark's draft
    model samples from its anchor row, so seven draft queries produce the seven
    proposals in an eight-token verify window.
    """
    # 译：返回本注意力后端看到的 query 宽度。
    #
    #     目标模型的验证会消费整个候选窗口。DSpark 的草稿模型从它的锚点行开始采样，所以 7 个草稿 query 在一个 8 个 token 的
    #     验证窗口中产生 7 个提议。
    width = int(server_args.speculative_num_draft_tokens)
    if is_draft and server_args.speculative_algorithm == "DSPARK":
        return width - 1
    return width


def is_block_drafter(speculative_algorithm: str | None, is_draft: bool) -> bool:
    """A draft that proposes a whole block in one forward and writes its KV at
    the target's cache locations (DFLASH / DSPARK)."""
    # 译：在一次前向中提出整个 block 的草稿、并把它的 KV 写到目标模型 cache 位置上的草稿模型（DFLASH / DSPARK）。
    return bool(is_draft and speculative_algorithm in ("DFLASH", "DSPARK"))


def block_draft_is_causal(hf_config) -> bool:
    """Whether a block draft's queries attend causally within their block.

    Only an explicit declaration makes a block causal: ``is_causal``, else
    ``dflash_config.causal`` (Kimi-K3.1's DSpark draft). A draft declaring
    neither (Kimi-K3 DSpark) attends the whole block; ``sliding_attention``
    layers only bound how far back a query reads.
    """
    # 译：块草稿的 query 在其 block 内是否按因果方式做注意力。
    #
    #     只有显式声明才会让 block 是因果的：``is_causal``，否则看 ``dflash_config.causal``（Kimi-K3.1 的 DSpark 草稿）。
    #     两者都没有声明的草稿（Kimi-K3 DSpark）对整个 block 做注意力；``sliding_attention`` 层只限制一个 query
    #     能往回读多远。
    is_causal = getattr(hf_config, "is_causal", None)
    if is_causal is not None:
        return bool(is_causal)
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    return bool(dflash_config.get("causal", False))


# 每层属于哪个 cache group 的标签（如 "full_attention"、"sliding_attention"），供 cache 规划使用（第 14 阶段）。
# Qwen3-8B 的 layer_types 全是 "full_attention"，所以只有一个全历史 cache group。
def resolve_cache_layer_types(
    hf_config,
    *,
    num_layers: int,
    is_draft: bool,
    draft_block_decode: bool,
) -> tuple[str, ...]:
    """Per-layer cache-group labels of one model side, read once for all
    softmax families.

    ``cache_layer_types`` wins over ``layer_types``: it can carry labels
    outside transformers' ``ALLOWED_LAYER_TYPES`` (Inkling's sliding
    sub-groups). Target-stack labels that do not fit a draft's depth are
    dropped so the recipe resolves the draft to full history. A block drafter
    writes at the target's cache locations, so its storage IS the target's
    full-history group whatever compute mask its layers apply: the label
    vector says so explicitly instead of losing the labels.
    """
    # 译：某一侧模型（目标或草稿）的每层 cache group 标签，为所有 softmax 系列统一读取一次。
    #
    #     ``cache_layer_types`` 优先于 ``layer_types``：它可以携带 transformers 的 ``ALLOWED_LAYER_TYPES`` 之外的标签
    #     （Inkling 的滑动窗口子组）。与草稿模型深度不符的目标模型标签会被丢弃，使 recipe 把草稿解析为全历史。块草稿器
    #     把 KV 写到目标模型的 cache 位置，所以不论它的层使用什么计算掩码，它的存储就是目标模型的全历史组：标签向量
    #     明确地这样说明，而不是丢掉这些标签。
    layer_types = tuple(
        getattr(hf_config, "cache_layer_types", None)
        or getattr(hf_config, "layer_types", None)
        or ()
    )
    if draft_block_decode:
        return (FULL_ATTENTION,) * num_layers
    if is_draft and layer_types and len(layer_types) != num_layers:
        return ()
    return layer_types


# 启动参数 --kv-cache-dtype -> torch dtype：
#   auto / bfloat16 -> bf16（Qwen3 主线）；fp8 / fp8_e4m3 -> FP8；mxfp8 -> FP8 数据 + 另存的 UE8M0 scale
def resolve_dtype(kv_cache_dtype_str: str) -> torch.dtype:
    if kv_cache_dtype_str == "auto" or kv_cache_dtype_str == "bfloat16":
        return torch.bfloat16
    elif kv_cache_dtype_str in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    elif kv_cache_dtype_str == "mxfp8":
        # fp8-e4m3 data; the UE8M0 scale sidecar lives in the pool's scale buffers (kv_cache_mxfp8)
        # 译：fp8-e4m3 数据；UE8M0 的 scale 附加数据存放在 KV 池的 scale 缓冲区中（kv_cache_mxfp8）
        return torch.float8_e4m3fn
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype_str!r}")


@dataclass(kw_only=True)
class AttnComponentSpec:
    """One attention mechanism's boot-constant facts.

    Subclasses are the identity; a mixed full+sliding softmax model stays ONE
    component carrying its per-layer vectors.
    """

    # 译：一种注意力机制在启动时就确定的参数。
    #
    #     子类本身就是身份标识；全注意力 + 滑动窗口混合的 softmax 模型仍然是一个组件，携带按层的向量。

    # The compute backend requested for this component; None lets the backend
    # registry pick the component's default.
    # 译：为该组件请求的计算后端；为 None 时由后端注册表选择该组件的默认后端。
    backend_name: str | None = None


@dataclass(kw_only=True)
# softmax 注意力系列（MHA / MLA / DSA / MSA）的公共字段：头数、KV 头数、head_dim、注意力 TP 宽度，
# 以及按层的 cache group 标签和滑动窗口。
# generate：由子类实现，根据启动参数和模型配置生成完整的 AttnConfig。
# cache_cell_size：每个 token 在一层中占用的 KV 字节数，cache 规划据此计算能容纳多少 token。
class SoftmaxAttnConfig(AttnComponentSpec):
    """Base of the softmax-attention families (MHA / MLA / DSA / MSA)."""

    # 译：softmax 注意力系列（MHA / MLA / DSA / MSA）的基类。

    is_dsa: ClassVar[bool] = False
    # Whether this family stores a latent KV page (MLA and its DSA variant) as
    # opposed to K/V head rows (MHA, MSA). A draft of the latter kind plans an
    # attention family of its own beside an MLA target.
    # 译：该系列存储的是潜在 KV 页（MLA 及其 DSA 变体），还是 K/V 头的行（MHA、MSA）。后一种草稿模型
    #     在 MLA 目标模型旁边规划自己的注意力系列。
    is_latent_kv: ClassVar[bool] = False

    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    attn_tp_size: int
    # Per-layer cache-group labels (the storage vocabulary of
    # recipes/spec.py, NOT the checkpoint's compute labels), consumed only by
    # the cache recipes; empty -> single full-history group. A layer's
    # compute mask lives on its PagedAttention (sliding_window_size).
    # 译：每层的 cache group 标签（recipes/spec.py 的存储词汇，不是 checkpoint 的计算标签），只由 cache recipe 使用；
    #     为空 -> 只有一个全历史组。一个层的计算掩码在它的 PagedAttention 上（sliding_window_size）。
    cache_layer_types: tuple[str, ...] = ()
    # Sliding window for per-layer cache recipes. Recipes that declare whole
    # groups, such as DeepSeek V4.1, put their windows on those groups.
    # 译：按层 cache recipe 使用的滑动窗口。声明整个组的 recipe（如 DeepSeek V4.1）把窗口放在那些组上。
    sliding_window_tokens: int | tuple[int | None, ...] | None = None

    @classmethod
    def generate(
        cls, server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
    ) -> AttnConfig:
        raise NotImplementedError("Not Implemented!")

    def cache_cell_size(self, config: AttnConfig) -> int:
        """Per-token cache bytes of this component's dominant field.

        A per-TOKEN concept, hence softmax-family only: linear-attention
        state is fixed-size per request, sized via its shape properties.
        Takes the ``AttnConfig`` for the model-wide facts (cache dtype,
        quant method) that deliberately do not live on the spec.
        """
        # 译：本组件主要字段在每个 token 上占用的 cache 字节数。
        #
        #     这是一个按 TOKEN 计的概念，所以只属于 softmax 系列：线性注意力的状态对每个请求是固定大小的，通过其形状属性
        #     确定大小。参数是 ``AttnConfig``，用于获取那些有意不放在 spec 上的模型范围参数（cache dtype、量化方法）。
        raise NotImplementedError("Not Implemented!")


@dataclass(kw_only=True)
# 模型的注意力配置。字段分组：
#   数据类型：dtype（激活）、kv_cache_dtype、kv_cache_quant_method、kv_cache_mxfp8
#   分页：kernel_page_size（None 表示由后端决定，见 kernel_page_sizes.py）
#   容量：context_len（每个请求的最大 KV 长度，含投机解码的少量余量）、max_bs（每个 DP 组的最大并发请求数）
#   投机解码：speculative_*、is_draft、draft_block_*（第 11 阶段）
#   并行：kvp_*（KV 按页分布）、qcp_*（query 按行分布）（第 12 阶段）
#   components：各注意力组件
# __post_init__ 校验组合是否合法：必须恰好一个 softmax 组件；不支持 INT8 KV；QCP、KVP 只能和特定后端搭配。
class AttnConfig:
    """The model's attention configuration: model-wide facts + components.

    Exactly one component is the softmax family (validated below); extra
    components (linear attention) ride alongside as peers. Built exclusively
    by ``registry._create_attn_config`` — one construction seam.
    """

    # 译：模型的注意力配置：模型范围的参数 + 组件。
    #
    #     恰好一个组件属于 softmax 系列（下面会校验）；额外的组件（线性注意力）作为同级成员并存。只由
    #     ``registry._create_attn_config`` 构建——唯一的构造入口。

    device: str
    dtype: torch.dtype
    kv_cache_dtype: torch.dtype
    kv_cache_quant_method: str
    # MXFP8 KV: UE8M0 vec-32 scale sidecar in pool buffers; kv_cache_dtype alone can't express it
    # 译：MXFP8 KV：UE8M0、每 32 个一组的 scale 附加数据存放在池的缓冲区中；单靠 kv_cache_dtype 无法表达
    kv_cache_mxfp8: bool = False
    # Tokens covered by one attention-kernel page. None picks the backend's
    # per-kernel default (flexible kernels use their group's block
    # granularity; fixed-page kernels use their own constant). For the
    # ordinary families the registry pins a fixed-page leaf's constant here
    # before planning, so the recipe sizes its block span to hold whole pages.
    # 译：一个注意力 kernel 页覆盖的 token 数。None 表示选用后端针对每个 kernel 的默认值（灵活的 kernel 使用其所在组的
    #     block 粒度；固定页的 kernel 使用它自己的常量）。对普通系列，registry 在规划之前把固定页叶子后端的常量写到这里，
    #     使 recipe 把 block 的跨度定为能容纳整数个页。
    kernel_page_size: int | None = None
    # Physical per-request KV extent: the model's logical context_len plus
    # ServerArgs.spec_context_pad (spec verify overshoot for a finished request
    # lingering one overlap step). Backends size page tables and clamp seq_lens
    # against this; user-facing limits stay on the logical context_len.
    # 译：每个请求物理上的 KV 范围：模型逻辑上的 context_len 加上 ServerArgs.spec_context_pad（投机验证在一个已结束、
    #     但还要停留一个 overlap 步的请求上的超出量）。后端据此确定页表的大小、截断 seq_lens；面向用户的限制仍然使用
    #     逻辑上的 context_len。
    context_len: int
    max_bs: int
    # True iff server_args.disaggregation_mode != "null"; cache recipes use
    # it to stamp transfer policies onto the cache group specs.
    # 译：当且仅当 server_args.disaggregation_mode != "null" 时为 True；cache recipe 用它在 cache group spec 上标注传输策略。
    pd_disaggregation_enabled: bool = False
    speculative_num_steps: int = 0
    speculative_num_draft_tokens: int = 1
    is_draft: bool = False
    # DFLASH drafts a whole block in one decode forward (q_len = spec_num_tokens
    # per request) instead of Eagle/MTP's per-step single-token decode. Backends
    # use this to expand decode metadata to spec_num_tokens rows per request.
    # 译：DFLASH 在一次 decode 前向中提出整个 block 的草稿（每个请求 q_len = spec_num_tokens），而不是 Eagle/MTP 那样每步
    #     一个 token 的 decode。后端据此把 decode 元数据扩展为每个请求 spec_num_tokens 行。
    draft_block_decode: bool = False
    # Causal within a block draft; otherwise every query sees the whole block.
    # 译：在块草稿内是否因果；否则每个 query 都能看到整个 block。
    draft_block_causal: bool = False
    # KV parallelism (page-cyclic KV page sharding over a consecutive subgroup
    # of attention TP): one topology for target and continuation/MTP views;
    # KVP does not add ranks. Decode attends the owned pages and merges the
    # partials by LSE (DCP).
    # 译：KV 并行（在注意力 TP 的一个连续子组上按页轮流分布 KV 页）：目标模型和续写 / MTP 视图共用一个拓扑；KVP 不增加 rank。
    #     decode 在自己拥有的页上做注意力，再按 LSE 合并部分结果（DCP）。
    kvp_size: int = 1
    kvp_rank: int = 0
    kvp_group: tuple[int, ...] = (0,)
    # Query context parallelism: each rank computes early and late segments
    # of every request; the group is the attention TP group.
    # 译：查询上下文并行：每个 rank 计算每个请求的靠前和靠后的若干段；组就是注意力 TP 组。
    qcp_size: int = 1
    qcp_rank: int = 0
    qcp_group: tuple[int, ...] = (0,)
    components: tuple[AttnComponentSpec, ...]

    def __post_init__(self):
        softmax_components = [
            c for c in self.components if isinstance(c, SoftmaxAttnConfig)
        ]
        if len(softmax_components) != 1:
            raise ValueError(
                "AttnConfig requires exactly one softmax-family component, got "
                f"{[type(c).__name__ for c in self.components] or 'none'}"
            )
        if self.kv_cache_dtype == torch.int8:
            raise ValueError("INT8 KV cache is not supported by the runtime")
        if self.qcp_size > 1:
            softmax = softmax_components[0]
            # DSA gathers paged history; V4.1 publishes its packed KV from
            # gathered projections and keeps full logical metadata.
            # 译：DSA 收集分页的历史；V4.1 从收集到的投影发布它打包的 KV，并保留完整的逻辑元数据。
            is_v41 = softmax.backend_name == "deepseek_v41"
            if not (
                is_v41 or (softmax.is_dsa and softmax.backend_name in (None, "dsa"))
            ):
                raise ValueError(
                    "query context parallelism requires GPU DSA or V4.1 attention; "
                    f"got {softmax.backend_name!r}"
                )
            if not is_v41 and self.qcp_size != self.kvp_size:
                raise ValueError(
                    "DSA query context parallelism requires KVP equal to QCP"
                )
            if torch.device(self.device).type != "cuda":
                raise ValueError("GPU query context parallelism requires CUDA")
            # The sharded KV write gathers the rotated latent and stores it
            # with latent_store, which writes native (bf16) rows only.
            # 译：分片的 KV 写入会收集旋转后的潜在向量，并用 latent_store 存储，而它只写原生（bf16）的行。
            if not is_v41 and (
                self.kv_cache_dtype is not torch.bfloat16
                or self.kv_cache_mxfp8
                or self.kv_cache_quant_method != "none"
            ):
                raise ValueError(
                    "query context parallelism requires a bf16 KV cache; got "
                    f"{self.kv_cache_dtype} (mxfp8={self.kv_cache_mxfp8}, quant "
                    f"method {self.kv_cache_quant_method!r})"
                )
        if self.kvp_size > 1:
            softmax = softmax_components[0]
            if self.is_draft and not softmax.is_latent_kv:
                # A block drafter of its own attention family (DFlash: a GQA
                # stack storing K/V head rows) shares the target's full-history
                # group (check_block_drafter_storage), so page-sharding that
                # group puts head-partitioned K/V inside a sharded group, which
                # PD refuses to transfer (ts-pd layout/route.rs). Refused here,
                # at startup, not per request once traffic arrives.
                # 译：使用自己注意力系列的块草稿器（DFlash：存储 K/V 头行的 GQA 堆栈）与目标模型共享全历史组
                #     （check_block_drafter_storage），所以对那个组做页分片会把按头切分的 K/V 放进一个分片组里，而 PD 拒绝传输这种组
                #     （ts-pd layout/route.rs）。在启动时就拒绝，而不是等流量到来后逐个请求地拒绝。
                raise ValueError(
                    "--kv-parallel-size > 1 page-shards the cache group a "
                    "block drafter shares with the target, but this draft "
                    "stores K/V head rows, which PD cannot transfer from a "
                    "page-sharded group; launch with --kv-parallel-size 1 "
                    "(and --query-context-parallel-size 1), or use a draft "
                    "that plans the target's own attention"
                )
            if softmax.backend_name == "flashmla":
                if torch.device(self.device).type != "cuda":
                    raise ValueError("FlashMLA DCP decode requires CUDA")
            elif softmax.backend_name == "tokenspeed_mla":
                pass
            elif softmax.is_dsa and softmax.backend_name in (None, "dsa"):
                device_type = torch.device(self.device).type
                if device_type not in ("cuda", "npu"):
                    raise ValueError("DSA DCP decode requires CUDA or Ascend NPU")
                if device_type == "cuda" and self.draft_block_decode:
                    raise ValueError(
                        "GPU DSA DCP decode does not yet support non-causal "
                        "block drafts"
                    )
            elif softmax.backend_name == "hybrid_linear_attn":
                # The registry resolves the user's full-attention leaf after
                # composing the hybrid components, and validates it before
                # cache allocation. The composite name is not a capability.
                # 译：registry 在组合混合组件之后才解析用户指定的全注意力叶子后端，并在分配 cache 之前校验它。这个组合名字本身不代表能力。
                pass
            elif softmax.backend_name == "deepseek_v41":
                if self.is_draft:
                    raise ValueError("V4.1 KVP requires the target attention backend")
            else:
                raise ValueError(
                    "--kv-parallel-size > 1 (DCP decode) currently requires "
                    "DSA, FlashMLA or CuTe MLA attention"
                )

    def component(self, cls: type[ComponentT]) -> ComponentT | None:
        """The first component that is a ``cls``, or None.

        The one generic accessor — ``config.component(LinearAttnConfig)``;
        new component classes need no change here.
        """
        # 译：第一个属于 ``cls`` 的组件，或 None。
        #
        #     唯一的通用访问方法——``config.component(LinearAttnConfig)``；新的组件类不需要修改这里。
        for component in self.components:
            if isinstance(component, cls):
                return component
        return None

    def cache_cell_size(self) -> int:
        return self.component(SoftmaxAttnConfig).cache_cell_size(self)


# 所有系列的 generate() 共用的 AttnConfig 字段。
# 其中 max_bs = 最大并发请求数 / attention DP 组数（每个 DP 组各自管理自己的请求）。
def model_wide_kwargs(
    server_args: ServerArgs,
    model_config: ModelConfig,
    is_draft: bool,
    *,
    kv_cache_dtype: torch.dtype,
    kv_cache_mxfp8: bool = False,
    draft_block_decode: bool = False,
    draft_block_causal: bool = False,
    speculative_num_draft_tokens: int | None = None,
    shards_query_context: bool = False,
) -> dict:
    """The AttnConfig fields shared by every family's generate().

    The softmax family passes in the facts its policies decide (cache dtype,
    block-decode mode, and — for families that bypass the DSpark width
    convention — the raw speculative width).

    ``shards_query_context`` says whether this family can compute a query
    context parallel shard (GPU DSA through the MLA config, and V4.1). It only
    affects a draft config; ``__post_init__`` refuses a target that cannot shard.
    """
    # 译：每个系列的 generate() 共用的 AttnConfig 字段。
    #
    #     softmax 系列传入由它的策略决定的参数（cache dtype、块 decode 模式，以及——对于绕过 DSpark 宽度约定的系列——
    #     原始的投机宽度）。
    #
    #     ``shards_query_context`` 说明该系列能否计算查询上下文并行的分片（经由 MLA 配置的 GPU DSA，以及 V4.1）。它只影响
    #     草稿模型的配置；``__post_init__`` 会拒绝不能分片的目标模型。
    kwargs = dict(
        device=server_args.device,
        dtype=model_config.dtype,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_mxfp8=kv_cache_mxfp8,
        kv_cache_quant_method=server_args.kv_cache_quant_method,
        context_len=model_config.context_len + server_args.spec_context_pad,
        max_bs=server_args.max_num_seqs
        // (server_args.data_parallel_size or server_args.mapping.attn.dp_size),
        pd_disaggregation_enabled=server_args.disaggregation_mode != "null",
        is_draft=is_draft,
        draft_block_decode=draft_block_decode,
        draft_block_causal=draft_block_causal,
    )
    attn_mapping = server_args.mapping.attn
    # QCP shards the target's extend rows. A draft takes that shard only when it
    # plans the target's own sharded attention (an MTP depth). A draft with its
    # own ordinary attention is handed the rows the target computed and runs
    # over them whole (the DFlash drafter builds its logits metadata with
    # ``query_shard=None``), so it is configured unsharded.
    # 译：QCP 对目标模型的 extend 行做分片。只有当草稿模型规划的是目标模型自己的分片注意力（某一 MTP 深度）时，它才使用
    #     这个分片。使用自己普通注意力的草稿模型拿到的是目标模型算出的行，并在这些完整的行上运行（DFlash 草稿器构建
    #     logits 元数据时使用 ``query_shard=None``），所以它被配置为不分片。
    draft_unsharded = is_draft and not shards_query_context
    kwargs.update(
        kvp_size=attn_mapping.kvp_size,
        kvp_rank=attn_mapping.kvp_rank,
        kvp_group=attn_mapping.kvp_group,
        qcp_size=1 if draft_unsharded else attn_mapping.qcp_size,
        qcp_rank=0 if draft_unsharded else attn_mapping.qcp_rank,
        qcp_group=(0,) if draft_unsharded else attn_mapping.qcp_group,
    )
    if server_args.speculative_algorithm is not None:
        kwargs.update(
            speculative_num_steps=server_args.speculative_num_steps,
            speculative_num_draft_tokens=(
                speculative_num_draft_tokens
                if speculative_num_draft_tokens is not None
                else resolve_speculative_num_tokens(server_args, is_draft)
            ),
        )
    return kwargs
