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

# 注意力组件的“装配厂”：启动时决定用哪个注意力后端、按什么 cache 规划分配 KV 显存，并把两者绑在一起。
#
# 你在这里：执行器启动时 execution/device.py 调用 create_attn_components（第 6 阶段），只调用一次
# （重建 arena 时可复用已有后端）。之后每次前向只用到它产出的 attn_backend 和 token_to_kv_pool。
#
# create_attn_components 的步骤（Qwen3-8B 单卡、--attention-backend 不指定）：
#   1. 解析模型“家族”（_resolve_attn_side）：按 architectures 判断是不是混合线性注意力、Inkling 等特殊模型
#      Qwen3ForCausalLM -> 普通模型，所有 is_* 都为 False
#   2. 必要时改写 server_args 中的后端名（_apply_backend_overrides，混合模型强制 hybrid_linear_attn）
#   3. 构造注意力配置 AttnConfig（_create_attn_config）：MHA 架构 -> MHAConfig.generate（见 configs/mha.py）
#   4. 选 cache recipe（_resolve_cache_family）：Qwen3 -> "mha"（普通家族）
#   5. 显存预算：profile_available_cache_memory_bytes 测出可用显存，减去 CUDA Graph 预留（utils.py）
#   6. recipe 规划 cache 布局（每个 cache group 的 block 大小、能放多少 block，第 14 阶段）
#   7. create_cache_arena 一次分配整块显存；create_cache_pool 为目标模型（和草稿模型）各建一个层窗口视图
#   8. 建后端：MHA 架构默认后端名 "mha" -> MHAAttnBackend 是分页叶子 -> 包成 CacheGroupRouter（create_paged_router）
#   9. backend.set_cache_pool(pool)：router 按 pool 发布的 cache group 建叶子
#   10. 返回 AttentionBuild（后端、pool、显存报告等）
#
# 本文件大部分代码是为了覆盖各种特殊模型（混合线性注意力、DSA、MSA、Inkling、DeepSeek V4.1、草稿模型、PP、KVP）；
# 主线读者只需看上面这条路径，以及 register_backend / _get_backend_cls / create_paged_router。
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.dots3_note_config import Dots3NoteConfig
from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    is_qwen4_exp,
)
from tokenspeed.runtime.configs.model_profile import ModelProfile
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import (
    DeepseekV41Config,
    is_deepseek_v41_config,
)
from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    Dots3NoteAttentionConfig,
)
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import (
    LinearAttnConfig,
    Mamba2Config,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import (
    MSAConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import (
    CachePool,
)
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ownership import (
    CacheLayerOwnership,
    cache_field_placement,
    pipeline_cache_ownership,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    CacheModelFamily,
    CachePoolSpec,
    cache_recipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    STATE_LAYER_TYPES,
)
from tokenspeed.runtime.layers.attention.utils import (
    check_probe_cache_fits,
    profile_available_cache_memory_bytes,
    reserve_cache_budget,
)

logger = logging.getLogger(__name__)

# 普通 cache 家族：没有专用 recipe、由通用（ordinary）recipe 规划的四种注意力。
_ORDINARY_CACHE_FAMILIES = frozenset({"mha", "mla", "dsa", "msa"})

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheMemoryPlan,
    )
    from tokenspeed.runtime.layers.attention.qsa.bf16_scratch import (
        QSABf16ScratchPlan,
    )
    from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclasses.dataclass(frozen=True)
class AttentionBuild:
    """Complete attention resources and cache placement for device assembly.

    Field placement and readiness are resolved from the same cache ownership
    used to select resident fields.
    ``logical_plan`` preserves the full PD wire layout when PP narrows the
    physical arena; it is None when the arena already holds the complete plan.
    Placement stays in the build result rather than on the allocation owner.
    """

    # 译：供设备装配使用的完整注意力资源和 cache 放置信息。
    #
    #     字段放置和就绪状态，由选择常驻字段时所用的同一份 cache 归属解析而来。
    #     PP 收窄物理 arena 时，``logical_plan`` 保留完整的 PD 传输布局；arena 已经持有完整规划时它为 None。
    #     放置信息保留在构建结果中，而不放在分配的拥有者上。

    # Resolved full-attention choices, including hybrid sub-backends. These
    # are startup compatibility facts; they do not expose backend internals.
    # 译：解析后的全注意力选择，包括混合模型的子后端。这些是启动时的兼容性事实；不暴露后端内部细节。
    # 主线关心的字段：attn_backend（Qwen3 为 CacheGroupRouter）、token_to_kv_pool（KV 池，见 kv_cache/mha.py）。
    # draft_* 在开启投机解码时才有值。
    attention_backend_name: str
    draft_attention_backend_name: str
    attn_backend: AttentionBackend
    token_to_kv_pool: CachePool
    draft_attn_backend: AttentionBackend | None
    draft_token_to_kv_pool: CachePool | None
    cache_storage: dict
    # The budget the memory profile gave before any CUDA-graph reserve.
    # 译：做任何 CUDA graph 预留之前，内存 profile 给出的预算。
    profiled_cache_bytes: int
    cache_fields_by_stage: tuple[tuple[str, ...], ...]
    producer_fields_by_step: tuple[tuple[str, ...], ...]
    logical_plan: CacheMemoryPlan | None


def _ordinary_cache_family(config: AttnConfig | None) -> CacheModelFamily | None:
    if config is None:
        return None
    softmax_attn = config.component(SoftmaxAttnConfig)
    if type(softmax_attn) is MHAConfig:
        return "mha"
    if type(softmax_attn) is MLAConfig:
        return "mla"
    if isinstance(softmax_attn, DSAConfig):
        return "dsa"
    if isinstance(softmax_attn, MSAConfig):
        return "msa"
    return None


def _resolve_heterogeneous_draft_family(
    target_family: CacheModelFamily,
    draft_family: CacheModelFamily | None,
    *,
    draft_family_declared: bool,
) -> CacheModelFamily | None:
    """Validate and return the supported heterogeneous draft family.

    ``None`` means the draft binds through the target family's own recipe —
    the pattern every custom family with an integrated draft view relies on
    (its recipe plans the draft layers). A draft whose PROFILE declares a
    different family is a layout contradiction, not that pattern, and is
    rejected instead of silently bound to the target's pool factory.
    """
    # 译：校验并返回受支持的异构草稿家族。
    #
    #     ``None`` 表示草稿通过目标家族自己的 recipe 绑定——这是每个带有集成草稿视图的自定义家族所依赖的模式
    #     （它的 recipe 规划草稿层）。PROFILE 声明了不同家族的草稿是布局上的矛盾，而不是这种模式，
    #     会被拒绝，而不是被悄悄绑定到目标模型的 pool 工厂。
    if draft_family is None or draft_family == target_family:
        return None
    if target_family == "kimi_k3":
        if draft_family != "mla":
            raise RuntimeError(
                "Kimi-K3 unified cache currently requires an ordinary MLA draft view"
            )
        return draft_family
    if target_family not in _ORDINARY_CACHE_FAMILIES:
        if draft_family_declared:
            raise RuntimeError(
                f"the draft declares cache family {draft_family!r}, but the "
                f"target family {target_family!r} has no heterogeneous draft "
                "views; a draft served from this pool must declare the "
                "target's family"
            )
        return None
    if draft_family != "mha":
        raise RuntimeError(
            "heterogeneous ordinary cache views currently require an MHA draft"
        )
    return draft_family


def _arena_allocated_bytes(arena) -> int:
    """Device bytes this model's cache owner actually occupies.

    Summing a pool's per-layer view sizes would answer a different question
    (and double-count aliased views), so read the owner directly.
    """
    # 译：本模型的 cache 拥有者实际占用的设备字节数。
    #
    #     对 pool 的逐层视图大小求和回答的是另一个问题（而且会把别名视图重复计算），所以直接读拥有者。
    return int(arena.allocated_device_bytes)


def _cache_storage_report(
    *,
    configured_cache_bytes: int,
    pool,
    fixed_workspace_bytes: int = 0,
) -> dict:
    """Describe cache storage from allocated tensors, not scheduler counts."""
    # 译：根据已分配的张量（而不是调度器的计数）描述 cache 存储。
    arena = pool.arena
    plan = arena.plan
    packing = {
        group.group_id: int(group.cache_blocks_per_lcm_block) for group in plan.groups
    }
    # The arena is the one definition of child-token capacity.
    # 译：arena 是子 token 容量的唯一定义。
    physical_token_capacity = int(arena.size)
    geometry = {
        "prefix_granularity": int(plan.prefix_granularity),
        "num_lcm_blocks": int(plan.num_lcm_blocks),
        "cache_blocks_per_lcm_block": packing,
        # Fraction of a parent each group's binding actually uses;
        # aliased slabs are sized by their widest tenant, so a narrow
        # binding strands the rest.
        # 译：每个组的绑定实际用到父块的比例；别名的 slab 按其最宽的租户确定大小，所以窄的绑定会浪费剩下的部分。
        "binding_utilization": {
            group_id: round(entry["binding_utilization"], 4)
            for group_id, entry in plan.capacity_report().items()
        },
    }

    # One arena: the draft view shares this allocation, so it already covers
    # both models' layers.
    # 译：只有一个 arena：草稿视图共享这次分配，所以它已经覆盖了两个模型的层。
    arena_bytes = _arena_allocated_bytes(arena)
    allocated_cache_bytes = arena_bytes + fixed_workspace_bytes
    storage = arena.storage_plan
    planned_peak_bytes = allocated_cache_bytes + storage.temporary_workspace_bytes
    if planned_peak_bytes > configured_cache_bytes:
        raise RuntimeError(
            "planned cache storage exceeds its profiled budget: "
            f"{planned_peak_bytes} > {configured_cache_bytes}"
        )
    return {
        "configured_cache_bytes": int(configured_cache_bytes),
        "allocated_cache_bytes": allocated_cache_bytes,
        "planned_peak_cache_bytes": planned_peak_bytes,
        "physical_token_capacity": physical_token_capacity,
        "capacity_source": "lcm_geometry",
        "geometry": geometry
        | {
            "arena_bytes": arena_bytes,
            "host_history_bytes": arena.allocated_host_bytes,
            "fixed_workspace_bytes": fixed_workspace_bytes,
            "offload_persistent_workspace_bytes": storage.persistent_workspace_bytes,
            "offload_temporary_workspace_bytes": storage.temporary_workspace_bytes,
        },
    }


# ---------- backend registry ----------
# 译：后端注册表

# Maps backend_name -> (supported archs, backend class)
# 译：映射 backend_name -> (支持的架构, 后端类)
# 各后端模块在导入时调用 register_backend 把自己登记进来，例如 paged/mha.py 末尾：
#   register_backend("mha" / "fa3" / "fa4" / "triton" / "flashinfer", {MHA}, MHAAttnBackend)
# paged/trtllm.py 末尾：register_backend("trtllm", {MHA}, TRTLLMMHAAttnBackend)。
# --attention-backend 的取值就是这里的键。
_BACKEND_REGISTRY: dict[str, tuple[set[AttentionArch], type[AttentionBackend]]] = {}


def register_backend(
    name: str,
    archs: set[AttentionArch],
    cls: type[AttentionBackend],
) -> None:
    _BACKEND_REGISTRY[name] = (archs, cls)


# The composite name the hybrid linear-attention wrapper runs under; it
# selects the wrapper, never a registered leaf.
# 译：混合线性注意力包装器运行时使用的复合名称；它选择的是包装器，而不是某个已注册的叶子。
HYBRID_LINEAR_ATTN_BACKEND = "hybrid_linear_attn"


def validate_attention_backend_name(name: str | None, *, flag: str) -> None:
    """Reject a launch backend name no in-tree or plugin backend registered.

    Runs after plugin discovery, so plugin backends are accepted exactly like
    in-tree ones.
    """
    # 译：拒绝没有任何树内或插件后端注册过的启动后端名。
    #
    #     在插件发现之后运行，所以插件后端与树内后端被同等接受。
    if name is None or name == HYBRID_LINEAR_ATTN_BACKEND:
        return
    if name not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown {flag} {name!r}; available: "
            f"{sorted([*_BACKEND_REGISTRY, HYBRID_LINEAR_ATTN_BACKEND])}"
        )


# 以下架构名集合用于识别特殊模型家族（第 13 阶段混合模型等），Qwen3 不在其中。
_HYBRID_GDN_ARCHITECTURES = {
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForConditionalGenerationNextN",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForConditionalGenerationNextN",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForCausalLMNextN",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForCausalLMNextN",
}
# Hybrid models whose linear layers are Mamba2 (SSD) and full-attention layers MHA.
# 译：线性层为 Mamba2（SSD）、全注意力层为 MHA 的混合模型。
_HYBRID_MAMBA2_ARCHITECTURES = {
    "NemotronHForCausalLM",
    "NemotronHForCausalLMNextN",
    "NemotronH_Nano_Omni_Reasoning_V3",
    "NemotronH_Omni_Reasoning_V3",
    "NemotronH_Omni_Reasoning_V3NextN",
}
# Hybrid linear-attention models whose full-attention layers are MLA (not MHA)
# and whose linear layers are KDA (per-channel gated delta rule), not GDN.
# They share the same HybridLinearAttnBackend wrapper and cache-group pool;
# the base sub-backend auto-resolves to MLA from the arch, and the linear
# sub-backend runs the KDA kernels (KdaAttnBackend).
# 译：混合线性注意力模型，其全注意力层是 MLA（不是 MHA），线性层是 KDA（逐通道门控 delta 规则），不是 GDN。
#     它们共用同一个 HybridLinearAttnBackend 包装器和 cache group pool；基础子后端根据架构自动解析为 MLA，
#     线性子后端运行 KDA kernel（KdaAttnBackend）。
_HYBRID_MLA_KDA_ARCHITECTURES = {
    "KimiK3ForConditionalGeneration",
}
# Kimi-K3.1: the K3 layout with DSA sparse MLA on its full-attention layers.
# 译：Kimi-K3.1：K3 的布局，全注意力层使用 DSA 稀疏 MLA。
_HYBRID_KIMI_DSA_KDA_ARCHITECTURES = {
    "KimiK31ForConditionalGeneration",
}
_HYBRID_DSA_KDA_TARGET_ARCHITECTURES = {
    "Glm53FlashForConditionalGeneration",
}
_HYBRID_DSA_KDA_ARCHITECTURES = {
    *_HYBRID_DSA_KDA_TARGET_ARCHITECTURES,
    "Glm53FlashForConditionalGenerationNextN",
}

# Inkling stays on the MHA path plus its thin sconv wrapper; it is not hybrid-GDN.
# 译：Inkling 留在 MHA 路径上，外加它那层很薄的 sconv 包装；它不是混合 GDN。
_INKLING_ARCHITECTURES = {
    "InklingForConditionalGeneration",
    "InklingForConditionalGenerationNextN",
}

_DSPARK_DRAFT_ARCHITECTURES = frozenset({"DeepseekV41ForCausalLMDSpark"})


@dataclasses.dataclass(frozen=True)
class _AttnSideProfile:
    """Architecture-derived family facts for one side (target or draft).

    Everything here resolves from the model's ``hf_config`` before any
    attention config exists, so the target and draft sides share one
    derivation instead of two interleaved copies.
    """

    # 译：由架构推导出的、某一侧（目标或草稿）的家族事实。
    #
    #     这里的一切都在任何注意力配置存在之前从模型的 ``hf_config`` 解析出来，所以目标侧和草稿侧共用一次推导，
    #     而不是两份交织的副本。

    architectures: tuple[str, ...]
    requested_backend: str | None
    is_hybrid_gdn: bool
    is_mamba2: bool
    is_kda: bool
    # KDA hybrid whose full-attention layers are DSA (GLM-5.3-Flash); a
    # subset of ``is_kda`` that selects the DSA history consumer and the
    # glm53_flash cache family.
    # 译：全注意力层为 DSA 的 KDA 混合模型（GLM-5.3-Flash）；是 ``is_kda`` 的子集，选择 DSA 历史消费者和 glm53_flash cache 家族。
    is_dsa_kda: bool
    # Kimi-K3.1: subset of is_kda that selects only the DSA history consumer.
    # 译：Kimi-K3.1：is_kda 的子集，只选择 DSA 历史消费者。
    is_kimi_dsa_kda: bool
    is_qwen4_exp: bool
    is_qsa: bool
    is_inkling: bool
    is_dspark: bool
    # The model's own profile, when it declares one; the facts above are then
    # all False and the profile answers for the model.
    # 译：模型自己的 profile（若它声明了一个）；此时上面的事实全为 False，由 profile 替模型回答。
    model_profile: ModelProfile | None

    @property
    def linear_attention(self) -> str | None:
        """The registered linear-attention backend of the model's linear
        layers, or None for a model without them."""
        # 译：模型线性层所用的已注册线性注意力后端；没有线性层的模型返回 None。
        if self.model_profile is not None:
            return self.model_profile.linear_attention
        if self.is_kda:
            return "kda"
        if self.is_hybrid_gdn:
            return "gdn"
        if self.is_mamba2:
            return "mamba2"
        return None

    @property
    def is_hybrid_linear(self) -> bool:
        # GDN and KDA both take the hybrid-linear path; they differ only in
        # the linear kernel (GDN scalar decay vs KDA per-channel) and the
        # base attn arch (MHA vs MLA vs DSA).
        # 译：GDN 和 KDA 都走混合线性路径；它们只在线性 kernel（GDN 标量衰减 vs KDA 逐通道）和
        #     基础注意力架构（MHA vs MLA vs DSA）上不同。
        return self.linear_attention is not None


def _resolve_attn_side(
    model_config: ModelConfig, requested_backend: str | None
) -> _AttnSideProfile:
    hf_config = model_config.hf_config
    architectures = getattr(hf_config, "architectures", None) or []
    if model_config.model_profile is not None:
        return _AttnSideProfile(
            architectures=tuple(architectures),
            requested_backend=requested_backend,
            is_hybrid_gdn=False,
            is_mamba2=False,
            is_kda=False,
            is_dsa_kda=False,
            is_kimi_dsa_kda=False,
            is_qwen4_exp=False,
            is_qsa=False,
            is_inkling=False,
            is_dspark=False,
            model_profile=model_config.model_profile,
        )
    text_config = getattr(hf_config, "text_config", hf_config)
    qwen4_exp = is_qwen4_exp(hf_config)
    is_dspark = any(a in _DSPARK_DRAFT_ARCHITECTURES for a in architectures)
    is_dsa_kda = any(a in _HYBRID_DSA_KDA_ARCHITECTURES for a in architectures)
    is_kimi_dsa_kda = any(
        a in _HYBRID_KIMI_DSA_KDA_ARCHITECTURES for a in architectures
    )
    return _AttnSideProfile(
        architectures=tuple(architectures),
        requested_backend=requested_backend,
        is_hybrid_gdn=any(a in _HYBRID_GDN_ARCHITECTURES for a in architectures),
        is_mamba2=any(a in _HYBRID_MAMBA2_ARCHITECTURES for a in architectures),
        is_kda=is_dsa_kda
        or is_kimi_dsa_kda
        or any(a in _HYBRID_MLA_KDA_ARCHITECTURES for a in architectures),
        is_dsa_kda=is_dsa_kda,
        is_kimi_dsa_kda=is_kimi_dsa_kda,
        is_qwen4_exp=qwen4_exp,
        is_qsa=qwen4_exp and getattr(text_config, "indexer_n_heads", None) is not None,
        is_inkling=any(a in _INKLING_ARCHITECTURES for a in architectures),
        is_dspark=is_dspark,
        model_profile=None,
    )


def _check_pd_support(
    server_args: ServerArgs,
    target: _AttnSideProfile,
    draft: _AttnSideProfile | None,
    *,
    has_draft_model: bool,
) -> None:
    """Every disaggregated-serving support gate, raised up front."""
    # 译：所有分离式服务（PD 分离）的支持性检查，提前报错。
    if server_args.disaggregation_mode not in ("prefill", "decode"):
        return
    if (target.is_inkling or (draft is not None and draft.is_inkling)) and (
        has_draft_model or server_args.speculative_algorithm is not None
    ):
        raise NotImplementedError(
            "Inkling PD supports target-only decoding; speculative/draft "
            "ShortConv checkpoint transfer is not implemented"
        )


def _apply_backend_overrides(
    server_args: ServerArgs,
    target: _AttnSideProfile,
    draft: _AttnSideProfile | None,
) -> None:
    """The one place family resolution writes back into ``server_args``.

    The mutation is deliberate, not a shortcut: ``_create_attn_config`` reads
    the backend choice through the generate() protocol. Must run before any
    ``_create_attn_config`` call. The user's pre-override choice survives
    as ``profile.requested_backend``.
    """
    # 译：家族解析写回 ``server_args`` 的唯一地方。
    #
    #     这个修改是有意为之，不是捷径：``_create_attn_config`` 通过 generate() 协议读取后端选择。必须在任何
    #     ``_create_attn_config`` 调用之前运行。用户在覆盖之前的选择保存在 ``profile.requested_backend`` 中。
    if "DeepseekV41ForCausalLM" in target.architectures:
        server_args.attention_backend = "deepseek_v41"

    if target.is_hybrid_linear:
        # GDN (Qwen3.5) / KDA (Kimi-K3) hybrid models always need
        # hybrid_linear_attn. The user's original choice stays in the profile
        # for the full-attention sub-backend (MHA for GDN, MLA for KDA).
        # 译：GDN（Qwen3.5）/ KDA（Kimi-K3）混合模型总是需要 hybrid_linear_attn。用户原来的选择保存在 profile 中，
        #     供全注意力子后端使用（GDN 用 MHA，KDA 用 MLA）。
        server_args.attention_backend = HYBRID_LINEAR_ATTN_BACKEND
        if (
            draft is not None
            and target.is_kda
            and not target.is_dsa_kda
            and server_args.kv_parallel_size > 1
            and server_args.drafter_attention_backend
            in (None, HYBRID_LINEAR_ATTN_BACKEND)
        ):
            # A K3 continuation must resolve its history consumer before
            # AttnConfig validates its DCP decode. Inherit the target's resolved leaf,
            # while preserving an explicitly requested draft leaf.
            # 译：K3 的延续层（continuation）必须在 AttnConfig 校验其 DCP decode 之前解析出历史消费者。继承目标模型
            #     已解析的叶子，同时保留显式请求的草稿叶子。
            server_args.drafter_attention_backend = _resolve_hybrid_full_backend_name(
                target.requested_backend,
                is_kda=True,
                is_dsa=False,
                is_qsa=False,
            )
    elif server_args.attention_backend == HYBRID_LINEAR_ATTN_BACKEND:
        logger.warning(
            "Ignoring hybrid_linear_attn backend for non-hybrid model architectures="
            f"{target.architectures!s}",
        )
        server_args.attention_backend = None
        if server_args.drafter_attention_backend == HYBRID_LINEAR_ATTN_BACKEND:
            logger.warning(
                "Ignoring hybrid_linear_attn backend for non-hybrid model "
                f"architectures={(draft.architectures if draft is not None else ())!s}",
            )
            server_args.drafter_attention_backend = None


def _resolve_full_attn_backend_name(
    profile: _AttnSideProfile, softmax_attn, hybrid_request: str | None
) -> str:
    """The name the full-attention layers run on (the hybrid sub-backend,
    or the config's own resolution)."""
    # 译：全注意力层运行所用的名字（混合模型的子后端，或配置自己的解析结果）。
    if profile.is_hybrid_linear:
        if profile.model_profile is not None:
            # The launch choice, already defaulted from the profile; None
            # selects the architecture's default leaf.
            # 译：启动时的选择，已经按 profile 给了默认值；None 选择该架构的默认叶子。
            return (
                None if hybrid_request == HYBRID_LINEAR_ATTN_BACKEND else hybrid_request
            )
        return _resolve_hybrid_full_backend_name(
            hybrid_request,
            is_kda=profile.is_kda,
            is_dsa=profile.is_dsa_kda or profile.is_kimi_dsa_kda,
            is_qsa=profile.is_qsa,
        )
    return softmax_attn.backend_name


def _cache_backend_name(
    softmax_attn: SoftmaxAttnConfig,
    full_attn_backend_name: str | None,
    arch: AttentionArch,
) -> str:
    """Identify the cache producer, including MSA's dense sub-backend.

    This is startup compatibility metadata only. MSA still constructs its
    dense/sparse routers from the original config, on the common path.
    """
    # 译：标识 cache 的生产者，包括 MSA 的稠密子后端。
    #
    #     这只是启动时的兼容性元数据。MSA 仍在公共路径上，从原始配置构造它的稠密 / 稀疏 router。
    name = full_attn_backend_name or _get_default_backend_name(arch)
    if isinstance(softmax_attn, MSAConfig):
        dense_name = softmax_attn.full_attn_backend_name or "mha"
        return f"{name}:{dense_name}"
    return name


def _has_state_layers(config: AttnConfig) -> bool:
    """The plan actually carries recurrent state (hybrid arch + state labels)."""
    # 译：规划中确实带有循环状态（混合架构 + 状态标签）。
    if config.component(LinearAttnConfig) is None:
        return False
    return any(
        layer_type in STATE_LAYER_TYPES
        for layer_type in config.component(SoftmaxAttnConfig).cache_layer_types
    )


def _resolve_cache_family(
    profile: _AttnSideProfile,
    config: AttnConfig,
) -> CacheModelFamily:
    """The one dispatch from family facts (plus built config) to the recipe."""
    # 译：从家族事实（加上已构建的配置）到 recipe 的唯一分派。
    if profile.model_profile is not None and profile.model_profile.cache_family:
        return profile.model_profile.cache_family
    if config.component(Dots3NoteAttentionConfig) is not None:
        return "dots3_note"
    if config.component(DeepseekV41Config) is not None:
        return "deepseek_v41"
    # PLE and QSA are cache consumers even in a view without GDN layers.
    # 译：即使在没有 GDN 层的视图中，PLE 和 QSA 也是 cache 消费者。
    if profile.is_qwen4_exp:
        return "qwen4_exp"
    if profile.is_hybrid_gdn and _has_state_layers(config):
        return "qwen_gdn"
    if profile.is_mamba2 and _has_state_layers(config):
        return "mamba2"
    if profile.is_dsa_kda:
        return "glm53_flash"
    if profile.is_kda:
        return "kimi_k3"
    if profile.is_inkling:
        return "inkling"
    family = _ordinary_cache_family(config)
    if family is None:
        raise RuntimeError(
            "No cache recipe is registered for "
            f"attention config {type(config.component(SoftmaxAttnConfig)).__name__}"
        )
    return family


# 后端名为 None（用户未指定）时按注意力架构取默认后端名。Qwen3 是 MHA 架构 -> "mha" -> MHAAttnBackend。
def _get_default_backend_name(arch: AttentionArch) -> str:
    if arch == AttentionArch.MLA:
        return "mla"
    if arch == AttentionArch.DSA:
        return "dsa"
    if arch == AttentionArch.MSA:
        return "msa"
    else:
        return "mha"


def _get_backend_cls(name: str, arch: AttentionArch) -> type[AttentionBackend]:
    if name is None:
        entry = _BACKEND_REGISTRY.get(_get_default_backend_name(arch))
        if entry is not None and arch in entry[0]:
            return entry[1]
        raise ValueError(
            f"No backend supports arch {arch}. Available: {list(_BACKEND_REGISTRY)}"
        )
    entry = _BACKEND_REGISTRY.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown attention backend: {name!r}. Available: {list(_BACKEND_REGISTRY)}"
        )
    supported_archs, cls = entry
    if arch not in supported_archs:
        raise ValueError(
            f"Backend {name!r} does not support arch {arch}. "
            f"Supported archs: {supported_archs}"
        )
    return cls


def create_paged_router(
    config: AttnConfig,
    arch: AttentionArch,
    *,
    backend_name: str | None = None,
) -> AttentionBackend:
    """Build the CacheGroupRouter for one side's paged attention.

    The router builds one ``PagedAttentionBackend`` leaf per paged
    (history) cache group of the pool view bound later via
    ``set_cache_pool``; each leaf's kernel page size resolves from the
    config override, the leaf class default, or the group's own block
    granularity (``PagedAttentionBackend.resolve_kernel_page_size``).
    """
    # 译：为某一侧的分页注意力构建 CacheGroupRouter。
    #
    #     router 为之后通过 ``set_cache_pool`` 绑定的 pool 视图中的每个分页（历史）cache group 构建一个
    #     ``PagedAttentionBackend`` 叶子；每个叶子的 kernel 页大小依次取自配置覆盖、叶子类默认值或组自己的 block 粒度
    #     （``PagedAttentionBackend.resolve_kernel_page_size``）。
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )

    spec = config.component(SoftmaxAttnConfig)
    if isinstance(spec, Dots3NoteAttentionConfig):
        from tokenspeed.runtime.layers.attention.backends.dots3_note import (
            create_dots3_note_router,
        )

        if backend_name not in (None, "dsa"):
            raise ValueError("dots3-note requires the DSA/SWA paged router")
        return create_dots3_note_router(config)
    name = backend_name if backend_name is not None else spec.backend_name
    if name == "hybrid_linear_attn":
        # The composite sentinel _apply_backend_overrides writes into
        # server_args (and MHAConfig.generate copies into the spec). It
        # names the WRAPPER; the leaf under it auto-resolves from the arch.
        # 译：这是 _apply_backend_overrides 写进 server_args 的复合哨兵值（MHAConfig.generate 会把它拷进 spec）。
        #     它命名的是“包装器”；包装器下面的叶子根据架构自动解析。
        name = None
    leaf_cls = _get_backend_cls(name, arch)

    # leaf_factory 此时只是定义，router 在 set_cache_pool 时才为每个 cache group 调用它（延迟构造）。
    # 例：Qwen3 一个 group，block 粒度 64 -> MHA 叶子 kernel 页大小 64；trtllm 叶子则固定为 64。
    def leaf_factory(group_id: str, block_granularity: int):
        del group_id
        kernel_page_size = leaf_cls.resolve_kernel_page_size(config, block_granularity)
        # A fresh spec, never a mutate-restore of the shared component: leaf
        # construction happens lazily at set_cache_pool, and several leaves
        # interpret backend_name themselves (MHA/MLA kernel-solution maps),
        # so a wrapper-selecting name like 'dsa' must not reach them.
        # 译：用一个全新的 spec，绝不对共享组件做“修改后再恢复”：叶子构造在 set_cache_pool 时才延迟发生，而且有几个叶子
        #     会自己解释 backend_name（MHA/MLA 的 kernel 实现映射），所以像 'dsa' 这种选择包装器的名字不能传到它们那里。
        leaf_spec = dataclasses.replace(spec, backend_name=name)
        return leaf_cls(config, leaf_spec, kernel_page_size=kernel_page_size)

    return CacheGroupRouter(
        leaf_factory,
        is_draft=bool(config.is_draft),
        spec_num_tokens=config.speculative_num_draft_tokens or 1,
        device=config.device,
        consumed_group_ids=(FULL_ATTENTION,) if name == "qsa" else None,
        query_sharded=config.qcp_size > 1,
    )


def _pin_fixed_kernel_page(
    config: AttnConfig,
    model_config: ModelConfig,
    backend_name: str | None,
) -> AttnConfig:
    """Pin a fixed-page leaf's kernel page on an ordinary side's config.

    The ordinary recipe makes one block span its prefix granularity and sizes
    that span to hold whole kernel pages (``OrdinaryRecipe.prefix_granularity``),
    so the page a leaf fixes must be visible before the plan. The leaf then
    resolves the same page from the pinned value. A flexible leaf runs at the
    block span and pins nothing; MSA builds a dense and a sparse leaf, and its
    recipe adds the sparse page itself.
    """
    # 译：在普通侧的配置上固定一个“固定页大小叶子”的 kernel 页。
    #
    #     普通 recipe 让一个 block 跨越其前缀粒度，并把这个跨度定为能容纳整数个 kernel 页（``OrdinaryRecipe.prefix_granularity``），
    #     所以叶子固定的页大小必须在规划之前可见。随后叶子从固定的值解析出同样的页大小。页大小灵活的叶子以 block 跨度运行，
    #     不固定任何东西；MSA 构建一个稠密叶子和一个稀疏叶子，其 recipe 自己加上稀疏的页大小。
    # 例：--attention-backend trtllm -> 叶子类默认页 64 -> 把 config.kernel_page_size 设为 64，
    # recipe 据此保证 block 粒度是 64 的整数倍。默认的 mha 后端 default_kernel_page_size=None，不做任何事。
    from tokenspeed.runtime.layers.attention.backends.paged.base import (
        PagedAttentionBackend,
    )

    if config.kernel_page_size is not None or isinstance(
        config.component(SoftmaxAttnConfig), MSAConfig
    ):
        return config
    leaf_cls = _get_backend_cls(backend_name, model_config.attention_arch)
    if (
        not issubclass(leaf_cls, PagedAttentionBackend)
        or leaf_cls.default_kernel_page_size is None
    ):
        return config
    return dataclasses.replace(
        config, kernel_page_size=int(leaf_cls.default_kernel_page_size)
    )


def _validate_lcm_page_size(
    config: AttnConfig,
    *,
    prefix_granularity: int,
) -> None:
    """Require the scheduler page to contain whole configured kernel pages.

    An unset kernel_page_size means the backend resolves its registry
    default itself and owns the divisibility check for it.
    """
    # 译：要求调度器页包含整数个配置的 kernel 页。
    #
    #     未设置 kernel_page_size 表示后端自己解析其 registry 默认值，并自己负责整除检查。
    if config.kernel_page_size is None:
        return
    kernel_page_size = int(config.kernel_page_size)
    if (
        prefix_granularity <= 0
        or kernel_page_size <= 0
        or prefix_granularity % kernel_page_size
    ):
        raise ValueError(
            "prefix granularity must be a positive multiple of kernel page "
            f"size, got {prefix_granularity} and {kernel_page_size}"
        )


# ---------- arch -> config class ----------
# 译：架构 -> 配置类

_CONFIG_CLS: dict[AttentionArch, type[SoftmaxAttnConfig]] = {
    AttentionArch.MHA: MHAConfig,
    AttentionArch.MLA: MLAConfig,
    AttentionArch.DSA: DSAConfig,
    AttentionArch.MSA: MSAConfig,
}

# Architectures declaring a linear-attention component, registered like
# _CONFIG_CLS. Whether a given checkpoint actually has linear layers is
# decided by generate() (NextN drafts may carry none).
# 译：声明了线性注意力组件的架构，像 _CONFIG_CLS 一样注册。某个 checkpoint 是否真的有线性层由 generate() 决定
#     （NextN 草稿可能一层都没有）。
_LINEAR_ATTN_CLS: dict[str, type[LinearAttnConfig]] = {
    arch: LinearAttnConfig
    for arch in (
        *_HYBRID_GDN_ARCHITECTURES,
        *_HYBRID_MLA_KDA_ARCHITECTURES,
        *_HYBRID_KIMI_DSA_KDA_ARCHITECTURES,
        # GLM NextN is one DSA layer. It reuses the target's mixed-layer
        # metadata but must not acquire a linear-attention component.
        # 译：GLM 的 NextN 是一个 DSA 层。它复用目标模型的混合层元数据，但不能获得线性注意力组件。
        *_HYBRID_DSA_KDA_TARGET_ARCHITECTURES,
    )
} | {arch: Mamba2Config for arch in _HYBRID_MAMBA2_ARCHITECTURES}


def _create_attn_config(
    server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
) -> AttnConfig:
    # 例：Qwen3-8B 无 model_profile、不是 Dots3 / DeepSeek V4.1 -> MHAConfig.generate(...)，
    # 得到只含一个 MHAConfig 组件的 AttnConfig（字段见 configs/base.py、configs/mha.py）；没有线性注意力组件。
    arch = model_config.attention_arch
    if arch not in _CONFIG_CLS:
        raise NotImplementedError(f"Not supported Attention Arch: {arch!r}")
    profile = model_config.model_profile
    if profile is not None and profile.create_attention_config is not None:
        config = profile.create_attention_config(server_args, model_config, is_draft)
        if not isinstance(config, AttnConfig):
            raise TypeError(
                "ModelProfile.create_attention_config must return AttnConfig"
            )
        if config.is_draft != is_draft:
            raise ValueError(
                "ModelProfile.create_attention_config must build the config for "
                f"is_draft={is_draft}, got is_draft={config.is_draft}"
            )
        if config.component(LinearAttnConfig) is not None:
            raise ValueError(
                "ModelProfile.create_attention_config must leave linear attention "
                "composition to the runtime"
            )
    else:
        if isinstance(model_config.hf_config, Dots3NoteConfig):
            config_cls = Dots3NoteAttentionConfig
        elif is_deepseek_v41_config(model_config.hf_config):
            config_cls = DeepseekV41Config
        else:
            config_cls = _CONFIG_CLS[arch]
        config = config_cls.generate(server_args, model_config, is_draft)
    # Extra components are built through the same generate() protocol and
    # composed into config.components (consumers look them up by class via
    # ``component()``).
    # 译：额外的组件通过同样的 generate() 协议构建，并组合进 config.components（消费者通过 ``component()`` 按类查找）。
    linear_attn = _linear_attn_component(server_args, model_config, is_draft)
    if linear_attn is not None:
        config = dataclasses.replace(
            config, components=config.components + (linear_attn,)
        )
    return config


def _linear_attn_component(
    server_args: ServerArgs, model_config: ModelConfig, is_draft: bool
) -> LinearAttnConfig | None:
    """The linear-attention component, or None for a model without one.

    A plugin profile that declares ``linear_attention`` must get it: serving
    the model through full attention alone would be silently wrong. The
    in-tree schema is the one linear-config reader today, so the checkpoint
    config must expose ``linear_layer_ids`` and the geometry fields
    ``LinearAttnConfig.generate`` reads.
    """
    # 译：线性注意力组件；没有线性注意力的模型返回 None。
    #
    #     声明了 ``linear_attention`` 的插件 profile 必须得到它：只用全注意力来服务这个模型会悄悄出错。
    #     目前树内的 schema 是唯一的线性配置读取者，所以 checkpoint 配置必须暴露 ``linear_layer_ids`` 以及
    #     ``LinearAttnConfig.generate`` 读取的几何字段。
    if model_config.model_profile is not None:
        profile_linear = model_config.model_profile.linear_attention
        if profile_linear is None:
            return None
        linear_attn = LinearAttnConfig.generate(server_args, model_config, is_draft)
        if linear_attn is None:
            raise ValueError(
                f"model profile declares linear_attention={profile_linear!r} "
                "but the checkpoint config exposes no linear_layer_ids; "
                "declare the linear geometry LinearAttnConfig reads, or drop "
                "linear_attention from the profile"
            )
        return linear_attn
    architectures = getattr(model_config.hf_config, "architectures", None) or ()
    linear_cls = next(
        (_LINEAR_ATTN_CLS[a] for a in architectures if a in _LINEAR_ATTN_CLS),
        None,
    )
    if linear_cls is None:
        return None
    return linear_cls.generate(server_args, model_config, is_draft)


def _create_attn_backend(
    arch: AttentionArch,
    config: AttnConfig,
) -> AttentionBackend:
    return _create_attn_backend_with_name(
        config.component(SoftmaxAttnConfig).backend_name, arch, config
    )


def _create_attn_backend_with_name(
    name: str | None,
    arch: AttentionArch,
    config: AttnConfig,
) -> AttentionBackend:
    from tokenspeed.runtime.layers.attention.backends.paged.base import (
        PagedAttentionBackend,
    )

    # 关键分支：注册的后端类若是分页叶子（PagedAttentionBackend 子类，如 MHA / trtllm），runner 拿到的不是叶子本身，
    # 而是包着叶子的 CacheGroupRouter；其他（非分页的复合后端）直接构造。
    cls = _get_backend_cls(name, arch)
    if issubclass(cls, PagedAttentionBackend):
        # Paged leaves are served through the cache-group router: one leaf
        # per history group, blocks -> kernel pages mapped in one place.
        # 译：分页叶子通过 cache group router 提供服务：每个历史组一个叶子，block -> kernel 页只在一个地方映射。
        return create_paged_router(config, arch, backend_name=name)
    spec = dataclasses.replace(
        config.component(SoftmaxAttnConfig),
        backend_name=name,
    )
    return cls(config, spec)


def _resolve_hybrid_full_backend_name(
    requested_name: str | None,
    *,
    is_kda: bool,
    is_dsa: bool,
    is_qsa: bool,
) -> str | None:
    """Resolve the compute backend that consumes the hybrid history cache."""
    # 译：解析消费混合历史 cache 的计算后端。
    name = None if requested_name == HYBRID_LINEAR_ATTN_BACKEND else requested_name
    if is_qsa:
        if name is not None:
            logger.warning(
                "Qwen4-Exp QSA pins its sparse dispatch to the qsa backend; "
                f"ignoring explicit attention backend {requested_name!r}",
            )
        return "qsa"
    if is_dsa and name is None:
        return "dsa"
    # NVIDIA K3 defaults to its CuteDSL history consumer. AMD keeps the
    # generic MLA backend; explicit user choices remain authoritative.
    # 译：NVIDIA 上的 K3 默认使用其 CuteDSL 历史消费者。AMD 保留通用的 MLA 后端；用户的显式选择始终优先。
    if is_kda and name is None and not current_platform().is_amd:
        return "tokenspeed_mla"
    return name


# 以下到 _create_hybrid_linear_attn_backend 为混合线性注意力模型专用（第 13 阶段），初读可跳过。
def _kda_linear_attn_backend(
    server_args: ServerArgs, config: AttnConfig
) -> AttentionBackend:
    from tokenspeed.runtime.layers.attention.backends.state.kda import (
        KdaAttnBackend,
        resolve_kda_backend,
    )

    return KdaAttnBackend(
        config,
        config.component(SoftmaxAttnConfig),
        enable_prefill_graph=not server_args.disable_kda_prefill_graph,
        kda_backend=resolve_kda_backend(server_args.kda_backend.strip().lower()),
    )


def _gdn_linear_attn_backend(
    server_args: ServerArgs, config: AttnConfig
) -> AttentionBackend:
    from tokenspeed.runtime.layers.attention.backends.state.gdn import (
        GdnAttnBackend,
    )

    return GdnAttnBackend(
        config,
        config.component(SoftmaxAttnConfig),
        writes_state_checkpoints_in_forward=(
            server_args.writes_state_checkpoints_in_forward
        ),
    )


def _mamba2_linear_attn_backend(
    server_args: ServerArgs, config: AttnConfig
) -> AttentionBackend:
    del server_args
    from tokenspeed.runtime.layers.attention.backends.state.mamba2 import (
        Mamba2AttnBackend,
    )

    return Mamba2AttnBackend(config, config.component(SoftmaxAttnConfig))


# Linear-attention backends by name: ``factory(server_args, config)`` builds
# the backend a hybrid model's linear layers run on. Plugins add entries via
# ``tokenspeed.runtime.plugins.registry.register_linear_attention_backend``.
# 译：按名字索引的线性注意力后端：``factory(server_args, config)`` 构建混合模型线性层运行所用的后端。
#     插件通过 ``tokenspeed.runtime.plugins.registry.register_linear_attention_backend`` 添加条目。
_LINEAR_ATTN_BACKENDS: dict[
    str, Callable[[ServerArgs, AttnConfig], AttentionBackend]
] = {
    "kda": _kda_linear_attn_backend,
    "gdn": _gdn_linear_attn_backend,
    "mamba2": _mamba2_linear_attn_backend,
}


def _create_hybrid_linear_attn_backend(
    server_args: ServerArgs,
    model_config: ModelConfig,
    config: AttnConfig,
    *,
    pool,
    full_attn_backend_name: str | None,
    linear_attention: str,
) -> AttentionBackend:
    """Create a hybrid backend for a linear-attention model.

    ``linear_attention`` names the registered backend of the linear layers:
    ``gdn`` (Qwen3.5, MHA base), ``kda`` (Kimi-K3, MLA base; GLM-5.3-Flash,
    DSA base) or a plugin's. Both sub-backends bind to the one shared cache
    pool through the wrapper's ``set_cache_pool``. ``pool`` is inspected here
    only to select the consumers local to this model view.
    """
    # 译：为线性注意力模型创建混合后端。
    #
    #     ``linear_attention`` 命名线性层的已注册后端：``gdn``（Qwen3.5，MHA 基础）、``kda``（Kimi-K3，MLA 基础；
    #     GLM-5.3-Flash，DSA 基础）或某个插件的。两个子后端都通过包装器的 ``set_cache_pool`` 绑定到同一个共享 cache pool。
    #     这里检查 ``pool`` 只是为了选出本模型视图本地的消费者。
    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )

    linear_factory = _LINEAR_ATTN_BACKENDS.get(linear_attention)
    if linear_factory is None:
        raise ValueError(
            f"Unknown linear attention backend {linear_attention!r}; available: "
            f"{sorted(_LINEAR_ATTN_BACKENDS)}"
        )
    hf_config = model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    full_attn_layers = text_config.full_attention_layer_ids
    # The paged full-attention router (MHA, MLA or DSA leaves by arch): the
    # user's original choice if provided, otherwise auto-selected.
    # 译：分页的全注意力 router（按架构为 MHA、MLA 或 DSA 叶子）：若用户提供了原始选择就用它，否则自动选择。
    full_attn_backend = _create_attn_backend_with_name(
        full_attn_backend_name,
        model_config.attention_arch,
        config,
    )

    # Create mamba/linear attention backend. Only propagate the configured
    # verify width when spec-dec is actually enabled — matches MLAConfig /
    # MHAConfig.generate. Otherwise the AttnConfig sentinel (1) wins so
    # non-spec hybrid decode doesn't get misclassified as target verify /
    # draft extend by `self.spec_num_tokens > 1`.
    # 译：创建 mamba / 线性注意力后端。只有真正开启投机解码时才传递配置的验证宽度——与 MLAConfig / MHAConfig.generate 一致。
    #     否则 AttnConfig 的哨兵值（1）生效，这样非投机的混合 decode 不会因为 `self.spec_num_tokens > 1`
    #     被误判为目标验证 / 草稿 extend。
    if server_args.speculative_algorithm is not None:
        if server_args.disaggregation_mode == "prefill":
            # The prefill role only executes committed prefill state. A KDA
            # backend starts allocating replay payloads in set_kv_pool, before
            # explicit workspace preparation, so its verify width must already
            # be one. Keep the original target/draft configs intact for
            # logical cache geometry and other backend views.
            # 译：prefill 角色只执行已提交的 prefill 状态。KDA 后端在 set_kv_pool 中、在显式准备工作区之前就开始分配重放负载，
            #     所以它的验证宽度必须已经是 1。保持原始的目标 / 草稿配置不变，供逻辑 cache 几何和其他后端视图使用。
            config = dataclasses.replace(config, speculative_num_draft_tokens=1)
        else:
            config.speculative_num_draft_tokens = (
                server_args.speculative_num_draft_tokens
            )

    # The linear component's presence decides whether this model actually
    # has any linear / mamba layers. A draft model on a hybrid-GDN target
    # (e.g. MTP on Qwen3.5) shares the same architecture class as the
    # target but commonly ships with *zero* mamba layers; such a view has no
    # state groups to consume, so the router alone serves it.
    # 译：线性组件是否存在决定本模型是否真的有线性 / mamba 层。混合 GDN 目标模型上的草稿模型（例如 Qwen3.5 上的 MTP）
    #     与目标模型共用同一个架构类，但通常一个 mamba 层都没有；这样的视图没有可消费的状态组，所以只由 router 服务。
    linear_attn = config.component(LinearAttnConfig)

    if linear_attn is None or not pool.state_group_by_layer:
        logger.info(
            f"Created hybrid_linear_attn backend: {len(full_attn_layers):d} full attn "
            "layers, 0 linear "
            "attn layers in this cache view (skipping linear backend)",
        )
        backend = full_attn_backend
    else:
        linear_attn_backend = linear_factory(server_args, config)
        backend = HybridLinearAttnBackend(
            full_attn_backend, linear_attn_backend, full_attn_layers
        )
        logger.info(
            f"Created hybrid_linear_attn backend: {len(full_attn_layers):d} full attn "
            f"layers, {len(linear_attn.layer_ids):d} linear attn layers, "
            f"{'LCM state fields'!s}",
        )
    if is_qwen4_exp(hf_config):
        from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
            Qwen4ExpBackend,
        )

        backend = Qwen4ExpBackend.from_cache_view(config, pool, backend)
    return backend


def _create_target_components(
    *,
    server_args,
    model_config,
    config,
    cache_spec: CachePoolSpec,
    arena: CacheArena,
    rank: int,
    full_attn_backend_name: str | None,
    linear_attention: str | None,
    is_inkling: bool,
    backend: AttentionBackend | None,
):
    """The target's compute view onto the shared arena + target backend.

    ``backend`` reuses an existing one, for a caller that is building a
    replacement arena for backends it already owns.
    """
    # 译：目标模型在共享 arena 上的计算视图 + 目标后端。
    #
    #     ``backend`` 复用一个已有的后端，供那些为自己已拥有的后端构建替换 arena 的调用方使用。
    # The arena owns every planned field; this view binds only the target
    # model's layer window.
    # 译：arena 拥有每个规划的字段；这个视图只绑定目标模型的层窗口。
    pool = create_cache_pool(
        cache_spec,
        config,
        arena,
        num_layers=len(cache_spec.layer_types),
        rank=rank,
    )
    if backend is not None:
        return backend, pool
    if linear_attention is not None:
        backend = _create_hybrid_linear_attn_backend(
            server_args,
            model_config,
            config,
            pool=pool,
            full_attn_backend_name=full_attn_backend_name,
            linear_attention=linear_attention,
        )
        return backend, pool

    # 普通模型（Qwen3）走这里：_create_attn_backend -> _create_attn_backend_with_name -> create_paged_router。
    backend = _create_attn_backend(model_config.attention_arch, config)
    if not is_inkling:
        return backend, pool

    from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
        InklingAttnBackend,
    )

    text_config = model_config.hf_config.get_text_config()
    backend = InklingAttnBackend.from_config(
        backend,
        text_config,
        config,
        num_layers=text_config.num_hidden_layers,
        is_draft=False,
        enable_layerwise_cache_ready=(
            server_args.disaggregation_mode == "prefill"
            and server_args.disaggregation_layerwise_interval > 0
        ),
    )
    return backend, pool


def _create_draft_components(
    *,
    server_args,
    model_config,
    config,
    pool,
    cache_spec: CachePoolSpec,
    num_target_layers: int,
    full_attn_backend_name: str | None,
    is_heterogeneous: bool,
    linear_attention: str | None,
    is_inkling: bool,
    backend: AttentionBackend | None,
):
    """Draft backend + the ONE arena viewed through the draft's layer window.

    One big model, one arena: draft layers are continuation layers of the
    merged plan, so the draft pool is a second compute view whose
    ``field_layer_offset`` places its LOCAL layer ids (a NextN draft's one
    layer is layer 0) onto the continuation range. Nothing remaps ids on the
    way through -- the view *is* the mapping, and an id outside the window is
    rejected rather than silently offset onto another model's planes.
    """
    # 译：草稿后端 + 通过草稿层窗口来看的“唯一一个” arena。
    #
    #     一个大模型，一个 arena：草稿层是合并规划中的延续层，所以草稿 pool 是第二个计算视图，它的 ``field_layer_offset``
    #     把它的“本地”层 id（NextN 草稿的那一层是第 0 层）放到延续范围上。一路上没有任何地方重映射 id——视图“就是”映射，
    #     窗口之外的 id 会被拒绝，而不是被悄悄偏移到另一个模型的平面上。
    # 例：Qwen3-8B（36 层）+ 1 层的 NextN 草稿：合并规划共 37 层，草稿视图 field_layer_offset=36，
    # 草稿的第 0 层对应 arena 中的第 36 层字段。
    if config is None:
        return None, None
    if is_heterogeneous and (linear_attention is not None or is_inkling):
        raise RuntimeError(
            "heterogeneous cache views currently support ordinary drafts only"
        )
    num_layers = model_config.num_attention_layers
    # The draft view's transfer counter stays local/None; heterogeneous PD is
    # rejected before construction.
    # 译：草稿视图的传输计数器保持本地 / None；异构 PD 在构造之前就被拒绝。
    draft_pool = create_cache_pool(
        cache_spec,
        config,
        pool.arena,
        num_layers=num_layers,
        rank=pool.rank,
        field_layer_offset=num_target_layers,
    )
    if backend is not None:
        return backend, draft_pool
    if linear_attention is not None:
        backend = _create_hybrid_linear_attn_backend(
            server_args,
            model_config,
            config,
            pool=draft_pool,
            full_attn_backend_name=full_attn_backend_name,
            linear_attention=linear_attention,
        )
        return backend, draft_pool

    backend = _create_attn_backend(model_config.attention_arch, config)
    if is_inkling:
        from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
            InklingAttnBackend,
        )

        # Depth layers carry conv checkpoint fields as continuation tenants
        # of the target's kvconv/hiddenconv groups; the draft gets the same
        # paged bridges (publish/restore) the target wrapper gets.
        # 译：深度层把卷积检查点字段作为目标模型 kvconv/hiddenconv 组的延续租户携带；草稿得到与目标包装器相同的分页桥接（发布 / 恢复）。
        text_config = model_config.hf_config.get_text_config()
        backend = InklingAttnBackend.from_config(
            backend,
            text_config,
            config,
            num_layers=num_layers,
            is_draft=True,
            enable_layerwise_cache_ready=False,
        )
    return backend, draft_pool


def _prepare_fixed_workspaces(
    *,
    server_args,
    config,
    backend,
    draft_backend,
    uses_paged_state_verify: bool,
    is_inkling: bool,
    qsa_bf16_scratch: QSABf16ScratchPlan | None,
    expected_bytes: int,
) -> None:
    """Allocate the fixed workspaces the recipe reserved from the cache budget
    (``CacheRecipe.workspace_bytes``) and check the bytes agree.

    The verify workspaces of the paged-state families, Qwen4-Exp's QSA BF16
    prefill scratch (``qsa_bf16_scratch``, the target's) and the
    query-context-parallel history gather workspace of GPU DSA: the target
    tree allocates the latter once and the draft tree gathers into the same
    buffers (the draft's extend step follows the target's forward on the
    same stream), so the plan budgets one workspace.
    """
    # 译：分配 recipe 从 cache 预算中预留的固定工作区（``CacheRecipe.workspace_bytes``），并检查字节数一致。
    #
    #     包括分页状态家族的验证工作区、Qwen4-Exp 的 QSA BF16 prefill 暂存区（``qsa_bf16_scratch``，目标模型的）以及
    #     GPU DSA 的 query-context 并行历史收集工作区：目标树分配后者一次，草稿树收集到同样的缓冲区中（草稿的 extend 步
    #     在同一个流上紧跟目标模型的前向），所以规划只为一个工作区做预算。
    # Qwen3（普通 MHA、无 QCP）在这里什么都不分配，直接返回。
    from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
        Qwen4ExpBackend,
        qwen4_exp_backend,
    )

    width = int(server_args.speculative_num_draft_tokens or 1)
    allocated = False
    actual_bytes = 0
    if isinstance(backend, Qwen4ExpBackend) or (
        uses_paged_state_verify and expected_bytes
    ):
        actual_bytes += backend.preallocate_verify_workspace(config.max_bs, width)
        allocated = True
    elif is_inkling:
        actual_bytes += backend.fixed_workspace_bytes()
        if draft_backend is not None:
            actual_bytes += draft_backend.fixed_workspace_bytes()
        allocated = True
    if qsa_bf16_scratch is not None:
        actual_bytes += qwen4_exp_backend(backend).preallocate_qsa_bf16_scratch(
            qsa_bf16_scratch
        )
        allocated = True
    # V4.1 allocates its gathered projections and per-request packed history
    # per forward; it has no DSA history gather workspace to reserve.
    # 译：V4.1 每次前向分配它收集的投影和逐请求打包的历史；它没有需要预留的 DSA 历史收集工作区。
    if config.qcp_size > 1 and config.component(DeepseekV41Config) is None:
        actual_bytes += backend.preallocate_history_gather_workspace(config.context_len)
        allocated = True
        if draft_backend is not None:
            workspace = backend.history_gather_workspace()
            if workspace is None:
                raise RuntimeError(
                    "query context parallelism needs the target's history gather "
                    "workspace for the draft to share"
                )
            draft_backend.adopt_history_gather_workspace(workspace)
    if not allocated:
        return
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            "planned fixed workspace does not match allocated tensors: "
            f"{expected_bytes} planned, {actual_bytes} allocated"
        )


# ---------- public API ----------
# 译：公共 API
def _narrow_spec_for_pp(
    spec: CachePoolSpec, ownership: CacheLayerOwnership
) -> CachePoolSpec:
    """Chunk-pipeline stage: physically allocate only this stage's layers'
    planes. The logical geometry (parents, packing, page math) stays the
    full model's so every rank's scheduler plans identically. The caller
    retains the complete plan separately for the PD wire contract.
    """
    # 译：分块流水线阶段：只为本阶段各层的平面做物理分配。逻辑几何（父块、打包方式、页计算）保持为完整模型的，
    #     这样每个 rank 的调度器规划都一致。调用方另外保留完整规划，用于 PD 传输契约。
    stage_start, stage_end = ownership.resident_cache_window
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.storage import (
        plan_cache_storage,
    )

    memory_plan = spec.memory_plan.narrow_to_layers(stage_start, stage_end)
    offload = None if spec.storage_plan is None else spec.storage_plan.offload
    return dataclasses.replace(
        spec,
        memory_plan=memory_plan,
        storage_plan=plan_cache_storage(
            memory_plan, spec.cache_group_specs, offload=offload
        ),
    )


def _validate_mla_dcp_backend(name: str | None, arch: AttentionArch) -> None:
    backend_cls = _get_backend_cls(name, arch)
    if not backend_cls.supports_mla_dcp:
        raise ValueError(
            f"Attention backend {name or _get_default_backend_name(arch)!r} "
            "does not support MLA DCP "
            "(supports_mla_dcp=False)"
        )


def _validate_hybrid_kvp_cache(spec: CachePoolSpec, *, kvp_size: int) -> None:
    """Validate declared storage, independent of recipe name or inheritance."""
    # 译：校验声明的存储，与 recipe 名字或继承关系无关。
    groups = spec.cache_group_specs
    if not any(group.attn_kind == "full_history" for group in groups):
        raise ValueError("Hybrid MLA KVP requires a full-history cache group")
    for group in groups:
        if group.attn_kind == "sliding_window":
            raise ValueError(
                f"Hybrid MLA KVP cache group {group.group_id!r} "
                "must be a full-history group"
            )
        expected = kvp_size if group.attn_kind != "state" else 1
        if group.shard_count != expected:
            raise ValueError(
                f"Hybrid MLA KVP cache group {group.group_id!r} ({group.attn_kind}) "
                f"requires shard_count={expected}, got {group.shard_count}"
            )


def create_attn_components(
    server_args: ServerArgs,
    model_config: ModelConfig,
    gpu_id: int,
    rank: int,
    gpu_memory: int,
    enable_memory_saver: bool = False,
    draft_model_config: ModelConfig | None = None,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    *,
    graph_reserve_bytes: int,
    post_profile_bytes: int,
    probe_batch_rows: int | None,
    profiled_cache_bytes: int | None,
    reuse_target_backend: AttentionBackend | None,
    reuse_draft_backend: AttentionBackend | None,
) -> AttentionBuild:
    """Build attention resources and return their explicit cache placement."""
    # 译：构建注意力资源，并返回它们显式的 cache 放置信息。
    if probe_batch_rows is not None and graph_reserve_bytes:
        # An arena sized by block count ignores the budget, so it cannot honour one.
        # 译：按 block 数确定大小的 arena 会忽略预算，所以无法满足预算约束。
        raise ValueError(
            f"a probe arena of {probe_batch_rows} blocks cannot also "
            f"reserve {graph_reserve_bytes} bytes: the two size the pool by "
            "different rules and only one of them can be applied"
        )
    validate_attention_backend_name(
        server_args.attention_backend, flag="--attention-backend"
    )
    validate_attention_backend_name(
        server_args.drafter_attention_backend, flag="--drafter-attention-backend"
    )
    target = _resolve_attn_side(model_config, server_args.attention_backend)
    draft = (
        _resolve_attn_side(draft_model_config, server_args.drafter_attention_backend)
        if draft_model_config is not None
        else None
    )
    _check_pd_support(
        server_args, target, draft, has_draft_model=draft_model_config is not None
    )
    _apply_backend_overrides(server_args, target, draft)

    # 步骤 3、4：构造 AttnConfig，选 cache recipe 家族，并确定全注意力层用的后端名
    # （Qwen3 未指定时为 None，之后按架构取默认 "mha"）。
    config = _create_attn_config(server_args, model_config)
    softmax_attn = config.component(SoftmaxAttnConfig)
    cache_family = _resolve_cache_family(target, config)
    target_full_attn_backend_name = _resolve_full_attn_backend_name(
        target, softmax_attn, hybrid_request=target.requested_backend
    )
    if (
        config.kvp_size > 1
        # V4.1 validates KVP in its config and specialized recipe.
        # 译：V4.1 在自己的配置和专用 recipe 中校验 KVP。
        and softmax_attn.backend_name != "deepseek_v41"
        and (
            model_config.attention_arch == AttentionArch.MLA or target.is_hybrid_linear
        )
    ):
        _validate_mla_dcp_backend(
            target_full_attn_backend_name, model_config.attention_arch
        )
    if config.kvp_size > 1 and target.is_hybrid_linear:
        resolved_softmax = dataclasses.replace(
            softmax_attn, backend_name=target_full_attn_backend_name
        )
        config = dataclasses.replace(
            config,
            components=tuple(
                resolved_softmax if component is softmax_attn else component
                for component in config.components
            ),
        )
        softmax_attn = resolved_softmax
    draft_attn_config = (
        _create_attn_config(server_args, draft_model_config, is_draft=True)
        if draft is not None and not draft.is_dspark
        else None
    )
    draft_softmax_attn = (
        draft_attn_config.component(SoftmaxAttnConfig)
        if draft_attn_config is not None
        else None
    )
    draft_full_attn_backend_name = (
        # The draft's hybrid sub-backend request is its config's own
        # resolution, not the user's target choice.
        # 译：草稿的混合子后端请求是它自己配置的解析结果，而不是用户对目标模型的选择。
        _resolve_full_attn_backend_name(
            draft, draft_softmax_attn, hybrid_request=draft_softmax_attn.backend_name
        )
        if draft_attn_config is not None
        else None
    )
    # A draft model's own profile is authoritative for its cache layout. One that
    # leaves cache_family None declares nothing: it binds like an undeclared
    # draft, so the same draft serves an ordinary target (as an MHA view) and a
    # custom family (through that family's own recipe).
    # 译：草稿模型自己的 profile 对其 cache 布局有决定权。cache_family 为 None 的 profile 什么都不声明：它像未声明的草稿一样绑定，
    #     所以同一个草稿既可以服务普通目标模型（作为 MHA 视图），也可以服务自定义家族（通过该家族自己的 recipe）。
    draft_profile = (
        draft_model_config.model_profile if draft_attn_config is not None else None
    )
    draft_family_declared = (
        draft_profile is not None and draft_profile.cache_family is not None
    )
    if draft_family_declared:
        draft_cache_family = draft_profile.cache_family
    else:
        draft_cache_family = _ordinary_cache_family(draft_attn_config)
    heterogeneous_draft_family = _resolve_heterogeneous_draft_family(
        cache_family,
        draft_cache_family,
        draft_family_declared=draft_family_declared,
    )
    # One profile per boot, where a boot without a reserve takes it; a rebuild reuses it.
    # 译：每次启动 profile 一次，由不带预留的那次启动来做；重建时复用它。
    # 步骤 5：显存预算。profile_available_cache_memory_bytes 测出加载权重后还能给 KV cache 用的字节数，
    # reserve_cache_budget 再扣掉 CUDA Graph 预留（见 utils.py）。
    if profiled_cache_bytes is None:
        profiled_cache_bytes = profile_available_cache_memory_bytes(
            attn_config=config,
            gpu_id=gpu_id,
            tp_size=server_args.mapping.world_size,
            gpu_memory_utilization=server_args.gpu_memory_utilization,
            total_gpu_memory=gpu_memory,
            post_profile_bytes=post_profile_bytes,
            world_group=server_args.mapping.world_group,
        )
    cache_memory = reserve_cache_budget(profiled_cache_bytes, graph_reserve_bytes)
    if cache_family in _ORDINARY_CACHE_FAMILIES:
        config = _pin_fixed_kernel_page(
            config, model_config, target_full_attn_backend_name
        )
        if draft_attn_config is not None:
            draft_attn_config = _pin_fixed_kernel_page(
                draft_attn_config, draft_model_config, draft_full_attn_backend_name
            )
    # 步骤 6：recipe 规划 cache 布局。Qwen3 走普通 MHA recipe：一个 full_history 组，
    # 每 token 每层 4096 字节（8 头 × 128 × 2（K、V）× 2 字节），36 层共 144 KiB / token。
    recipe = cache_recipe(
        cache_family,
        server_args=server_args,
        model_config=model_config,
        attn_config=config,
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=cache_memory,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
        probe_batch_rows=probe_batch_rows,
    )
    # Reserved from the budget by the recipe, allocated below; a family
    # without QSA layers refuses --enable-qsa-prefill-bf16-scratch here.
    # 译：由 recipe 从预算中预留、在下面分配；没有 QSA 层的家族在这里拒绝 --enable-qsa-prefill-bf16-scratch。
    qsa_bf16_scratch = recipe.qsa_bf16_scratch_plan
    num_draft_cache_layers = recipe.num_draft_layers
    num_target_cache_layers = len(recipe.layer_types) - num_draft_cache_layers
    from tokenspeed.runtime.distributed.pp_stage import pp_target_cache_windows

    owners = pipeline_cache_ownership(
        num_target_cache_layers,
        num_draft_cache_layers,
        pp_target_cache_windows(
            server_args.mapping, model_config, num_target_cache_layers
        ),
    )
    layer_ownership = owners[server_args.mapping.pp_rank]
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.capacity import (
        agree_cache_capacity,
    )

    capacity_group = (
        pg_manager.get_process_group("gloo", server_args.mapping.world_group)
        if server_args.mapping.world_size > 1
        else None
    )
    cache_setup = recipe.setup_for_layers(
        resident_cache_window=layer_ownership.resident_cache_window,
        synchronize_capacity=partial(
            agree_cache_capacity,
            cpu_group=capacity_group,
            probe=probe_batch_rows is not None,
        ),
    )
    if probe_batch_rows is not None:
        check_probe_cache_fits(cache_setup.cache_budget_bytes, profiled_cache_bytes)
    spec = cache_setup.spec
    if config.kvp_size > 1 and target.is_hybrid_linear:
        _validate_hybrid_kvp_cache(spec, kvp_size=config.kvp_size)
    cache_fields_by_stage, stage_schedules = cache_field_placement(
        spec.memory_plan, owners
    )
    target_spec = spec
    draft_view_spec = None
    if num_draft_cache_layers:
        # Transfer fields need one owner even when target and draft share a
        # cache family, so both compute views use disjoint layer windows.
        # 译：传输字段即使在目标与草稿共享一个 cache 家族时也需要唯一的拥有者，所以两个计算视图使用不相交的层窗口。
        target_spec = spec.layer_view(
            first_layer=0,
            num_layers=num_target_cache_layers,
        )
        draft_view_spec = spec.layer_view(
            first_layer=num_target_cache_layers,
            num_layers=num_draft_cache_layers,
            family=heterogeneous_draft_family,
        )
    prefix_granularity = spec.memory_plan.prefix_granularity
    _validate_lcm_page_size(
        config,
        prefix_granularity=prefix_granularity,
    )
    if draft_attn_config is not None:
        _validate_lcm_page_size(
            draft_attn_config,
            prefix_granularity=prefix_granularity,
        )
    cache_budget_bytes = cache_setup.cache_budget_bytes
    fixed_workspace_bytes = cache_setup.fixed_workspace_bytes
    # A probe arena is not the served geometry; it must not read as one.
    # 译：探测用的 arena 不是实际服务的几何，不能被当成实际几何（日志级别降为 DEBUG）。
    logger.log(
        logging.DEBUG if probe_batch_rows is not None else logging.INFO,
        f"Cache profile: parent_bytes={spec.memory_plan.lcm_block_bytes:d}, P="
        f"{spec.memory_plan.prefix_granularity:d}, parents="
        f"{spec.memory_plan.num_lcm_blocks:d}, token_capacity={spec.token_capacity:d}, "
        f"layers={len(spec.layer_types):d} (draft {num_draft_cache_layers:d}), "
        "groups="
        f"{ {group.group_id: group.cache_blocks_per_lcm_block for group in spec.memory_plan.groups}!s}",
    )

    # One model, one arena: the merged plan's single allocation, which every
    # compute view below (target, draft) is a layer window onto.
    # 译：一个模型，一个 arena：合并规划的单次分配，下面的每个计算视图（目标、草稿）都是它上面的一个层窗口。
    logical_plan = None
    if server_args.mapping.has_pp:
        logical_plan = spec.memory_plan
        spec = _narrow_spec_for_pp(spec, layer_ownership)
        target_spec = dataclasses.replace(target_spec, memory_plan=spec.memory_plan)
        if draft_view_spec is not None:
            draft_view_spec = dataclasses.replace(
                draft_view_spec, memory_plan=spec.memory_plan
            )
    logger.log(
        logging.DEBUG if probe_batch_rows is not None else logging.INFO,
        f"Cache resident: pp_rank={server_args.mapping.pp_rank}, "
        f"window={layer_ownership.resident_cache_window}, "
        f"parent_bytes={spec.memory_plan.arena_bytes // (spec.memory_plan.num_lcm_blocks + 1)}, "
        f"parents={spec.memory_plan.num_lcm_blocks}, "
        f"local_limit={cache_setup.local_parent_limit}, "
        f"agreed_limit={cache_setup.agreed_parent_limit}, "
        f"arena_bytes={spec.memory_plan.arena_bytes}, "
        f"fixed_workspace_bytes={fixed_workspace_bytes}, "
        f"budget_bytes={cache_budget_bytes}",
    )
    # 步骤 7、8：一次分配 arena，再为目标模型（和草稿模型）建 pool 视图与后端。
    arena = create_cache_arena(
        spec,
        device=config.device,
        enable_memory_saver=enable_memory_saver,
    )
    backend, pool = _create_target_components(
        backend=reuse_target_backend,
        server_args=server_args,
        model_config=model_config,
        config=config,
        cache_spec=target_spec,
        arena=arena,
        rank=rank,
        full_attn_backend_name=target_full_attn_backend_name,
        linear_attention=target.linear_attention,
        is_inkling=target.is_inkling,
    )
    draft_attn_backend, draft_pool = _create_draft_components(
        backend=reuse_draft_backend,
        server_args=server_args,
        model_config=draft_model_config,
        config=(
            draft_attn_config
            if not server_args.mapping.has_pp or layer_ownership.owns_draft_cache
            else None
        ),
        pool=pool,
        cache_spec=draft_view_spec,
        num_target_layers=num_target_cache_layers,
        full_attn_backend_name=draft_full_attn_backend_name,
        is_heterogeneous=heterogeneous_draft_family is not None,
        linear_attention=draft.linear_attention if draft is not None else None,
        is_inkling=draft is not None and draft.is_inkling,
    )

    # Bind the pools before CUDA-graph state allocation: backends learn
    # their group geometry (and buffer sizing) from the pool's published
    # specs. Every LCM pool publishes a cache contract, so there is no
    # separate contract-marking step.
    # 译：在分配 CUDA graph 状态之前绑定 pool：后端从 pool 发布的规格中得知其组几何（以及缓冲区尺寸）。
    #     每个 LCM pool 都发布 cache 契约，所以没有单独的契约标记步骤。
    for side_backend, side_pool in ((backend, pool), (draft_attn_backend, draft_pool)):
        if side_backend is None or side_pool is None:
            continue
        side_backend.set_cache_pool(side_pool)

    _prepare_fixed_workspaces(
        server_args=server_args,
        config=config,
        backend=backend,
        draft_backend=draft_attn_backend,
        uses_paged_state_verify=cache_setup.uses_paged_state_verify,
        is_inkling=cache_family == "inkling",
        qsa_bf16_scratch=qsa_bf16_scratch,
        expected_bytes=fixed_workspace_bytes,
    )

    cache_storage = _cache_storage_report(
        configured_cache_bytes=cache_budget_bytes,
        pool=pool,
        fixed_workspace_bytes=fixed_workspace_bytes,
    )

    return AttentionBuild(
        attention_backend_name=_cache_backend_name(
            softmax_attn, target_full_attn_backend_name, model_config.attention_arch
        ),
        draft_attention_backend_name=(
            _cache_backend_name(
                draft_softmax_attn,
                draft_full_attn_backend_name,
                draft_model_config.attention_arch,
            )
            if draft_attn_backend is not None
            else ""
        ),
        attn_backend=backend,
        token_to_kv_pool=pool,
        draft_attn_backend=draft_attn_backend,
        draft_token_to_kv_pool=draft_pool,
        cache_storage=cache_storage,
        profiled_cache_bytes=profiled_cache_bytes,
        cache_fields_by_stage=cache_fields_by_stage,
        producer_fields_by_step=stage_schedules[server_args.mapping.pp_rank],
        logical_plan=logical_plan,
    )
