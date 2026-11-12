"""Concrete cache-pool construction from a prepared cache spec."""

# 译：根据准备好的 cache spec 构造具体的 cache 池。

# 创建 arena 和 KV 池的工厂（由 layers/attention/registry.py 在启动时调用）。
#   create_cache_arena：按 cache 规划的结果（CachePoolSpec）分配 arena，一个模型一个
#   create_cache_pool：按“cache 家族”（spec.family）选择 KV 池的类，在 arena 上创建视图
# 家族与 KV 池的对应见 _POOL_FACTORIES。Qwen3 的家族是 "mha"，创建 MHATokenToKVPool（bf16）。
from collections.abc import Callable

from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.dsa import (
    DSAConfig,
    dsa_kernel_page_size,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import CachePoolSpec


def create_cache_arena(
    spec: CachePoolSpec,
    *,
    device: str,
    enable_memory_saver: bool,
) -> CacheArena:
    """Allocate the one arena every compute view of this spec shares."""
    # 译：分配本 spec 的所有计算视图共享的那一个 arena。
    return CacheArena(
        spec.memory_plan,
        device,
        cache_group_specs=spec.cache_group_specs,
        token_capacity=spec.token_capacity,
        enable_memory_saver=enable_memory_saver,
        storage_plan=spec.storage_plan,
    )


def _mha_pool_class(family: str, *, mxfp8: bool) -> type[CachePool]:
    """The MHA-shaped pool for one family, in its plain or mxfp8 variant.

    All supported families take the same arguments; only the recurrent-state
    aliasing (and the scale planes) differ, which is the class's business, not
    the caller's.
    """
    # 译：某个家族使用的 MHA 形状的池，普通版或 mxfp8 版。
    #
    #     所有支持的家族接受相同的参数；不同之处只在于递归状态的共用（以及 scale 平面），这是类自己的事，而不是调用方的事。
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_inkling import (
        HybridInklingTokenToKVPool,
        HybridInklingTokenToKVPoolMXFP8,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
        HybridMHATokenToKVPool,
        HybridMHATokenToKVPoolMXFP8,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPool,
        MHATokenToKVPoolMXFP8,
    )

    by_family = {
        "mha": (MHATokenToKVPool, MHATokenToKVPoolMXFP8),
        "inkling": (HybridInklingTokenToKVPool, HybridInklingTokenToKVPoolMXFP8),
        "qwen_gdn": (HybridMHATokenToKVPool, HybridMHATokenToKVPoolMXFP8),
        "qwen4_exp": (HybridMHATokenToKVPool, HybridMHATokenToKVPoolMXFP8),
        "mamba2": (HybridMHATokenToKVPool, HybridMHATokenToKVPoolMXFP8),
    }
    plain, scaled = by_family[family]
    return scaled if mxfp8 else plain


def _softmax_config(config: AttnConfig, family: str, expected: type):
    softmax_attn = config.component(SoftmaxAttnConfig)
    if not isinstance(softmax_attn, expected):
        raise TypeError(
            f"cache family {family!r} is incompatible with "
            f"{type(softmax_attn).__name__}"
        )
    return softmax_attn


def _glm53_flash_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int,
) -> CachePool:
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_glm53_flash import (
        HybridGlm53FlashTokenToKVPool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.glm53_flash import (
        Glm53FlashPoolOptions,
    )

    options = spec.pool_options
    if not isinstance(options, Glm53FlashPoolOptions):
        raise TypeError("GLM-5.3-Flash cache spec is missing pool options")
    softmax_attn = config.component(SoftmaxAttnConfig)
    if not isinstance(softmax_attn, DSAConfig):
        raise TypeError("GLM-5.3-Flash cache requires DSA attention")
    if options.index_head_dim != softmax_attn.index_head_dim:
        raise ValueError("GLM-5.3-Flash cache spec and attention config disagree")

    return HybridGlm53FlashTokenToKVPool(
        arena=arena,
        dtype=config.kv_cache_dtype,
        model_dtype=config.dtype,
        quant_method=config.kv_cache_quant_method,
        kv_lora_rank=softmax_attn.kv_lora_rank,
        qk_rope_head_dim=softmax_attn.qk_rope_head_dim,
        layer_num=num_layers,
        rank=rank,
        pool_options=options,
        layer_types=spec.layer_types,
        field_layer_offset=field_layer_offset,
    )


def _deepseek_v41_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int,
) -> CachePool:
    del spec, config
    from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v41 import (
        DeepseekV41CachePool,
    )

    return DeepseekV41CachePool(
        arena=arena,
        layer_num=num_layers,
        rank=rank,
        field_layer_offset=field_layer_offset,
    )


def _dsa_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int,
) -> CachePool:
    from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool

    softmax_attn = _softmax_config(config, spec.family, DSAConfig)
    return DSATokenToKVPool(
        arena,
        dtype=config.kv_cache_dtype,
        model_dtype=config.dtype,
        quant_method=config.kv_cache_quant_method,
        kv_lora_rank=softmax_attn.kv_lora_rank,
        qk_rope_head_dim=softmax_attn.qk_rope_head_dim,
        layer_num=num_layers,
        rank=rank,
        index_head_dim=softmax_attn.index_head_dim,
        field_layer_offset=field_layer_offset,
    )


def _dots3_note_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int,
) -> CachePool:
    from tokenspeed.runtime.layers.attention.kv_cache.dots3_note import (
        Dots3NoteTokenToKVPool,
    )

    softmax_attn = _softmax_config(config, spec.family, MLAConfig)
    return Dots3NoteTokenToKVPool(
        arena,
        dtype=config.kv_cache_dtype,
        model_dtype=config.dtype,
        quant_method=config.kv_cache_quant_method,
        kv_lora_rank=softmax_attn.kv_lora_rank,
        qk_rope_head_dim=softmax_attn.qk_rope_head_dim,
        layer_num=num_layers,
        rank=rank,
        index_head_dim=128,
        field_layer_offset=field_layer_offset,
    )


def _msa_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int,
) -> CachePool:
    from tokenspeed.runtime.layers.attention.kv_cache.msa import MSATokenToKVPool

    softmax_attn = _softmax_config(config, spec.family, MSAConfig)
    return MSATokenToKVPool(
        arena=arena,
        dtype=config.kv_cache_dtype,
        head_num=max(softmax_attn.num_kv_heads // softmax_attn.attn_tp_size, 1),
        head_dim=softmax_attn.head_dim,
        layer_num=num_layers,
        rank=rank,
        field_layer_offset=field_layer_offset,
    )


# MHA 形状的池：按 TP 切分后的 KV 头数（至少 1）、head_dim、层数，以及异构头数信息。
# Qwen3-8B、TP=1：head_num = 8 // 1 = 8，head_dim = 128，layer_num = 36
def _mha_shaped_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int,
) -> CachePool:
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
        HybridMHATokenToKVPool,
    )

    softmax_attn = _softmax_config(config, spec.family, MHAConfig)
    pool_cls = _mha_pool_class(spec.family, mxfp8=bool(config.kv_cache_mxfp8))
    # Only the hybrid pools route layers by label; a plain MHA pool has no
    # state layers to tell apart.
    # 译：只有混合池按标签区分各层；普通 MHA 池没有需要区分的 state 层。
    hybrid_kwargs = (
        {"layer_types": spec.layer_types}
        if issubclass(pool_cls, HybridMHATokenToKVPool)
        else {}
    )
    return pool_cls(
        arena=arena,
        dtype=config.kv_cache_dtype,
        head_num=max(softmax_attn.num_kv_heads // softmax_attn.attn_tp_size, 1),
        head_dim=softmax_attn.head_dim,
        layer_num=num_layers,
        rank=rank,
        layer_kv_head_counts=spec.layer_kv_head_counts,
        kv_alloc_head_count=softmax_attn.num_kv_heads,
        field_layer_offset=field_layer_offset,
        **hybrid_kwargs,
    )


def _mla_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int,
) -> CachePool:
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    softmax_attn = _softmax_config(config, spec.family, MLAConfig)
    return MLATokenToKVPool(
        arena,
        dtype=config.kv_cache_dtype,
        model_dtype=config.dtype,
        quant_method=config.kv_cache_quant_method,
        kv_lora_rank=softmax_attn.kv_lora_rank,
        qk_rope_head_dim=softmax_attn.qk_rope_head_dim,
        layer_num=num_layers,
        rank=rank,
        field_layer_offset=field_layer_offset,
    )


def _hybrid_kda_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int,
) -> CachePool:
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
        HybridKDADSATokenToKVPool,
        HybridKDATokenToKVPool,
    )

    softmax_attn = _softmax_config(config, spec.family, MLAConfig)
    common = dict(
        arena=arena,
        dtype=config.kv_cache_dtype,
        model_dtype=config.dtype,
        quant_method=config.kv_cache_quant_method,
        kv_lora_rank=softmax_attn.kv_lora_rank,
        qk_rope_head_dim=softmax_attn.qk_rope_head_dim,
        layer_num=num_layers,
        rank=rank,
        layer_types=spec.layer_types,
        field_layer_offset=field_layer_offset,
    )
    if isinstance(softmax_attn, DSAConfig):
        # Kimi-K3.1: MLA layers carry index-K planes in the DSA leaf's kernel pages.
        # 译：Kimi-K3.1：MLA 层在 DSA 叶子的 kernel 页中携带 index-K 平面。
        return HybridKDADSATokenToKVPool(
            index_head_dim=softmax_attn.index_head_dim,
            index_k_page_size=dsa_kernel_page_size(config),
            **common,
        )
    return HybridKDATokenToKVPool(**common)


# family -> the pool its recipe's plan binds to. A factory takes
# ``(spec, config, arena, *, num_layers, rank, field_layer_offset)``. Plugins
# add families via ``tokenspeed.runtime.plugins.registry.register_cache_pool``.
# 译：家族 -> 其 recipe 的计划所绑定的池。工厂接受 ``(spec, config, arena, *, num_layers, rank, field_layer_offset)``。
#     插件通过 ``tokenspeed.runtime.plugins.registry.register_cache_pool`` 增加家族。
_POOL_FACTORIES: dict[str, Callable[..., CachePool]] = {
    "mha": _mha_shaped_pool,
    "inkling": _mha_shaped_pool,
    "qwen_gdn": _mha_shaped_pool,
    "qwen4_exp": _mha_shaped_pool,
    "mamba2": _mha_shaped_pool,
    "mla": _mla_pool,
    "kimi_k3": _hybrid_kda_pool,
    "dsa": _dsa_pool,
    "dots3_note": _dots3_note_pool,
    "msa": _msa_pool,
    "glm53_flash": _glm53_flash_pool,
    "deepseek_v41": _deepseek_v41_pool,
}


def create_cache_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int = 0,
) -> CachePool:
    """Bind one model's compute views to an already-allocated arena.

    ``field_layer_offset`` places this view's local layer ids onto the
    merged plan's global layer window, so a draft view names the
    continuation fields the target's plan already reserved.
    """
    # 译：把一个模型的计算视图绑定到已经分配好的 arena 上。
    #
    #     ``field_layer_offset`` 把本视图的局部层 id 放到合并计划的全局层窗口上，这样草稿视图就能指向目标模型的计划
    #     已经预留的续写字段。
    factory = _POOL_FACTORIES.get(spec.family)
    if factory is None:
        raise TypeError(f"no cache pool is registered for family {spec.family!r}")
    return factory(
        spec,
        config,
        arena,
        num_layers=num_layers,
        rank=rank,
        field_layer_offset=field_layer_offset,
    )
