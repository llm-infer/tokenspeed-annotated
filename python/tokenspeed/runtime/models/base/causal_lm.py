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

"""Base causal language model: model + lm_head + logits_processor."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    non_unit_kv_scale_message,
    record_non_unit_kv_scales,
)
from tokenspeed.runtime.models.base.transformer_model import BaseTransformerModel
from tokenspeed.runtime.utils import add_prefix

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.moe import MoECheckpointLoader


# 装饰器，记录本次加载写入了哪些参数
# 
# 为什么需要？
#   普通加载：完整权重 -> 内存 -> 显存 -> 加载完成
#   分块加载：chunk1、chunk2、chunk3
#                 ↓  多次调用 load_weights
#               加载完成
#   分块加载需要检查所有分块是否完整、遗漏、重复
def _record_loaded_names(load_weights: Callable) -> Callable:
    """Track physical parameter writes and the canonical source stream.

    A delegating override and its ``super()`` loader share one stream. Only
    the outermost wrapper checks source names; inner loaders may rename them
    to fused or rank-local parameter names before loading.
    """

    @functools.wraps(load_weights)
    def wrapper(self: BaseCausalLM, *args: Any, **kwargs: Any) -> Any:
        # 记录加载函数的嵌套深度
        # 在最外层最适合检查checkpoint权重名称
        # 内层可能把名称改成了合并参数，或者rank本地参数名
        depth = self._weight_update_load_depth
        self._weight_update_load_depth = depth + 1
        checked_stream = None
        try:
            if depth == 0:
                # 发现 KV scale≠1 就记到 rejected
                rejected = self._weight_update_non_unit_kv_scales
                # 每个名字检查"是否意外/重复"
                complete = self._weight_update_required_names is not None
                if rejected is not None or complete:
                    weights = args[0] if args else kwargs["weights"]
                    if complete:
                        weights = self._record_weight_update_sources(weights)
                    if rejected is not None:
                        weights = record_non_unit_kv_scales(weights, rejected)
                    weights = iter(weights)
                    if complete:
                        checked_stream = weights
                    if args:
                        args = (weights, *args[1:])
                    else:
                        kwargs["weights"] = weights
            loaded = load_weights(self, *args, **kwargs)
            # 检查是不是所有权重都读了一遍
            if checked_stream is not None:
                # Receiving every required name is not sufficient if a loader
                # returned early and left part of its input unconsumed.
                sentinel = object()
                if next(checked_stream, sentinel) is not sentinel:
                    raise ValueError(
                        f"{type(self).__name__}: load_weights did not consume "
                        "the complete checkpoint stream"
                    )
            if isinstance(loaded, set):
                self.record_loaded_weights(loaded)
            return loaded
        except BaseException:
            if self._weight_update_required_names is not None:
                self._weight_update_load_failed = True
            raise
        finally:
            self._weight_update_load_depth = depth

    return wrapper


def _derive_once(post_load_weights: Callable) -> Callable:
    """Defer a ``post_load_weights`` override while a session is active."""

    @functools.wraps(post_load_weights)
    def wrapper(self: BaseCausalLM, *args: Any, **kwargs: Any) -> None:
        # 判断当前是否处于在线更新
        #   是：设置标记
        #   否：立刻执行
        # 没有它:  块1→派生  块2→派生  块3→派生 ...   (N 次,且每次都在半新半旧的权重上)
        # 有了它:  块1→记待办 块2→记待办 块3→记待办 ... end→派生 1 次
        if self._weight_update_active:
            self._weight_update_derive_pending = True
            return
        post_load_weights(self, *args, **kwargs)

    return wrapper


# BaseCasualLM是所有因果模型的公共基类：主干BaseTransformerModel+lm_head+LogitsProcessor
#   主干model：计算hidden states
#   lm_head：把hidden变成各词的分数
#   LogitProcessor：围绕lm_head的流程
#       挑出要采样的行：每个请求一行
#       调用lm_head做矩阵乘
#       按lm_head的切法整理结果
#       打包成LogitsProcessorOutput
#
# 再加上通用的权重加载流程，具体模型只需要指定model_cls，必要时提供权重名映射
# 本身不包含任何模型实现，只是制定流程骨架，具体实现由子类提供
class BaseCausalLM(nn.Module):
    """Model + lm_head + logits_processor, plus the live weight-update session.

    Weight-loading contract for subclasses:

    * ``load_weights(weights)`` consumes a (possibly partial) checkpoint
      stream, ends by calling ``post_load_weights()`` and returns the
      ``named_parameters()`` names that received data (``set[str]``) when it
      can tell; the base class records the returned names for the session.
    * ``post_load_weights()`` derives state from the loaded parameters. It
      must be safe to re-run: write into derived tensors that already exist
      (``bind_or_copy``; captured CUDA graphs hold their addresses) and apply
      one-shot in-place transforms only to parameters that were reloaded
      (``_weight_update_loaded_names``; None means the initial load).

    An RL trainer rewrites the parameters of a serving model in place through
    many partial ``load_weights`` calls (one per NCCL broadcast, or whatever
    chunking the Model Updater SDK streams), so the per-call derivation would
    run once per chunk on a half-updated model. ``begin_weight_update`` /
    ``end_weight_update`` bracket the update: ``__init_subclass__`` wraps
    every subclass-defined ``post_load_weights`` so that, while the session
    is active, it only marks the derivation pending, and ``end_weight_update``
    runs it exactly once over the whole update. Overrides therefore need no
    session awareness of their own; subclasses that keep pairing state across
    chunks (a fused parameter assembled from several checkpoint tensors)
    extend ``begin_weight_update`` / ``end_weight_update`` /
    ``abort_weight_update`` to reset and verify it. The session also screens
    every chunk for KV-cache scales other than one and rejects the update at
    its end (``end_weight_update``), the check the initial load applies to
    the checkpoint, whoever drives the ``load_weights`` calls.

    A complete session also checks the canonical source names consumed by
    the outermost loader against ``required_weight_update_names``. Every
    required source must arrive exactly once; the session validates all
    models before any post-load derivation. The model declares discard
    support separately and lists its non-checkpoint preserved tensors in
    ``weight_discard_preserved_tensors``.

    The session fields are class-level defaults so a subclass that builds
    itself without ``BaseCausalLM.__init__`` (the speculative drafts) still
    takes part in sessions.
    """

    # post_load_weights必须可重复执行
    # 1. 写入已存在的张量，而不是新建张量。因为CUDA Graph在捕获时记的是张量显存地址
    # 2. 只对被重新加载的参数做原地变换，比如把权重乘个系数。如果对没更新的参数也做一遍,就乘了两次

    # 子类必须指定的主干类型
    model_cls: type[BaseTransformerModel]
    # 默认值，子类可覆盖
    logits_processor_cls: ClassVar[type[LogitsProcessor]] = LogitsProcessor
    # Whether the model's MoE layers route through the process-global expert
    # placement (redundant replicas, load counters; ``moe/expert_location.py``)
    # and its loader fills every placed slot. ``build_expert_placement``
    # refuses the placement flags for a model that does not, so a placement
    # is never installed only to be ignored.

    # MoE 是否支持专家放置(冗余副本、负载计数)。不支持的模型被传了相关参数会直接拒绝
    supports_expert_placement: ClassVar[bool] = False
    # Opt in only when complete checkpoint loading and non-parameter state
    # restoration are both defined for this model.
    
    # 是否支持“丢弃权重存储”，要求定义了完整的 checkpoint 加载和非参数状态的恢复
    supports_weight_discard: ClassVar[bool] = False
    # Opt in only when ``self.model`` returns the taps ``layers_to_capture`` names (EAGLE3, DFLASH).
    
    # self.model会返回layers_to_capture时（EAGLE3、DFLASH），才开启
    supports_aux_capture: ClassVar[bool] = False

    @property
    def routed_experts_weights_of_layer(self) -> dict[int, list[torch.Tensor]]:
        """Each MoE layer's slot tensors, ``[num_local_slots, ...]`` in slot order.

        The online expert rebalance (``--enable-eplb``) moves these between
        slots and reserves a staging buffer of one layer's worth at startup.
        Keyed by the placement's layer id. Every model that opts in to expert
        placement provides it; the processed parameters (quantized weights and
        scales included) are what moves, so nothing is re-derived.

        One shape for every MoE model: a read-only property collecting each
        MoE block's ``get_moe_routed_weights()``. It is never assigned (an
        assignment to a setter-less property raises at load time).
        """
        # 每个MoE层的权重张量
        # eplb均衡负载时，会移动这些张量
        raise NotImplementedError(
            f"{type(self).__name__} does not expose its routed expert weights per "
            "layer; --enable-eplb needs routed_experts_weights_of_layer"
        )

    # Live weight-update session (see the class docstring). Declared here
    # with defaults and assigned again in ``__init__``.
    # 是否在在线更新权重
    _weight_update_active: bool = False
    # Parameter names ``load_weights`` touched during the active session;
    # None outside a session (the initial load touches everything).
    # 本次更新写过哪些参数
    _weight_update_loaded_names: set[str] | None = None
    # A ``post_load_weights`` call was deferred by the session.
    # 有没有被推迟的会话
    _weight_update_derive_pending: bool = False
    # KV-cache scales other than one seen by this session's ``load_weights``
    # calls (KV caches are written and read at unit scale); None outside a
    # session, where the loader screens the checkpoint itself.
    # 本次会话中不等于 1 的 KV cache scale
    _weight_update_non_unit_kv_scales: list[str] | None = None
    # Canonical checkpoint names, distinct from the physical parameter names
    # above. Complete sessions require each source exactly once.
    _weight_update_required_names: set[str] | None = None
    _weight_update_source_names: set[str] | None = None
    _weight_update_load_failed: bool = False
    _weight_update_load_depth: int = 0

    # 定义子类时自动执行：如果子类自己定义了 load_weights / post_load_weights，
    # 就给它们套上上面两个装饰器。这样子类不用关心在线权重更新会话的存在。
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        load_weights = cls.__dict__.get("load_weights")
        if inspect.isfunction(load_weights):
            cls.load_weights = _record_loaded_names(load_weights)
        post_load_weights = cls.__dict__.get("post_load_weights")
        if inspect.isfunction(post_load_weights):
            cls.post_load_weights = _derive_once(post_load_weights)

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        encoder_only: bool = False,
    ) -> None:

        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.capture_aux_hidden_states: bool = False
        self._weight_update_active = False
        self._weight_update_loaded_names = None
        self._weight_update_derive_pending = False
        self._weight_update_non_unit_kv_scales = None
        self._weight_update_required_names = None
        self._weight_update_source_names = None
        self._weight_update_load_failed = False
        self._weight_update_load_depth = 0

        self.encoder_only = encoder_only
        if encoder_only:
            # Vision-only role (EPD encode): never allocate the LM / lm_head /
            # logits processor (the LM allocation is the OOM at encode TP=1).
            # self.config is already set above for the vision path
            # (separate_deepstack_embeds needs self.config.hidden_size).
            # 纯视觉模型，不需要语言模型、lm_head、logits processor
            self.model = None
            self.lm_head = None
            self.logits_processor = None
        else:
            self.model = self.resolve_model(config, mapping, quant_config, prefix)
            # pp并行，只有最后一个stage需要lm_head和logits
            # 不开pp时默认是1，本rank就是最后一个stage
            if mapping.is_last_pp_rank:
                self.lm_head = self.resolve_lm_head(config, quant_config, prefix)
                self.logits_processor = self.resolve_logits_processor(config)
            else:
                # Mid-pipeline stages emit hidden states, never logits.
                # 流水线并行的中间阶段只生成hidden state，不生成logits
                self.lm_head = None
                self.logits_processor = None
        self.post_init()

    # 创建主干模型实例
    def resolve_model(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> BaseTransformerModel:

        return self.model_cls(
            config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

    # 创建 lm_head
    # 1. tie_word_embeddings：lm_head与embedding共享一张[V, H]表
    #       [V, H]是id -> embedding的映射，embedding是根据id查embedding，
    #       lm_head是根据hidden state计算每个embedding的相似度，两者可以复用一张表
    # 2. 
    def resolve_lm_head(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:

        if getattr(config, "tie_word_embeddings", False):
            return self.model.embed_tokens

        # mapping.lm_head follows attention TP without attention DP and is
        # replicated (tp 1) under it unless --lm-head-tp-size widens it.
        
        # lm_head的切分条件为什么要特殊考虑DP？
        # 
        # 情况1：无DP，lm_head_tp=2：所有rank都拿着同一份hidden，各算半个词表
        # rank0 与 rank1 的 hidden 完全相同 [301, 4096]
        # │
        # ├─ ① 挑行(gather_ids = [299, 300])
        # │     hidden_sel [2, 4096]            ← 两个 rank 一样
        # │
        # ├─ ② 各算自己那半词表
        # │     rank0: [2,4096] × W[0:75968]^T      → [2, 75968]
        # │     rank1: [2,4096] × W[75968:151936]^T → [2, 75968]
        # │
        # └─ ③ all-gather(沿词表维拼接)
        #         两个 rank 都得到 [2, 151936]
        # 
        # 情况2：attn DP=2，lm_head不切，每个rank只算自己的请求，零通信
        # 
        # 情况3：attn DP=2，lm_head_tp=2，dp_lm_head_tp=true
        #               rank0：有 A         rank1：有B
        #                   │                 │
        #   ① 各自挑行:   [1,4096]       [1,4096]
        #                   │                 │
        #   ② 先把 hidden 跨组收集(all-gather)
        #       两个 rank 都得到 [2, 4096](A 和 B 的行都有)
        #   ③ 各算自己那半词表
        #   ④ 把结果送回各请求的主人
        if self.mapping.attn.has_dp and not self.mapping.lm_head.has_tp:
            return ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        return ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            tp_rank=self.mapping.lm_head.tp_rank,
            tp_size=self.mapping.lm_head.tp_size,
            tp_group=self.mapping.lm_head.tp_group,
        )

    # 创建 LogitsProcessor，根据lm_head的切分方式，
    # 决定如何通过各rank输出的logits拼回完整词表
    def resolve_logits_processor(self, config: PretrainedConfig) -> LogitsProcessor:

        return self.logits_processor_cls(
            config,
            skip_all_gather=self.mapping.attn.has_dp,
            tp_rank=self.mapping.lm_head.tp_rank,
            tp_size=self.mapping.lm_head.tp_size,
            tp_group=self.mapping.lm_head.tp_group,
            attn_tp_size=self.mapping.attn.tp_size,
            dp_lm_head_tp=self.mapping.attn.has_dp and self.mapping.lm_head.has_tp,
        )

    def post_init(self) -> None:
        """Hook for subclasses that need derived state after shared modules exist."""

    def require_aux_capture(self) -> None:
        """Refuse a capture request this model's forward would silently ignore."""
        if not self.supports_aux_capture:
            raise NotImplementedError(
                f"{type(self).__name__} does not return aux hidden states, so "
                "EAGLE3 and DFLASH drafts cannot attach to it"
            )

    # 投机解码时，由草稿器调用，指定目标模型需要捕获哪些层的hidden
    # 层号i捕获的是第i层的输入，也就是第i-1层的输出
    # 
    # EAGLE3：默认捕获靠前、中间、靠后三层的输入
    # 显示给出layer_ids时，它们指的是层输出，需要+1转为层的残差（即下一层的输入）
    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:

        self.require_aux_capture()
        self.capture_aux_hidden_states = True

        if layer_ids is None:

            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]

        else:

            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    # 捕获DFLASH/DSPARK草稿器所需目标模型的hidden states
    # 例子:L=36,layer_ids=[1,17,35]:
    #   捕获层号是 [2, 18, 37]（layer_ids+1）
    #   _dflash_capture_idx_map = {2:0, 18:1, 37:2}（记录每个层号的顺序）
    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture the target hidden states a DFLASH/DSpark draft consumes.

        Checkpoints name layer *outputs*, but a layer captures the residual
        entering it -- hence the ``+ 1`` shift, same as EAGLE3. The last
        layer's output enters no layer: it is the final norm's residual,
        captured after the loop under EAGLE3's ``num_layers + 1`` id. Each
        forward that wants the taps handed over as they are produced attaches
        a ``ctx.target_capture_sink``; otherwise they are only collected.
        """

        self.require_aux_capture()
        num_layers = len(self.model.layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("DFLASH target_layer_ids must be unique.")
        invalid = [val for val in layer_ids if val < 0 or val >= num_layers]
        if invalid:
            raise ValueError(
                "DFLASH target_layer_ids must map to capturable target layer "
                f"outputs. Got invalid ids {invalid}; valid range is "
                f"[0, {num_layers - 1}] for {num_layers} target layers."
            )

        self.capture_aux_hidden_states = True
        capture_layers = sorted(
            val + 1 if val + 1 < num_layers else num_layers + 1 for val in layer_ids
        )
        self.model.layers_to_capture = capture_layers
        # The draft concatenates captures in ascending layer order.
        self.model._dflash_capture_idx_map = {
            layer_idx: i for i, layer_idx in enumerate(capture_layers)
        }

    # no_grad：跳过反向图计算
    #   最后一层：交给logits_processor
    #   其它层：返回hidden_states给下一层
    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        model_kwargs = self.prepare_model_kwargs(ctx, input_ids, kwargs)

        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            ctx,
            **model_kwargs,
        )
        if not self.mapping.is_last_pp_rank:
            # Mid-pipeline stage: the executor sends this boundary state to
            # the next stage; there are no logits here.
            return hidden_states
        return self.exit_logits(input_ids, hidden_states, aux_hidden_states, ctx)

    def exit_logits(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        ctx: ForwardContext,
    ):
        """The model exit: prompt logprobs, sampled-row selection, the LM head.

        ``hidden_states`` are the rows this rank computed -- under query
        context parallelism its shard (``ctx.query_shard``). The logits
        processor scores the planned prompt rows on them, then selects the
        sampled rows (one per request; gathered across the group in request
        order on a shard) for the LM head; a full-hidden capture for the
        drafter stays the rows as given, which is the drafter's extend input.
        """
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            LogitsMetadata.from_forward_context(ctx),
            aux_hidden_states,
        )

    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        """Hook for subclasses to pass model-specific tensors."""
        model_kwargs = {}
        for key in ("input_embeds", "inputs_embeds", "pp_inbound"):
            if kwargs.get(key) is not None:
                model_kwargs[key] = kwargs[key]
        return model_kwargs

    # Weight loading.

    # checkpoint 权重名 -> 合并参数名的映射
    #   每项 (合并参数名, checkpoint权重名，shard_id)
    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:

        return []

    # 忽略名字包含这些字符串的checkpoint权重名
    # 有些checkpoint带着RoPE的inv_freq缓存，本项目自己算cos/sin表，用不到该缓存
    def get_skip_weight_names(self) -> list[str]:

        return ["rotary_emb.inv_freq"]

    # MoE模型返回专家权重的加载器，稠密模型返回None
    def get_expert_checkpoint_loader(self) -> MoECheckpointLoader | None:
        """The loader for this rank's routed experts, or None without experts."""

        return None

    # 通用的权重加载流程
    # weights：checkpoint张量迭代器，由model_loader从safetensor文件读出
    # 每个张量按顺序尝试四条规则
    #   1. 命中get_skip_weight_names，跳过
    #   2. 是MoE专家权重，交给专家加载器；不属于本rank的专家（在其他EP rank），跳过
    #   3. 命中get_stacked_params_mapping -> 改名为合并参数，传入shard_id调用weight_loader
    #   4. 其余：同名参数，调用torch.Param的weight_loader（没有weight_loader的话直接用默认加载器）
    # weight_loader挂载参数对象上（由Linear/Embedding等层在参数创建时设置）
    # 负责从完整的checkpoint张量里切出本rank的那一分，写到参数正确的位置
    #
    # 例（Qwen3，单卡）：
    #   "model.layers.0.self_attn.q_proj.weight" [4096, 4096]
    #     -> 规则 3：改名为 "model.layers.0.self_attn.qkv_proj.weight"，shard_id="q"
    #     -> 写入 qkv_proj.weight [6144, 4096] 的第 0~4095 行
    #   "model.layers.0.input_layernorm.weight" [4096]
    #     -> 规则 4：同名参数，直接拷贝
    @_record_loaded_names
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
    ) -> set[str]:
        """Load a (possibly partial) stream of checkpoint tensors.

        Returns the names of the parameters that received data; see the
        class docstring for the session contract.
        """

        stacked_params_mapping = self.get_stacked_params_mapping()
        skip_patterns = self.get_skip_weight_names()
        params_dict: dict[str, nn.Parameter] = dict(self.named_parameters())
        expert_loader = self.get_expert_checkpoint_loader()
        loaded: set[str] = set()

        for name, loaded_weight in weights:

            if any(pattern in name for pattern in skip_patterns):
                continue

            if expert_loader is not None and expert_loader.is_expert_checkpoint_weight(
                name
            ):
                # Experts this rank does not match belong to another EP rank.
                # 本rank匹配不上的专家，属于另一个EP rank
                if expert_loader.matches(name):
                    loaded.add(expert_loader.load(name, loaded_weight))
                continue

            # for ... else：break表示命中了规则3；
            # 一条都没命中，才会进入else
            for param_name, weight_name, shard_id in stacked_params_mapping:

                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                # 模型不用bias，但checkpoint带了bias，直接跳过
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)

                break

            else:

                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        # 全部权重加载完成后，开始其他任务，比如转成kernel要求的布局
        self.post_load_weights()
        return loaded

    # Live weight updates (see the class docstring).

    # 完整更新需要的规范 checkpoint 名字
    def required_weight_update_names(self) -> set[str]:
        """Canonical checkpoint names required for a complete model update.

        Models that support discarding weights return their rank-local source
        contract, excluding aliases loaded by another model in the session.
        These names precede any loader renaming or fused-parameter mapping.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support complete weight updates"
        )

    # 丢弃权重后必须保留
    def weight_discard_preserved_tensors(self) -> tuple[torch.Tensor, ...]:
        """Non-checkpoint tensors that must survive discarded weight storage.

        An opted-in model lists only immutable state that cannot be rebuilt
        from the new weights. Model parameters and reconstructible derived
        tensors must not be retained as a CPU weight backup.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support weight discard"
        )

    def _record_weight_update_sources(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        required = self._weight_update_required_names
        received = self._weight_update_source_names
        assert required is not None and received is not None
        for name, tensor in weights:
            if name not in required:
                raise ValueError(
                    f"{type(self).__name__}: unexpected checkpoint weight {name!r}"
                )
            if name in received:
                raise ValueError(
                    f"{type(self).__name__}: duplicate checkpoint weight {name!r}"
                )
            received.add(name)
            yield name, tensor

    def _validate_weight_update_complete(self) -> None:
        required = self._weight_update_required_names
        if required is None:
            return
        if self._weight_update_load_failed:
            raise ValueError(
                f"{type(self).__name__}: a complete weight update failed while loading"
            )
        assert self._weight_update_source_names is not None
        missing = required - self._weight_update_source_names
        if missing:
            raise ValueError(
                f"{type(self).__name__}: missing checkpoint weights: {sorted(missing)}"
            )

    def begin_weight_update(self) -> None:
        """Enter a live weight-update session.

        Raises:
            RuntimeError: A session is already active.
        """
        if self._weight_update_active:
            raise RuntimeError(
                f"{type(self).__name__}: a weight-update session is already active"
            )
        self._weight_update_active = True
        self._weight_update_loaded_names = set()
        self._weight_update_derive_pending = False
        self._weight_update_non_unit_kv_scales = []
        self._weight_update_required_names = None
        self._weight_update_source_names = None
        self._weight_update_load_failed = False
        self._weight_update_load_depth = 0

    def end_weight_update(self) -> None:
        """Leave the session and run the deferred derivation once.

        ``post_load_weights`` runs only if a chunk asked for it, and with
        ``_weight_update_loaded_names`` still populated so a model can
        restrict derivations that are not idempotent to the parameters this
        update actually replaced. The session is closed whether or not the
        derivation raises. An update that streamed a KV-cache scale other
        than one is still loaded to completion and derived, so the model
        stays consistent, and then rejected.

        Raises:
            RuntimeError: No session is active.
            ValueError: The update carried a KV-cache scale other than one.
        """
        if not self._weight_update_active:
            raise RuntimeError(
                f"{type(self).__name__}: no weight-update session is active"
            )
        try:
            self._validate_weight_update_complete()
        except BaseException:
            self.abort_weight_update()
            raise
        # Leave the session first: the ``post_load_weights`` wrapper defers
        # only while one is active.
        self._weight_update_active = False
        rejected = self._weight_update_non_unit_kv_scales
        try:
            if self._weight_update_derive_pending:
                self.post_load_weights()
        finally:
            self.abort_weight_update()
        if rejected:
            # A subclass loader that hands the stream to ``super()`` screens
            # the same tensors twice; report each once.
            raise ValueError(
                f"{type(self).__name__}: "
                f"{non_unit_kv_scale_message(list(dict.fromkeys(rejected)))}"
            )

    def abort_weight_update(self) -> None:
        """Leave the session without deriving state (the update failed)."""
        self._weight_update_active = False
        self._weight_update_loaded_names = None
        self._weight_update_derive_pending = False
        self._weight_update_non_unit_kv_scales = None
        self._weight_update_required_names = None
        self._weight_update_source_names = None
        self._weight_update_load_failed = False
        self._weight_update_load_depth = 0

    def record_loaded_weights(self, names: Iterable[str]) -> None:
        """Remember which parameters this session's ``load_weights`` touched.

        The base class records what ``load_weights`` returns; calling this
        explicitly is harmless.
        """
        if self._weight_update_loaded_names is not None:
            self._weight_update_loaded_names.update(names)

    def post_load_weights(self) -> None:
        """Derive state from the loaded parameters (see the class docstring)."""

    def get_embed_and_head(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The embedding and LM-head weights a draft may share with this target.

        A pipeline stage holds only its own side: the embedding lives on the
        first stage and the head on the last, so the missing side is None
        rather than a dereference of an absent module.
        """
        embed_tokens = self.model.embed_tokens
        embed = embed_tokens.weight if embed_tokens is not None else None
        head = self.lm_head.weight if self.lm_head is not None else None
        return embed, head

    def set_embed_and_head(
        self, embed: torch.Tensor | None, head: torch.Tensor
    ) -> None:
        """Alias the target's embedding and LM-head weights into this draft.

        ``embed=None`` means the target shares no embedding (a pipeline's last
        stage: it lives on the first stage). A generic draft keeps no embedding
        of its own, so it refuses; a pipeline-capable draft overrides this to
        keep the ``embed_tokens`` its checkpoint ships, and the factory checks
        ``get_embed_and_head`` still reports one afterwards.
        """
        if embed is None:
            raise ValueError(
                f"{type(self).__name__} shares the target embedding and cannot "
                "keep its own; it is not a pipeline-capable draft."
            )
        del self.model.embed_tokens.weight
        del self.lm_head.weight

        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
