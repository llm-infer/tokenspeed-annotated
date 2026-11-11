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

# 译：因果语言模型基类：model + lm_head + logits_processor。

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

# 所有因果语言模型的公共基类：主干（BaseTransformerModel）+ lm_head + LogitsProcessor，
# 再加上权重加载的通用流程。具体模型（如 Qwen3ForCausalLM）只需指定 model_cls，
# 必要时提供权重名映射（get_stacked_params_mapping）。
#
# 你在这里：ModelRunner.forward -> 【BaseCausalLM.forward】 -> BaseTransformerModel.forward
#           -> LogitsProcessor -> 返回 LogitsProcessorOutput 给 ModelExecutor 去采样
# 形状记号见 models/qwen3.py 文件头。
#
# 本文件分两部分：
#   1. 模型的构建与前向（主线）：__init__、resolve_*、forward、exit_logits
#   2. 权重加载：load_weights 的通用流程（主线），以及“在线权重更新会话”
#      （强化学习训练时训练端不断推送新权重，第 8 阶段详解）
#
# 前向数据流（单卡，Qwen3-8B，A 的 300 个 token + B 的 1 个 token，T=301）
#   input_ids [301], positions [301], ctx
#        │ self.model（BaseTransformerModel）
#        ▼
#   hidden [301, 4096]
#        │ self.logits_processor（第 5 阶段）
#        │   只取每个请求要采样的那一行：A 的最后一行（第 299 行）、B 的唯一一行（第 300 行）
#        │   乘 lm_head 权重 [V, H]^T
#        ▼
#   LogitsProcessorOutput（其中 next_token_logits [2, 151936]）


# 下面两个装饰器服务于“在线权重更新会话”（见 BaseCausalLM 的 docstring，第 8 阶段详解）。
# 它们由 __init_subclass__ 自动套在每个子类自己定义的 load_weights / post_load_weights 上，
# 子类不需要手动使用。
#
# _record_loaded_names：包装 load_weights。
#   - 记录本次加载写入了哪些参数（返回值的名字集合）
#   - 会话中要求“完整更新”时，检查 checkpoint 的每个权重名恰好出现一次、且全部被消费
#   - 筛查 KV cache 的 scale 是否为 1（KV cache 按单位 scale 读写，其他值会被拒绝）
#   子类的 load_weights 往往会调用 super().load_weights，形成嵌套；
#   depth 计数保证这些检查只在最外层做一次。
def _record_loaded_names(load_weights: Callable) -> Callable:
    """Track physical parameter writes and the canonical source stream.

    A delegating override and its ``super()`` loader share one stream. Only
    the outermost wrapper checks source names; inner loaders may rename them
    to fused or rank-local parameter names before loading.
    """
    # 译：记录物理参数的写入，以及规范的来源权重流。
    #
    #     一个委托给 ``super()`` 加载器的覆盖方法与其父类加载器共用同一个权重流。只有最外层的包装器
    #     检查来源名字；内层加载器在加载前可能会把名字改成合并参数名或 rank 本地的参数名。

    @functools.wraps(load_weights)
    def wrapper(self: BaseCausalLM, *args: Any, **kwargs: Any) -> Any:
        depth = self._weight_update_load_depth
        self._weight_update_load_depth = depth + 1
        checked_stream = None
        try:
            if depth == 0:
                rejected = self._weight_update_non_unit_kv_scales
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
            if checked_stream is not None:
                # Receiving every required name is not sufficient if a loader
                # returned early and left part of its input unconsumed.
                # 译：如果某个加载器提前返回、留下部分输入没有消费，那么“收到了每个必需的名字”并不足以说明加载完整。
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


# _derive_once：包装 post_load_weights（加载后从参数派生其他状态，例如把权重转换成 kernel
# 需要的布局）。训练端会把一次权重更新拆成很多块分别调用 load_weights，每块都会触发
# post_load_weights；如果每次都派生，就会在“半新半旧”的权重上做派生。
# 会话进行中时，这里只记一个“待派生”标记，等 end_weight_update 时统一派生一次。
def _derive_once(post_load_weights: Callable) -> Callable:
    """Defer a ``post_load_weights`` override while a session is active."""
    # 译：在会话进行中时，推迟 ``post_load_weights`` 覆盖方法的执行。

    @functools.wraps(post_load_weights)
    def wrapper(self: BaseCausalLM, *args: Any, **kwargs: Any) -> None:
        if self._weight_update_active:
            self._weight_update_derive_pending = True
            return
        post_load_weights(self, *args, **kwargs)

    return wrapper


# 因果语言模型基类。主要成员：
#   model：主干（类型由子类的 model_cls 指定）
#   lm_head：hidden -> logits 的投影；只在流水线并行的最后一个 stage 上存在
#   logits_processor：从 hidden 中挑出要采样的行、乘 lm_head、处理 logprobs 等（第 5 阶段）
# 类级别的几个 supports_* 开关声明模型是否支持某个可选特性（默认都不支持），
# 启动时据此拒绝不支持的参数组合，而不是悄悄忽略。
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

    # 译：model + lm_head + logits_processor，外加在线权重更新会话。
    #
    #     子类的权重加载约定：
    #
    #     * ``load_weights(weights)`` 消费一个（可能是部分的）checkpoint 权重流，最后调用
    #       ``post_load_weights()``，并在能判断的情况下返回接收到数据的 ``named_parameters()`` 名字
    #       （``set[str]``）；基类为会话记录这些返回的名字。
    #     * ``post_load_weights()`` 从加载好的参数派生状态。它必须可以安全地重复执行：写入已经存在的
    #       派生张量（``bind_or_copy``；已捕获的 CUDA Graph 持有它们的地址），并且只对被重新加载的参数
    #       做一次性的原地变换（``_weight_update_loaded_names``；None 表示初次加载）。
    #
    #     强化学习训练端通过多次部分的 ``load_weights`` 调用（每次 NCCL 广播一次，或按 Model Updater SDK
    #     推送的任意分块）原地改写正在服务的模型的参数，因此每次调用都派生的话，就会在半更新的模型上
    #     每块派生一次。``begin_weight_update`` / ``end_weight_update`` 把一次更新括起来：
    #     ``__init_subclass__`` 包装每个子类自己定义的 ``post_load_weights``，使它在会话进行中只标记
    #     “待派生”，由 ``end_weight_update`` 在整个更新结束后恰好执行一次。因此覆盖方法本身不需要感知会话；
    #     在多个分块之间保存配对状态的子类（由多个 checkpoint 张量拼装成的合并参数）扩展
    #     ``begin_weight_update`` / ``end_weight_update`` / ``abort_weight_update`` 来重置和校验这些状态。
    #     会话还会筛查每个分块中不等于 1 的 KV cache scale，并在结束时（``end_weight_update``）拒绝该更新，
    #     这与初次加载对 checkpoint 做的检查相同，不论是谁在驱动 ``load_weights`` 调用。
    #
    #     完整会话还会用 ``required_weight_update_names`` 检查最外层加载器消费的规范来源名字。
    #     每个必需的来源必须恰好到达一次；会话在任何加载后派生之前校验所有模型。模型另外声明是否支持
    #     丢弃权重，并在 ``weight_discard_preserved_tensors`` 中列出需要保留的非 checkpoint 张量。
    #
    #     会话字段是类级别的默认值，这样不经过 ``BaseCausalLM.__init__`` 构建自身的子类
    #     （投机解码的草稿模型）也能参与会话。

    # 子类必须指定主干类型，例如 Qwen3ForCausalLM.model_cls = Qwen3Model
    model_cls: type[BaseTransformerModel]
    # LogitsProcessor 的类型，个别模型（需要特殊 logits 处理）会覆盖
    logits_processor_cls: ClassVar[type[LogitsProcessor]] = LogitsProcessor
    # Whether the model's MoE layers route through the process-global expert
    # placement (redundant replicas, load counters; ``moe/expert_location.py``)
    # and its loader fills every placed slot. ``build_expert_placement``
    # refuses the placement flags for a model that does not, so a placement
    # is never installed only to be ignored.
    # 译：模型的 MoE 层是否经由进程全局的专家放置（冗余副本、负载计数器；``moe/expert_location.py``）
    #     进行路由，且其加载器会填满每个放置的槽位。对于不满足这一点的模型，``build_expert_placement``
    #     会拒绝相关的放置参数，从而不会出现“安装了放置却被忽略”的情况。
    supports_expert_placement: ClassVar[bool] = False
    # Opt in only when complete checkpoint loading and non-parameter state
    # restoration are both defined for this model.
    # 译：只有当该模型同时定义了完整的 checkpoint 加载和非参数状态的恢复时，才开启此项。
    supports_weight_discard: ClassVar[bool] = False
    # Opt in only when ``self.model`` returns the taps ``layers_to_capture`` names (EAGLE3, DFLASH).
    # 译：只有当 ``self.model`` 会返回 ``layers_to_capture`` 所列层的捕获结果时（EAGLE3、DFLASH），才开启此项。
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
        # 译：每个 MoE 层的槽位张量，形状 ``[num_local_slots, ...]``，按槽位顺序排列。
        #
        #     在线专家重平衡（``--enable-eplb``）在槽位之间移动这些张量，并在启动时预留一层大小的暂存缓冲区。
        #     以放置中的层 id 为键。每个开启专家放置的模型都提供它；移动的是处理后的参数（包括量化后的权重和
        #     scale），所以不需要重新派生。
        #
        #     所有 MoE 模型共用一种形式：一个只读属性，收集每个 MoE 块的 ``get_moe_routed_weights()``。
        #     它从不被赋值（给没有 setter 的属性赋值会在加载时报错）。
        raise NotImplementedError(
            f"{type(self).__name__} does not expose its routed expert weights per "
            "layer; --enable-eplb needs routed_experts_weights_of_layer"
        )

    # Live weight-update session (see the class docstring). Declared here
    # with defaults and assigned again in ``__init__``.
    # 译：在线权重更新会话（见类的 docstring）。这里声明带默认值的字段，并在 ``__init__`` 中再次赋值。
    _weight_update_active: bool = False
    # Parameter names ``load_weights`` touched during the active session;
    # None outside a session (the initial load touches everything).
    # 译：活动会话期间 ``load_weights`` 写入过的参数名；不在会话中时为 None（初次加载会写入全部参数）。
    _weight_update_loaded_names: set[str] | None = None
    # A ``post_load_weights`` call was deferred by the session.
    # 译：某次 ``post_load_weights`` 调用被会话推迟了。
    _weight_update_derive_pending: bool = False
    # KV-cache scales other than one seen by this session's ``load_weights``
    # calls (KV caches are written and read at unit scale); None outside a
    # session, where the loader screens the checkpoint itself.
    # 译：本次会话的 ``load_weights`` 调用中出现的、不等于 1 的 KV cache scale（KV cache 按单位 scale 读写）；
    #     不在会话中时为 None，此时由加载器自己筛查 checkpoint。
    _weight_update_non_unit_kv_scales: list[str] | None = None
    # Canonical checkpoint names, distinct from the physical parameter names
    # above. Complete sessions require each source exactly once.
    # 译：规范的 checkpoint 名字，与上面的物理参数名不同。完整会话要求每个来源恰好出现一次。
    _weight_update_required_names: set[str] | None = None
    _weight_update_source_names: set[str] | None = None
    _weight_update_load_failed: bool = False
    _weight_update_load_depth: int = 0

    # 定义子类时自动执行：如果子类自己定义了 load_weights / post_load_weights，
    # 就给它们套上上面两个装饰器。这样子类不用关心在线权重更新会话的存在。
    # （只包装子类自己 __dict__ 里的定义，继承来的不会被重复包装。）
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
        # 是否需要返回中间层的 hidden（投机解码的草稿模型要用），由 set_*_layers_to_capture 打开
        self.capture_aux_hidden_states: bool = False
        # 在线权重更新会话的状态，初始为“不在会话中”（含义见类级别的同名字段）
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
            # 译：纯视觉角色（EPD 的编码节点）：绝不分配语言模型、lm_head、logits processor
            #     （在编码节点 TP=1 时，分配语言模型正是 OOM 的原因）。
            #     self.config 已经在上面为视觉路径设置好了（separate_deepstack_embeds 需要 self.config.hidden_size）。
            self.model = None
            self.lm_head = None
            self.logits_processor = None
        else:
            self.model = self.resolve_model(config, mapping, quant_config, prefix)
            # 流水线并行时只有最后一个 stage 需要 lm_head 和 logits 处理；
            # 单卡或不开流水线时本 rank 就是最后一个 stage
            if mapping.is_last_pp_rank:
                self.lm_head = self.resolve_lm_head(config, quant_config, prefix)
                self.logits_processor = self.resolve_logits_processor(config)
            else:
                # Mid-pipeline stages emit hidden states, never logits.
                # 译：流水线中间的 stage 输出 hidden states，从不输出 logits。
                self.lm_head = None
                self.logits_processor = None
        self.post_init()

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

    # 创建 lm_head，三种情况：
    #   1. tie_word_embeddings：lm_head 与 embedding 共用同一张 [V, H] 表，直接返回 embed_tokens
    #   2. 开了 attention DP、lm_head 又不做 TP：每个 DP 组处理各自的请求，
    #      每个 rank 持有完整的 lm_head（ReplicatedLinear，不切分）
    #   3. 其余情况（包括单卡）：词表并行的 ParallelLMHead，按 mapping.lm_head 的 TP 组切分词表
    # Qwen3-8B 的 tie_word_embeddings=False，单卡时走情况 3（tp_size=1，即完整的一份）。
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
        # 译：没有 attention DP 时，mapping.lm_head 跟随注意力的 TP；有 attention DP 时它是复制的（tp 为 1），
        #     除非用 --lm-head-tp-size 把它加宽。
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

    # 创建 LogitsProcessor。参数描述 lm_head 的切分方式，决定算出各 rank 的部分 logits 后
    # 怎样拼回完整词表（第 5 阶段）：
    #   skip_all_gather：attention DP 下不在这里做跨 rank 的 all-gather
    #   tp_*：lm_head 所在的 TP 组
    #   dp_lm_head_tp：attention DP 且 lm_head 仍做 TP 的组合
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
        # 译：供子类使用的钩子：在共享模块创建之后派生所需的状态。

    def require_aux_capture(self) -> None:
        """Refuse a capture request this model's forward would silently ignore."""
        # 译：拒绝那些会被本模型前向静默忽略的捕获请求。
        if not self.supports_aux_capture:
            raise NotImplementedError(
                f"{type(self).__name__} does not return aux hidden states, so "
                "EAGLE3 and DFLASH drafts cannot attach to it"
            )

    # 下面两个方法在启动投机解码时由草稿器调用，指定目标模型要捕获哪些层的 hidden
    # （第 11 阶段）。层号约定见 BaseTransformerModel.layers_to_capture：
    # 层号 i 捕获的是第 i 层的输入，也就是第 i-1 层的输出，所以“第 k 层的输出”要写成 k+1。
    #
    # EAGLE3：默认捕获靠前、中间、靠后三层的输入（Qwen3-8B，L=36：第 2、18、33 层）；
    # 显式给出 layer_ids 时，它们指的是层的输出，所以整体加 1。
    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:

        self.require_aux_capture()
        self.capture_aux_hidden_states = True

        if layer_ids is None:

            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]

        else:

            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture the target hidden states a DFLASH/DSpark draft consumes.

        Checkpoints name layer *outputs*, but a layer captures the residual
        entering it -- hence the ``+ 1`` shift, same as EAGLE3. The last
        layer's output enters no layer: it is the final norm's residual,
        captured after the loop under EAGLE3's ``num_layers + 1`` id. Each
        forward that wants the taps handed over as they are produced attaches
        a ``ctx.target_capture_sink``; otherwise they are only collected.
        """
        # 译：捕获 DFLASH / DSpark 草稿模型所需的目标模型 hidden states。
        #
        #     checkpoint 中给出的是层的“输出”，而一个层捕获的是进入它的残差——所以要加 1，和 EAGLE3 一样。
        #     最后一层的输出不进入任何层：它是 final norm 处的残差，在循环之后以 EAGLE3 的
        #     ``num_layers + 1`` 号捕获。希望捕获结果一产生就被交出的前向会挂上 ``ctx.target_capture_sink``；
        #     否则它们只是被收集起来。

        # DFlash：layer_ids 是“第几层的输出”，换算成捕获层号时加 1；
        # 最后一层（L-1）的输出不是任何层的输入，换成特殊层号 L+1（final norm 处的残差）。
        #   例：L=36，layer_ids=[1, 17, 35] -> 捕获层号 [2, 18, 37]
        # _dflash_capture_idx_map 记录每个捕获层号在草稿模型输入里的顺序（按层号升序）。
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
        # 译：草稿模型按层号升序拼接各个捕获结果。
        self.model._dflash_capture_idx_map = {
            layer_idx: i for i, layer_idx in enumerate(capture_layers)
        }

    # 模型前向的入口，由 ModelRunner.forward 调用（第 4 阶段）。
    #   ctx：前向上下文；input_ids [T]、positions [T]：来自固定输入缓冲区
    #   kwargs：可选输入，例如多模态的 input_embeds、流水线上一 stage 传来的 pp_inbound
    # 返回：
    #   最后一个 stage：LogitsProcessorOutput（logits 及 logprobs 等）
    #   流水线中间 stage：主干输出的 hidden，由执行器发送给下一个 stage
    # @torch.no_grad()：推理不需要构建反向图，省显存和开销。
    # 注意参数顺序是 (ctx, input_ids, positions)，与主干的 (input_ids, positions, ctx) 不同。
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
            # 译：流水线中间的 stage：执行器把这个边界状态发送给下一个 stage；这里没有 logits。
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
        # 译：模型出口：prompt 的 logprobs、选出要采样的行、LM head。
        #
        #     ``hidden_states`` 是本 rank 计算的行——在查询上下文并行下是它的分片（``ctx.query_shard``）。
        #     logits processor 先在这些行上为计划中的 prompt 行打分，然后选出要采样的行（每个请求一行；
        #     在分片上按请求顺序在组内收集）送入 LM head；为草稿器捕获的完整 hidden 保持传入时的行，
        #     那就是草稿器 extend 的输入。
        # LogitsMetadata 从 ctx 中取出 LogitsProcessor 需要的信息（要采样的行号 gather_ids、
        # 是否要算 prompt 的 logprobs 等）。lm_head 作为参数传入，因为它可能就是 embedding。
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            LogitsMetadata.from_forward_context(ctx),
            aux_hidden_states,
        )

    # 从 forward 的 kwargs 中挑出主干认识的参数传下去，其余忽略。
    # 需要额外输入的模型（多模态、带特殊位置编码的模型等）覆盖它。
    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        """Hook for subclasses to pass model-specific tensors."""
        # 译：供子类传递模型特有张量的钩子。
        model_kwargs = {}
        for key in ("input_embeds", "inputs_embeds", "pp_inbound"):
            if kwargs.get(key) is not None:
                model_kwargs[key] = kwargs[key]
        return model_kwargs

    # Weight loading.
    # 译：权重加载。

    # checkpoint 权重名 -> 合并参数名的映射，默认没有合并参数。
    # 每项 (合并参数名片段, checkpoint 名片段, shard_id)，示例见 Qwen3ForCausalLM。
    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:

        return []

    # 名字里包含这些片段的 checkpoint 张量直接跳过。
    # 有些 checkpoint 带着 RoPE 的 inv_freq 缓存，而本项目的 RoPE 在启动时自己计算 cos/sin 表。
    def get_skip_weight_names(self) -> list[str]:

        return ["rotary_emb.inv_freq"]

    # MoE 模型返回专家权重的加载器（第 10 阶段）；稠密模型没有专家，返回 None。
    def get_expert_checkpoint_loader(self) -> MoECheckpointLoader | None:
        """The loader for this rank's routed experts, or None without experts."""
        # 译：本 rank 路由专家的加载器；没有专家时为 None。

        return None

    @_record_loaded_names
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
    ) -> set[str]:
        """Load a (possibly partial) stream of checkpoint tensors.

        Returns the names of the parameters that received data; see the
        class docstring for the session contract.
        """
        # 译：加载一个（可能是部分的）checkpoint 张量流。
        #
        #     返回接收到数据的参数名；会话约定见类的 docstring。
        # 通用的权重加载流程。weights 是 (checkpoint 权重名, 张量) 的迭代器，
        # 由 model_loader 从 safetensors 文件逐个读出（第 8 阶段），这里逐个分发到参数上。
        # 每个张量按顺序尝试四条规则，命中一条就处理并进入下一个张量：
        #   1. 名字命中 get_skip_weight_names -> 跳过
        #   2. 是 MoE 专家权重 -> 交给专家加载器；不属于本 rank 的专家（EP 下在别的 rank）跳过
        #   3. 命中 get_stacked_params_mapping -> 改名为合并参数，带 shard_id 调用 weight_loader
        #   4. 其余 -> 同名参数，调用它的 weight_loader（没有的话用直接拷贝的默认加载器）
        # weight_loader 挂在参数对象上（由 Linear / Embedding 等层在创建参数时设置），
        # 负责“从完整的 checkpoint 张量里切出本 rank 的那一份，写到参数的正确位置”（第 2 阶段）。
        #
        # 例（Qwen3，单卡）：
        #   "model.layers.0.self_attn.q_proj.weight" [4096, 4096]
        #     -> 规则 3：改名为 "model.layers.0.self_attn.qkv_proj.weight"，shard_id="q"
        #     -> 写入 qkv_proj.weight [6144, 4096] 的第 0~4095 行
        #   "model.layers.0.input_layernorm.weight" [4096]
        #     -> 规则 4：同名参数，直接拷贝

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
                # 译：本 rank 匹配不上的专家属于另一个 EP rank。
                if expert_loader.matches(name):
                    loaded.add(expert_loader.load(name, loaded_weight))
                continue

            # for ... else：循环中 break 表示命中了规则 3；循环正常结束（一条都没命中）
            # 才会进入 else 分支，走规则 4
            for param_name, weight_name, shard_id in stacked_params_mapping:

                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                # 模型里没有对应参数时静默跳过（例如 checkpoint 带了 bias 而模型不用 bias，
                # 或者该层属于流水线的其他 stage）。注意：改名之后才发现不存在时，
                # 这里的 continue 会接着尝试映射表中的下一项，最终落到 else 分支再查一次。
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)

                break

            else:

                # 同上：模型里没有的参数静默跳过
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        # 全部张量加载完后，让子类从参数派生其他状态（例如转换成 kernel 要求的布局）。
        # 在线权重更新会话中，这次调用会被推迟到会话结束时（见 _derive_once）。
        self.post_load_weights()
        return loaded

    # ---- 在线权重更新会话（第 8 阶段详解）----
    # 强化学习训练时，训练端会在推理服务运行中把新权重分多块推送过来，每块调用一次
    # load_weights。begin_weight_update / end_weight_update 把这些调用括成一个会话：
    #   begin：进入会话，清空记录
    #   多次 load_weights：只写参数，派生被推迟
    #   end：校验完整性 -> 统一派生一次 -> 退出会话；发现 KV scale 不为 1 时最后报错拒绝
    #   abort：失败时退出会话，不做派生
    # Live weight updates (see the class docstring).
    # 译：在线权重更新（见类的 docstring）。

    def required_weight_update_names(self) -> set[str]:
        """Canonical checkpoint names required for a complete model update.

        Models that support discarding weights return their rank-local source
        contract, excluding aliases loaded by another model in the session.
        These names precede any loader renaming or fused-parameter mapping.
        """
        # 译：完整的模型更新所必需的规范 checkpoint 名字。
        #
        #     支持丢弃权重的模型返回本 rank 的来源约定，不包括由会话中另一个模型加载的别名。
        #     这些名字是在任何加载器改名或合并参数映射之前的名字。
        raise NotImplementedError(
            f"{type(self).__name__} does not support complete weight updates"
        )

    def weight_discard_preserved_tensors(self) -> tuple[torch.Tensor, ...]:
        """Non-checkpoint tensors that must survive discarded weight storage.

        An opted-in model lists only immutable state that cannot be rebuilt
        from the new weights. Model parameters and reconstructible derived
        tensors must not be retained as a CPU weight backup.
        """
        # 译：在权重存储被丢弃后仍必须保留的非 checkpoint 张量。
        #
        #     开启此功能的模型只列出无法从新权重重建的不可变状态。模型参数以及可以重建的派生张量
        #     不能作为 CPU 上的权重备份保留。
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
        # 译：进入在线权重更新会话。
        #
        #     异常：
        #         RuntimeError：已经有一个会话在进行中。
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
        # 译：退出会话，并执行一次被推迟的派生。
        #
        #     只有某个分块请求过时才运行 ``post_load_weights``，并且此时 ``_weight_update_loaded_names``
        #     仍然保留着内容，这样模型可以把非幂等的派生限制在本次更新实际替换的参数上。无论派生是否抛出
        #     异常，会话都会被关闭。推送了不等于 1 的 KV cache scale 的更新仍会被完整加载并派生，以保持
        #     模型一致，然后再被拒绝。
        #
        #     异常：
        #         RuntimeError：没有进行中的会话。
        #         ValueError：更新中带有不等于 1 的 KV cache scale。
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
        # 译：先退出会话：``post_load_weights`` 的包装器只在会话进行中时才推迟。
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
            # 译：把权重流交给 ``super()`` 的子类加载器会把同样的张量筛查两次；每个只报告一次。
            raise ValueError(
                f"{type(self).__name__}: "
                f"{non_unit_kv_scale_message(list(dict.fromkeys(rejected)))}"
            )

    def abort_weight_update(self) -> None:
        """Leave the session without deriving state (the update failed)."""
        # 译：退出会话，不做派生（更新失败了）。
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
        # 译：记住本次会话的 ``load_weights`` 写入了哪些参数。
        #
        #     基类会记录 ``load_weights`` 的返回值；显式调用本方法也无妨。
        if self._weight_update_loaded_names is not None:
            self._weight_update_loaded_names.update(names)

    # 子类覆盖：从加载好的参数派生其他状态。默认什么也不做（Qwen3 不需要）。
    def post_load_weights(self) -> None:
        """Derive state from the loaded parameters (see the class docstring)."""
        # 译：从加载好的参数派生状态（见类的 docstring）。

    # 下面两个方法用于投机解码时草稿模型与目标模型共享 embedding 和 lm_head（第 11 阶段）。
    # 流水线并行时 embedding 只在第一个 stage、lm_head 只在最后一个 stage，缺的一侧返回 None。
    def get_embed_and_head(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The embedding and LM-head weights a draft may share with this target.

        A pipeline stage holds only its own side: the embedding lives on the
        first stage and the head on the last, so the missing side is None
        rather than a dereference of an absent module.
        """
        # 译：草稿模型可以与本目标模型共享的 embedding 和 LM head 权重。
        #
        #     一个流水线 stage 只持有自己这一侧：embedding 在第一个 stage，head 在最后一个 stage，
        #     所以缺失的一侧为 None，而不是去访问一个不存在的模块。
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
        # 译：把目标模型的 embedding 和 LM head 权重作为别名放进本草稿模型。
        #
        #     ``embed=None`` 表示目标模型不共享 embedding（流水线的最后一个 stage：embedding 在第一个 stage 上）。
        #     普通草稿模型自己不保留 embedding，所以会拒绝；支持流水线的草稿模型会覆盖本方法，保留它的
        #     checkpoint 自带的 ``embed_tokens``，之后工厂会检查 ``get_embed_and_head`` 是否仍能报告一个 embedding。
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
