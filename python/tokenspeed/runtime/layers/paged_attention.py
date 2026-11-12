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

"""Paged attention.

A ``PagedAttention`` layer declares its *visibility* -- how far back the
kernel may look (``sliding_window_size``, a ``window_left`` mask) -- and
nothing about storage. Which cache group its KV lives in is the cache plan's
decision, bound onto the layer at executor startup by
:func:`bind_cache_groups`, which also checks the one relation the two
contracts must satisfy: a group must retain every token its layers can see.
"""

# 译：分页注意力。
#
#     ``PagedAttention`` 层声明的是它的“可见范围”——kernel 最远可以往回看多远（``sliding_window_size``，
#     即 ``window_left`` 掩码）——而不涉及任何存储。它的 KV 放在哪个 cache group 由 cache 规划决定，
#     在执行器启动时由 :func:`bind_cache_groups` 绑定到层上；该函数同时检查两份契约必须满足的唯一关系：
#     一个组必须保留它的层能看到的每一个 token。

# PagedAttention：模型里的注意力层（Qwen3Attention 中的 self.attn，见 models/qwen3.py）。
# 它本身没有权重（q/k/v/o 投影在模型的注意力模块里），负责两件事：
#   1. prologue（前奏）：对投影后的 q/k 做 QK-Norm、RoPE，把 k/v 写进 KV 池，返回处理后的 q
#      ——一个融合 kernel（tokenspeed_kernel 的 gqa_prologue）完成，设计见 docs/design/attention-prologue.md
#   2. attend：调用注意力后端（ctx.attn_backend，即 CacheGroupRouter）做核心注意力
#
# 一层注意力的数据流（Qwen3-8B 单卡，MIXED 批 T=301：A extend 300 + B decode 1）：
#   qkv_proj 输出拆成 q [301, 4096]、k [301, 1024]、v [301, 1024]
#       │ forward -> prologue（可被 CUDA Graph 捕获）
#       │   写入位置 slots = router.padded_write_locations(...)  [301]（A 的 300 个 + B 的 1 个）
#       │   gqa_prologue：q、k 做逐头 RMSNorm + RoPE；k、v 写入 k_buffer/v_buffer[slots]
#       ▼
#   q [301, 32, 128]（非 decode 时还返回 k、v）
#       │ attend（@break_point：breakable CUDA Graph 在这里切回 eager）
#       ▼   router.forward -> MHA 叶子 -> 注意力 kernel
#   输出 [301, 4096] -> o_proj
#
# 注意：与“先算注意力再写 cache”的写法不同，这里 KV 总是先写进 cache，核心注意力只读，后端不再写 KV。
# decode 时 prologue 不返回 k/v（return_kv=False），因为 decode kernel 直接从 cache 读。
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import torch
from tokenspeed_kernel.ops.attention.prologue import (
    FP8TensorScales,
    GQAPrologueOutput,
    HeadKVCache,
    HeadNorm,
    KVCacheFormat,
    LatentKVCache,
    MLAExpandedKV,
    MLAPrologueOutput,
    Rotary,
    gqa_prologue,
    latent_store,
    mla_prologue,
)
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import (
    token_all_gather,
    token_all_gather_rows,
)
from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots
from tokenspeed.runtime.model_loader.weight_utils import (
    K_CACHE_SCALE_SUFFIXES,
    V_CACHE_SCALE_SUFFIXES,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import DecodeKVView


# QueryShardGather 和后面的 latent_prologue* 属于 MLA 与 query-context 并行（第 12 阶段），初读可跳过。
@dataclass(frozen=True)
class QueryShardGather:
    """Where a query shard's latent rows come together before the KV write.

    Under query context parallelism every rank rotates its own rows and the
    prologue all-gathers the rotated latent over ``group`` (per-rank counts
    ``plan.row_counts``) and puts it in span order (``plan.to_span_order``)
    before the owner-masked store of the whole extend span, so page ownership
    stays independent of the query shard.

    Attributes:
        plan: The forward's shard plan (``ctx.query_shard``).
        group: The query-context-parallel group (``mapping.attn.qcp_group``).
    """

    # 译：query 分片的潜在行在写 KV 之前于何处汇合。
    #
    #     在 query-context 并行下，每个 rank 旋转自己的行，prologue 在 ``group`` 上 all-gather 旋转后的潜在向量
    #     （每个 rank 的行数为 ``plan.row_counts``），并把它们排成段顺序（``plan.to_span_order``），然后对整个 extend 段
    #     做按所有者掩码的存储，这样页的归属与 query 分片无关。
    #
    #     属性：
    #         plan：本次前向的分片计划（``ctx.query_shard``）。
    #         group：query-context 并行组（``mapping.attn.qcp_group``）。

    plan: QueryShardPlan
    group: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.group) != self.plan.size:
            raise ValueError(
                f"query shard group of {len(self.group)} ranks does not match a "
                f"plan over {self.plan.size}"
            )


def hf_sliding_window_to_window_left(sliding_window: int) -> int:
    """HF sliding windows count the current token; kernels take the number of
    earlier tokens still visible (``window_left``)."""
    # 译：HF 的滑动窗口包含当前 token；kernel 接受的是仍然可见的“更早 token 数”（``window_left``）。
    # 例：HF sliding_window=4096 -> window_left=4095（当前 token 加上前 4095 个，共 4096 个可见）。
    return int(sliding_window) - 1


class HeadRotary(Protocol):
    """A rotary embedding the prologue applies."""

    # 译：prologue 应用的旋转位置编码。

    def as_rotary(self, positions: torch.Tensor) -> Rotary: ...


class HeadRMSNorm(Protocol):
    """A per-head RMSNorm the prologue applies: ``x * rsqrt(mean(x^2) + eps) *
    (weight + weight_offset)``."""

    # 译：prologue 应用的逐头 RMSNorm：``x * rsqrt(mean(x^2) + eps) * (weight + weight_offset)``。

    weight: torch.Tensor
    weight_offset: float
    variance_epsilon: float


def head_norm(q_norm: HeadRMSNorm, k_norm: HeadRMSNorm) -> HeadNorm:
    """The prologue's norm step for a layer's per-head query and key norms."""
    # 译：为一层的逐头 query norm 和 key norm 构造 prologue 的 norm 步骤。
    # Qwen3 的 q_norm / k_norm：weight [128]，weight_offset=0，eps=1e-6（Gemma 系的 offset 为 1）。
    # 两者共用 eps 和 offset，所以融合 kernel 只需一组标量。
    if (q_norm.weight_offset, q_norm.variance_epsilon) != (
        k_norm.weight_offset,
        k_norm.variance_epsilon,
    ):
        raise ValueError("query and key norms must share epsilon and weight offset")
    return HeadNorm(
        q_norm.weight, k_norm.weight, q_norm.weight_offset, q_norm.variance_epsilon
    )


class PagedAttention(nn.Module):
    """
    The attention layer implementation.
    """

    # 译：注意力层的实现。

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        sliding_window_sink: int = 0,
        *,
        rotary_emb: HeadRotary | None,
        qk_norm: tuple[HeadRMSNorm, HeadRMSNorm] | None,
    ):
        """``rotary_emb`` and ``qk_norm`` are the steps the attention prologue
        applies before core attention; ``None`` skips a step."""
        # 译：``rotary_emb`` 和 ``qk_norm`` 是注意力 prologue 在核心注意力之前应用的步骤；``None`` 表示跳过该步。
        # Qwen3-8B 单卡：num_heads=32、num_kv_heads=8、head_dim=128、scaling=128^-0.5、sliding_window_size=-1（全注意力）。
        # 注意：num_heads / num_kv_heads 是本 rank 的头数（已除以注意力 TP），所以字段名带 tp_ 前缀。
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        # window_left of the compute mask: -1 for full attention, 0 for the current token only.
        # 译：计算掩码的 window_left：全注意力为 -1，只看当前 token 为 0。
        if sliding_window_size is None or sliding_window_size < -1:
            raise ValueError(
                f"PagedAttention layer_id={layer_id}: sliding_window_size is a "
                f"window_left >= 0 or -1 for full attention, got "
                f"{sliding_window_size!r}"
            )
        self.sliding_window_size = int(sliding_window_size)
        # Leading tokens kept visible outside the window (attention sinks).
        # 译：在窗口外仍保持可见的开头若干 token（attention sink）。
        if sliding_window_sink is None or sliding_window_sink < 0:
            raise ValueError(
                f"PagedAttention layer_id={layer_id}: sliding_window_sink is a "
                f"token count >= 0, got {sliding_window_sink!r}"
            )
        if sliding_window_sink and sliding_window_size < 0:
            raise ValueError(
                f"PagedAttention layer_id={layer_id}: sliding_window_sink="
                f"{sliding_window_sink} needs a sliding window; full attention "
                "already keeps those rows"
            )
        self.sliding_window_sink = int(sliding_window_sink)
        # The cache group this layer's KV rides, bound at startup by bind_cache_groups.
        # 译：本层 KV 所在的 cache group，启动时由 bind_cache_groups 绑定。
        self._group_id: str | None = None
        self.rotary_emb = rotary_emb
        self.qk_norm = qk_norm
        # Per-tensor FP8 KV-cache scales (None at unit); a non-FP8 cache ignores them.
        # 译：逐张量的 FP8 KV cache scale（缺省为 None，即 scale 为 1）；非 FP8 的 cache 忽略它们。
        self.fp8_kv_scales: FP8TensorScales | None = None

    def load_kv_cache_scale(self, name: str, scale: torch.Tensor) -> None:
        """Take a checkpoint's key, value or shared KV-cache scale tensor."""
        # 译：接收 checkpoint 中的 key、value 或共享的 KV cache scale 张量。
        if scale.numel() != 1:
            raise NotImplementedError(
                f"PagedAttention layer_id={self.layer_id}: {name} holds "
                f"{scale.numel()} scales; only per-tensor KV-cache scales are "
                "supported"
            )
        k, v = (
            (1.0, 1.0)
            if self.fp8_kv_scales is None
            else (self.fp8_kv_scales.k, self.fp8_kv_scales.v)
        )
        if name.endswith(K_CACHE_SCALE_SUFFIXES):
            k = float(scale.item())
        elif name.endswith(V_CACHE_SCALE_SUFFIXES):
            v = float(scale.item())
        elif name.endswith(".kv_scale"):
            k = v = float(scale.item())
        else:
            raise KeyError(f"{name!r} is not a KV-cache scale tensor")
        self.fp8_kv_scales = None if k == v == 1.0 else FP8TensorScales(k, v)

    @property
    def group_id(self) -> str:
        if self._group_id is None:
            raise RuntimeError(
                f"PagedAttention layer_id={self.layer_id} has no cache group "
                "bound; bind_cache_groups runs at executor startup, before any "
                "forward."
            )
        return self._group_id

    def bind_cache_group(self, group_id: str) -> None:
        """Bind the plan's group for this layer; rebinding to another group
        is a contract bug, not a rename."""
        # 译：绑定规划为本层指定的组；重新绑定到另一个组是契约错误，而不是改名。
        if not group_id:
            raise ValueError(
                f"PagedAttention layer_id={self.layer_id}: cache group id must "
                "be nonempty"
            )
        if self._group_id is not None and self._group_id != group_id:
            raise ValueError(
                f"PagedAttention layer_id={self.layer_id} is bound to cache "
                f"group {self._group_id!r}; cannot rebind to {group_id!r}"
            )
        self._group_id = group_id

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        positions: torch.Tensor | None,
        ctx: ForwardContext,
        **kwargs,
    ) -> torch.Tensor:
        """Run this layer's attention.

        A GQA layer given K/V takes its projected rows: the prologue normalizes
        and rotates them and writes K/V at the backend's write locations, padded
        to the rows the forward carries so a graph can record it; core attention
        runs in the eager break (:meth:`attend`). ``k = v = None`` means the
        inputs are prepared and the cache written (by :meth:`prologue`, or by
        an MLA layer's :meth:`latent_prologue`).
        """
        # 译：运行本层的注意力。
        #
        #     给了 K/V 的 GQA 层接收的是投影后的行：prologue 对它们做归一化和旋转，并把 K/V 写到后端给出的写入位置，
        #     写入位置 padding 到本次前向所携带的行数，以便 graph 可以记录它；核心注意力在 eager 断点中运行（:meth:`attend`）。
        #     ``k = v = None`` 表示输入已经准备好、cache 已经写入（由 :meth:`prologue`，或由 MLA 层的 :meth:`latent_prologue`）。
        # idle（attention DP 下本 rank 没有请求的空跑前向）不跑 prologue，直接走 attend。
        if k is not None and v is None:
            raise ValueError("v must be provided when k is provided.")
        if k is not None and not ctx.forward_mode.is_idle():
            out = self.prologue(q, k, v, positions, ctx)
            q, k, v = out.q, out.k, out.v
        return self.attend(q, k, v, ctx, **kwargs)

    @break_point
    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        ctx: ForwardContext,
        **kwargs,
    ) -> torch.Tensor:
        """Core attention over prepared inputs, the eager break: the prologue's
        query and, for a forward that is not a decode, its returned key and
        value rows. A model that runs :meth:`prologue` inside its own stream
        scope calls this afterwards."""
        # 译：对准备好的输入做核心注意力，即 eager 断点：prologue 产出的 query，以及（非 decode 前向时）它返回的 key 和 value 行。
        #     在自己的流作用域内运行 :meth:`prologue` 的模型会在之后调用这个方法。
        # save_kv_cache=False：KV 已由 prologue 写好，后端只读（router 中有断言）。
        # ctx.attn_backend 是 CacheGroupRouter，它按本层的 group_id 找到叶子后端（Qwen3 为 MHA 叶子）。
        if k is not None:
            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        return ctx.attn_backend.forward(
            q,
            k,
            v,
            self,
            ctx.token_to_kv_pool,
            ctx.forward_mode,
            ctx.bs,
            save_kv_cache=False,
            ctx=ctx,
            **kwargs,
        )

    def decode_kv_view(self, ctx: ForwardContext) -> "DecodeKVView | None":
        """This layer's paged cache and decode metadata, for a kernel that runs
        its decode attention inside a larger launch, or None when the backend
        does not expose them (:meth:`AttentionBackend.decode_kv_view`) or the
        forward is not a decode."""
        # 译：本层的分页 cache 和 decode 元数据，供在更大的一次启动中运行其 decode 注意力的 kernel 使用；
        #     若后端不暴露它们（:meth:`AttentionBackend.decode_kv_view`）或本次前向不是 decode，则返回 None。
        if not ctx.forward_mode.is_decode():
            return None
        return ctx.attn_backend.decode_kv_view(self, ctx.token_to_kv_pool)

    def prologue(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
    ) -> GQAPrologueOutput:
        """Run a GQA layer's prologue alone: prepare ``q`` and write K/V.

        :meth:`forward` calls this before core attention; a narrowed draft step
        calls it and then attends its live rows.

        Returns:
            The prepared query, and the key/value rows unless the forward
            decodes (decode attention reads the cache).
        """
        # 译：单独运行 GQA 层的 prologue：准备 ``q`` 并写入 K/V。
        #
        #     :meth:`forward` 在核心注意力之前调用它；收窄的草稿步调用它，然后只关注其活跃行。
        #
        #     返回：
        #         准备好的 query，以及 key/value 行（decode 前向除外：decode 注意力从 cache 读）。
        # gqa_prologue 的参数：
        #   norm / rotary：QK-Norm 和 RoPE（positions 决定每个 token 的旋转角）
        #   cache：写入目标（KV 池中本层的 k/v 缓冲区 + 写入槽位），见 _write_target
        #   query_dtype：返回的 q 的 dtype（FP8 cache 的 decode 直接输出 FP8 query，省一次转换）
        #   solution / override：kernel 选择，None 表示由 tokenspeed_kernel registry 自动选择
        norm, rotary = self._steps(positions)
        return gqa_prologue(
            q,
            k,
            v,
            norm=norm,
            rotary=rotary,
            cache=self._write_target(q, ctx),
            query_dtype=ctx.attn_backend.query_dtype(self, ctx.forward_mode, q.dtype),
            return_kv=not ctx.forward_mode.is_decode(),
            solution=None,
            override=None,
        )

    def store_prepared(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ForwardContext,
    ) -> GQAPrologueOutput:
        """Write rows whose projection or collective already applied norm/RoPE.

        The producer must match this layer's prologue numerical contract.
        This method owns the usual padded write slots and cache conversion;
        call :meth:`attend` with its result to avoid repeating the head steps.
        """
        # 译：写入那些在投影或集合通信中已经应用过 norm/RoPE 的行。
        #
        #     生产者必须符合本层 prologue 的数值契约。这个方法负责通常的 padding 写入槽位和 cache 格式转换；
        #     用它的结果调用 :meth:`attend`，以免重复做逐头的步骤。
        return gqa_prologue(
            q,
            k,
            v,
            norm=None,
            rotary=None,
            cache=self._write_target(q, ctx),
            query_dtype=ctx.attn_backend.query_dtype(self, ctx.forward_mode, q.dtype),
            return_kv=not ctx.forward_mode.is_decode(),
            solution=None,
            override=None,
        )

    def _steps(
        self, positions: torch.Tensor | None
    ) -> tuple[HeadNorm | None, Rotary | None]:
        return (
            None if self.qk_norm is None else head_norm(*self.qk_norm),
            None if self.rotary_emb is None else self.rotary_emb.as_rotary(positions),
        )

    def _write_target(
        self, q: torch.Tensor, ctx: ForwardContext
    ) -> HeadKVCache | LatentKVCache:
        # rows = q.shape[0] = 本次前向的 token 行数（可能含 prefill graph 的 padding 行）。
        # 例：MIXED 批 301 行 -> slots [301]；padding 行的槽位为 0（哑页）。
        slots = ctx.attn_backend.padded_write_locations(
            self, ctx.forward_mode, q.shape[0]
        )
        return self._local_target(slots, ctx)

    def _local_target(
        self, slots: torch.Tensor, ctx: ForwardContext
    ) -> HeadKVCache | LatentKVCache:
        """The pool's target for ``slots``, local to this rank's shard under KVP."""
        # 译：``slots`` 在 pool 中的写入目标；在 KVP 下只限于本 rank 的分片。
        # 单卡 / 不开 KVP 时 cache_placement 为 None，slots 原样返回；kv_write_target 返回本层 k/v 缓冲区与槽位的组合
        # （HeadKVCache，见 kv_cache/mha.py）。
        slots, owned = resolve_cache_slots(
            slots, ctx.attn_backend.cache_placement(self)
        )
        target = ctx.token_to_kv_pool.kv_write_target(self.layer_id, slots, owned)
        if (
            self.fp8_kv_scales is not None
            and isinstance(target, HeadKVCache)
            and target.format is KVCacheFormat.FP8
        ):
            # Every reader of this cache must apply the scales it was written at.
            # 译：这个 cache 的每个读取者都必须应用写入时所用的 scale。
            if not ctx.attn_backend.reads_fp8_kv_scales(self):
                raise NotImplementedError(
                    f"layer {self.layer_id} has per-tensor KV-cache scales "
                    f"{self.fp8_kv_scales}, which "
                    f"{type(ctx.attn_backend).__name__} does not read"
                )
            target = replace(target, scales=self.fp8_kv_scales)
        return target

    def attend_live_rows(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """A narrowed draft's first step: write every row's K/V, then attend
        only the live rows (``ctx.gather_ids``) as a decode over the accepted
        prefix."""
        # 译：收窄草稿的第一步：写入每一行的 K/V，然后只把活跃行（``ctx.gather_ids``）当作基于已接受前缀的 decode 来做注意力。
        ctx.draft_narrowing.publish_accepted_prefix()
        decode_ctx = replace(ctx, forward_mode=ForwardMode.DECODE)
        q = self.prologue(q, k, v, positions, decode_ctx).q
        return self.forward(
            q.index_select(0, ctx.gather_ids),
            k=None,
            v=None,
            positions=None,
            ctx=decode_ctx,
            # The DECODE dispatch would skip the PD cache step a catch-up round records.
            # 译：DECODE 分派会跳过追赶轮所记录的 PD cache 步。
            record_kv_cache=not ctx.forward_mode.is_decode_or_idle(),
        )

    def latent_prologue(
        self,
        query: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        *,
        slots: torch.Tensor,
        expanded: MLAExpandedKV | None,
        key_rows: QueryShardGather | None,
    ) -> MLAPrologueOutput:
        """Run an MLA layer's prologue: rotate, write the latent rows to
        ``slots``, and return attention inputs (FP8 for an FP8 cache, not
        per-token-head planes).

        Args:
            query: ``[num_tokens, num_heads, q_nope_dim + rope_dim]`` whose
                leading channels hold the query's non-RoPE part.
            q_pe: Unrotated query RoPE part; it may alias ``query``.
            latent_cache: Normalized latent and unrotated key RoPE part.
            positions: Token positions.
            ctx: Forward context.
            slots: Cache slots of the leading latent rows to write. Under a
                query shard this is the whole extend span (every rank holds
                it) while the rows above are this rank's shard.
            expanded: Per-head keys and values for non-absorbed attention.
            key_rows: The query shard's gather, or ``None`` when the rows
                are the whole span. With a shard the prologue rotates the
                local rows, all-gathers the rotated latent to the span and
                stores it owner-masked, so every rank writes the pages it
                owns whichever rank computed the row.
        """
        # 译：运行 MLA 层的 prologue：旋转，把潜在行写到 ``slots``，并返回注意力的输入（FP8 cache 时为 FP8，不是逐 token 逐头的平面）。
        #
        #     参数：
        #         query：``[num_tokens, num_heads, q_nope_dim + rope_dim]``，开头的通道放着 query 的非 RoPE 部分。
        #         q_pe：未旋转的 query RoPE 部分；可以与 ``query`` 是同一块存储。
        #         latent_cache：归一化后的潜在向量和未旋转的 key RoPE 部分。
        #         positions：token 位置。
        #         ctx：前向上下文。
        #         slots：要写入的开头若干潜在行的 cache 槽位。在 query 分片下，这是整个 extend 段（每个 rank 都持有），
        #             而上面的那些行是本 rank 的分片。
        #         expanded：非吸收（non-absorbed）注意力用的逐头 key 和 value。
        #         key_rows：query 分片的收集信息；行就是整个段时为 ``None``。有分片时，prologue 旋转本地行，把旋转后的
        #             潜在向量 all-gather 成整个段，再按所有者掩码存储，这样无论哪个 rank 计算了某行，每个 rank 都写它拥有的页。
        rotary = (
            None if self.rotary_emb is None else self.rotary_emb.as_rotary(positions)
        )
        if key_rows is None:
            return mla_prologue(
                query,
                q_pe,
                latent_cache,
                expanded=expanded,
                rotary=rotary,
                cache=self._local_target(slots, ctx),
                solution=None,
                override=None,
            )
        if expanded is not None:
            raise ValueError(
                "a query shard writes its latent through the absorbed prologue; "
                "expanded attention cannot gather per-head keys"
            )
        plan = key_rows.plan
        if latent_cache.shape[0] != plan.local_rows:
            raise ValueError(
                f"query shard rank {plan.rank} rotates {plan.local_rows} rows, "
                f"got {latent_cache.shape[0]}"
            )
        if slots.numel() != plan.total_rows:
            raise ValueError(
                f"a sharded latent store takes the whole span of {plan.total_rows} "
                f"slots, got {slots.numel()}"
            )
        if plan.local_rows > 0:
            out = mla_prologue(
                query,
                q_pe,
                latent_cache,
                expanded=None,
                rotary=rotary,
                cache=None,
                solution=None,
                override=None,
            )
        else:
            # An empty shard rotates nothing but still joins the gather and
            # stores the rows it owns of what the other ranks computed.
            # 译：空分片什么都不旋转，但仍参与收集，并存储其他 rank 计算的行中自己拥有的那些。
            out = MLAPrologueOutput(
                query=query, key=None, value=None, latent=latent_cache
            )
        gathered = plan.to_span_order(
            token_all_gather(
                out.latent.contiguous(), key_rows.group, list(plan.row_counts)
            )
        )
        latent_store(
            gathered,
            kv_lora_rank=out.latent.shape[-1] - q_pe.shape[-1],
            cache=self._local_target(slots, ctx),
        )
        return out

    def latent_prologue_with_rows(
        self,
        query: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        *,
        slots: torch.Tensor,
        key_rows: QueryShardGather,
        extra_rows: torch.Tensor,
    ) -> tuple[MLAPrologueOutput, torch.Tensor]:
        """A query shard's prologue with extra rows riding its gather.

        ``extra_rows`` (``[local_rows, width]``, the latent's dtype) are
        concatenated after the rotated latent columns and all-gathered in the
        same collective as the shard's latent rows; they come back in span
        order. ``width`` must keep the latent half of the payload 16-byte
        aligned (a multiple of 8 fp16/bf16 elements), which the latent store
        slices require. The latent is rotated and stored exactly as
        :meth:`latent_prologue` does -- only the wiring differs, so the store
        writes the same bytes. Returns the prologue output and the gathered
        extra rows (``[total_rows, width]``).
        """
        # 译：query 分片的 prologue，并让额外的行搭乘它的收集。
        #
        #     ``extra_rows``（``[local_rows, width]``，与潜在向量同 dtype）被拼接在旋转后的潜在列之后，
        #     与分片的潜在行在同一次集合通信中 all-gather；它们以段顺序返回。``width`` 必须让负载中潜在向量那一半
        #     保持 16 字节对齐（8 个 fp16/bf16 元素的倍数），这是潜在存储的切片所要求的。潜在向量的旋转和存储与
        #     :meth:`latent_prologue` 完全相同——只是连接方式不同，所以存储写入的字节相同。返回 prologue 的输出和
        #     收集到的额外行（``[total_rows, width]``）。
        plan = key_rows.plan
        if latent_cache.shape[0] != plan.local_rows:
            raise ValueError(
                f"query shard rank {plan.rank} rotates {plan.local_rows} rows, "
                f"got {latent_cache.shape[0]}"
            )
        if slots.numel() != plan.total_rows:
            raise ValueError(
                f"a sharded latent store takes the whole span of {plan.total_rows} "
                f"slots, got {slots.numel()}"
            )
        if (
            extra_rows.dim() != 2
            or extra_rows.shape[0] != plan.local_rows
            or extra_rows.dtype != latent_cache.dtype
        ):
            raise ValueError(
                "extra rows must be a [local_rows, width] tensor of the latent's "
                f"dtype: got {tuple(extra_rows.shape)} {extra_rows.dtype} for "
                f"{plan.local_rows} rows of {latent_cache.dtype}"
            )
        if extra_rows.shape[1] * extra_rows.element_size() % 16:
            raise ValueError(
                "extra rows must keep the gathered latent half 16-byte aligned: "
                f"got {extra_rows.shape[1]} elements of "
                f"{extra_rows.element_size()} bytes, which is not a multiple of 16"
            )
        rotary = (
            None if self.rotary_emb is None else self.rotary_emb.as_rotary(positions)
        )
        if plan.local_rows > 0:
            out = mla_prologue(
                query,
                q_pe,
                latent_cache,
                expanded=None,
                rotary=rotary,
                cache=None,
                solution=None,
                override=None,
            )
        else:
            out = MLAPrologueOutput(
                query=query, key=None, value=None, latent=latent_cache
            )
        width = extra_rows.shape[1]
        payload = torch.cat((extra_rows, out.latent), dim=1).contiguous()
        gathered = plan.to_span_order(
            token_all_gather_rows(payload, key_rows.group, list(plan.row_counts))
        )
        latent_store(
            gathered[:, width:],
            kv_lora_rank=out.latent.shape[-1] - q_pe.shape[-1],
            cache=self._local_target(slots, ctx),
        )
        return out, gathered[:, :width].contiguous()


class _CacheGroupSpecLike(Protocol):
    group_id: str
    attn_kind: str
    sliding_window_tokens: int | None


class _CacheArenaLike(Protocol):
    @property
    def cache_group_specs(self) -> tuple[_CacheGroupSpecLike, ...]: ...


class _CacheViewLike(Protocol):
    """What :func:`bind_cache_groups` reads from a cache pool (duck-typed:
    the pool package imports this module)."""

    # 译：这是 :func:`bind_cache_groups` 从 cache pool 读取的内容（鸭子类型：pool 包会导入本模块）。

    arena: _CacheArenaLike

    def history_group_by_layer(self) -> Mapping[int, str]: ...


def bind_cache_groups(model: nn.Module, cache_pool: _CacheViewLike) -> None:
    """Bind every ``PagedAttention`` layer to the cache group the plan
    declared its KV planes in, checking retention covers visibility.

    Fails fast (ValueError) at startup instead of a KeyError deep in the
    backend, possibly during graph capture. The plan is the single record
    of layer -> group, so the model side carries no group vocabulary at all;
    the one thing a layer must satisfy is that its group keeps every token
    its mask can reach: a full-visibility layer cannot ride a sliding group,
    and a sliding mask must fit inside the group's retention window.
    """
    # 译：把每个 ``PagedAttention`` 层绑定到规划中声明其 KV 平面所在的 cache group，并检查保留范围覆盖可见范围。
    #
    #     在启动时快速失败（ValueError），而不是在后端深处（可能在 graph 捕获期间）抛 KeyError。规划是“层 -> 组”
    #     的唯一记录，所以模型侧完全不涉及组的概念；层唯一必须满足的是：它的组保留其掩码能触及的每个 token：
    #     全可见的层不能放在滑动窗口组里，滑动窗口掩码必须落在组的保留窗口之内。
    # 例：Qwen3-8B 36 层都是全注意力，规划只有一个 full_history 组，36 个 PagedAttention 都绑到它上面。
    specs = {str(spec.group_id): spec for spec in cache_pool.arena.cache_group_specs}
    group_by_layer = cache_pool.history_group_by_layer()
    model_name = type(model).__name__
    for name, module in model.named_modules():
        if not isinstance(module, PagedAttention):
            continue
        group_id = group_by_layer.get(module.layer_id)
        if group_id is None:
            raise ValueError(
                f"{model_name}: attention layer {name!r} (layer_id="
                f"{module.layer_id}) has no history cache group in the "
                f"pool's plan (planned layers: {sorted(group_by_layer)})."
            )
        _check_visibility_within_retention(model_name, name, module, specs[group_id])
        module.bind_cache_group(group_id)


def _check_visibility_within_retention(
    model_name: str, name: str, layer: PagedAttention, spec: _CacheGroupSpecLike
) -> None:
    if spec.attn_kind != "sliding_window":
        return
    window_left = layer.sliding_window_size
    retained = spec.sliding_window_tokens
    if window_left < 0:
        raise ValueError(
            f"{model_name}: attention layer {name!r} (layer_id={layer.layer_id}) "
            f"sees the full history but its cache group {spec.group_id!r} "
            f"retains only a {retained}-token window."
        )
    # The scheduler keeps the last `retained - 1` computed tokens ahead of
    # the next position (GroupGeometry::ExpiredBlocksAt), exactly the HF
    # window minus the current token.
    # 译：调度器在下一个位置之前保留最近 `retained - 1` 个已计算的 token（GroupGeometry::ExpiredBlocksAt），
    #     正好是 HF 窗口减去当前 token。
    if retained is None or window_left + 1 > retained:
        raise ValueError(
            f"{model_name}: attention layer {name!r} (layer_id={layer.layer_id}) "
            f"masks to window_left={window_left} but its cache group "
            f"{spec.group_id!r} retains only a {retained}-token window."
        )


def check_block_drafter_storage(
    draft_model: nn.Module, target_cache_pool: _CacheViewLike
) -> None:
    """A block drafter writes at the target's cache locations, so every one of
    its layers must ride a full-history group the target's own layers share.

    Bind the draft first (:func:`bind_cache_groups`). Without a target
    full-history group there is nothing to borrow: the draft's KV would sit
    in a group of its own while the target router hands it another group's
    slots.

    A draft with its own group would read history through its own router while
    its KV lands at the target's locations, so the two stop naming the same
    rows. Nothing raises; the speculation is simply wasted.
    """
    # 译：块草稿模型在目标模型的 cache 位置写入，所以它的每一层都必须放在一个与目标模型自身层共享的全历史组上。
    #
    #     先绑定草稿（:func:`bind_cache_groups`）。没有目标模型的全历史组就没有可借用的：草稿的 KV 会落在
    #     它自己的组里，而目标 router 交给它的却是另一个组的槽位。
    #
    #     拥有自己组的草稿会通过自己的 router 读历史，而它的 KV 却落在目标模型的位置上，于是两者不再指向同样的行。
    #     不会报任何错；投机只是白白浪费了。
    target_groups = set(target_cache_pool.history_group_by_layer().values())
    specs = {
        str(spec.group_id): spec for spec in target_cache_pool.arena.cache_group_specs
    }
    model_name = type(draft_model).__name__
    for name, module in draft_model.named_modules():
        if not isinstance(module, PagedAttention):
            continue
        group_id = module.group_id
        if group_id not in target_groups or specs[group_id].attn_kind != "full_history":
            raise ValueError(
                f"{model_name}: block drafter layer {name!r} (layer_id="
                f"{module.layer_id}) rides cache group {group_id!r}, but a block "
                "drafter must share a full-history group with the target's "
                f"layers (target groups: {sorted(target_groups)})."
            )
