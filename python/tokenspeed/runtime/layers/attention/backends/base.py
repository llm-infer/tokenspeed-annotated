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

"""The runner-facing attention backend contract.

An ``AttentionBackend`` is what ``ForwardStepRunner`` / ``ModelExecutor`` /
``PrefillGraph`` talk to: it receives the scheduler bridge's per-group
``block_tables`` (raw scheduler blocks, batch-ordered) on every metadata
call and answers the model's ``forward``. Three kinds of node implement it:

* ``CacheGroupRouter`` — the paged-KV composite that maps blocks to kernel
  pages once and fans out to ``PagedAttentionBackend`` leaves (``paged.py``);
* composites that wrap a router next to a state consumer (hybrid GDN/KDA,
  Inkling's conv columns, MSA's sparse layers);
* block consumers that read raw tables of their own state groups (Mamba /
  KDA state paging, DeepSeek V4.1's bespoke multi-group backend).

``block_tables`` is always a complete mapping — the runner synthesizes
placeholder tables for capture, idle and warmup — so no implementation
carries a "no tables" arm. Padding is the consumer's job: requests in
``[actual_bs, bs)`` are dummies it must route to the null page itself.
"""

# 译：面向 runner 的注意力后端契约。
#
#     ``AttentionBackend`` 是 ``ForwardStepRunner`` / ``ModelExecutor`` / ``PrefillGraph`` 打交道的对象：
#     每次元数据调用它都会收到调度器桥接层的按组 ``block_tables``（原始调度器 block，按批次排序），
#     并负责响应模型的 ``forward``。有三类节点实现它：
#
#     * ``CacheGroupRouter``——分页 KV 的复合节点，一次性把 block 映射为 kernel 页，
#       再扇出到 ``PagedAttentionBackend`` 叶子（``paged.py``）；
#     * 把 router 与一个状态消费者包装在一起的复合节点（混合 GDN/KDA、Inkling 的卷积列、MSA 的稀疏层）；
#     * 读取自己状态组原始表的 block 消费者（Mamba / KDA 的状态分页、DeepSeek V4.1 定制的多组后端）。
#
#     ``block_tables`` 总是一个完整的映射——runner 会为捕获、idle 和预热合成占位表——所以没有实现需要
#     “没有表”的分支。padding 是消费者的职责：``[actual_bs, bs)`` 中的请求是哑请求，消费者必须自己把它们导向空页。

# 注意力后端的抽象基类。所有注意力后端（router、各种叶子、复合节点）都继承 AttentionBackend。
#
# 后端构成一棵树（Qwen3-8B 只有两层）：
#   CacheGroupRouter（runner 直接持有的根节点，见 paged/router.py）
#     └── MHAAttnBackend（叶子，按 cache group 一个，见 paged/mha.py）
# 混合模型会在 router 外面再包一层复合节点（第 13 阶段）。树上的很多方法（绑定 pool、offload、draft tree）
# 都是“对每个子节点广播”的默认实现，child_backends() 返回子节点列表。
#
# 一次前向中 runner 调用后端的顺序：
#   1. 元数据：init_forward_metadata（extend / mixed / idle）或 refresh_decode_metadata（decode，含 graph 回放）
#   2. 每层 prologue：padded_write_locations -> 把本层 K/V 写入 KV 池
#   3. 每层核心注意力：forward -> forward_extend / forward_decode
#
# 注意：decode 元数据只有“原地刷新”一条路径（refresh_decode_metadata），没有每步新建张量的 decode 分支。
# 这是 eager decode 与 CUDA Graph 回放共用一条路径的前提（docs/design/unified_path.md）。
from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.execution.slot_state import (
    pack_slot_rows,
    slot_state_image_bytes,
    unpack_slot_rows,
)
from tokenspeed.runtime.layers.attention.backends.support import (  # noqa: F401
    CudaGraphSupport,
    TreeSupport,
    resolve_cuda_graph_support,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.query_shard import QueryShardPlan
    from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
        TreeDraftInputs,
        TreeVerifyInputs,
    )
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.dcp.cache import (
        HistoryGatherWorkspace,
    )
    from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.pd.utils import StepCounter


# decode monokernel（第 18 阶段）会把整层融合成一个 kernel，它需要直接拿到 KV cache 和页表，这里打包给它。
@dataclass(frozen=True)
class DecodeKVView:
    """The paged cache and decode metadata one layer's decode attention reads.

    For a kernel that runs this layer's decode attention inside a larger
    launch instead of through the backend's ``forward``.
    """

    # 译：一层的 decode 注意力读取的分页 cache 和 decode 元数据。
    #
    #     供那些在更大的一次启动中运行本层 decode 注意力、而不是通过后端 ``forward`` 运行的 kernel 使用。

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen_k: int


# 稀疏注意力（DSA / QSA）专用，第 12 阶段再看。
@dataclass
class SparseTopKShare:
    """The sparse indexer's selection shared across one forward's layers.

    Sparse attention (GLM DSA, Qwen4-Exp QSA) selects the KV positions each
    query row attends to with an indexer; a model may declare layers that
    carry no indexer and reuse the previous indexer layer's selection
    (``indexer_types == "shared"``), and an MTP head may reuse the target's
    selection for its draft steps. The indexer layer that computes a
    selection publishes it here; the consumers read it. It is the sparse
    kernels' per-forward metadata, so it lives on the backend like the rest
    — as scratch outside the graph-guarded metadata slots — and the node
    clears it when it builds the next forward's metadata; between draft
    steps the drafter carries it across explicitly.

    ``prefill`` covers the extend rows, ``decode`` the decode rows (or the
    verify window); each family stores its own record type. ``qsa_metadata``
    holds QSA's layer-invariant row geometry so it is built once per forward;
    ``qsa_selection`` retains its logical candidates alongside physical slots.
    """

    # 译：稀疏索引器的选择结果，在一次前向的各层之间共享。
    #
    #     稀疏注意力（GLM DSA、Qwen4-Exp QSA）用索引器为每个 query 行选出要关注的 KV 位置；模型可以声明一些
    #     不带索引器、复用上一个索引器层选择结果的层（``indexer_types == "shared"``），MTP 头也可以在草稿步中复用
    #     目标模型的选择。计算出选择的索引器层把它发布在这里；消费者读取它。它是稀疏 kernel 的逐前向元数据，
    #     所以和其余元数据一样放在后端上——作为 graph 守护的元数据槽之外的暂存——节点在构建下一次前向的元数据时
    #     清空它；在草稿步之间由草稿模型显式携带过去。
    #
    #     ``prefill`` 覆盖 extend 行，``decode`` 覆盖 decode 行（或验证窗口）；每个家族存放自己的记录类型。
    #     ``qsa_metadata`` 保存 QSA 与层无关的行几何信息，所以每次前向只构建一次；
    #     ``qsa_selection`` 在物理槽位之外还保留其逻辑候选。

    prefill: Any | None = None
    decode: Any | None = None
    qsa_metadata: Any | None = None
    qsa_selection: Any | None = None

    def clear(self) -> None:
        self.prefill = None
        self.decode = None
        self.qsa_metadata = None
        self.qsa_selection = None


class CachePoolBinding:
    """A node's bound cache pool."""

    # 译：一个节点所绑定的 cache pool。

    #: Slot-sized tensors a retraction image leaves out (``SlotStateExporter``):
    #: none by default; a node that keeps some names them.
    # 译：撤回（retraction）镜像所排除的槽位大小的张量（``SlotStateExporter``）：
    #     默认没有；保留了这类张量的节点自己列出它们的名字。
    token_derived_slot_state: tuple[str, ...] = ()
    constant_slot_state: tuple[str, ...] = ()

    def _init_pool_binding(self) -> None:
        self.cache_pool: CachePool | None = None

    def child_backends(self) -> tuple[CachePoolBinding, ...]:
        """The nodes this one composes (graph support, pointer walk, binding); leaves: ()."""
        # 译：本节点组合的子节点（graph 支持、指针遍历、绑定都用到）；叶子返回 ()。
        return ()

    def validate_cache_pool(self, cache_pool: CachePool) -> None:
        """Raise if this node or any child cannot rebind to ``cache_pool``."""
        # 译：如果本节点或任何子节点无法重新绑定到 ``cache_pool``，就报错。
        for backend in self.child_backends():
            backend.validate_cache_pool(cache_pool)

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        """Bind: every node agrees, then the children publish, then this node."""
        # 译：绑定：先让每个节点都同意，然后子节点发布，最后是本节点。
        self.validate_cache_pool(cache_pool)
        self._bind(cache_pool)

    def _bind(self, cache_pool: CachePool) -> None:
        for backend in self.child_backends():
            backend._bind(cache_pool)
        self._publish_cache_pool(cache_pool)

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        """A node's own binding work: read the old pool before super(), use the new one after."""
        # 译：节点自身的绑定工作：在调用 super() 之前读旧 pool，之后使用新 pool。
        self.cache_pool = cache_pool

    def cache_placement(self, layer: PagedAttention) -> CachePlacement | None:
        """Return logical-slot ownership, or None for local/replicated storage."""
        # 译：返回逻辑槽位的归属；本地 / 复制存储时返回 None。
        return None

    def decode_kv_view(
        self, layer: PagedAttention, token_to_kv_pool
    ) -> DecodeKVView | None:
        """``layer``'s decode cache view (:class:`DecodeKVView`), or None when
        this backend's decode needs more than the cache and metadata (block
        or per-tensor scales, a split schedule, another kernel family)."""
        # 译：``layer`` 的 decode cache 视图（:class:`DecodeKVView`）；若该后端的 decode 需要的不止 cache 和元数据
        #     （block 级或逐张量的 scale、拆分的调度、别的 kernel 家族），则返回 None。
        return None

    def reads_fp8_kv_scales(self, layer: PagedAttention) -> bool:
        """Whether ``layer``'s reads apply the per-tensor scales its FP8 cache
        was written at (``PagedAttention.fp8_kv_scales``)."""
        # 译：``layer`` 的读取是否应用其 FP8 cache 写入时所用的逐张量 scale（``PagedAttention.fp8_kv_scales``）。
        return False

    def query_dtype(
        self, layer: PagedAttention, forward_mode: ForwardMode, activation: torch.dtype
    ) -> torch.dtype:
        """The dtype ``layer``'s attention reads its query in this forward,
        which the GQA prologue returns it in so no separate cast runs."""
        # 译：本次前向中 ``layer`` 的注意力读取 query 所用的 dtype；GQA prologue 直接以该 dtype 返回 query，免去单独的类型转换。
        return activation

    # ------------------------------------------------------------------
    # Slot-state image (SlotStateExporter)
    # ------------------------------------------------------------------
    # 译：槽位状态镜像（SlotStateExporter）

    # 以下槽位状态相关方法服务于“撤回”（retraction，第 4 阶段 execution/slot_state.py）：请求被撤回时，除了 KV 页，
    # 后端私有的、按请求槽位存放的状态也要一起快照 / 恢复。普通 MHA 后端没有这类状态，返回空列表。
    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        """This node's own per-slot rows a retraction snapshot images.

        Backend-private state keyed by ``req_pool_index`` -- a recurrent
        ring, a partial index pool not yet in a page -- is the exception
        (``AGENTS.md``); a backend that keeps some lists it here, in a fixed
        order, so a restore carries it with the pages. Paged state is in
        cache groups and never listed. A node lists its own rows only; the
        tree is flattened by :meth:`slot_state_exporters`. A leaf's default
        is empty.
        """
        # 译：撤回快照为本节点镜像的、本节点自己的每槽位行。
        #
        #     以 ``req_pool_index`` 为键的后端私有状态——循环环形缓冲、尚未放进页的部分索引池——是例外情况
        #     （``AGENTS.md``）；保留这类状态的后端在这里按固定顺序列出它们，这样恢复时它们随页一起被带回。
        #     分页状态在 cache group 中，从不在此列出。节点只列出自己的行；整棵树由 :meth:`slot_state_exporters` 展平。
        #     叶子的默认值为空。
        del slot
        return []

    def slot_state_exporters(self) -> tuple[CachePoolBinding, ...]:
        """This node and every node below it, each imaging its own rows.

        The model executor lists these in the blob, so a composite's rows and
        its children's are separate segments laid out once by the Host cache
        executor rather than re-measured at every export.
        """
        # 译：本节点及其下的每个节点，每个都镜像自己的行。
        #
        #     模型执行器把它们列在 blob 中，所以复合节点的行和其子节点的行是分开的段，由 Host cache 执行器
        #     一次性排布好，而不是在每次导出时重新测量。
        exporters = [self]
        for child in self.child_backends():
            exporters.extend(child.slot_state_exporters())
        return tuple(exporters)

    def slot_state_bytes(self) -> int:
        return slot_state_image_bytes(self.slot_state_rows(0))

    def export_slot_state(
        self, slot: int, out: torch.Tensor, stream, *, request_id: str
    ) -> None:
        # Backend rows are rewritten by every forward and prepared at a PD
        # landing, never keyed by request: always valid, nothing to mark.
        # 译：后端的行每次前向都会被重写，并在 PD 落地时准备好，从不以请求为键：总是有效，无需标记。
        del request_id
        pack_slot_rows(self.slot_state_rows(slot), out, stream)

    def import_slot_state(
        self, slot: int, src: torch.Tensor, stream, *, request_id: str
    ) -> None:
        del request_id
        unpack_slot_rows(self.slot_state_rows(slot), src, stream)

    # 以下四个方法服务于 KV offload（KV 放在 Host 内存，按需搬到 GPU，第 14 阶段），默认向子节点广播。
    def prepare_cache_batch(
        self, request_slots: torch.Tensor, *, num_extends: int, stream: torch.Stream
    ) -> None:
        """Begin KV offloading access for this target batch's request slots.

        Broadcast to child backends; leaves without offloading do nothing.
        Clear last-step read/write mappings, mask null/padded requests, and
        order residency work on stream. This does not allocate history.
        num_extends identifies prefill rows, which offloaded arenas reject.
        """
        # 译：为这个目标批次的请求槽位开始 KV offload 访问。
        #
        #     向子后端广播；没有 offload 的叶子什么都不做。清除上一步的读 / 写映射，屏蔽空请求 / padding 请求，
        #     并在 stream 上安排驻留工作。这里不分配历史。num_extends 标识 prefill 行，offload 的 arena 会拒绝它们。
        for child in self.child_backends():
            child.prepare_cache_batch(
                request_slots, num_extends=num_extends, stream=stream
            )

    def writeback_accepted_kv(self, accept_lengths: torch.Tensor) -> None:
        """Persist accepted KV from offloading compute buffers to Host history.

        Broadcast to child backends; leaves without offloading do nothing.
        Join writeback to the execution stream before publishing completion.
        accept_lengths counts accepted target-input rows, not output tokens.
        """
        # 译：把已接受的 KV 从 offload 计算缓冲区持久化到 Host 历史。
        #
        #     向子后端广播；没有 offload 的叶子什么都不做。在发布完成之前，把写回操作与执行流汇合。
        #     accept_lengths 统计的是已接受的目标模型输入行，而不是输出 token。
        for child in self.child_backends():
            child.writeback_accepted_kv(accept_lengths)

    def invalidate_cache_residency(
        self, request_slots: torch.Tensor, *, stream: torch.Stream
    ) -> None:
        """Fence KV offloading work and invalidate reused request-slot tags.

        Broadcast to child backends; leaves without offloading do nothing.
        Reset hot-cache tags, seed state, and LRU on stream. This does not
        erase authoritative history or release scheduler-owned cache blocks.
        """
        # 译：对 KV offload 工作做栅栏同步，并使被复用的请求槽位标签失效。
        #
        #     向子后端广播；没有 offload 的叶子什么都不做。在 stream 上重置热 cache 标签、种子状态和 LRU。
        #     这不会擦除权威历史，也不会释放调度器拥有的 cache block。
        for child in self.child_backends():
            child.invalidate_cache_residency(request_slots, stream=stream)

    def prefetch_sparse_kv(
        self,
        layer: PagedAttention,
        selection: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Overlap sparse KV offloading loads for this producer's consumers.

        layer identifies the selection producer. Its recipe declares which
        offloaded consumer fields share selection. Load those fields on the
        prefetch stream when overlap is enabled; otherwise do nothing.
        Each consumer must still call prepare_sparse_kv_access to join its
        ready event and publish compute slots before projection/attention.
        """
        # 译：为这个生产者的消费者重叠执行稀疏 KV offload 加载。
        #
        #     layer 标识选择结果的生产者。它的 recipe 声明了哪些 offload 的消费者字段共享该选择结果。启用重叠时，
        #     在预取流上加载这些字段；否则什么都不做。每个消费者仍必须调用 prepare_sparse_kv_access，
        #     以等待其就绪事件，并在投影 / 注意力之前发布计算用的槽位。


class AttentionBackend(CachePoolBinding, ABC):
    """The runner-facing contract; see the module docstring.

    A subclass that skips ``__init__`` calls ``_init_pool_binding()`` itself.
    """

    # 译：面向 runner 的契约；见模块 docstring。
    #
    #     跳过 ``__init__`` 的子类要自己调用 ``_init_pool_binding()``。

    # Cache-group kinds this node consumes from the pool contract (startup
    # validation: every published kind must have a consumer); composites
    # union their children's. Paged backends read rows of either history kind.
    # 译：本节点从 pool 契约中消费的 cache group 种类（启动校验：每个发布的种类都必须有消费者）；
    #     复合节点取其子节点的并集。分页后端读取任一种历史种类的行。
    cache_consumer_attn_kinds: frozenset[str] = frozenset(
        {"full_history", "sliding_window"}
    )

    # Whether this side computes a query-context shard of the extend span. The
    # runner reads it to decide whether to hand a side the forward's shard plan:
    # the draft of a sharded target may itself be unsharded, and a backend that
    # cannot attend a shard refuses one (``reject_query_shard``). A subclass that
    # skips ``__init__`` sets it from its config or the child it wraps.
    # 译：本侧是否计算 extend 段的 query-context 分片。runner 读取它来决定是否把本次前向的分片计划交给某一侧：
    #     分片目标模型的草稿本身可能不分片，而不能处理分片的后端会拒绝分片（``reject_query_shard``）。
    #     跳过 ``__init__`` 的子类从其配置或它包装的子节点设置该值。
    query_sharded: bool = False

    @property
    def stores_in_forward_state_checkpoints(self) -> bool:
        """Whether every State group this node consumes stores in-forward checkpoints.

        With ``--enable-state-checkpoint-in-forward``, the state at a checkpoint
        goes into the third block of an extend row. A node consuming no State
        group has none to miss; a State consumer opts in, and a composite answers
        for its children.
        """
        # 译：本节点消费的每个 State 组是否都存储前向内检查点。
        #
        #     开启 ``--enable-state-checkpoint-in-forward`` 时，检查点处的状态写入 extend 行的第三个 block。
        #     不消费 State 组的节点没有可遗漏的；State 消费者需选择加入，复合节点替其子节点回答。
        return "state" not in self.cache_consumer_attn_kinds

    # Only backends whose model skips incomplete-prefill outputs opt in.
    # 译：只有那些模型会跳过未完成 prefill 输出的后端才选择开启。
    skips_incomplete_prefill_outputs: bool = False
    # MLA sharded history reads/writes and global attention-partial merging,
    # independent of whether the model also has linear-attention layers.
    # 译：MLA 的分片历史读 / 写与全局注意力部分结果的合并，与模型是否还带有线性注意力层无关。
    supports_mla_dcp: bool = False
    supports_mla_projected_value_decode: bool = False
    # Bound by register_step_counter (PD layerwise transfer); None otherwise.
    # 译：由 register_step_counter 绑定（PD 逐层传输）；否则为 None。
    step_counter: StepCounter | None = None
    # Static CUDA-graph capability of this class; the executor AND-composes
    # it over the target+draft trees (resolve_cuda_graph_support).
    # 译：该类静态的 CUDA graph 能力；执行器在目标 + 草稿两棵树上做 AND 组合（resolve_cuda_graph_support）。
    cuda_graph_support: CudaGraphSupport = CudaGraphSupport()
    # This backend forwards each layer's ``sliding_window_size`` to its kernels.
    # Left False, a declared window silently widens to full-history attention.
    # 译：该后端会把每层的 ``sliding_window_size`` 传给其 kernel。
    #     若保持 False，声明的窗口会被悄悄放宽成全历史注意力。
    supports_layer_sliding_window: bool = False
    # This backend keeps each layer's ``sliding_window_sink`` leading tokens
    # visible outside the window. Left False, a declared sink silently
    # degrades to plain sliding-window attention: no error, only a quality
    # loss, so a drafter refuses the pairing up front.
    # 译：该后端让每层前 ``sliding_window_sink`` 个 token 在窗口外仍可见。若保持 False，声明的 sink 会悄悄
    #     退化成普通的滑动窗口注意力：不报错，只损失质量，所以草稿模型会提前拒绝这种组合。
    supports_layer_window_sink: bool = False

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        # Qwen3-8B 单卡（attn_tp_size=1）：num_qo_heads=32、num_kv_heads=8、head_dim=128；TP=2 时为 16、4、128。
        # spec_num_tokens：不开投机解码时为 1；开启时为验证宽度 N（每个请求每步写 N 个 token）。
        self.device = config.device
        self.dtype = config.dtype
        self.is_draft = bool(config.is_draft)
        self.spec_num_tokens = max(int(config.speculative_num_draft_tokens or 1), 1)
        self.num_qo_heads = spec.num_attention_heads // spec.attn_tp_size
        self.num_kv_heads = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.head_dim = spec.head_dim
        self._init_pool_binding()

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    # 译：结构

    def _init_pool_binding(self) -> None:
        """The binding lifecycle fields; wrappers that skip __init__ call this."""
        # 译：绑定生命周期的字段；跳过 __init__ 的包装器调用它。
        super()._init_pool_binding()
        self._sparse_topk = SparseTopKShare()

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        """Record the pool and clear shared sparse-forward metadata."""
        # 译：记录 pool 并清空共享的稀疏前向元数据。
        super()._publish_cache_pool(cache_pool)
        self._sparse_topk.clear()

    def configure_runtime(self, **kwargs) -> None:
        """Post-load configuration hook (information unavailable at
        construction, e.g. sliding window sizes). Default: no-op."""
        # 译：加载后的配置钩子（构造时还拿不到的信息，例如滑动窗口大小）。默认：什么都不做。

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        """Allocate static buffers the breakable prefill graphs bake.
        Default: no-op — attention stays eager at the break points."""
        # 译：分配可中断（breakable）prefill graph 固化进去的静态缓冲区。
        #     默认：什么都不做——注意力在断点处保持 eager 执行。

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Allocate graph-stable target verify buffers and return their bytes.

        ``max_bs`` is the serving batch capacity; ``draft_token_num`` is the
        verify width. Composites delegate to their verify consumers. Backends
        without verify buffers return zero.
        """
        # 译：分配 graph 中地址稳定的目标模型验证缓冲区，并返回其字节数。
        #
        #     ``max_bs`` 是服务的批次容量；``draft_token_num`` 是验证宽度。复合节点委托给它们的验证消费者。
        #     没有验证缓冲区的后端返回零。
        return 0

    def preallocate_history_gather_workspace(self, max_model_len: int) -> int:
        """Allocate the query-context-parallel history gather workspace and
        return its bytes (the recipe reserved them from the cache budget;
        the registry checks the two agree). Only a tree with the GPU DSA
        sharded extend arm has one."""
        # 译：分配 query-context 并行的历史收集工作区，并返回其字节数（recipe 已从 cache 预算中预留了这部分；
        #     registry 会检查两者一致）。只有带 GPU DSA 分片 extend 分支的树才有它。
        raise NotImplementedError(
            f"{type(self).__name__} has no sharded extend arm; query context "
            "parallelism needs GPU DSA attention"
        )

    def history_gather_workspace(self) -> HistoryGatherWorkspace | None:
        """The allocated history gather workspace of this tree, or ``None``
        before :meth:`preallocate_history_gather_workspace` ran (or on a tree
        without the sharded extend arm)."""
        # 译：本树已分配的历史收集工作区；在 :meth:`preallocate_history_gather_workspace` 运行之前
        #     （或在没有分片 extend 分支的树上）为 ``None``。
        return None

    def adopt_history_gather_workspace(self, workspace: HistoryGatherWorkspace) -> None:
        """Share another tree's history gather workspace: the draft tree
        gathers into the target's buffers, which are idle while the draft
        runs. A leaf without the sharded extend arm ignores it."""
        # 译：共享另一棵树的历史收集工作区：草稿树收集到目标模型的缓冲区中，草稿运行时这些缓冲区是空闲的。
        #     没有分片 extend 分支的叶子忽略它。
        raise NotImplementedError(
            f"{type(self).__name__} has no sharded extend arm to share a history "
            "gather workspace with"
        )

    # 以下三个方法服务于 prefill 的 breakable CUDA Graph（第 7 阶段）：默认不支持，注意力在断点处 eager 执行。
    @property
    def prefill_metadata_is_capture_ready(self) -> bool:
        """Whether the current execution metadata supports a captured forward."""
        # 译：当前的执行元数据是否支持被捕获的前向。
        return False

    def admits_prefill_graph(
        self, token_capacity: int, bs: int, forward_mode: ForwardMode
    ) -> bool:
        """Whether a captured prefill graph can include attention for this shape.

        The question ``prepare_prefill_metadata`` answers on its way to doing
        the work, asked on its own so a caller can decide before anything is
        written. Must read no forward context and change nothing.
        """
        # 译：被捕获的 prefill graph 能否在这个形状下包含注意力。
        #
        #     这是 ``prepare_prefill_metadata`` 在做实际工作的过程中顺带回答的问题，单独拿出来问，
        #     使调用方可以在写入任何东西之前做决定。不得读取任何前向上下文，也不得改变任何东西。
        return False

    def prepare_prefill_metadata(
        self, token_capacity: int, bs: int, forward_mode: ForwardMode, *, capture: bool
    ) -> bool:
        """Prepare execution metadata before eager forward or graph replay.

        ``capture`` only retains startup buffers at their captured addresses;
        it must not select different computation. Returns the same answer as
        :meth:`admits_prefill_graph`, except that a forward writing in-forward
        State checkpoints keeps its live metadata and gets False from a
        backend whose fixed-capacity metadata stores none; the prefill
        capture raises if an override refuses a shape that query admitted.
        Other backends keep
        their existing metadata and attention breaks. Call on the consumer
        stream, after the scheduler-derived metadata is built and before any
        layer consumes it.
        """
        # 译：在 eager 前向或 graph 回放之前准备执行元数据。
        #
        #     ``capture`` 只是把启动时的缓冲区保留在其被捕获的地址上；它不得选择不同的计算。返回值与
        #     :meth:`admits_prefill_graph` 相同，例外是：写入前向内 State 检查点的前向保留其实时元数据，
        #     并从固定容量元数据中不存这些检查点的后端那里得到 False；若某个覆盖实现拒绝了查询所接受的形状，
        #     prefill 捕获会报错。其他后端保留现有元数据，注意力处断开。在消费者的流上调用，
        #     时机在由调度器推导的元数据构建之后、任何层消费它之前。
        return self.admits_prefill_graph(token_capacity, bs, forward_mode)

    # ------------------------------------------------------------------
    # Metadata (docs/design/unified_path.md)
    # ------------------------------------------------------------------
    # 译：元数据（docs/design/unified_path.md）

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        """Allocate the persistent decode buffers, sized by ``max_bs`` (the
        max decode bs, never the capture ladder). Runs unconditionally at
        wrapper construction, ``enforce_eager`` included.

        Args:
            max_bs: Persistent-buffer row capacity.
            **kwargs: Runner extras every node accepts
                (``cache_group_specs``, ``cache_group_page_counts``,
                ``max_tokens_per_req``, ``overlap_schedule_depth``); a
                narrower signature TypeErrors at boot.
        """
        # 译：分配持久的 decode 缓冲区，尺寸按 ``max_bs``（最大 decode bs，从不按捕获阶梯）。
        #     在包装器构造时无条件运行，``enforce_eager`` 时也运行。
        #
        #     参数：
        #         max_bs：持久缓冲区的行容量。
        #         **kwargs：每个节点都接受的 runner 额外参数
        #             （``cache_group_specs``、``cache_group_page_counts``、``max_tokens_per_req``、
        #             ``overlap_schedule_depth``）；签名更窄的实现会在启动时抛 TypeError。
        raise NotImplementedError(
            f"{type(self).__name__} must implement init_cuda_graph_state"
        )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_replay_lens_cpu: torch.Tensor,
        extend_prompt_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        query_shard: QueryShardPlan | None,
        **kwargs,
    ) -> None:
        """Build metadata for an extend / mixed (or idle warmup) forward.

        Decode metadata goes through :meth:`refresh_decode_metadata`; a pure
        DECODE call here is a contract violation.

        Args:
            bs: Requests in the batch (extend requests first, then decode
                requests).
            num_extends: Leading extend requests.
            req_pool_indices: ``[>= bs]`` request-pool slots.
            seq_lens: ``[>= bs]`` total cache lengths after this step.
            forward_mode: EXTEND, MIXED or IDLE.
            block_tables: ``group_id -> [>= bs, cols]`` int32 raw scheduler
                tables for every published group (placeholders on warmup).
            extend_*: ``[>= num_extends]`` per-request new-token / prefix
                lengths and their pinned host mirrors (empty on idle warmup).
            extend_replay_lens_cpu: ``[>= num_extends]`` host-only leading
                input rows per request that re-feed cached prompt positions
                (bounded replay). Positions ``[prefix, prefix + replay)``
                regenerate replayable cache groups only; a node that cannot
                honour that calls :func:`reject_bounded_replay`.
            extend_prompt_lens_cpu: ``[>= num_extends]`` host-only whole
                prompt lengths, so a node can tell the chunk that ends a
                prompt from an intermediate one.
            extend_with_prefix: Whether any extend row continues a cached or
                chunked prefix (some ``extend_prefix_lens`` entry is non-zero).
            query_shard: The rows of the extend span this rank computes under
                query context parallelism, or ``None`` when every rank computes
                every row. The lengths above describe the whole span on every
                rank; a node that cannot attend a query shard calls
                :func:`reject_query_shard`.
            **kwargs: Model-side extras (positions, capture mode, ...) a
                node may ignore.
        """
        # 译：为 extend / mixed（或 idle 预热）前向构建元数据。
        #
        #     decode 元数据走 :meth:`refresh_decode_metadata`；在这里传入纯 DECODE 属于违反契约。
        #
        #     参数：
        #         bs：批次中的请求数（先是 extend 请求，然后是 decode 请求）。
        #         num_extends：排在前面的 extend 请求数。
        #         req_pool_indices：``[>= bs]`` 请求池槽位。
        #         seq_lens：``[>= bs]`` 本步之后的 cache 总长度。
        #         forward_mode：EXTEND、MIXED 或 IDLE。
        #         block_tables：``group_id -> [>= bs, cols]`` int32 原始调度器表，覆盖每个发布的组（预热时为占位表）。
        #         extend_*：``[>= num_extends]`` 每个请求的新 token 数 / 前缀长度，以及它们锁页的 Host 镜像
        #             （idle 预热时为空）。
        #         extend_replay_lens_cpu：``[>= num_extends]`` 仅 Host 端，每个请求开头重新输入已缓存 prompt 位置的
        #             输入行数（有界重放）。位置 ``[prefix, prefix + replay)`` 只重新生成可重放的 cache group；
        #             做不到这一点的节点调用 :func:`reject_bounded_replay`。
        #         extend_prompt_lens_cpu：``[>= num_extends]`` 仅 Host 端，完整的 prompt 长度，使节点能区分
        #             结束一个 prompt 的块和中间块。
        #         extend_with_prefix：是否有任何 extend 行接续已缓存的或分块的前缀（某个 ``extend_prefix_lens`` 项非零）。
        #         query_shard：在 query-context 并行下本 rank 计算的 extend 段中的行；每个 rank 都计算全部行时为 ``None``。
        #             上面的长度在每个 rank 上都描述整个段；不能处理 query 分片的节点调用 :func:`reject_query_shard`。
        #         **kwargs：节点可以忽略的模型侧额外参数（positions、捕获模式等）。
        # 贯穿例子（MIXED 批，A extend 300 个新 token、无前缀，B decode 写位置 1000）：
        #   bs=2，num_extends=1，seq_lens=[300, 1001]
        #   extend_seq_lens=[300]，extend_prefix_lens=[0]，extend_with_prefix=False，query_shard=None
        #   block_tables={"<gid>": [2, cols] int32}，A 的行为 [12, 3, 40, 7, 21, -1, ...]（-1 为不规则 padding）
        raise NotImplementedError(
            f"{type(self).__name__} must implement init_forward_metadata"
        )

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        block_tables: Mapping[str, torch.Tensor],
        num_extends: int = 0,
        for_graph_replay: bool = False,
        **kwargs,
    ) -> None:
        """The single decode metadata path — eager decode and graph replay.

        Refreshes the persistent decode buffers in place; there is no
        fresh-allocation decode arm anywhere (capture runs the idle arm of
        this refresh, replay refreshes before ``graph.replay()``, eager
        refreshes before the same forward the graph recorded).

        Args:
            bs: Requests to prepare (the padded capture batch under replay);
                eager passes ``bs == actual_bs``.
            actual_bs: Live requests; ``[actual_bs, bs)`` are padding the node
                routes to the null page. ``0`` is the idle replay / capture
                seeding.
            req_pool_indices: ``[>= bs]`` request-pool slots.
            seq_lens: ``[>= bs]`` live cache lengths (padding requests hold 1).
            forward_mode: A decode mode.
            block_tables: ``group_id -> [>= actual_bs, cols]`` raw scheduler
                tables for every published group (placeholders when idle).
            num_extends: Leading extend requests of a MIXED round whose decode
                half this refresh describes; 0 for pure decode.
            for_graph_replay: A graph is in play (live replay or capture
                seeding). Branch on it only for graph-mechanics asymmetries.
        """
        # 译：唯一的 decode 元数据路径——eager decode 和 graph 回放共用。
        #
        #     原地刷新持久的 decode 缓冲区；任何地方都没有新分配张量的 decode 分支（捕获运行这个刷新的 idle 分支，
        #     回放在 ``graph.replay()`` 之前刷新，eager 在 graph 所记录的同一个前向之前刷新）。
        #
        #     参数：
        #         bs：要准备的请求数（回放时为 padding 后的捕获批大小）；eager 传 ``bs == actual_bs``。
        #         actual_bs：活跃请求数；``[actual_bs, bs)`` 是由节点导向空页的 padding。``0`` 表示 idle 回放 / 捕获时的初始填充。
        #         req_pool_indices：``[>= bs]`` 请求池槽位。
        #         seq_lens：``[>= bs]`` 活跃的 cache 长度（padding 请求为 1）。
        #         forward_mode：某种 decode 模式。
        #         block_tables：``group_id -> [>= actual_bs, cols]`` 原始调度器表，覆盖每个发布的组（idle 时为占位表）。
        #         num_extends：本次刷新描述其 decode 半批的那个 MIXED 轮中，排在前面的 extend 请求数；纯 decode 为 0。
        #         for_graph_replay：有 graph 参与（实际回放或捕获时的初始填充）。只为 graph 机制上的不对称而据此分支。
        # 例：纯 decode 一步，真实 3 个请求，graph 桶为 4 -> bs=4、actual_bs=3，第 4 行是 padding：
        # seq_lens 为 1、页表行全 0，写入槽位为 0（哑页）。
        raise NotImplementedError(
            f"{type(self).__name__} must implement refresh_decode_metadata"
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> None:
        """Default capture: the idle refresh (``actual_bs=0``,
        ``for_graph_replay=True``) over the same persistent buffers replay
        refreshes, against the runner's placeholder tables and the seeded
        ``seq_lens``. Idempotent. Override only for a kernel-imposed
        capture asymmetry (unified_path.md, "Capture is inherited")."""
        # 译：默认的捕获：在回放时刷新的同一批持久缓冲区上做 idle 刷新（``actual_bs=0``，``for_graph_replay=True``），
        #     使用 runner 的占位表和预置的 ``seq_lens``。幂等。只有在 kernel 强加了捕获上的不对称时才覆盖
        #     （unified_path.md，“Capture is inherited”）。
        if not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"{type(self).__name__} CUDA graphs record decode only, got {forward_mode}"
            )
        self.refresh_decode_metadata(
            bs,
            0,
            req_pool_indices,
            seq_lens,
            forward_mode=forward_mode,
            block_tables=block_tables,
            num_extends=0,
            for_graph_replay=True,
            **kwargs,
        )

    # 以下几个草稿钩子属于投机解码（第 11 阶段），初读可跳过。
    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """Publish a drafter's in-graph seq_lens edits (Eagle chains); nodes
        without per-request decode lengths ignore it."""
        # 译：发布草稿模型在 graph 内对 seq_lens 的修改（Eagle 链）；没有逐请求 decode 长度的节点忽略它。

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        """Vanilla MTP re-anchors the draft requests to the committed frontier."""
        # 译：普通 MTP 把草稿请求重新锚定到已提交的前沿。

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """DFLASH: broadcast block-end lengths to each request's materialized
        decode entries."""
        # 译：DFLASH：把块末尾的长度广播到每个请求已物化的 decode 条目。

    @contextmanager
    def override_num_extends(self, num_extends: int):
        """Temporarily override the decode-row slice discriminator (MLA
        family). Default no-op."""
        # 译：临时覆盖 decode 行切片的判别量（MLA 家族）。默认什么都不做。
        yield

    @property
    def sparse_topk(self) -> SparseTopKShare:
        """This node's layer-shared sparse selection for the current forward
        (:class:`SparseTopKShare`). Every runner-facing node clears it when it
        builds a forward's metadata; composites fronting a paged child route
        to the child's, so the model, the indexer and the drafter meet the
        same object whichever level of the tree they hold."""
        # 译：本节点在当前前向中层间共享的稀疏选择结果（:class:`SparseTopKShare`）。每个面向 runner 的节点
        #     在构建一次前向的元数据时清空它；包在分页子节点外面的复合节点路由到子节点的那一个，
        #     所以无论模型、索引器和草稿模型持有树的哪一层，它们碰到的都是同一个对象。
        return self._sparse_topk

    def supports_narrowed_draft_decode(self, forward_mode: ForwardMode) -> bool:
        """Whether a narrowed draft step in a round of ``forward_mode`` can
        attend its live rows as a DECODE dispatch (Eagle3's first step)."""
        # 译：在一个 ``forward_mode`` 的轮次中，收窄的草稿步能否以 DECODE 分派来关注其活跃行（Eagle3 的第一步）。
        return False

    # ------------------------------------------------------------------
    # Write locations
    # ------------------------------------------------------------------
    # 译：写入位置

    def draft_write_locations_uniform(
        self, out: torch.Tensor, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Resolve ``num_tokens`` KV write slots per request into a
        caller-owned scratch buffer (in-graph safe: fixed table address).

        For slots a FORWARD will consume, use
        :meth:`publish_draft_step_locations` instead — this variant serves
        side writes that must not clobber the published step window (e.g.
        DFLASH copying target-row KV into the draft cache).
        """
        # 译：把每个请求 ``num_tokens`` 个 KV 写入槽位解析到调用方拥有的暂存缓冲区中（在 graph 内安全：表地址固定）。
        #
        #     对“会被某次前向消费”的槽位，请改用 :meth:`publish_draft_step_locations`——这个变体服务于那些
        #     不能覆盖已发布步窗口的旁路写入（例如 DFLASH 把目标模型行的 KV 拷进草稿 cache）。
        raise NotImplementedError(
            f"{type(self).__name__} owns no draft write locations"
        )

    def publish_draft_step_locations(
        self, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Publish a draft step's write window (``num_tokens`` slots per
        request starting at ``cache_start``) so the next forward's
        ``write_locations`` serve it. The drafters declare the window; the
        math and the address-stable storage live in the backend. In-graph
        safe."""
        # 译：发布一个草稿步的写入窗口（每个请求从 ``cache_start`` 开始的 ``num_tokens`` 个槽位），使下一次前向的
        #     ``write_locations`` 提供它。草稿模型声明窗口；计算和地址稳定的存储在后端中。在 graph 内安全。
        raise NotImplementedError(
            f"{type(self).__name__} owns no draft write locations"
        )

    def tree_support(self) -> TreeSupport:
        """This node's own draft-tree capability; ``resolve_tree_support`` asks
        every node before any bind. Nodes without a tree path keep this."""
        # 译：本节点自身的 draft tree 能力；``resolve_tree_support`` 在任何绑定之前询问每个节点。没有树路径的节点保持此默认值。
        name = type(self).__name__
        return TreeSupport(
            verify_blocker=f"{name} has no draft-tree verify path",
            draft_blocker=f"{name} has no draft-tree lane path",
        )

    def bind_tree_verify(self, inputs: TreeVerifyInputs) -> None:
        """Arm draft-tree verify (--speculative-eagle-topk > 1) on every node,
        after ``resolve_tree_support``; leaves that verify trees keep ``inputs``."""
        # 译：在 ``resolve_tree_support`` 之后，在每个节点上启用 draft tree 验证（--speculative-eagle-topk > 1）；
        #     验证树的叶子保存 ``inputs``。
        for backend in self.child_backends():
            backend.bind_tree_verify(inputs)

    def bind_tree_draft(self, inputs: TreeDraftInputs) -> None:
        """Arm draft-tree lanes on every node of the drafter's backend, after
        ``resolve_tree_support``; leaves that draft trees keep ``inputs``."""
        # 译：在 ``resolve_tree_support`` 之后，在草稿模型后端的每个节点上启用 draft tree 通道；起草树的叶子保存 ``inputs``。
        for backend in self.child_backends():
            backend.bind_tree_draft(inputs)

    def compact_verify_window(self, path: torch.Tensor) -> None:
        """After a draft-tree verify, move each request's accepted path to the
        front of its verify window in the cache this node owns; nodes that own
        none forward to their children.

        Args:
            path: ``[bs, N]`` int32 accepted window row at each depth, root
                first, ``-1`` past the path.
        """
        # 译：draft tree 验证之后，在本节点拥有的 cache 中，把每个请求被接受的路径移到其验证窗口的前部；
        #     不拥有 cache 的节点转发给子节点。
        #
        #     参数：
        #         path：``[bs, N]`` int32，每个深度上被接受的窗口行，根在前，路径之后为 ``-1``。
        for backend in self.child_backends():
            backend.compact_verify_window(path)

    def write_locations(
        self, layer: PagedAttention, forward_mode: ForwardMode
    ) -> torch.Tensor:
        """Return this layer's authoritative history write slots.

        Index-K uses these before selection, independently of KV offloading.
        Offloaded latent KV writers use the compute slots returned by
        prepare_sparse_kv_access instead. This query does not copy KV.
        """
        # 译：返回本层权威的历史写入槽位。
        #
        #     索引 K 在选择之前使用它们，与 KV offload 无关。offload 的潜在 KV 写入者改用
        #     prepare_sparse_kv_access 返回的计算槽位。这个查询不拷贝 KV。
        raise NotImplementedError(
            f"{type(self).__name__} owns no paged write locations"
        )

    def forward_write_locations(
        self, layer: PagedAttention, forward_mode: ForwardMode
    ) -> torch.Tensor:
        """Slots for the K/V rows a forward in ``forward_mode`` carries: the
        attention prologue's write target."""
        # 译：本次 ``forward_mode`` 前向所携带的 K/V 行的槽位：注意力 prologue 的写入目标。
        raise NotImplementedError(
            f"{type(self).__name__} owns no paged write locations"
        )

    def padded_write_locations(
        self, layer: PagedAttention, forward_mode: ForwardMode, rows: int
    ) -> torch.Tensor:
        """:meth:`forward_write_locations` widened to the ``rows`` a graph-padded
        forward carries, the extra rows landing in the dummy slot 0, so the
        prologue can be captured; a backend without a padded span serves exact
        counts only."""
        # 译：把 :meth:`forward_write_locations` 加宽到 graph padding 后的前向所携带的 ``rows`` 行，多出的行落到哑槽位 0，
        #     这样 prologue 就可以被捕获；没有 padding 段的后端只提供精确数量。
        locations = self.forward_write_locations(layer, forward_mode)
        if locations.numel() != rows:
            raise ValueError(f"{locations.numel()} write slots for {rows} rows")
        return locations

    def prepare_sparse_kv_access(
        self,
        layer: PagedAttention,
        selection: torch.Tensor,
        positions: torch.Tensor,
        *,
        forward_mode: ForwardMode,
    ) -> torch.Tensor:
        """Prepare sparse KV offloading and return this layer's compute writes.

        selection contains history row IDs, including masks and duplicates;
        positions contains token positions. Decode loads history or joins
        prefetch, then publishes hot reads/writes. Extend reserves projection
        staging; sparse-prefill iteration later flushes and gathers history.
        Use the returned slots for KV projection before attention. Layers
        without offloading return ordinary history writes without copying KV.
        Wrappers route by layer and preserve the returned tensor.
        """
        # 译：准备稀疏 KV offload，并返回本层的计算写入位置。
        #
        #     selection 包含历史行 ID（含掩码和重复项）；positions 包含 token 位置。decode 加载历史或等待预取完成，
        #     然后发布热读 / 写位置。extend 预留投影暂存区；稀疏 prefill 的迭代稍后会刷新并收集历史。
        #     在注意力之前，用返回的槽位做 KV 投影。没有 offload 的层返回普通的历史写入位置，不拷贝 KV。
        #     包装器按层路由，并原样保留返回的张量。
        return self.write_locations(layer, forward_mode)

    # ------------------------------------------------------------------
    # PD / speculative side state
    # ------------------------------------------------------------------
    # 译：PD / 投机解码的旁路状态

    def prepare_remote_cache_slots(self, slot_indices: list[int]) -> None:
        """Clear model-specific restore state before remote cache admission."""
        # 译：在远程 cache 准入之前，清除模型特定的恢复状态。
        del slot_indices

    def mark_remote_cache_ready(self, slot_index: int) -> None:
        """Arm model-specific hydration after a remote cache transfer succeeds."""
        # 译：远程 cache 传输成功后，启用模型特定的状态填充（hydration）。
        del slot_index

    def register_step_counter(self, step_counter: StepCounter) -> None:
        self.step_counter = step_counter

    def commit_state_after_acceptance(
        self,
        accepted_lengths: torch.Tensor,
        *,
        num_extends: int,
        accepted_path: torch.Tensor | None,
    ) -> None:
        """Commit live acceptance after decode/mixed execution or replay.

        ``num_extends == 0`` identifies pure decode; otherwise extend requests
        lead the mixed batch. ``accepted_path`` is the ``[bs, N]`` accepted
        draft-tree path (root first, ``-1`` past it), ``None`` for a chain.
        Stateless backends inherit this no-op.
        """
        # 译：在 decode / mixed 执行或回放之后，提交实际的接受结果。
        #
        #     ``num_extends == 0`` 表示纯 decode；否则 extend 请求排在 mixed 批的前面。``accepted_path`` 是 ``[bs, N]``
        #     被接受的 draft tree 路径（根在前，路径之后为 ``-1``），链式草稿时为 ``None``。无状态的后端继承这个空操作。

    # PD 分离（第 15 阶段）的逐层传输：prefill 端每写完一层 KV 就记一步，传输线程据此可以边算边传。
    # 单机不分离时 step_counter 为 None，这里什么都不做。
    @contextmanager
    def record_pd_cache_step(
        self,
        forward_mode: ForwardMode,
        writes_in_call: bool,
        record_kv_cache: bool | None,
    ):
        """Anchor the PD layerwise cache-step record to the layer's last cache
        write: after the wrapped call when it writes a cache field
        (``writes_in_call``), before it otherwise. No-op without a step
        counter."""
        # 译：把 PD 逐层 cache 步记录锚定到该层最后一次 cache 写入：被包装的调用会写 cache 字段时
        #     （``writes_in_call``）在调用之后记录，否则在调用之前记录。没有步计数器时什么都不做。
        if record_kv_cache is None:
            record_cache = not forward_mode.is_decode() and not forward_mode.is_idle()
        else:
            record_cache = record_kv_cache
        record_cache = record_cache and self.step_counter is not None
        if record_cache and not writes_in_call:
            self.step_counter.record_cache()
        yield
        if record_cache and writes_in_call:
            self.step_counter.record_cache()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    # 译：前向

    # @break_point：breakable CUDA Graph 的断点（第 7 阶段）。CacheGroupRouter 覆盖了这个方法（见 paged/router.py）。
    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool: CachePool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        **kwargs,
    ):
        """Run one attention layer; KV write locations come from
        :meth:`write_locations` (there is no caller-supplied location
        vector). ``record_kv_cache`` overrides the PD layerwise recording
        (None: record on the extend-side path)."""
        # 译：运行一个注意力层；KV 写入位置来自 :meth:`write_locations`（没有调用方提供的位置向量）。
        #     ``record_kv_cache`` 覆盖 PD 逐层记录（None：在 extend 一侧的路径上记录）。
        out_cache_loc = self.write_locations(layer, forward_mode)
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                return self.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    **kwargs,
                )
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                forward_mode=forward_mode,
                **kwargs,
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        raise NotImplementedError()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        raise NotImplementedError()


def reject_query_shard(query_shard: QueryShardPlan | None, node: str) -> None:
    """Fail loud when a forward shards its query rows over a node that attends
    every row.

    Under query context parallelism the model feeds a node its shard of the
    extend rows while the metadata describes the whole span; only a backend
    whose extend arm gathers each request's history for its local rows may
    accept the shard.

    Args:
        query_shard: The forward's shard plan, or ``None`` when not sharded.
        node: Backend name for the diagnostic.
    """
    # 译：当一次前向把 query 行分片给一个会关注每一行的节点时，直接报错。
    #
    #     在 query-context 并行下，模型把 extend 行的分片喂给节点，而元数据描述的是整个段；只有那些 extend 分支
    #     会为本地行收集每个请求历史的后端才能接受分片。
    #
    #     参数：
    #         query_shard：本次前向的分片计划；不分片时为 ``None``。
    #         node：用于诊断信息的后端名。
    if query_shard is not None and query_shard.size > 1:
        raise RuntimeError(
            f"{node} cannot attend a query shard; query context parallelism "
            "needs the gathered-history DSA extend arm"
        )


def reject_bounded_replay(extend_replay_lens_cpu: torch.Tensor, node: str) -> None:
    """Fail loud when a forward re-feeds cached positions a node cannot mask.

    Replayed rows must not rewrite the groups whose rows the prefix hit
    already holds; only a backend that plans its writes around
    ``extend_replay_lens_cpu`` may accept them.

    Args:
        extend_replay_lens_cpu: ``[num_extends]`` host replay lengths.
        node: Backend name for the diagnostic.
    """
    # 译：当一次前向重新输入了节点无法屏蔽的已缓存位置时，直接报错。
    #
    #     重放的行不得重写那些其行已被前缀命中所持有的组；只有围绕 ``extend_replay_lens_cpu`` 规划写入的后端才能接受它们。
    #
    #     参数：
    #         extend_replay_lens_cpu：``[num_extends]`` Host 端重放长度。
    #         node：用于诊断信息的后端名。
    if extend_replay_lens_cpu.numel() and bool((extend_replay_lens_cpu != 0).any()):
        raise RuntimeError(
            f"{node} cannot mask bounded-replay rows; its cache groups must not "
            "be replayable"
        )
