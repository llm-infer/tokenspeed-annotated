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

"""The paged-KV attention leaf: the kernel-facing contract.

A ``PagedAttentionBackend`` serves the layers of exactly one cache group and
perceives exactly what a paged attention kernel perceives — a kernel page
table, cache sequence lengths and write slots. It never sees cache groups,
scheduler block tables, the pool's contract or which side (target/draft)
it runs on beyond the ``is_draft`` verify-floor geometry. The
``CacheGroupRouter`` owns all of that and hands every leaf a fully resolved
``page_table`` (kernel pages, batch-ordered, padded to ``[bs,
max_num_pages]``) on every metadata call; the leaf copies it into its own
graph-recorded buffers. This is the interface the backends had before cache
groups existed, minus the scheduler-table gather the router now performs.
"""

# 译：分页 KV 注意力叶子：面向 kernel 的契约。
#
#     一个 ``PagedAttentionBackend`` 恰好服务一个 cache group 的那些层，并且它看到的恰好就是一个分页注意力 kernel
#     看到的东西——kernel 页表、cache 序列长度和写入槽位。它从不看到 cache group、调度器 block table、pool 的契约，
#     也不知道自己运行在哪一侧（目标 / 草稿），除了 ``is_draft`` 带来的验证下限几何。``CacheGroupRouter`` 拥有
#     所有这些，并在每次元数据调用时交给每个叶子一张完全解析好的 ``page_table``（kernel 页，按批次排序，
#     padding 到 ``[bs, max_num_pages]``）；叶子把它拷贝进自己由 graph 记录的缓冲区。这就是 cache group 出现之前
#     后端已有的接口，只是去掉了现在由 router 完成的调度器表收集。

# 叶子后端的抽象基类。具体实现：MHA 系 mha.py（Qwen3 默认）、trtllm.py；MLA 系 flashmla.py、trtllm_mla.py 等。
#
# 叶子的职责边界（与 router 对照）：
#   router（router.py）：block -> kernel 页的换算、所有写入槽位、按 group 分派
#   叶子（本类）       ：拿到已换算好的 [bs, max_num_pages] 页表和 seq_lens，建 kernel 需要的元数据，调 kernel
#
# 叶子持有的持久 decode 缓冲区（init_cuda_graph_state 分配，CUDA Graph 记录其地址）：
#   page_table_buf  [max_bs, max_num_pages] int32   每次 decode 刷新时从 router 的页表原地拷入
#   seq_lens_buf    [max_bs] int32                  同上，拷入 seq_lens（按 verify_floor 截断下限）
#   Qwen3-8B（context_len=40960，P=64）：max_num_pages = ceil(40960 / 64) = 640
#
# extend 元数据则每轮按传入的页表重建（不被 graph 记录；prefill graph 在注意力处断开）。
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from tokenspeed.runtime.layers.attention.backends.base import CachePoolBinding
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeCascadeRows,
)
from tokenspeed.runtime.layers.attention.backends.support import (
    CudaGraphSupport,
    TreeSupport,
)
from tokenspeed.runtime.layers.attention.dcp.cache import PrefixHistoryGather
from tokenspeed.runtime.utils.common import ceil_div

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
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


class PagedAttentionBackend(CachePoolBinding, ABC):
    """One cache group's paged attention kernels and their metadata.

    Persistent decode state is leaf-owned: ``init_cuda_graph_state`` allocates
    the page-table buffer (``[entries, max_num_pages]``) and the cache-seqlens
    buffer (``[entries]``) a captured graph records, where ``entries`` is
    ``max_bs * block_decode_expansion`` — one metadata entry per request,
    materialized per block position under block decode (see
    :attr:`block_decode_expansion`) — and every ``refresh_decode_metadata``
    copies the router-supplied table and lengths into them in place. Extend
    metadata is rebuilt per round from the supplied table (never
    graph-recorded; the prefill graph breaks at attention).
    """

    # 译：一个 cache group 的分页注意力 kernel 及其元数据。
    #
    #     持久 decode 状态归叶子所有：``init_cuda_graph_state`` 分配被捕获 graph 所记录的页表缓冲区
    #     （``[entries, max_num_pages]``）和 cache 序列长度缓冲区（``[entries]``），其中 ``entries`` 为
    #     ``max_bs * block_decode_expansion``——每个请求一个元数据条目，块 decode 时按块内位置各物化一个
    #     （见 :attr:`block_decode_expansion`）——每次 ``refresh_decode_metadata`` 都把 router 提供的表和长度原地拷贝进去。
    #     extend 元数据每轮按提供的表重建（从不被 graph 记录；prefill graph 在注意力处断开）。

    # Static CUDA-graph capability of this leaf class (see support.py).
    # 译：该叶子类静态的 CUDA graph 能力（见 support.py）。
    cuda_graph_support: CudaGraphSupport = CudaGraphSupport()
    # Whether a block draft (DFLASH/DSpark) runs its whole non-causal block
    # in one decode forward (see block_decode_active / block_decode_expansion);
    # set from config by leaves that support it.
    # 译：块草稿（DFLASH/DSpark）是否在一次 decode 前向中跑完它整个非因果的块
    #     （见 block_decode_active / block_decode_expansion）；由支持它的叶子根据配置设置。
    draft_block_decode: bool = False
    # PD layerwise transfer counter, installed by the router; the MLA
    # leaves record the step inside their chunked prefill.
    # 译：PD 逐层传输计数器，由 router 安装；MLA 叶子在自己的分块 prefill 内部记录步。
    step_counter = None
    # The kernel's page span when the config does not override it
    # (kernel_page_sizes.py). None means the kernel is page-size flexible and
    # views the cache at its group's own block granularity (no expansion).
    # 译：配置未覆盖时 kernel 的页跨度（kernel_page_sizes.py）。None 表示 kernel 对页大小灵活，
    #     以其所在组自己的 block 粒度来看 cache（不做展开）。
    default_kernel_page_size: int | None = None
    # This leaf forwards each layer's ``sliding_window_size`` to its kernels.
    # Left False, a declared window silently widens to full-history attention.
    # Declared here as well as on AttentionBackend: the refactor made the two
    # separate roots, so a paged leaf inherits only this one.
    # 译：该叶子会把每层的 ``sliding_window_size`` 传给其 kernel。若保持 False，声明的窗口会被悄悄放宽成全历史注意力。
    #     这里和 AttentionBackend 上都声明一遍：重构使两者成为各自独立的根类，所以分页叶子只继承这一个。
    supports_layer_sliding_window: bool = False
    # This leaf keeps each layer's ``sliding_window_sink`` leading tokens
    # visible outside the window. Left False, a declared sink silently
    # degrades to plain sliding-window attention: no error, only a quality
    # loss. Declared alongside the window flag for the same reason.
    # 译：该叶子让每层前 ``sliding_window_sink`` 个 token 在窗口外仍可见。若保持 False，声明的 sink 会悄悄退化成
    #     普通的滑动窗口注意力：不报错，只损失质量。出于同样原因与窗口标志一起声明在这里。
    supports_layer_window_sink: bool = False
    supports_mla_dcp: bool = False
    prefill_needs_host_page_table: bool = False

    @classmethod
    def resolve_kernel_page_size(
        cls, config: AttnConfig, block_granularity: int
    ) -> int:
        """The page span this leaf class runs at for a group of
        ``block_granularity`` tokens per scheduler block: the config override
        when given, else the class default, else the group's own grain."""
        # 译：该叶子类在一个“每调度器 block 含 ``block_granularity`` 个 token”的组上运行时的页跨度：
        #     给了配置覆盖就用它，否则用类默认值，再否则用该组自己的粒度。
        # 例：MHA 后端 default_kernel_page_size 为 None，未覆盖时 kernel 页大小 = 组的 block 粒度（ratio=1）。
        # TRT-LLM 等 kernel 只支持特定页大小时设类默认值，此时 router 把一个 block 展开成多个 kernel 页。
        if config.kernel_page_size is not None:
            return int(config.kernel_page_size)
        if cls.default_kernel_page_size is not None:
            return int(cls.default_kernel_page_size)
        return int(block_granularity)

    def __init__(
        self, config: AttnConfig, spec: SoftmaxAttnConfig, *, kernel_page_size: int
    ) -> None:
        """Args:
        config: The side's attention config (device, dtypes, spec width).
        spec: The softmax attention component this leaf serves.
        kernel_page_size: Tokens per kernel page, resolved by the caller
            from the kernel registry (``kernel_page_sizes.py``) and the
            group's block granularity.
        """
        # 译：参数：
        #     config：本侧的注意力配置（设备、dtype、投机宽度）。
        #     spec：本叶子服务的 softmax 注意力组件。
        #     kernel_page_size：每个 kernel 页的 token 数，由调用方根据 kernel registry（``kernel_page_sizes.py``）
        #         和该组的 block 粒度解析得到。
        self.device = config.device
        self.dtype = config.dtype
        self.is_draft = bool(config.is_draft)
        self.spec_num_tokens = max(int(config.speculative_num_draft_tokens or 1), 1)
        self.max_context_len = int(config.context_len)
        if kernel_page_size <= 0:
            raise ValueError(
                f"kernel_page_size must be positive, got {kernel_page_size}"
            )
        self.kernel_page_size = int(kernel_page_size)
        self.max_num_pages = ceil_div(self.max_context_len, self.kernel_page_size)
        self.num_qo_heads = spec.num_attention_heads // spec.attn_tp_size
        self.num_kv_heads = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.head_dim = spec.head_dim
        self._init_pool_binding()
        # Persistent decode buffers (``init_cuda_graph_state``) and the cached
        # per-bs metadata views over them (each leaf's ``_decode_views``).
        # 译：持久 decode 缓冲区（``init_cuda_graph_state``）以及其上按 bs 缓存的元数据视图（每个叶子的 ``_decode_views``）。
        self.page_table_buf: torch.Tensor | None = None
        self.seq_lens_buf: torch.Tensor | None = None
        self._decode_views_by_bs: dict[int, Any] = {}
        # Draft-tree verify inputs; unset for chains and for draft leaves.
        # 译：draft tree 验证的输入；链式草稿和草稿叶子上不设置。
        self.tree_verify: TreeVerifyInputs | None = None
        # Draft-tree lane inputs; set on the drafter's leaves only.
        # 译：draft tree 通道的输入；只在草稿模型的叶子上设置。
        self.tree_draft: TreeDraftInputs | None = None
        # Draft-tree verify: committed keys per request, the cascade's prefix.
        # 译：draft tree 验证：每个请求已提交的 key 数，即级联注意力的前缀。
        self.tree_prefix_lens_buf: torch.Tensor | None = None
        # A cache dealt over owners: the prefill reads of its cached prefix,
        # from the first _prepare_prefix_history; None reads this rank's cache.
        # 译：分发给多个所有者的 cache：其已缓存前缀的 prefill 读取，来自第一次 _prepare_prefix_history；
        #     为 None 时读本 rank 的 cache。
        self._prefix_history: PrefixHistoryGather | None = None

    # 以下 tree_* 方法属于 draft tree 投机解码（第 11 阶段），初读可跳过。
    def tree_support(self) -> TreeSupport:
        name = type(self).__name__
        return TreeSupport(
            verify_blocker=f"{name} has no tree verify path; use --attention-backend trtllm (MLA: trtllm_mla or tokenspeed_mla)",
            draft_blocker=f"{name} has no tree lane path; use --drafter-attention-backend trtllm (MLA: trtllm_mla or tokenspeed_mla)",
        )

    def bind_tree_verify(self, inputs: TreeVerifyInputs) -> None:
        self.tree_verify = inputs

    def bind_tree_draft(self, inputs: TreeDraftInputs) -> None:
        self.tree_draft = inputs

    @property
    def tree_lane_step_active(self) -> bool:
        return self.tree_draft is not None and self.tree_draft.active

    def _refresh_tree_prefix(self, bs: int, verify_seq_lens: torch.Tensor) -> None:
        """Publish the verify cascade's prefix (``verify_seq_lens`` minus the
        window) when a tree is bound; ``verify_seq_lens`` is already clamped
        to the window width."""
        # 译：绑定了树时，发布验证级联的前缀（``verify_seq_lens`` 减去窗口）；``verify_seq_lens`` 已截断到窗口宽度。
        if self.tree_verify is not None:
            torch.sub(
                verify_seq_lens[:bs],
                self.tree_verify.num_nodes,
                out=self.tree_prefix_lens_buf[:bs],
            )

    def _tree_lane_rows(self, bs: int) -> TreeCascadeRows | None:
        """The drafting lanes' cascade while a lane forward runs, else None."""
        # 译：通道前向运行时，返回起草通道的级联；否则返回 None。
        if not self.tree_lane_step_active:
            return None
        return self.tree_draft.cascade_rows(self, bs)

    def _tree_verify_rows(
        self, bs: int, q_len: int, page_table: torch.Tensor, seq_lens: torch.Tensor
    ) -> TreeCascadeRows | None:
        """The verify cascade when a tree is bound and this forward verifies
        (``q_len > 1``), else None; ``seq_lens`` include the window."""
        # 译：绑定了树且本次前向是验证（``q_len > 1``）时返回验证级联，否则返回 None；``seq_lens`` 包含窗口。
        if self.tree_verify is None or q_len <= 1:
            return None
        nodes = self.tree_verify.num_nodes
        return TreeCascadeRows(
            page_table=page_table,
            seq_lens=seq_lens,
            prefix_lens=self.tree_prefix_lens_buf[:bs],
            mask=self.tree_verify.mask[: bs * nodes],
            rows=nodes,
            window=nodes,
        )

    # ------------------------------------------------------------------
    # Static shape / lifecycle
    # ------------------------------------------------------------------
    # 译：静态形状 / 生命周期

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        # Graph buffers and views return with init_cuda_graph_state.
        # 译：graph 缓冲区和视图随 init_cuda_graph_state 重新建立。
        self.page_table_buf = None
        self.seq_lens_buf = None
        self.tree_prefix_lens_buf = None
        self._decode_views_by_bs = {}
        # Its gathers read the previous pool's layers.
        # 译：它的收集操作读的是上一个 pool 的层。
        self._prefix_history = None

    def configure_runtime(self, **kwargs) -> None:
        """Post-load configuration hook (e.g. sliding window sizes)."""
        # 译：加载后的配置钩子（例如滑动窗口大小）。

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        """Allocate static buffers the breakable prefill graphs bake; most
        leaves keep attention eager at the break points and need none."""
        # 译：分配可中断 prefill graph 固化进去的静态缓冲区；大多数叶子在断点处让注意力保持 eager，不需要这些缓冲区。

    @property
    def verify_floor(self) -> int:
        """Minimum per-request cache seq_len decode metadata must present.

        The target's verify window spans ``seq - N .. seq - 1`` so its
        requests need ``seq_len >= N``; plain decode and drafts have floor 1,
        where the clamp is the identity.
        """
        # 译：decode 元数据必须呈现的每请求 cache seq_len 最小值。
        #
        #     目标模型的验证窗口覆盖 ``seq - N .. seq - 1``，所以其请求需要 ``seq_len >= N``；
        #     普通 decode 和草稿的下限为 1，此时截断是恒等操作。
        # 例：N=4 的投机验证，padding 行 seq_len=1 被截到 4，否则窗口起点 seq-N 为负；不开投机时 N=1，不影响。
        return self.spec_num_tokens if not self.is_draft else 1

    @property
    def block_decode_active(self) -> bool:
        """DFLASH/DSpark block draft: every decode forward runs a whole
        non-causal block of ``spec_num_tokens`` positions per request."""
        # 译：DFLASH/DSpark 块草稿：每次 decode 前向为每个请求运行一个包含 ``spec_num_tokens`` 个位置的完整非因果块。
        return self.draft_block_decode and self.spec_num_tokens > 1

    @property
    def block_decode_expansion(self) -> int:
        """Decode metadata entries materialized per request in the persistent
        buffers.

        ``spec_num_tokens`` under block decode: one entry per block position,
        because the block is non-causal (all positions share the block-end
        length) and the drafter rewrites those lengths in-graph
        (:meth:`fill_block_decode_seq_lens`), so they must exist as storage.
        Otherwise 1: plain decode and a vanilla MTP draft have one position
        per request, and target verify derives its causal per-position
        lengths at forward time from the request's single entry. Leaves that
        keep one entry per request even under block decode (FlashMLA,
        TRT-LLM MLA: they repeat it across the block's queries at forward
        time) override this to 1 together with ``fill_block_decode_seq_lens``.
        """
        # 译：持久缓冲区中为每个请求物化的 decode 元数据条目数。
        #
        #     块 decode 时为 ``spec_num_tokens``：每个块内位置一个条目，因为块是非因果的（所有位置共享块末尾的长度），
        #     而且草稿模型在 graph 内重写这些长度（:meth:`fill_block_decode_seq_lens`），所以它们必须作为存储存在。
        #     否则为 1：普通 decode 和普通 MTP 草稿每个请求只有一个位置，目标模型验证在前向时从请求的单个条目推导出
        #     逐位置的因果长度。即使在块 decode 下也每个请求只保留一个条目的叶子（FlashMLA、TRT-LLM MLA：它们在前向时
        #     把这个条目在块的各个 query 上重复）会把这里覆盖为 1，并一并覆盖 ``fill_block_decode_seq_lens``。
        return self.spec_num_tokens if self.block_decode_active else 1

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    # 译：元数据

    def init_cuda_graph_state(self, max_bs: int) -> None:
        """Allocate the persistent decode buffers for
        ``max_bs * block_decode_expansion`` metadata entries. Runs
        unconditionally at wrapper construction, enforce-eager included: the
        unified refresh serves eager decode from the same buffers. A leaf
        with more persistent decode state extends this (``super()`` first).
        """
        # 译：为 ``max_bs * block_decode_expansion`` 个元数据条目分配持久 decode 缓冲区。在包装器构造时无条件运行，
        #     enforce-eager 时也运行：统一的刷新路径用同一批缓冲区服务 eager decode。有更多持久 decode 状态的叶子
        #     扩展这个方法（先调用 ``super()``）。
        entries = max_bs * self.block_decode_expansion
        # The router hands this leaf [bs, max_num_pages] tables (context_len
        # already carries the spec-verify overshoot); null page 0 is the
        # padding contract's dereferenceable dummy.
        # 译：router 交给本叶子 [bs, max_num_pages] 的表（context_len 已经包含了投机验证的超出部分）；
        #     空页 0 是 padding 约定中可解引用的哑页。
        self.page_table_buf = torch.zeros(
            (entries, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        # Own the cache-seqlens buffer; every refresh copies the live lengths
        # in, so graph state never depends on a shared mutable tensor. Under
        # block decode the drafter fills it in-graph; seed the block width so
        # any pre-broadcast read stays in range.
        # 译：自己持有 cache 序列长度缓冲区；每次刷新都把实际长度拷进来，所以 graph 状态从不依赖一个共享的可变张量。
        #     块 decode 时草稿模型在 graph 内填充它；先用块宽度初始化，使广播之前的任何读取都保持在合法范围内。
        self.seq_lens_buf = torch.full(
            (entries,),
            self.spec_num_tokens if self.block_decode_active else 0,
            dtype=torch.int32,
            device=self.device,
        )
        self.tree_prefix_lens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        # Buffers were (re)allocated: cached per-bs views must rebuild.
        # 译：缓冲区被（重新）分配：缓存的按 bs 视图必须重建。
        self._decode_views_by_bs = {}

    @abstractmethod
    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        query_shard: QueryShardPlan | None,
        page_table_cpu: torch.Tensor | None,
        **kwargs,
    ) -> None:
        """Build extend/mixed (or idle warmup) metadata.

        Args:
            bs: Requests in the batch (extend requests first, then any decode
                requests).
            num_extends: Leading extend requests; ``bs`` for a pure extend.
            seq_lens: ``[>= bs]`` total cache lengths after this step.
            page_table: ``[bs, max_num_pages]`` kernel page table,
                batch-ordered and padded (request i is batch position i).
            forward_mode: EXTEND, MIXED or IDLE; a DECODE call is a contract
                violation (decode metadata is :meth:`refresh_decode_metadata`).
            extend_*: ``[>= num_extends]`` per-request new-token / prefix
                lengths with their pinned host mirrors (empty on idle warmup).
            extend_with_prefix: Whether any extend request continues a cached or
                chunked prefix (some ``extend_prefix_lens`` entry is non-zero);
                leaves that size ragged-vs-paged prefill metadata read it.
            query_shard: The rows of the extend span this rank computes under
                query context parallelism, or ``None`` when every rank computes
                every row. Every length above describes the whole span; a leaf
                without a gathered-history extend arm calls
                :func:`reject_query_shard`.
            page_table_cpu: Host mirror of ``page_table``'s extend rows
                (``[num_extends, cols]`` int32 kernel pages, unpadded) when the
                forward is sharded, or when this leaf's group is KVP
                page-sharded and the forward extends a prefix or is a pure
                extend; else ``None``. The only host-side table a leaf sees:
                the sharded extend arm and the KVP prefix-chunk gather count
                how many history rows each page owner holds from it, so the
                gather's per-rank split never waits on the device, and the DSA
                leaf bounds the index pages each rank owns from it.
        """
        # 译：构建 extend / mixed（或 idle 预热）元数据。
        #
        #     参数：
        #         bs：批次中的请求数（先是 extend 请求，然后是可能有的 decode 请求）。
        #         num_extends：排在前面的 extend 请求数；纯 extend 时等于 ``bs``。
        #         seq_lens：``[>= bs]`` 本步之后的 cache 总长度。
        #         page_table：``[bs, max_num_pages]`` kernel 页表，按批次排序并已 padding（请求 i 就是批次位置 i）。
        #         forward_mode：EXTEND、MIXED 或 IDLE；传入 DECODE 属于违反契约（decode 元数据走 :meth:`refresh_decode_metadata`）。
        #         extend_*：``[>= num_extends]`` 每个请求的新 token 数 / 前缀长度，以及它们锁页的 Host 镜像（idle 预热时为空）。
        #         extend_with_prefix：是否有任何 extend 请求接续已缓存的或分块的前缀（某个 ``extend_prefix_lens`` 项非零）；
        #             需要在 ragged 与分页 prefill 元数据之间确定尺寸的叶子会读它。
        #         query_shard：在 query-context 并行下本 rank 计算的 extend 段中的行；每个 rank 都计算全部行时为 ``None``。
        #             上面每个长度都描述整个段；没有“收集历史”extend 分支的叶子调用 :func:`reject_query_shard`。
        #         page_table_cpu：``page_table`` 中 extend 行的 Host 镜像（``[num_extends, cols]`` int32 kernel 页，不 padding），
        #             在前向被分片时提供，或在本叶子的组按 KVP 分页且前向接续前缀或是纯 extend 时提供；否则为 ``None``。
        #             这是叶子能看到的唯一 Host 端表：分片 extend 分支和 KVP 前缀块收集据此统计每个页所有者持有多少历史行，
        #             这样收集的逐 rank 划分从不等待设备；DSA 叶子据此界定每个 rank 拥有的索引页。

    @abstractmethod
    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_extends: int = 0,
        for_graph_replay: bool = False,
    ) -> None:
        """The single decode metadata path — eager decode and graph replay.

        Copies ``seq_lens`` (clamped to :attr:`verify_floor`) and
        ``page_table`` into the persistent buffers in place and points
        ``forward_decode_metadata`` at the pointer-stable per-bs views over
        them. Requests in ``[actual_bs, bs)`` are padding (already null pages in
        ``page_table``, seq_len 1 in ``seq_lens``); ``actual_bs == 0`` is the
        idle replay and the capture seeding.

        Args:
            bs: Requests to prepare (the padded graph batch under replay).
            actual_bs: Live requests.
            seq_lens: ``[>= bs]`` live cache lengths.
            page_table: ``[bs, max_num_pages]`` kernel page table for these
                requests, batch-ordered and padded.
            num_extends: Leading extend requests of a MIXED round whose decode
                half this refresh describes; 0 for pure decode.
            for_graph_replay: A graph is in play (live replay or the capture
                seeding). Branch on it only for graph-mechanics asymmetries
                a shared in-place refresh cannot express.
        """
        # 译：唯一的 decode 元数据路径——eager decode 和 graph 回放共用。
        #
        #     把 ``seq_lens``（截断到 :attr:`verify_floor` 以上）和 ``page_table`` 原地拷贝进持久缓冲区，
        #     并让 ``forward_decode_metadata`` 指向它们之上指针稳定的按 bs 视图。``[actual_bs, bs)`` 中的请求是 padding
        #     （在 ``page_table`` 中已是空页，在 ``seq_lens`` 中 seq_len 为 1）；``actual_bs == 0`` 表示 idle 回放和捕获时的初始填充。
        #
        #     参数：
        #         bs：要准备的请求数（回放时为 padding 后的 graph 批大小）。
        #         actual_bs：活跃请求数。
        #         seq_lens：``[>= bs]`` 活跃的 cache 长度。
        #         page_table：这些请求的 ``[bs, max_num_pages]`` kernel 页表，按批次排序并已 padding。
        #         num_extends：本次刷新描述其 decode 半批的那个 MIXED 轮中，排在前面的 extend 请求数；纯 decode 为 0。
        #         for_graph_replay：有 graph 参与（实际回放或捕获时的初始填充）。只为共享的原地刷新无法表达的
        #             graph 机制上的不对称而据此分支。

    def decode_buffer_targets(
        self, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The ``(seq_lens, page_table)`` buffer views a decode refresh copies into.

        A leaf whose :meth:`refresh_decode_metadata` is exactly those two
        copies (seq_lens clamped to :attr:`decode_seq_lens_floor`) followed
        by :meth:`publish_decode_metadata` returns them, so the router can
        fill every such leaf with one batched copy; a leaf with any other
        refresh returns None and refreshes itself.

        Args:
            bs: Requests to prepare (the padded graph batch under replay).
        """
        # 译：decode 刷新要拷入的 ``(seq_lens, page_table)`` 缓冲区视图。
        #
        #     若某个叶子的 :meth:`refresh_decode_metadata` 恰好就是这两次拷贝（seq_lens 截断到 :attr:`decode_seq_lens_floor`）
        #     再加 :meth:`publish_decode_metadata`，它就返回这两个视图，使 router 可以用一次批量拷贝填好所有这类叶子；
        #     刷新方式不同的叶子返回 None，自己刷新。
        #
        #     参数：
        #         bs：要准备的请求数（回放时为 padding 后的 graph 批大小）。
        return None

    @property
    def decode_seq_lens_floor(self) -> int:
        """The minimum the router clamps ``seq_lens`` to before copying them
        into :meth:`decode_buffer_targets`. Decode ``seq_lens`` are at least
        1 (padding rows replay at 1), so 1 copies them unchanged."""
        # 译：router 在把 ``seq_lens`` 拷入 :meth:`decode_buffer_targets` 之前截断到的下限。decode 的 ``seq_lens`` 至少为 1
        #     （padding 行以 1 回放），所以下限为 1 时原样拷贝。
        return 1

    def publish_decode_metadata(
        self, bs: int, seq_lens: torch.Tensor, *, num_extends: int
    ) -> None:
        """The rest of :meth:`refresh_decode_metadata` once the router has
        filled :meth:`decode_buffer_targets`; arguments as there."""
        # 译：router 填好 :meth:`decode_buffer_targets` 之后，:meth:`refresh_decode_metadata` 余下的部分；参数同那里。
        raise NotImplementedError(
            f"{type(self).__name__} returns no decode_buffer_targets"
        )

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, seq_lens: torch.Tensor, page_table: torch.Tensor
    ) -> None:
        """Capture seeding: the idle refresh over the same buffers replay
        refreshes. Override only for a kernel-imposed capture asymmetry."""
        # 译：捕获时的初始填充：在回放时刷新的同一批缓冲区上做 idle 刷新。只有在 kernel 强加了捕获上的不对称时才覆盖。
        self.refresh_decode_metadata(bs, 0, seq_lens, page_table, for_graph_replay=True)

    def preallocate_history_gather_workspace(self, max_model_len: int) -> int:
        """A leaf without the sharded extend arm reserves nothing; the router
        sums its leaves and the registry checks the total against the plan."""
        # 译：没有分片 extend 分支的叶子不预留任何东西；router 对其叶子求和，registry 对照规划检查总数。
        del max_model_len
        return 0

    def adopt_history_gather_workspace(self, workspace: HistoryGatherWorkspace) -> None:
        """A leaf without the sharded extend arm has nothing to gather into."""
        # 译：没有分片 extend 分支的叶子没有可以收集进去的东西。
        del workspace

    # 以下三个草稿钩子属于投机解码（第 11 阶段），初读可跳过。
    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """Publish a drafter's in-graph seq_lens edits into this leaf's own
        cache-seqlens buffer (one token per request per step)."""
        # 译：把草稿模型在 graph 内对 seq_lens 的修改发布到本叶子自己的 cache 序列长度缓冲区（每步每个请求一个 token）。
        buf = self.decode_seq_lens_buffer
        bs = seq_lens.shape[0]
        buf[:bs].copy_(seq_lens[:bs])

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        """Publish a multi-depth MTP drafter's re-anchored window: every depth
        re-runs ``spec_num_tokens`` rows per request ending at ``frontier``
        (``[bs]`` committed lengths), in-graph.

        The decode kernels derive each row's causal bound from the request's
        single cache length, so for most leaves this is the same seq_lens
        edit as :meth:`advance_draft_forward_metadata`. A leaf holding
        per-row decode metadata (DSA's per-token indexer rows) re-expands it
        to the k-row shape here instead.
        """
        # 译：发布多深度 MTP 草稿模型重新锚定的窗口：每个深度为每个请求重新运行以 ``frontier``（``[bs]`` 已提交长度）
        #     结尾的 ``spec_num_tokens`` 行，在 graph 内完成。
        #
        #     decode kernel 从请求的单个 cache 长度推导每行的因果边界，所以对大多数叶子来说，这与
        #     :meth:`advance_draft_forward_metadata` 是同样的 seq_lens 修改。持有逐行 decode 元数据的叶子
        #     （DSA 的逐 token 索引器行）则在这里把它重新展开为 k 行的形状。
        self.advance_draft_forward_metadata(frontier)

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """DFLASH: broadcast each request's block-end length to its
        ``block_decode_expansion`` materialized entries (uniform, non-causal),
        clamped to the block width and the context limit."""
        # 译：DFLASH：把每个请求的块末尾长度广播到它物化的 ``block_decode_expansion`` 个条目（统一、非因果），
        #     并截断到块宽度和上下文上限之间。
        expansion = self.block_decode_expansion
        self.decode_seq_lens_buffer[: bs * expansion].view(bs, expansion).copy_(
            block_seq_lens[:bs]
            .clamp(self.spec_num_tokens, self.max_context_len)
            .unsqueeze(1)
        )

    @property
    def decode_seq_lens_buffer(self) -> torch.Tensor:
        """The persistent cache-seqlens buffer decode metadata views."""
        # 译：decode 元数据视图所在的持久 cache 序列长度缓冲区。
        return self.seq_lens_buf

    @contextmanager
    def override_num_extends(self, num_extends: int):
        """Temporarily override the decode-request slice discriminator (MLA
        family: drafter step 0 slices ``[num_extends:]``, step 1+ ``[0:]``).
        Default no-op for leaves with separate prefill/decode slots."""
        # 译：临时覆盖 decode 请求的切片判别量（MLA 家族：草稿第 0 步切 ``[num_extends:]``，第 1 步及以后切 ``[0:]``）。
        #     对 prefill / decode 槽分开的叶子默认什么都不做。
        yield

    def supports_narrowed_draft_decode(self, forward_mode: ForwardMode) -> bool:
        return False

    def set_request_slots(self, req_pool_indices: torch.Tensor) -> None:
        """Publish this forward's ``[bs]`` request-pool slots (batch order).

        The one side channel beyond ``page_table`` / ``seq_lens``: a leaf that
        owns per-request side state indexed by pool slot (DSA's KPool tails)
        reads it; paged KV leaves need nothing but the table. The router
        calls this after every metadata build (extend init, decode refresh,
        capture seeding), before the model runs. Default: no-op.
        """
        # 译：发布本次前向的 ``[bs]`` 请求池槽位（批次顺序）。
        #
        #     这是 ``page_table`` / ``seq_lens`` 之外唯一的旁路通道：持有按池槽位索引的逐请求旁路状态的叶子
        #     （DSA 的 KPool 尾部）会读它；分页 KV 叶子只需要页表。router 在每次元数据构建（extend 初始化、decode 刷新、
        #     捕获初始填充）之后、模型运行之前调用它。默认：什么都不做。
        del req_pool_indices

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    # 译：前向

    @abstractmethod
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        **kwargs,
    ) -> torch.Tensor:
        """Decode attention over the current ``forward_decode_metadata``; the
        prologue has already written this forward's KV rows."""
        # 译：基于当前 ``forward_decode_metadata`` 做 decode 注意力；prologue 已经写好了本次前向的 KV 行。

    @abstractmethod
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        **kwargs,
    ) -> torch.Tensor:
        """Extend attention over the current extend metadata; the prologue has
        already written this forward's KV rows."""
        # 译：基于当前 extend 元数据做 extend 注意力；prologue 已经写好了本次前向的 KV 行。

    # 以下与 MLA 的分块前缀重放、KV 跨 rank 分布（DCP/KVP）有关，第 12 阶段再看。
    def forward_extend_chunked(self, *args, **kwargs):
        """DeepSeek's chunked prefix replay (MLA family); others never call it."""
        # 译：DeepSeek 的分块前缀重放（MLA 家族）；其他叶子从不调用它。
        raise NotImplementedError(
            f"{type(self).__name__} has no chunked prefix-replay prefill"
        )

    def _prepare_prefix_history(
        self,
        chunk_kv_indices_list: Sequence[torch.Tensor],
        *,
        page_table_cpu: torch.Tensor | None,
        chunk_bounds_cpu: torch.Tensor,
    ) -> None:
        """Record a prefill forward's prefix-chunk loops for
        :meth:`read_prefix_history`.

        A leaf declaring ``supports_mla_dcp`` calls this from its prefill metadata
        init with the loops it cut (``chunk.build_chunked_prefill_metadata_arrays``);
        while its cache is local this does nothing.

        Args:
            chunk_kv_indices_list: Each loop's ``[rows]`` virtual slots.
            page_table_cpu: ``[requests, columns]`` host kernel-page table of
                the extend requests; required when the cache is dealt over
                owners and there are loops.
            chunk_bounds_cpu: ``[loops + 1, requests]`` host prefix positions
                delimiting exactly the same loops as the slot indices.
        """
        # 译：为 :meth:`read_prefix_history` 记录一次 prefill 前向的前缀块循环。
        #
        #     声明了 ``supports_mla_dcp`` 的叶子在其 prefill 元数据初始化中，用它切出的循环
        #     （``chunk.build_chunked_prefill_metadata_arrays``）调用这个方法；cache 在本地时这里什么都不做。
        #
        #     参数：
        #         chunk_kv_indices_list：每个循环的 ``[rows]`` 虚拟槽位。
        #         page_table_cpu：extend 请求的 ``[requests, columns]`` Host kernel 页表；cache 分发给多个所有者
        #             且存在循环时必须提供。
        #         chunk_bounds_cpu：``[loops + 1, requests]`` Host 端前缀位置，界定的循环与槽位下标完全相同。
        if self._prefix_history is None:
            # A leaf deals every layer's pages the same way.
            # 译：一个叶子对每一层的页都用同样的方式分发。
            placement = self.cache_placement(None)
            if placement is None:
                return
            self._prefix_history = PrefixHistoryGather.on_history_groups(
                placement, page_size=self.kernel_page_size
            )
        self._prefix_history.prepare(
            chunk_kv_indices_list,
            page_table_cpu=page_table_cpu,
            chunk_bounds_cpu=chunk_bounds_cpu,
        )

    def read_prefix_history(
        self,
        pool: CachePool,
        layer: PagedAttention,
        chunk_kv_indices_list: Sequence[torch.Tensor],
        *,
        dst_dtype: torch.dtype,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """The latent and RoPE keys of ``layer`` at every prefix-chunk loop's
        virtual slots (``chunked_prefill_metadata``), one pair per loop in
        loop order, for DeepSeek's chunked prefix replay (MLA family).

        A local cache is read in place; one dealt over owners is gathered
        from them as :meth:`_prepare_prefix_history` recorded.
        """
        # 译：``layer`` 在每个前缀块循环的虚拟槽位（``chunked_prefill_metadata``）处的潜在 key 和 RoPE key，
        #     按循环顺序每个循环一对，用于 DeepSeek 的分块前缀重放（MLA 家族）。
        #
        #     本地 cache 原地读取；分发给多个所有者的 cache 则按 :meth:`_prepare_prefix_history` 记录的方式从它们那里收集。
        if self._prefix_history is not None:
            return self._prefix_history.read(
                pool, layer, chunk_kv_indices_list, dst_dtype=dst_dtype
            )
        if self.cache_placement(layer) is not None:
            raise NotImplementedError(
                f"{type(self).__name__} deals history pages over owners but "
                "prepared no prefix read (_prepare_prefix_history)"
            )
        return (
            pool.get_mla_kv_buffer(layer, slots, dst_dtype)
            for slots in chunk_kv_indices_list
        )
