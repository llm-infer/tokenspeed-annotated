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

# 并行映射（Mapping）：给定本 rank 的全局编号和各维度的并行大小，算出它在每种并行里的位置和通信组。
#
# 你在这里：启动时按命令行参数构造一个 Mapping，传给模型；各层从中取自己需要的部分：
#   mapping.attn：注意力层（Qwen3Attention 用 mapping.attn.tp_*）
#   mapping.dense：稠密 MLP（GatedMLP 用 mapping.dense.tp_*）
#   mapping.moe：MoE 层（第 10 阶段）
#   mapping.lm_head：LM head
#   mapping.vision / mapping.linear_attn：视觉编码器、线性注意力层
#   以及流水线并行的 pp_*
# 不同类型的层可以用不同的并行方式，这是本项目和“全模型统一 TP”做法的主要区别。
# 例如 attention DP（注意力按请求切分到多组，每组内部再 TP）+ MoE 用 EP（专家并行）。
# 单卡时所有维度都是 1，所有 group 都是 (0,)。
#
# rank 编号的排布：越“内层”的维度，对应编号越相邻的 rank（stride 越小）：
#   一个流水线 stage 内：TP 最内层（stride 1），DP 在外（stride = tp_size）；
#   流水线 PP 在最外层：world 先切成 pp_size 个连续段，每段是一个 stage。
#
# 例：8 卡，注意力 TP=4、DP=2，看 rank 5：
#   tp_rank = 5 % 4 = 1，tp_group = (4, 5, 6, 7)    ← 相邻的 4 个 rank 一起做 TP
#   dp_rank = (5 // 4) % 2 = 1，dp_group = (1, 5)   ← 间隔 4 的 rank 属于同一个 DP 组
import math
from functools import cached_property

# 通信组的表示：组内所有 rank 的全局编号组成的元组，例如 (4, 5, 6, 7)
Group = tuple[int, ...]

# The named copy of the attention KVP group that page-sharded MLA prefill
# all-gathers its cached history on (``ProcessGroupManager.init_process_group``):
# its own communicators let the transfers overlap the attention TP collectives
# over the same ranks instead of queueing with them.
# 译：页分片的 MLA prefill 在其上 all-gather 缓存历史的、注意力 KVP 组的具名副本
#     （``ProcessGroupManager.init_process_group``）：它有自己的通信器，使这些传输可以与同一组 rank 上的
#     注意力 TP 集合通信重叠，而不是和它们一起排队。
KVP_HISTORY_GROUP_NAME = "kvp_history"


# 解析各维度的并行大小：sizes 中为 None 的维度自动推导。
# 规则：已给出的维度相乘后，world_size 剩下的倍数全部给第一个 None 的维度，其余 None 维度取 1。
#   例：world=8，(tp=None, dp=2) -> (4, 2)；(tp=None, dp=None) -> (8, 1)
# 最后要求所有维度相乘恰好等于 world_size。
def _resolve_parallelism_sizes(world_size: int, *sizes: int | None) -> tuple[int, ...]:
    """Resolve the parallelism sizes given world_size.

    `sizes` is ordered innermost (fastest-varying) to outermost.
    """
    # 译：根据 world_size 解析各个并行维度的大小。
    #
    #     `sizes` 按从最内层（变化最快）到最外层的顺序排列。
    assert all(x is None or x > 0 for x in sizes)

    resolved = [x for x in sizes]
    num_to_resolve = sum(x is None for x in sizes)
    if num_to_resolve > 0:
        provided_size = math.prod(x for x in sizes if x is not None)
        assert provided_size <= world_size
        assert world_size % provided_size == 0
        resolved_size = world_size // provided_size

        for index, size in enumerate(resolved):
            if size is None:
                resolved[index] = resolved_size
                resolved_size = 1

    assert math.prod(resolved) == world_size
    return tuple(resolved)


# 下面两个函数校验注意力 TP 组内的两种子划分（都不增加 world 的维度）：
#   KVP（KV 并行）：KV cache 的页在 TP 组内连续的子组上轮流存放（第 12 阶段）
#   QCP（查询上下文并行）：一次 extend 的 query 行分到整个注意力 TP 组上（第 12 阶段），
#     只能是 1（关闭）或等于注意力 TP 大小
def _resolve_kvp_size(tp_size: int, kvp_size: int) -> int:
    """Validate KV parallelism within resolved attention TP.

    KVP shards the KV cache pages page-cyclically over a consecutive subgroup
    of attention TP; it adds no world-size dimension.
    """
    # 译：校验 KV 并行在已解析的注意力 TP 之内。
    #
    #     KVP 把 KV cache 的页按页轮流（page-cyclically）分布到注意力 TP 中一个连续的子组上；它不增加 world size 的维度。
    if isinstance(kvp_size, bool) or not isinstance(kvp_size, int) or kvp_size < 1:
        raise ValueError("kvp_size must be a positive integer")
    if tp_size % kvp_size:
        raise ValueError("attention TP size must be divisible by KVP size")
    return kvp_size


def _resolve_qcp_size(tp_size: int, qcp_size: int) -> int:
    """Validate query context parallelism within resolved attention TP.

    QCP shards an extend forward's query rows over the attention TP group;
    it adds no world-size dimension. The shard spans the whole group (1 is
    off): a partial shard would leave the ranks holding the same rows strided
    by the shard width, which no group here (the head group, the KVP group)
    is built with. This is the one place the rule lives; the server's
    ``validate_qcp`` and ``validate_context_parallel_roles`` cover what the
    mapping cannot see (the engine role, the other parallel groups).
    """
    # 译：校验查询上下文并行在已解析的注意力 TP 之内。
    #
    #     QCP 把一次 extend 前向的 query 行分布到注意力 TP 组上；它不增加 world size 的维度。分片跨越整个组
    #     （1 表示关闭）：部分分片会让持有相同行的 rank 以分片宽度为间隔交错分布，而这里没有任何组（head 组、
    #     KVP 组）是这样构建的。这是该规则唯一所在的地方；服务端的 ``validate_qcp`` 和
    #     ``validate_context_parallel_roles`` 覆盖 mapping 看不到的部分（引擎角色、其他并行组）。
    if isinstance(qcp_size, bool) or not isinstance(qcp_size, int) or qcp_size < 1:
        raise ValueError("qcp_size must be a positive integer")
    if tp_size % qcp_size:
        raise ValueError("attention TP size must be divisible by QCP size")
    if qcp_size not in (1, tp_size):
        raise ValueError(
            f"a query shard spans the whole attention TP group: qcp_size={qcp_size} "
            f"must be 1 or the attention TP size {tp_size}"
        )
    return qcp_size


# 两个核心公式（所有 *_rank / *_group 属性都由它们算出）：
#   某维度的编号 = (rank // stride) % size
#   所在组 = 从 base 开始、间隔 stride 的 size 个 rank，base 是把本 rank 在该维度上的编号清零后的 rank
#   例：rank=5，size=2，stride=4 -> 编号 (5//4)%2 = 1；base = 5 - 1*4 = 1；组 = (1, 5)
def _make_parallelism_rank(rank: int, size: int, stride: int = 1) -> int:
    """Return the rank of given size and stride."""
    # 译：返回给定大小和步长下的编号。
    return (rank // stride) % size


def _make_parallelism_group(rank: int, size: int, stride: int = 1) -> Group:
    """Return the group of ranks of given size and stride."""
    # 译：返回给定大小和步长下的 rank 组。
    base = rank - (rank // stride % size) * stride
    return tuple(base + j * stride for j in range(size))


# 所有映射的基类：保存全局 rank 和本映射的 world 大小。
# rank 可以先空着（None），之后通过 rank 属性赋值一次（构造 Mapping 时还不知道自己是哪个进程的情况）；
# 赋值时触发 _on_rank_initialized，让组合映射把 rank 传给各子映射。
class MappingBase:

    def __init__(self, rank: int | None = None, world_size: int = 1):
        assert rank is None or rank >= 0
        self._rank = rank
        assert world_size > 0
        self._world_size = world_size

    @property
    def rank(self) -> int:
        assert self._rank is not None, "rank is not initialized"
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        assert self._rank is None, "rank is already initialized"
        assert rank >= 0
        self._rank = rank
        self._on_rank_initialized(rank)

    def _on_rank_initialized(self, rank: int):
        return None

    @property
    def world_size(self) -> int:
        return self._world_size

    @cached_property
    def world_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.world_size, stride=1)


# 稠密 MLP 的映射：TP（内层）× DP（外层）。
# dense TP 可以与注意力 TP 不同。例如注意力用 DP、MLP 用更宽的 TP，此时层与层之间需要按 token 行
# 重新分布（由 CommManager 处理，第 9 阶段）。
class DenseLayerMapping(MappingBase):

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        dp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, dp_size
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)


# 注意力 TP 组内持有“相同行”的 rank 数：不开 QCP 时是整个 TP 组；开 QCP 时每个 rank 的行都不同，为 1
def _attention_row_group_size(tp_size: int, qcp_size: int) -> int:
    """Ranks of the attention TP group that hold the same attention rows.

    Attention TP replicates a forward's rows over its group; query context
    parallelism (QCP) shards an extend's rows over that group instead, so
    only ``tp_size // qcp_size`` ranks hold the same rows -- one, since a
    query shard spans the whole group (``_resolve_qcp_size``), which is what
    lets the groups below stay contiguous.
    """
    # 译：注意力 TP 组中持有相同注意力行的 rank 数。
    #
    #     注意力 TP 在其组内复制一次前向的所有行；查询上下文并行（QCP）则把一次 extend 的行分布到该组上，
    #     所以只有 ``tp_size // qcp_size`` 个 rank 持有相同的行——由于查询分片跨越整个组
    #     （``_resolve_qcp_size``），这个数为 1，这也让下面的组可以保持连续。
    return tp_size // qcp_size


# 注意力 head TP 的宽度：默认等于持有相同行的 rank 数（即普通的注意力 TP）。
# 可以设得更宽，让持有不同行的 rank（attention DP 的各组、或 QCP 的各分片）也按头切分投影权重，
# 此时注意力前后要在“按 token 分”与“按头分”之间交换数据（第 12 阶段，DeepSeek V4.1 等使用）。
def _resolve_head_tp_size(
    tp_size: int, qcp_size: int, world_size: int, head_tp_size: int | None
) -> int:
    """Resolve the attention head-TP width.

    ``None`` keeps the head projections on the ranks that hold the same
    attention rows (today's layout: the attention TP group; head-replicated
    under query context parallelism, whose shards hold different rows). A
    wider head group shards ``q_b_proj`` / ``kv_b_proj`` / ``o_proj`` by
    heads over ranks that hold different rows -- attention-DP ranks, or the
    query shards of a QCP group -- and the attention forward exchanges heads
    for tokens around core attention. It therefore requires that no two
    ranks of the group hold the same rows (attention TP 1, or a query shard
    spanning the attention TP group), is the QCP group itself under query
    sharding, and must tile the stage world.
    """
    # 译：解析注意力 head-TP 的宽度。
    #
    #     ``None`` 表示把 head 投影保留在持有相同注意力行的 rank 上（目前的布局：注意力 TP 组；在查询上下文并行下
    #     head 是复制的，因为各分片持有不同的行）。更宽的 head 组把 ``q_b_proj`` / ``kv_b_proj`` / ``o_proj``
    #     按头切分到持有不同行的 rank 上——attention-DP 的 rank，或一个 QCP 组的各个查询分片——注意力前向在核心
    #     注意力前后用头交换 token。因此它要求组内没有两个 rank 持有相同的行（注意力 TP 为 1，或查询分片跨越
    #     注意力 TP 组），在查询分片下它就是 QCP 组本身，并且必须铺满 stage 的 world。
    row_group_size = _attention_row_group_size(tp_size, qcp_size)
    if head_tp_size is None:
        return row_group_size
    if (
        isinstance(head_tp_size, bool)
        or not isinstance(head_tp_size, int)
        or head_tp_size < 1
    ):
        raise ValueError("attention head TP size must be a positive integer")
    if head_tp_size == row_group_size:
        return head_tp_size
    if row_group_size != 1:
        raise ValueError(
            "attention head TP shards heads over ranks that hold different rows "
            "(attention-DP ranks, or the query shards of a QCP group) and needs "
            "attention TP 1 or a query shard over the attention TP group, "
            f"got attn_tp_size={tp_size} with qcp_size={qcp_size}"
        )
    if qcp_size != 1 and head_tp_size != qcp_size:
        raise ValueError(
            "under query context parallelism the attention head TP group is the "
            f"query-shard group: head_tp_size={head_tp_size} must equal "
            f"qcp_size={qcp_size}"
        )
    if world_size % head_tp_size:
        raise ValueError(
            f"attention head TP size {head_tp_size} must divide the stage "
            f"world size {world_size}"
        )
    return head_tp_size


# 注意力层的映射：TP（内层）× DP（外层），外加 TP 组内的 KVP、QCP 子划分和 head TP。
# 主线（单卡）只用到 tp_rank / tp_size / tp_group，都是 0 / 1 / (0,)。
# 例（8 卡，TP=4、DP=2，rank 5）：tp_rank=1，tp_group=(4,5,6,7)，dp_rank=1，dp_group=(1,5)
class AttentionLayerMapping(MappingBase):

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        dp_size: int | None = None,
        kvp_size: int = 1,
        head_tp_size: int | None = None,
        qcp_size: int = 1,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, dp_size
        )
        self.kvp_size = _resolve_kvp_size(self.tp_size, kvp_size)
        self.qcp_size = _resolve_qcp_size(self.tp_size, qcp_size)
        # Width of the group the head projections (q_b/kv_b/o_proj) shard
        # over. Equal to the ranks holding the same rows (tp_size, or 1 under
        # query sharding) unless head TP widens it over DP ranks or over the
        # query shards.
        # 译：head 投影（q_b/kv_b/o_proj）在其上切分的组的宽度。等于持有相同行的 rank 数（tp_size，或在查询分片下为 1），
        #     除非 head TP 把它扩展到 DP 的 rank 上或查询分片上。
        self.head_tp_size = _resolve_head_tp_size(
            self.tp_size, self.qcp_size, self.world_size, head_tp_size
        )

    @property
    def has_head_tp(self) -> bool:
        """Heads are sharded over a group wider than the ranks that hold the
        same attention rows (the attention TP group, or one query shard of it
        under QCP), so the attention forward exchanges heads for tokens around
        core attention."""
        # 译：头被切分到一个比“持有相同注意力行的 rank”（注意力 TP 组，或 QCP 下它的一个查询分片）更宽的组上，
        #     因此注意力前向要在核心注意力前后用头交换 token。
        return self.head_tp_size != _attention_row_group_size(
            self.tp_size, self.qcp_size
        )

    @property
    def head_tp_serves_decode_only(self) -> bool:
        """Head TP over attention-DP ranks serves decode rows only: an expanded
        prefill needs every head's K/V for the cached prefix, which the
        head-sharded ``kv_b_proj`` cannot produce. The engine therefore tunes
        on a decode step and runs no prefill graph; it needs no admission
        rule for it, because a retracted request resumes by restore, not by
        a recovery prefill. Over the query shards of a QCP group the sparse
        prefill is absorbed and the extend rows take the exchange, so that
        layout serves the prefill role."""
        # 译：跨 attention-DP rank 的 head TP 只服务 decode 行：展开的 prefill 需要缓存前缀上每个头的 K/V，
        #     而按头切分的 ``kv_b_proj`` 产生不了。因此引擎在一个 decode 步上做调优，也不运行 prefill graph；
        #     它不需要为此设置准入规则，因为被抢占的请求通过恢复（restore）继续，而不是通过重新 prefill。
        #     在 QCP 组的查询分片上，稀疏 prefill 被吸收，extend 行参与交换，所以那种布局可以承担 prefill 角色。
        return self.has_head_tp and not self.has_qcp

    @cached_property
    def head_tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.head_tp_size, stride=1)

    @cached_property
    def head_tp_group(self) -> Group:
        """Contiguous ranks sharing one set of head projections; equals
        ``tp_group`` without head TP (``(rank,)`` under query sharding, whose
        default is head-replicated) and ``qcp_group`` under head TP over the
        query shards."""
        # 译：共享一组 head 投影的连续 rank；没有 head TP 时等于 ``tp_group``（在查询分片下为 ``(rank,)``，
        #     其默认是 head 复制的），在跨查询分片的 head TP 下等于 ``qcp_group``。
        return _make_parallelism_group(self.rank, self.head_tp_size, stride=1)

    @property
    def has_kvp(self) -> bool:
        return self.kvp_size > 1

    @property
    def has_qcp(self) -> bool:
        return self.qcp_size > 1

    @cached_property
    def qcp_rank(self) -> int:
        """Rank within the consecutive query-context-parallel subgroup of
        attention TP; it is this rank's query shard."""
        # 译：在注意力 TP 中连续的查询上下文并行子组内的编号；即本 rank 的查询分片。
        return _make_parallelism_rank(self.rank, self.qcp_size, stride=1)

    @cached_property
    def qcp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.qcp_size, stride=1)

    @cached_property
    def kvp_rank(self) -> int:
        """Rank within the consecutive KVP subgroup of attention TP; it selects
        the KV pages this rank stores."""
        # 译：在注意力 TP 中连续的 KVP 子组内的编号；它决定本 rank 存储哪些 KV 页。
        return _make_parallelism_rank(self.rank, self.kvp_size, stride=1)

    @cached_property
    def kvp_replica_rank(self) -> int:
        return _make_parallelism_rank(
            self.rank, self.tp_size // self.kvp_size, stride=self.kvp_size
        )

    @cached_property
    def kvp_group(self) -> Group:
        """The ranks that together hold one copy of the KV pages."""
        # 译：共同持有一份完整 KV 页的那些 rank。
        return _make_parallelism_group(self.rank, self.kvp_size, stride=1)

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)

    # 本 rank 在“按 DP 组为主、组内按 TP 编号”排列的 token 计数表中的下标。
    # CommManager 用这张表记录每个 rank 持有多少 token 行（RSAG 布局，第 9 阶段）。
    def scatter_index(self, rank: int) -> int:
        """Index of ``rank`` in a dp-major/tp-minor scattered token count
        table."""
        # 译：``rank`` 在 dp 为主序、tp 为次序的分散 token 计数表中的下标。
        tp_rank = _make_parallelism_rank(rank, self.tp_size, stride=1)
        dp_rank = _make_parallelism_rank(rank, self.dp_size, stride=self.tp_size)
        return dp_rank * self.tp_size + tp_rank


# MoE 层的映射：TP（最内层）× EP（专家并行）× DP（最外层）。
# tp_ep_group：TP 与 EP 合起来的连续组，MoE 的 token 分发和合并在这个组内进行（第 10 阶段）。
class MoeLayerMapping(MappingBase):
    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.ep_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, ep_size, dp_size
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_ep(self) -> bool:
        return self.ep_size > 1

    @cached_property
    def ep_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.ep_size, stride=self.tp_size)

    @cached_property
    def ep_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.ep_size, stride=self.tp_size)

    @cached_property
    def has_tp_ep(self) -> bool:
        return self.tp_ep_size > 1

    @cached_property
    def tp_ep_size(self) -> int:
        return self.tp_size * self.ep_size

    @cached_property
    def tp_ep_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_ep_size, stride=1)

    @cached_property
    def tp_ep_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_ep_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(
            self.rank, self.dp_size, stride=self.tp_size * self.ep_size
        )

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(
            self.rank, self.dp_size, stride=self.tp_size * self.ep_size
        )


# 视觉编码器的映射（第 16 阶段）
class VisionTowerMapping(MappingBase):
    """Parallel mapping for colocated multimodal encoders.

    ``tp_size`` controls weight tensor parallelism inside the encoder, while
    ``dp_size`` controls item data parallelism. The mapping's world is one
    attention TP group, so item-DP composes independently with outer request
    data parallelism.
    """

    # 译：同机部署的多模态编码器的并行映射。
    #
    #     ``tp_size`` 控制编码器内部的权重张量并行，``dp_size`` 控制按条目（item）的数据并行。该映射的 world
    #     是一个注意力 TP 组，因此按条目的 DP 与外层的请求数据并行相互独立地组合。

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        dp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, dp_size
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)


# 线性注意力层（KDA、GDN）的映射（第 13 阶段）
class LinearAttnLayerMapping(MappingBase):
    """Parallel mapping for linear-attention layers (KDA, GDN).

    Linear-attention layers default to the attention TP width, which
    preserves the historical behavior on every existing deployment. Unlike
    MLA — whose per-token latent KV cannot shard by heads and therefore
    needs attention-DP — linear attention is TP-friendly: weights and the
    per-head recurrent state both shard by head. A wider ``tp_size`` (up to
    the stage world) head-shards them across ranks that are data-parallel
    for the full-attention layers (the MLA-DP + linear-attn-TP hybrid).
    The TP group is contiguous (stride 1) inside a pipeline stage.
    """

    # 译：线性注意力层（KDA、GDN）的并行映射。
    #
    #     线性注意力层默认使用注意力 TP 的宽度，这保持了所有现有部署上的历史行为。与 MLA 不同——MLA 每个 token
    #     的潜在 KV 无法按头切分，因此需要 attention-DP——线性注意力对 TP 友好：权重和每个头的递归状态都按头切分。
    #     更宽的 ``tp_size``（最大到 stage 的 world）会把它们按头切分到那些对全注意力层而言是数据并行的 rank 上
    #     （MLA-DP + 线性注意力 TP 的混合）。在一个流水线 stage 内，TP 组是连续的（步长 1）。

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        # dp is the implicit complement: the number of KDA-TP replicas.
        # 译：dp 是隐含的补数：KDA-TP 副本的个数。
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, None
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)


# LM head 的映射
class LmHeadMapping(MappingBase):
    """Parallel mapping for the vocabulary-sharded LM head.

    Without attention DP the head follows the attention TP group, as it
    always has. Under attention DP the head is replicated by default
    (``tp_size`` 1); a wider ``tp_size`` vocab-shards it over a contiguous
    group of attention-DP ranks, which then gather their tokens before the
    logits GEMM and transpose the vocab shards back to their own rows.
    """

    # 译：词表切分的 LM head 的并行映射。
    #
    #     没有 attention DP 时，head 跟随注意力 TP 组，一直如此。在 attention DP 下，head 默认是复制的
    #     （``tp_size`` 为 1）；更宽的 ``tp_size`` 会把它按词表切分到一个连续的 attention-DP rank 组上，
    #     这些 rank 在 logits GEMM 之前收集各自的 token，之后再把词表分片转置回各自的行。

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, None
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)


def _resolve_lm_head_tp_size(
    attn: AttentionLayerMapping, lm_head_tp_size: int | None
) -> int:
    """Resolve the LM-head TP width against the attention layout.

    The default is today's layout: the attention TP width without attention
    DP, replicated (1) with it. An explicit width under attention DP must
    tile the stage world and align with attention TP groups; without DP it
    must equal the attention TP.
    """
    # 译：根据注意力布局解析 LM head 的 TP 宽度。
    #
    #     默认是目前的布局：没有 attention DP 时为注意力 TP 宽度，有 attention DP 时为复制（1）。在 attention DP 下
    #     显式给出的宽度必须铺满 stage 的 world，并与注意力 TP 组对齐；没有 DP 时它必须等于注意力 TP。
    if lm_head_tp_size is None:
        return 1 if attn.has_dp else attn.tp_size
    if (
        isinstance(lm_head_tp_size, bool)
        or not isinstance(lm_head_tp_size, int)
        or lm_head_tp_size < 1
    ):
        raise ValueError("LM head TP size must be a positive integer")
    if not attn.has_dp:
        if lm_head_tp_size != attn.tp_size:
            raise ValueError(
                "without attention DP the LM head follows the attention TP "
                f"group; got lm_head_tp_size={lm_head_tp_size} with "
                f"attn_tp_size={attn.tp_size}"
            )
        return lm_head_tp_size
    if attn.world_size % lm_head_tp_size:
        raise ValueError(
            f"LM head TP size {lm_head_tp_size} must divide the stage world "
            f"size {attn.world_size}"
        )
    if max(lm_head_tp_size, attn.tp_size) % min(lm_head_tp_size, attn.tp_size):
        raise ValueError("LM head and attention TP groups must tile each other")
    return lm_head_tp_size


# 总映射：模型拿到的就是它。持有各类层的子映射，以及流水线并行和 GPU 编号相关的信息。
# 所有参数都是关键字参数；没给的维度按 _resolve_parallelism_sizes 的规则推导。
# 单卡：Mapping(rank=0, world_size=1) 即可，所有子映射的 tp/dp/ep 都是 1。
class Mapping(MappingBase):

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        *,
        attn_tp_size: int | None = None,
        attn_dp_size: int | None = None,
        attn_kvp_size: int = 1,
        attn_head_tp_size: int | None = None,
        lm_head_tp_size: int | None = None,
        attn_qcp_size: int = 1,
        dense_tp_size: int | None = None,
        dense_dp_size: int | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        moe_dp_size: int | None = None,
        vision_tp_size: int | None = None,
        vision_dp_size: int | None = None,
        linear_attn_tp_size: int | None = None,
        pp_size: int = 1,
        pp_layer_partition: tuple[int, ...] | None = None,
        nprocs_per_node: int | None = None,
        nnodes: int | None = None,
        base_gpu_id: int = 0,
        gpu_id_step: int = 1,
    ):
        super().__init__(rank, world_size)
        # Pipeline parallelism is the outermost dimension: the world splits
        # into pp_size contiguous stages and every per-layer-type mapping
        # resolves inside one stage. Sub-mappings keep the GLOBAL rank — the
        # stride arithmetic stays correct because a stage base is a multiple
        # of every intra-stage stride.
        # 译：流水线并行是最外层的维度：world 被切成 pp_size 个连续的 stage，每种层类型的映射都在一个 stage 内解析。
        #     子映射保留全局 rank——步长的计算仍然正确，因为一个 stage 的起始编号是 stage 内每个步长的倍数。
        assert pp_size > 0
        assert (
            world_size % pp_size == 0
        ), f"world_size {world_size} must be divisible by pp_size {pp_size}"
        self.pp_size = pp_size
        # Optional explicit per-stage layer counts (front to back); validated
        # against the model's layer count where the split is resolved
        # (pp_stage_windows).
        # 译：可选的、显式给出的每个 stage 的层数（从前到后）；在解析切分的地方（pp_stage_windows）
        #     根据模型的层数进行校验。
        self.pp_layer_partition = (
            tuple(int(count) for count in pp_layer_partition)
            if pp_layer_partition
            else None
        )
        stage_world_size = world_size // pp_size
        self.attn = AttentionLayerMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=attn_tp_size,
            dp_size=attn_dp_size,
            kvp_size=attn_kvp_size,
            head_tp_size=attn_head_tp_size,
            qcp_size=attn_qcp_size,
        )
        self.lm_head = LmHeadMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=_resolve_lm_head_tp_size(self.attn, lm_head_tp_size),
        )
        self.dense = DenseLayerMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=dense_tp_size,
            dp_size=dense_dp_size,
        )
        self.moe = MoeLayerMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=moe_tp_size,
            ep_size=moe_ep_size,
            dp_size=moe_dp_size,
        )
        # The vision mapping is local to each attention TP group. With both
        # sizes omitted it resolves to weight TP, preserving the legacy mode.
        # 译：视觉映射在每个注意力 TP 组内部。两个大小都省略时它解析为权重 TP，保留旧的模式。
        self.vision = VisionTowerMapping(
            rank=rank,
            world_size=self.attn.tp_size,
            tp_size=vision_tp_size,
            dp_size=vision_dp_size,
        )
        # Linear-attention layers follow the attention TP width unless
        # overridden — the default is behavior-identical to reading
        # mapping.attn.tp_size.
        # 译：线性注意力层沿用注意力 TP 的宽度，除非被覆盖——默认行为与读取 mapping.attn.tp_size 完全相同。
        self.linear_attn = LinearAttnLayerMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=(
                linear_attn_tp_size
                if linear_attn_tp_size is not None
                else self.attn.tp_size
            ),
        )
        # 节点数 × 每节点进程数 = world_size；gpu_id = base_gpu_id + 本节点内编号 * gpu_id_step
        self.nprocs_per_node, self.nnodes = _resolve_parallelism_sizes(
            self.world_size, nprocs_per_node, nnodes
        )
        assert base_gpu_id >= 0
        assert gpu_id_step > 0
        self.base_gpu_id = base_gpu_id
        self.gpu_id_step = gpu_id_step

    def _on_rank_initialized(self, rank: int):
        self.attn.rank = rank
        self.lm_head.rank = rank
        self.dense.rank = rank
        self.moe.rank = rank
        self.vision.rank = rank
        self.linear_attn.rank = rank

    # 给块草稿模型（DFlash 等，第 11 阶段）用的映射：只把稠密 MLP 改为不切分（每个 rank 持有完整权重），
    # 其余维度保持不变。原因见下面的 docstring。
    def replicated_for_draft(self) -> "Mapping":
        """This layout with a replicated dense group, every other axis unchanged.

        A heterogeneous block draft (a DFlash stack, not an MTP depth) joins none
        of the target's dense parallelism: its MLP holds whole weights and
        reduces nothing. That matters beyond the duplicated bytes. Only the
        attention-DP rank holding a request runs the draft; the other ranks'
        idle forward skips it on purpose (``ModelExecutor.execute_idle_forward``),
        so a collective inside the draft that one rank of a group enters and the
        rest skip never completes.

        The ATTENTION width is left alone: a block drafter writes K/V at the
        target's cache locations and its cache cell is sized from the attention
        TP width (``MhaSpec.cache_cell_size``), so its projections must keep
        producing exactly the heads its cache rows hold.
        """
        # 译：本布局，但稠密组改为复制的，其他所有轴不变。
        #
        #     异构的块草稿模型（DFlash 堆叠，而不是 MTP 的某一深度）不参与目标模型的任何稠密并行：它的 MLP 持有完整
        #     权重，不做任何归约。这不只关系到重复的字节。只有持有某个请求的那个 attention-DP rank 才运行草稿模型；
        #     其他 rank 的空跑前向有意跳过它（``ModelExecutor.execute_idle_forward``），因此草稿模型内部的集合通信
        #     如果组内一个 rank 进入、其余 rank 跳过，就永远无法完成。
        #
        #     注意力的宽度保持不变：块草稿器把 K/V 写到目标模型的 cache 位置，它的 cache 单元大小由注意力 TP 宽度
        #     决定（``MhaSpec.cache_cell_size``），所以它的投影必须恰好产生它的 cache 行所持有的那些头。
        stage_world_size = self.world_size // self.pp_size
        return Mapping(
            rank=self._rank,
            world_size=self.world_size,
            attn_tp_size=self.attn.tp_size,
            attn_dp_size=self.attn.dp_size,
            attn_kvp_size=self.attn.kvp_size,
            attn_head_tp_size=self.attn.head_tp_size,
            lm_head_tp_size=self.lm_head.tp_size,
            attn_qcp_size=self.attn.qcp_size,
            dense_tp_size=1,
            dense_dp_size=stage_world_size,
            moe_tp_size=self.moe.tp_size,
            moe_ep_size=self.moe.ep_size,
            moe_dp_size=self.moe.dp_size,
            pp_size=self.pp_size,
            pp_layer_partition=self.pp_layer_partition,
            nprocs_per_node=self.nprocs_per_node,
            nnodes=self.nnodes,
            base_gpu_id=self.base_gpu_id,
            gpu_id_step=self.gpu_id_step,
        )

    # 流水线并行相关：pp_rank = 第几个 stage；pp_prev_rank / pp_next_rank 是上下游 stage 中
    # stage 内位置相同的 rank（流水线前向在它们之间传递中间结果，第 9 阶段）
    @cached_property
    def has_pp(self) -> bool:
        return self.pp_size > 1

    @cached_property
    def stage_world_size(self) -> int:
        return self.world_size // self.pp_size

    @cached_property
    def pp_rank(self) -> int:
        return _make_parallelism_rank(
            self.rank, self.pp_size, stride=self.stage_world_size
        )

    @cached_property
    def pp_group(self) -> Group:
        return _make_parallelism_group(
            self.rank, self.pp_size, stride=self.stage_world_size
        )

    @cached_property
    def is_first_pp_rank(self) -> bool:
        return self.pp_rank == 0

    @cached_property
    def is_last_pp_rank(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @cached_property
    def pp_prev_rank(self) -> int:
        """Global rank of the same intra-stage position one stage upstream."""
        # 译：上游一个 stage 中、stage 内位置相同的 rank 的全局编号。
        assert not self.is_first_pp_rank
        return self.rank - self.stage_world_size

    @cached_property
    def pp_next_rank(self) -> int:
        """Global rank of the same intra-stage position one stage downstream."""
        # 译：下游一个 stage 中、stage 内位置相同的 rank 的全局编号。
        assert not self.is_last_pp_rank
        return self.rank + self.stage_world_size

    @cached_property
    def has_attn_tp(self) -> bool:
        return self.attn.has_tp

    @cached_property
    def has_attn_dp(self) -> bool:
        return self.attn.has_dp

    # 节点编号、节点内编号，以及本 rank 使用的 GPU 编号
    @cached_property
    def node_rank(self) -> int:
        return self.rank // self.nprocs_per_node

    @cached_property
    def local_rank(self) -> int:
        return self.rank % self.nprocs_per_node

    @cached_property
    def gpu_id(self) -> int:
        return self.base_gpu_id + self.local_rank * self.gpu_id_step

    def __repr__(self) -> str:
        rank_str = str(self._rank) if self._rank is not None else "?"
        lines = [
            f"Mapping(rank={rank_str}, world_size={self.world_size})",
            f"  Cluster : {self.nnodes} node(s) x {self.nprocs_per_node} proc(s)",
            f"  Pipeline: pp={self.pp_size}",
            f"  Attention: tp={self.attn.tp_size}  kvp={self.attn.kvp_size}  "
            f"qcp={self.attn.qcp_size}  dp={self.attn.dp_size}  "
            f"head_tp={self.attn.head_tp_size}",
            f"    Vision: tp={self.vision.tp_size}  item_dp={self.vision.dp_size}",
            f"  LM head : tp={self.lm_head.tp_size}",
            f"  Dense   : tp={self.dense.tp_size}  dp={self.dense.dp_size}",
            f"  MoE     : tp={self.moe.tp_size}  ep={self.moe.ep_size}  dp={self.moe.dp_size}",
        ]
        return "\n".join(lines)
