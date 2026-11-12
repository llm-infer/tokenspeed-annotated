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

# 层内通信管理：解码层与层之间的所有集合通信，以及它们和 RMSNorm 的融合，都集中在这里。
#
# 你在这里：BaseDecoderLayer 的 ①~⑧ 各步（models/base/decoder_layer.py 文件头的流程图）、
#           BaseTransformerModel 的 final norm、LogitsProcessor 收集要采样的行。
#
# 为什么集中在一处：o_proj、down_proj 都是行并行、都设了 reduce_results=False，自己不求和。
# 求和放在层与层的交界处统一处理，好处有两个：
#   1. 可以和紧接着的“相加残差 + RMSNorm”合成一个 kernel（通信融合）
#   2. 注意力和 MLP 可以用不同的并行方式，交界处按需要改变 token 行的分布，模型代码不用跟着改
#
# 单卡：所有通信都是空操作，CommManager 退化为“相加残差 + RMSNorm”。
#
# 两种行布局（多卡时才有区别）
#   all-reduce 布局：注意力 TP 组内每个 rank 都持有本组的全部 token 行。
#     层出口：部分和 all-reduce，每个 rank 得到完整结果。
#     当注意力的 TP 宽度 = MLP 所在组（稠密 TP 组，或 MoE 的 TP×EP 组）的宽度时使用。
#   RSAG 布局（reduce-scatter + all-gather）：每个 rank 只持有一部分 token 行。
#     层出口：部分和 reduce-scatter，每个 rank 只得到自己那份行的完整结果；
#     下一处需要完整行时（注意力入口、MLP 入口）再 all-gather。
#     当两者宽度不同时使用，例如 attention DP（注意力组更窄）+ 更宽的 MLP TP 或 MoE EP。
#   token 行按 scatter_count 均分，前面的 rank 多分一行：
#     例：本组 301 行、TP=2 -> rank 0 持有第 0~150 行（151 行），rank 1 持有第 151~300 行（150 行）
#   RSAG 的好处：残差相加、RMSNorm 只在自己那份行上算，计算量和显存都按 TP 宽度缩小。
#
# TP=2、all-reduce 布局下一层的通信（Qwen3，T=301）：
#   ① input_reduce_norm：上一层 MLP 的部分和 all-reduce + 相加残差 + RMSNorm（可融合为一个 kernel）
#   ④ 注意力：每个 rank 算一半的头，o_proj 输出 [301, 4096] 的部分和
#   ⑤ post_attn_reduce_norm：all-reduce + 相加残差 + RMSNorm（可融合）
#   ⑦ MLP：每个 rank 算一半的 intermediate，down_proj 输出部分和
#   ⑧ post_mlp_fused：可融合时什么也不做（推迟到下一层的 ①），否则在这里 all-reduce
#
# 本文件还处理查询上下文并行（QCP，第 12 阶段）、attention DP 的空跑前向、head TP 等情况，
# 这些分支第 9、12 阶段回访。
from collections.abc import Sequence

import torch

from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    all_to_all_transpose,
    token_all_gather,
    token_all_gather_rows,
    token_reduce_scatter,
)
from tokenspeed.runtime.distributed.mapping import Group, Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.query_shard import QueryShardPlan, scatter_count


# QCP 下（第 12 阶段）：各 rank 持有不同的 token 行，每个请求的最后一行（要采样的行）落在某一个 rank 上。
# 这里把各 rank 上要采样的行 all-gather 到每个 rank，并按请求顺序排好，供 LogitsProcessor 使用。
def gather_sampled_rows(
    hidden_states: torch.Tensor,
    plan: QueryShardPlan,
    *,
    group: tuple[int, ...],
) -> torch.Tensor:
    """Gather the sampled rows of a sharded extend forward to every rank.

    The plan assigns the last row of every request to exactly one rank;
    each rank selects its rows with ``plan.local_sampled_index``. One
    all-gather with the plan's per-rank sampled-row counts concatenates them
    rank by rank, and ``sampled_to_request`` puts them in request order.
    The logits processor runs this in place of its ``hidden_states[gather_ids]``
    selection on a sharded forward, for the
    final hidden rows and for a LAST capture's aux hidden rows alike; the
    gather is byte-preserving (:func:`token_all_gather_rows`), so any row
    dtype and width travel, the per-rank sampled-row counts being whatever
    the batch makes them.

    Args:
        hidden_states: ``[local_rows, hidden]`` this rank's final rows.
        plan: The forward's query shard.
        group: The query-context-parallel group.

    Returns:
        ``[bs, hidden]`` sampled rows in request order, on every rank.
    """
    # 译：把一次分片 extend 前向中要采样的行收集到每个 rank。
    #
    #     分片计划把每个请求的最后一行恰好分给一个 rank；每个 rank 用 ``plan.local_sampled_index`` 选出自己的行。
    #     用计划中每个 rank 的采样行数做一次 all-gather，把它们按 rank 顺序拼接起来，再用 ``sampled_to_request``
    #     排成请求顺序。logits processor 在分片前向上用它代替 ``hidden_states[gather_ids]`` 的选择，对最终的
    #     hidden 行以及 LAST 捕获的辅助 hidden 行都是如此；这次 gather 是按字节保持的（:func:`token_all_gather_rows`），
    #     所以任何行的 dtype 和宽度都能传输，每个 rank 的采样行数由 batch 决定。
    #
    #     参数：
    #         hidden_states：``[local_rows, hidden]``，本 rank 的最终行。
    #         plan：本次前向的查询分片。
    #         group：查询上下文并行组。
    #
    #     返回：
    #         ``[bs, hidden]``，按请求顺序排列的采样行，每个 rank 上都有。
    if hidden_states.shape[0] != plan.local_rows:
        raise ValueError(
            f"query shard rank {plan.rank} holds {plan.local_rows} rows, got "
            f"{hidden_states.shape[0]}"
        )
    local = hidden_states.index_select(0, plan.local_sampled_index)
    gathered = token_all_gather_rows(local, group, list(plan.sampled_rows_per_rank))
    return gathered.index_select(0, plan.sampled_to_request)


# 算出 MoE 层输入由哪几段行组成、每段有多少真实 token（其余是 padding 行）。
# 用于 CUDA Graph 回放等“行数被补齐”的前向：MoE 的路由、专家计算可以跳过 padding 行（第 10 阶段）。
# 三种情况与 CommManager 的行布局一一对应：
#   MoE 组宽于注意力组：MoE 输入是 MoE 组内各 rank 的 reduce-scatter 分片 all-gather 起来的，每个 rank 一段
#   注意力 TP 下 RSAG 布局：MoE 输入就是本 rank 的那一段
#   all-reduce 布局：MoE 输入是本 attention-DP 组的全部行，一段
def moe_input_row_segments(
    mapping: Mapping,
    *,
    padded_global_num_tokens: Sequence[int],
    live_global_num_tokens: Sequence[int],
) -> list[tuple[int, int]]:
    """``(rows, live_rows)`` of each segment of the MoE layers' input, in row order.

    Mirrors the row layout ``CommManager.post_attn_comm`` and ``pre_moe_comm``
    produce for a padded forward (a decode graph replayed at a ladder batch
    size, a prefill graph replayed at a bucket): every rank's rows are padded
    to the same count and only a prefix of each rank's rows carries real
    tokens. The MoE input is either this rank's own rows -- the whole
    attention-DP group's rows after an all-reduce, or this rank's contiguous
    reduce-scatter shard of them -- or the all-gather of such shards over the
    MoE TP-EP group, one segment per group rank in group order.

    Args:
        mapping: The parallel layout.
        padded_global_num_tokens: Rows every rank feeds the model, indexed by
            global rank (the ranks of one attention-DP group share a value).
        live_global_num_tokens: Real token rows per global rank, same index;
            a rank's live rows are the first ones of its padded rows.

    Returns:
        One ``(rows, live_rows)`` per segment, ``live_rows <= rows``, summing
        to the MoE input's row count.
    """
    # 译：MoE 层输入中每一段的 ``(rows, live_rows)``，按行的顺序排列。
    #
    #     对应 ``CommManager.post_attn_comm`` 和 ``pre_moe_comm`` 为补齐的前向（按某一档 batch 大小回放的 decode graph、
    #     按某个桶回放的 prefill graph）产生的行布局：每个 rank 的行都补齐到相同的数量，而每个 rank 只有行的一个
    #     前缀承载真实 token。MoE 的输入要么是本 rank 自己的行——all-reduce 之后整个 attention-DP 组的行，或者
    #     本 rank 对这些行做 reduce-scatter 得到的连续分片——要么是在 MoE TP-EP 组上把这样的分片 all-gather 起来，
    #     组内每个 rank 一段，按组内顺序排列。
    #
    #     参数：
    #         mapping：并行布局。
    #         padded_global_num_tokens：每个 rank 送入模型的行数，按全局 rank 编号（同一 attention-DP 组的 rank 值相同）。
    #         live_global_num_tokens：每个全局 rank 的真实 token 行数，下标相同；一个 rank 的真实行是其补齐行中的前若干行。
    #
    #     返回：
    #         每段一个 ``(rows, live_rows)``，``live_rows <= rows``，总和等于 MoE 输入的行数。
    attn = mapping.attn
    world_size = mapping.world_size
    if len(padded_global_num_tokens) != world_size:
        raise ValueError(
            f"padded_global_num_tokens has {len(padded_global_num_tokens)} "
            f"entries, world_size={world_size}"
        )
    if len(live_global_num_tokens) != world_size:
        raise ValueError(
            f"live_global_num_tokens has {len(live_global_num_tokens)} "
            f"entries, world_size={world_size}"
        )

    def dp_group_rows(rank: int) -> tuple[int, int, int]:
        """``(padded, live, tp_rank)`` of ``rank``'s attention-DP group."""
        # 译：``rank`` 所在 attention-DP 组的 ``(补齐行数, 真实行数, tp_rank)``。
        dp_rank, tp_rank = divmod(attn.scatter_index(rank), attn.tp_size)
        # The count table is indexed by global rank with the DP stride.
        # 译：计数表以全局 rank 为下标，DP 维的步长为 tp_size。
        first = dp_rank * attn.tp_size
        padded = int(padded_global_num_tokens[first])
        live = int(live_global_num_tokens[first])
        if not 0 <= live <= padded:
            raise ValueError(
                f"rank {rank}: {live} live rows do not fit {padded} padded rows"
            )
        return padded, live, tp_rank

    def shard(rank: int) -> tuple[int, int]:
        """``rank``'s reduce-scatter shard: a contiguous run of its group's rows."""
        # 译：``rank`` 的 reduce-scatter 分片：它所在组的行中连续的一段。
        padded, live, tp_rank = dp_group_rows(rank)
        lengths = scatter_count(padded, attn.tp_size)
        offset = sum(lengths[:tp_rank])
        # 本分片的真实行数 = 组内真实行数减去前面 rank 的分片偏移，再截到 [0, 分片大小]
        return lengths[tp_rank], min(max(live - offset, 0), lengths[tp_rank])

    moe_all_reduce = attn.tp_size == mapping.moe.tp_ep_size
    if mapping.moe.has_tp_ep and not moe_all_reduce:
        return [shard(rank) for rank in mapping.moe.tp_ep_group]
    if mapping.has_attn_tp and not moe_all_reduce:
        return [shard(mapping.rank)]
    padded, live, _ = dp_group_rows(mapping.rank)
    return [(padded, live)]


# 下面几个函数服务于 head TP（注意力头切分到持有不同行的 rank 上，第 12 阶段）的数据交换：
# 交换前要知道组内每个 rank 各持有多少行。
#   dp_group_row_counts：从“按全局 rank 编号”的 DP 行数表中取出组内各 rank 的行数，并核对本 rank 的值
#   forward_input_row_table：前向输入的行数表（每个 rank 的 token 数）
#   forward_collective_row_table：集合通信按其确定大小的行数表（草稿模型可能只算部分行，见 ForwardContext）
#   head_tp_row_counts：统一入口，attention DP 和 QCP 两种布局下都返回组内各 rank 的行数
def dp_group_row_counts(
    table: list[int] | None,
    group: Group,
    rank: int,
    num_rows: int,
) -> list[int]:
    """Per-rank row counts of ``group`` read from a world-indexed DP table.

    ``num_rows`` is the rows this rank holds; the table's entry for it must
    agree, so a layout mistake fails here instead of hanging in the
    collective. The caller picks the table by the forward phase (input rows
    or collective rows, see ``forward_input_row_table`` and
    ``forward_collective_row_table``), never by matching ``num_rows``: a rank
    with no rows of its own could not tell the tables apart, and every rank
    of the group must pick the same one.
    """
    # 译：从以 world 为下标的 DP 表中读出 ``group`` 中每个 rank 的行数。
    #
    #     ``num_rows`` 是本 rank 持有的行数；表中对应的项必须与之一致，这样布局错误会在这里报错，而不是在集合通信中卡住。
    #     调用方按前向的阶段选择表（输入行或集合通信行，见 ``forward_input_row_table`` 和
    #     ``forward_collective_row_table``），绝不通过匹配 ``num_rows`` 来选：自己没有行的 rank 无法区分这两张表，
    #     而组内每个 rank 必须选同一张。
    if table is None:
        raise ValueError(
            "DP row counts are unavailable: the forward carries no global token "
            "table (attention DP is required)"
        )
    if table[rank] != num_rows:
        raise ValueError(
            f"rank {rank} holds {num_rows} rows but the forward's DP row table "
            f"gives it {table[rank]}"
        )
    return [table[peer] for peer in group]


def forward_input_row_table(ctx: ForwardContext) -> list[int] | None:
    """Per-rank input rows of the forward (the token counts)."""
    # 译：本次前向每个 rank 的输入行数（token 数）。
    return ctx.global_num_tokens


def forward_collective_row_table(ctx: ForwardContext) -> list[int] | None:
    """Per-rank rows the forward's collectives size by: the sizing a model
    reported (a drafter narrowing to its live rows), else the input rows."""
    # 译：本次前向的集合通信按其确定大小的每个 rank 的行数：模型报告的大小（草稿器缩小到它的有效行），否则为输入行数。
    if ctx.collective_global_num_tokens is not None:
        return ctx.collective_global_num_tokens
    return ctx.global_num_tokens


def head_tp_row_counts(
    ctx: ForwardContext, mapping: Mapping, num_rows: int, *, collective: bool
) -> list[int]:
    """Per-rank rows of the head group for one leg of the head-TP exchange.

    Head TP shards the head projections over ranks that hold different rows,
    and its exchanges split by how many rows each of them holds. Under
    attention DP those counts are the forward's world-indexed DP tables;
    under query context parallelism the head group is the query-shard group
    and the counts are the shard plan's. One resolver, so no model code
    branches on the layout.

    Args:
        ctx: The forward.
        mapping: The parallel layout (``mapping.attn.head_tp_group`` is the
            group).
        num_rows: Rows this rank holds in this leg; the table's entry for it
            must agree, so a layout mistake fails here instead of hanging in
            the collective.
        collective: Which rows the leg moves. ``False`` is the forward's input
            rows (the legs up to core attention); ``True`` the rows the
            forward's collectives size by -- the live rows a narrowing drafter
            reported, else the input rows (the legs after core attention).
            Every rank of the group picks the same leg, never by matching
            ``num_rows``: a rank with no rows could not tell the tables apart.

    Returns:
        One count per rank of the head group, in group order.

    Raises:
        ValueError: The forward carries no table for the layout -- no DP
            table under attention DP, or a replicated-row forward under query
            sharding (every rank holds every row; the attention exchanges
            nothing there) -- or ``num_rows`` disagrees with it.
    """
    # 译：head-TP 交换的某一段中，head 组每个 rank 的行数。
    #
    #     head TP 把 head 投影切分到持有不同行的 rank 上，它的交换按每个 rank 持有多少行来切分。在 attention DP 下，
    #     这些计数是前向的以 world 为下标的 DP 表；在查询上下文并行下，head 组就是查询分片组，计数来自分片计划。
    #     只有一个解析函数，所以模型代码不会按布局分支。
    #
    #     参数：
    #         ctx：本次前向。
    #         mapping：并行布局（``mapping.attn.head_tp_group`` 就是这个组）。
    #         num_rows：本 rank 在这一段中持有的行数；表中对应的项必须与之一致，这样布局错误会在这里报错，
    #             而不是在集合通信中卡住。
    #         collective：这一段移动哪些行。``False`` 是前向的输入行（核心注意力之前的各段）；``True`` 是前向的集合
    #             通信按其确定大小的行——缩小的草稿器报告的有效行，否则是输入行（核心注意力之后的各段）。
    #             组内每个 rank 选择同一段，绝不通过匹配 ``num_rows`` 来选：没有行的 rank 无法区分这两张表。
    #
    #     返回：
    #         head 组每个 rank 一个计数，按组内顺序排列。
    #
    #     异常：
    #         ValueError：前向没有携带该布局所需的表——attention DP 下没有 DP 表，或者在查询分片下是复制行的前向
    #             （每个 rank 持有每一行；注意力在那里什么也不交换）——或者 ``num_rows`` 与之不一致。
    attn = mapping.attn
    if not attn.has_qcp:
        table = (
            forward_collective_row_table(ctx)
            if collective
            else forward_input_row_table(ctx)
        )
        return dp_group_row_counts(table, attn.head_tp_group, mapping.rank, num_rows)
    plan = ctx.query_shard
    if plan is None or plan.size == 1:
        raise ValueError(
            "head-TP row counts are unavailable: the forward carries no query "
            "shard, so every rank holds every row and the attention exchanges "
            "nothing"
        )
    if plan.size != len(attn.head_tp_group):
        raise ValueError(
            f"query shard over {plan.size} ranks does not match the head TP group "
            f"of {len(attn.head_tp_group)}"
        )
    counts = (
        plan.rows_for_collective(ctx.collective_num_tokens)
        if collective
        else plan.row_counts
    )
    if counts[plan.rank] != num_rows:
        raise ValueError(
            f"query shard rank {plan.rank} holds {num_rows} rows but the shard "
            f"table gives it {counts[plan.rank]}"
        )
    return list(counts)


# 每个解码层一个 CommManager。成员：
#   mapping：并行映射；layer_id：层号
#   is_moe / prev_is_moe：本层、上一层的 MLP 是否为 MoE（决定出口和入口用哪种布局）
#   input_layernorm / post_attn_layernorm：本层的两个 norm（通信与 norm 融合时要用）
#   query_sharded：模型是否按 QCP 切分 extend 行（Qwen3 为 False）
#   layer_boundary_norm / moe_combine_order：两个数值相关的启动选项（见 __init__ 中的注释）
#   dense_batch_invariant：是否用“批大小无关”的稠密 MLP 收尾方式
class CommManager:
    """Manages communication patterns (all_reduce vs RSAG) for each decoder layer.

    ``query_sharded`` declares the model slices its extend rows by
    ``ctx.query_shard`` (query context parallelism). Under that mapping an
    attention output row is complete wherever it is computed -- the
    attention weights are head-replicated on every rank, or, under head TP
    over the query shards (``--attn-head-tp-size``), the attention's own
    tail (``DeepseekV3AttentionMLA.project_output``) returns complete rows --
    so the attention legs are identity on every forward, a sharded extend
    and the drafter's replicated decode steps alike; nothing is ever
    scattered by attention, so nothing is gathered before it or at the final
    norm either. The dense and MoE legs follow the forward: with a shard they
    run the existing all-gather / reduce-scatter path over the shard's
    per-rank row table, without one (decode steps, idle) the replicated
    all-reduce legs, which is why dense TP and the MoE TP x EP group must
    each be 1 or the attention TP width (``validate_qcp``). The logits
    processor gathers only the sampled rows of the shard
    (:func:`gather_sampled_rows` over its TP group). The flag must agree with
    ``mapping.attn.has_qcp``: a model that does not slice its rows cannot run
    under a query-sharding mapping, and one that does cannot run without it.

    ``dense_batch_invariant`` selects the TP-batch-invariant dense tail: the
    layer's ``down_proj`` is column-parallel on hidden and ``post_dense_comm``
    transposes its ``[T_full, H / W]`` output back to this rank's rows instead
    of reduce-scattering partial sums.
    """

    # 译：为每个解码层管理通信模式（all_reduce 还是 RSAG）。
    #
    #     ``query_sharded`` 声明模型按 ``ctx.query_shard`` 切分它的 extend 行（查询上下文并行）。在这种映射下，注意力的
    #     输出行无论在哪里计算都是完整的——注意力权重在每个 rank 上按头复制，或者在跨查询分片的 head TP 下
    #     （``--attn-head-tp-size``），注意力自己的收尾（``DeepseekV3AttentionMLA.project_output``）返回完整的行——
    #     所以注意力的各段在每次前向中都是恒等的，分片的 extend 和草稿器复制的 decode 步都是如此；注意力从不分散
    #     任何东西，所以在它之前、在 final norm 处也都不需要收集。稠密和 MoE 的各段跟随前向：有分片时，它们按分片的
    #     每 rank 行表走现有的 all-gather / reduce-scatter 路径；没有分片时（decode 步、空跑）走复制的 all-reduce 路径，
    #     这就是为什么稠密 TP 和 MoE 的 TP x EP 组都必须是 1 或注意力 TP 的宽度（``validate_qcp``）。logits processor
    #     只在它的 TP 组上收集分片中要采样的行（:func:`gather_sampled_rows`）。这个标志必须与
    #     ``mapping.attn.has_qcp`` 一致：不切分其行的模型不能在查询分片映射下运行，切分的模型也不能在没有它时运行。
    #
    #     ``dense_batch_invariant`` 选择 TP 批大小无关的稠密收尾：该层的 ``down_proj`` 在 hidden 维上做列并行，
    #     ``post_dense_comm`` 把它的 ``[T_full, H / W]`` 输出转置回本 rank 的行，而不是对部分和做 reduce-scatter。

    def __init__(
        self,
        mapping: Mapping,
        layer_id: int,
        is_moe: bool,
        prev_is_moe: bool,
        dense_batch_invariant: bool,
        input_layernorm: torch.nn.Module | None = None,
        post_attn_layernorm: torch.nn.Module | None = None,
        *,
        query_sharded: bool,
    ) -> None:
        # query_sharded 必须与 mapping 一致：模型声明的切分方式和启动参数不符时直接报错
        if query_sharded != mapping.attn.has_qcp:
            raise ValueError(
                f"CommManager(query_sharded={query_sharded}) disagrees with the "
                f"attention mapping (qcp_size={mapping.attn.qcp_size}): a model "
                "declares query sharding only when it slices its extend rows by "
                "ctx.query_shard, and must when the mapping shards queries"
            )
        if query_sharded and (
            mapping.dense.tp_size not in (1, mapping.attn.tp_size)
            or mapping.moe.tp_ep_size not in (1, mapping.attn.tp_size)
        ):
            # A replicated-row forward (the drafter's decode steps) keeps its
            # rows whole through attention, so a dense / MoE group narrower
            # than attention TP would gather rows nobody scattered.
            # 译：复制行的前向（草稿器的 decode 步）在注意力中保持行完整，所以比注意力 TP 更窄的稠密 / MoE 组会去收集
            #     没有人分散过的行。
            raise ValueError(
                "query sharding needs dense TP and the MoE TP x EP group to be 1 "
                f"or the attention TP width {mapping.attn.tp_size}; got dense "
                f"{mapping.dense.tp_size}, MoE {mapping.moe.tp_ep_size}"
            )
        self.mapping = mapping
        self.layer_id = layer_id
        self.is_moe = is_moe
        self.prev_is_moe = prev_is_moe
        self.input_layernorm = input_layernorm
        self.post_attn_layernorm = post_attn_layernorm
        self.query_sharded = query_sharded
        # utils.env imports server_args, which imports this package: resolve
        # the launch options lazily, as the fusion predicates below do.
        # 译：utils.env 导入 server_args，而 server_args 又导入本包：像下面的融合判断那样，延迟解析启动选项。
        from tokenspeed.runtime.utils.env import global_server_args_dict

        # --layer-boundary-norm: how input_reduce_norm and final_norm add the
        # residual (docs/design/numerics.md, alignment.trainer).
        # 译：--layer-boundary-norm：input_reduce_norm 和 final_norm 如何加上残差（docs/design/numerics.md，alignment.trainer）。
        # "unfused" 时按训练端的方式：先把 bf16 的 hidden + residual 算出来，再对它做 RMSNorm，
        # 使推理的数值与训练框架逐位一致；默认用融合的 add + RMSNorm（相加在 fp32 中进行）。
        self.layer_boundary_norm: str = global_server_args_dict["layer_boundary_norm"]
        # --moe-combine-order: whether the MoE leaf already combined the routed
        # output across the MoE TP-EP group (post_moe_comm).
        # 译：--moe-combine-order：MoE 叶子是否已经在 MoE TP-EP 组内合并了路由输出（post_moe_comm）。
        self.moe_combine_order: str = global_server_args_dict["moe_combine_order"]
        if dense_batch_invariant and (
            not mapping.dense.has_tp or self.use_all_reduce(is_moe=False)
        ):
            raise ValueError(
                "the batch-invariant dense tail replaces a token reduce-scatter "
                "and needs a dense TP group wider than attention TP"
            )
        self.dense_batch_invariant = dense_batch_invariant

    # 下面几个方法回答：本层各个通信组里，每个 rank 持有多少 token 行。
    # 集合通信（all-gather、reduce-scatter）需要知道每个 rank 的行数才能正确拼接或拆分。
    # ---- Scattered token counts ----
    # 译：---- 分散的 token 计数 ----

    def _shard(self, ctx: ForwardContext) -> QueryShardPlan | None:
        """The forward's query shard when the rows this layer holds are a
        shard; None for replicated rows."""
        # 译：当本层持有的行是一个分片时，返回本次前向的查询分片；行是复制的时返回 None。
        plan = ctx.query_shard
        if plan is None or plan.size == 1:
            return None
        if not self.query_sharded:
            raise RuntimeError(
                "a query-sharded forward reached a CommManager whose model did "
                "not declare query_sharded"
            )
        return plan

    # 返回 (所有 rank 的行数之和, 单个 rank 的最大行数)，供 MoE 确定分发缓冲区大小。
    # 单卡：scattered_num_tokens = [T]，返回 (T, T)；Qwen3 的例子是 (301, 301)。
    def get_num_tokens(self, ctx: ForwardContext):
        scattered = self.scattered_num_tokens(ctx)
        return sum(scattered), max(scattered)

    # 按 RSAG 布局，世界中每个 rank 分到的行数，排列顺序为“DP 组为主、组内按 TP 编号”。
    #   有 attention DP 时：每个 DP 组的 token 数在组内按 scatter_count 均分
    #   没有 DP 时：本组 token 数在 TP 组内均分
    #   例：无 DP、TP=2、T=301 -> [151, 150]；单卡 -> [301]
    def scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        plan = self._shard(ctx)
        if plan is not None:
            # The shard is the scattered table (attention DP is refused under
            # query sharding, so one group); a model that narrowed its rows to
            # the sampled rows reports that through collective_num_tokens.
            # 译：分片就是分散计数表（查询分片下拒绝 attention DP，所以只有一个组）；把行缩小到采样行的模型
            #     通过 collective_num_tokens 报告这一点。
            return list(plan.rows_for_collective(ctx.collective_num_tokens))
        global_counts = (
            ctx.collective_global_num_tokens
            if ctx.collective_global_num_tokens is not None
            else ctx.global_num_tokens
        )
        if global_counts is not None:
            scattered = []
            for attn_dp_rank in range(self.mapping.attn.dp_size):
                # global_counts is indexed by global rank with dp stride
                # tp_size.
                # 译：global_counts 以全局 rank 为下标，dp 维的步长为 tp_size。
                num_tokens = global_counts[attn_dp_rank * self.mapping.attn.tp_size]
                scattered.extend(scatter_count(num_tokens, self.mapping.attn.tp_size))
            return scattered
        num_tokens = (
            ctx.collective_num_tokens
            if ctx.collective_num_tokens is not None
            else ctx.input_num_tokens
        )
        return scatter_count(num_tokens, self.mapping.attn.tp_size)

    # 从上面的总表中取出某个通信组的那一段：注意力 TP 组、稠密 TP 组、MoE 的 TP×EP 组
    def attn_tp_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        start = self.mapping.attn.tp_size * self.mapping.attn.dp_rank
        end = start + self.mapping.attn.tp_size
        return self.scattered_num_tokens(ctx)[start:end]

    def dense_tp_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        start = self.mapping.dense.tp_size * self.mapping.dense.dp_rank
        end = start + self.mapping.dense.tp_size
        return self.scattered_num_tokens(ctx)[start:end]

    def moe_tp_ep_group_scattered_num_tokens(self, ctx: ForwardContext) -> list[int]:
        tp_ep_size = self.mapping.moe.tp_ep_size
        global_counts = (
            ctx.collective_global_num_tokens
            if ctx.collective_global_num_tokens is not None
            else ctx.global_num_tokens
        )
        # Without DP, all ranks share the batch and the scattered table needs
        # no global metadata, so the lookup below stays valid.
        # 译：没有 DP 时，所有 rank 共享同一个 batch，分散计数表不需要全局元数据，因此下面的查找仍然有效。
        if global_counts is not None or not self.mapping.attn.has_dp:
            # After post_attn_comm reduce-scatter, each rank holds its
            # scattered share of its attn dp group's tokens, not the raw
            # global count; MoE collectives must size from those rows.
            # 译：在 post_attn_comm 的 reduce-scatter 之后，每个 rank 持有的是它所在 attn dp 组 token 的分散份额，
            #     而不是原始的全局计数；MoE 的集合通信必须按这些行确定大小。
            scattered = self.scattered_num_tokens(ctx)
            return [
                scattered[self.mapping.attn.scatter_index(rank)]
                for rank in self.mapping.moe.tp_ep_group
            ]
        # With DP but no gathered metadata, other dp groups' counts are
        # unknown; only the local rank's contribution can be reported.
        # 译：有 DP 但没有收集到的元数据时，其他 dp 组的计数未知；只能报告本 rank 的那一份。
        num_tokens = (
            ctx.collective_num_tokens
            if ctx.collective_num_tokens is not None
            else ctx.input_num_tokens
        )
        result = [0] * tp_ep_size
        result[self.mapping.moe.tp_ep_rank] = num_tokens
        return result

    # ---- Communication patterns ----
    # 译：---- 通信模式 ----

    # 本层用 all-reduce 布局还是 RSAG 布局：注意力 TP 宽度等于 MLP 所在组（稠密 TP 组或 MoE 的 TP×EP 组）
    # 的宽度时用 all-reduce，否则用 RSAG。单卡时两者都是 1，恒为 all-reduce（实际上不通信）。
    def use_all_reduce(self, is_moe: bool):
        if is_moe:
            return self.mapping.attn.tp_size == self.mapping.moe.tp_ep_size
        return self.mapping.attn.tp_size == self.mapping.dense.tp_size

    # ③ 处是否要 all-gather：上一层用的是 RSAG 布局（各 rank 只有部分行），而注意力需要完整行
    def needs_pre_attn_all_gather(self) -> bool:
        """Whether attention preparation must gather the previous layer's rows
        (replicated-row layouts whose attention legs reduce-scatter; a
        query-sharded model's attention never scatters, so never)."""
        # 译：注意力的准备阶段是否必须收集上一层的行（复制行的布局中，注意力的各段做 reduce-scatter；
        #     查询分片模型的注意力从不分散，所以从不需要）。
        return (
            not self.query_sharded
            and self.layer_id > 0
            and self.mapping.has_attn_tp
            and not self.use_all_reduce(self.prev_is_moe)
        )

    # ③ 注意力入口：需要时在注意力 TP 组内 all-gather 回完整行；单卡原样返回
    def pre_attn_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.needs_pre_attn_all_gather():
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )

    # 把残差收集成完整行（捕获层输入时用，见 models/base/decoder_layer.py 的 capture_layer_input）
    def gather_residual(self, residual: torch.Tensor, ctx: ForwardContext):
        """All-gather a residual left scattered by the previous layer's RSAG
        path (e.g. for aux hidden capture); no-op when rows are already full.

        Mirrors the pre_attn_comm gather conditions.
        """
        # 译：对被上一层的 RSAG 路径留在分散状态的残差做 all-gather（例如用于辅助 hidden 捕获）；行已经完整时什么也不做。
        #
        #     与 pre_attn_comm 的收集条件相同。
        if not self.needs_pre_attn_all_gather():
            return residual
        return self.gather_scattered_rows(residual, ctx)

    # ---- Row layouts ----
    # 译：---- 行布局 ----
    #
    # An all-reduce layer holds every row of its attention DP group on each
    # attention-TP rank; an RSAG layer holds this rank's scattered share. The
    # two helpers below convert between the layouts for a model whose MLPs do
    # not all follow one pattern (LongCat runs a MoE beside dense MLPs). On a
    # sharded forward the rows a layer holds ARE the scattered share and no
    # layer ever holds full rows, so both conversions are identity there; a
    # model never branches on the shard for its row layout.
    # 译：all-reduce 层在每个注意力 TP rank 上持有其 attention DP 组的每一行；RSAG 层持有本 rank 的分散份额。
    #     下面两个辅助函数在这两种布局之间转换，供 MLP 不全遵循同一模式的模型使用（LongCat 在稠密 MLP 旁边
    #     运行一个 MoE）。在分片的前向上，一个层持有的行就是分散份额，没有任何层持有完整的行，所以两种转换在那里
    #     都是恒等的；模型从不按分片对其行布局做分支。

    def slice_scattered_rows(
        self, hidden_states: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        """Keep this rank's scattered share of the full rows (no collective);
        identity on a sharded forward, whose rows are already the share."""
        # 译：保留完整行中本 rank 的分散份额（不做集合通信）；在分片的前向上是恒等的，因为其行已经是那个份额。
        if self._shard(ctx) is not None:
            return hidden_states
        token_list = self.attn_tp_group_scattered_num_tokens(ctx)
        if hidden_states.shape[0] != sum(token_list):
            raise RuntimeError(
                "slice_scattered_rows expects the full rows of the attention "
                f"DP group: got {hidden_states.shape[0]} rows for "
                f"scattered counts {token_list}"
            )
        offset = sum(token_list[: self.mapping.attn.tp_rank])
        return hidden_states[offset : offset + token_list[self.mapping.attn.tp_rank]]

    def gather_scattered_rows(
        self, hidden_states: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        """All-gather the scattered shares back into full rows; identity on a
        sharded forward, which never holds full rows."""
        # 译：把分散的份额 all-gather 回完整的行；在分片的前向上是恒等的，因为它从不持有完整的行。
        if self._shard(ctx) is not None:
            return hidden_states
        token_list = self.attn_tp_group_scattered_num_tokens(ctx)
        if hidden_states.shape[0] != token_list[self.mapping.attn.tp_rank]:
            raise RuntimeError(
                "gather_scattered_rows expects this rank's scattered share: "
                f"got {hidden_states.shape[0]} rows for scattered counts "
                f"{token_list} at attention-TP rank {self.mapping.attn.tp_rank}"
            )
        if sum(token_list) == 0:
            return hidden_states
        return token_all_gather(
            hidden_states,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=token_list,
        )

    # ⑤ 注意力出口的通信（不融合时调用，见 post_attn_reduce_norm）：
    #   all-reduce 布局：o_proj 的部分和 all-reduce；如果上一层是 RSAG 布局，残差还只有部分行，要 all-gather 补齐
    #   RSAG 布局：部分和 reduce-scatter，每个 rank 只留自己那份行；如果残差还是完整行（第 0 层，或上一层是
    #     all-reduce 布局），就从中切出本 rank 的那份
    #   单卡（没有注意力 TP）：原样返回
    def post_attn_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        # A query-sharded model's attention output rows are complete on every
        # forward (sharded extend or replicated decode step): head-replicated
        # weights, or the head-TP tail that already returned this rank's rows.
        # Nothing to reduce or scatter.
        # 译：查询分片模型的注意力输出行在每次前向中都是完整的（分片的 extend 或复制的 decode 步）：权重按头复制，
        #     或者 head-TP 的收尾已经返回了本 rank 的行。没有什么要归约或分散的。
        if self.query_sharded or not self.mapping.has_attn_tp:
            return hidden_states, residual

        if self.use_all_reduce(self.is_moe):
            hidden_states = all_reduce(hidden_states, self.mapping.attn.tp_group)
            # The output residual is expected to have attn_tp_num_tokens.
            # For first layer, the input residual has attn_tp_num_tokens.
            # Otherwise, if this layer experiences a RSAG -> AR switch, residual needs allgather.
            # 译：输出的残差应当有 attn_tp_num_tokens 行。
            #     对第一层，输入的残差有 attn_tp_num_tokens 行。
            #     否则，如果本层经历了 RSAG -> AR 的切换，残差需要 all-gather。
            if self.layer_id > 0 and not self.use_all_reduce(self.prev_is_moe):
                # Keep this residual-producing collective free of early PDL
                # triggers: the next gated combine-norm may preload its result.
                # 译：让这次产生残差的集合通信不带提前的 PDL 触发：下一个门控的合并 norm 可能会预取它的结果。
                residual = token_all_gather(
                    residual,
                    group=self.mapping.attn.tp_group,
                    scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
                )
        else:
            token_list = self.attn_tp_group_scattered_num_tokens(ctx)
            hidden_states = token_reduce_scatter(
                hidden_states,
                group=self.mapping.attn.tp_group,
                scattered_num_tokens=token_list,
            )
            # The output residual is expected to have scattered_num_tokens.
            # For first layer, the input residual has attn_tp_num_tokens, so needs slice.
            # Otherwise, if this layer experiences a AR -> RSAG switch, residual needs slice.
            # 译：输出的残差应当有 scattered_num_tokens 行。
            #     对第一层，输入的残差有 attn_tp_num_tokens 行，所以需要切片。
            #     否则，如果本层经历了 AR -> RSAG 的切换，残差需要切片。
            if self.layer_id == 0 or self.use_all_reduce(self.prev_is_moe):
                offset = sum(token_list[: self.mapping.attn.tp_rank])
                residual = residual[offset : offset + hidden_states.size(0)]

        return hidden_states, residual

    # ⑥ MLP 入口：MLP 所在组比注意力组宽（或处于 QCP 分片）时，先在 MLP 组内 all-gather 出 MLP 要处理的全部行；
    # all-reduce 布局下各 rank 已经有完整行，原样返回。稠密 MLP 和 MoE 分别用各自的组。
    def pre_mlp_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if self.is_moe:
            return self.pre_moe_comm(hidden_states, ctx)
        else:
            return self.pre_dense_comm(hidden_states, ctx)

    def pre_dense_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.mapping.dense.has_tp:
            return hidden_states

        if self._shard(ctx) is None and self.use_all_reduce(is_moe=False):
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.dense.tp_group,
            scattered_num_tokens=self.dense_tp_group_scattered_num_tokens(ctx),
        )

    def pre_moe_comm(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        if not self.mapping.moe.has_tp_ep:
            return hidden_states

        if self._shard(ctx) is None and self.use_all_reduce(is_moe=True):
            return hidden_states

        return token_all_gather(
            hidden_states,
            group=self.mapping.moe.tp_ep_group,
            scattered_num_tokens=self.moe_tp_ep_group_scattered_num_tokens(ctx),
        )

    # ⑧ MLP 出口的通信（不融合时调用，见 post_mlp_fused）：
    #   all-reduce 布局：down_proj 部分和 all-reduce
    #   RSAG 布局：reduce-scatter，每个 rank 拿回自己那份行
    #   批大小无关模式：all-to-all 转置（见下）
    # 注意这里不处理残差：残差的相加推迟到下一层开头。
    def post_mlp_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if self.is_moe:
            return self.post_moe_comm(hidden_states, residual, ctx)
        else:
            return self.post_dense_comm(hidden_states, residual, ctx)

    def post_dense_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.mapping.dense.has_tp:
            return hidden_states, residual

        if self._shard(ctx) is None and self.use_all_reduce(is_moe=False):
            hidden_states = all_reduce(hidden_states, self.mapping.dense.tp_group)
            return hidden_states, residual
        if self.dense_batch_invariant:
            # The column-parallel down_proj already reduced the whole
            # intermediate dim on each rank: [T_full, H / W] holds every
            # token's hidden shard, so only the rows move -- no sum.
            # 译：列并行的 down_proj 已经在每个 rank 上归约了整个 intermediate 维：[T_full, H / W] 持有每个 token 的 hidden 分片，
            #     所以只需要移动行——不需要求和。
            hidden_states = all_to_all_transpose(
                hidden_states,
                group=self.mapping.dense.tp_group,
                input_split_sizes=self.dense_tp_group_scattered_num_tokens(ctx),
            )
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            group=self.mapping.dense.tp_group,
            scattered_num_tokens=self.dense_tp_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    def moe_output_all_reduced(self, ctx: ForwardContext) -> bool:
        """Whether post_moe_comm sums the MoE output with a plain all-reduce."""
        # 译：post_moe_comm 是否用一次普通的 all-reduce 对 MoE 输出求和。
        return (
            self.mapping.moe.has_tp_ep
            and self.moe_combine_order != "slot"
            and self._shard(ctx) is None
            and self.use_all_reduce(is_moe=True)
        )

    # MoE 出口的通信。与稠密 MLP 类似，多了 --moe-combine-order 的区别：
    #   rank（默认）：MoE 返回本 rank 的部分和，这里 all-reduce 或 reduce-scatter
    #   slot：MoE 内部已经把每个 token 在各专家上的结果合并好了，行已完整，不再求和，
    #     RSAG 布局下只需取回本 rank 的行
    def post_moe_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        """Bring the routed-expert output back to this rank's layout.

        Under ``--moe-combine-order rank`` the MoE leaf returned this rank's
        partial and the group sums them here (all-reduce, or reduce-scatter
        in the RSAG layout). Under ``slot`` the leaf already folded every
        token's slots across the EP group, so the rows are complete on every
        rank and nothing is reduced; in the RSAG layout this rank still takes
        back its own token rows, as the reduce-scatter would have.
        """
        # 译：把路由专家的输出带回本 rank 的布局。
        #
        #     在 ``--moe-combine-order rank`` 下，MoE 叶子返回本 rank 的部分和，由组在这里求和（all-reduce，或在 RSAG 布局中
        #     reduce-scatter）。在 ``slot`` 下，叶子已经在 EP 组内合并了每个 token 的各个槽位，所以每个 rank 上的行都是完整的，
        #     不需要归约；在 RSAG 布局中，本 rank 仍要取回自己的 token 行，就像 reduce-scatter 会做的那样。
        if not self.mapping.moe.has_tp_ep:
            return hidden_states, residual

        # A query shard always took the all-gather leg in pre_moe_comm, so it
        # takes the reduce-scatter back -- or, under the slot-ordered combine,
        # the shard's rows out of the combined span: ``replicated`` is the one
        # predicate both branches follow, never ``use_all_reduce`` alone.
        # 译：查询分片在 pre_moe_comm 中总是走了 all-gather 那一段，所以它取回 reduce-scatter——或者在按槽位顺序合并时，
        #     从合并后的范围中取出分片的行：``replicated`` 是两个分支都遵循的同一个判断，绝不单独使用 ``use_all_reduce``。
        replicated = self._shard(ctx) is None and self.use_all_reduce(is_moe=True)
        if self.moe_combine_order == "slot":
            if replicated:
                return hidden_states, residual
            token_list = self.moe_tp_ep_group_scattered_num_tokens(ctx)
            offset = sum(token_list[: self.mapping.moe.tp_ep_rank])
            own = token_list[self.mapping.moe.tp_ep_rank]
            return hidden_states[offset : offset + own], residual

        if replicated:
            hidden_states = all_reduce(hidden_states, self.mapping.moe.tp_ep_group)
            return hidden_states, residual
        hidden_states = token_reduce_scatter(
            hidden_states,
            group=self.mapping.moe.tp_ep_group,
            scattered_num_tokens=self.moe_tp_ep_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    # final norm 之后是否要 all-gather：最后一层是 RSAG 布局时各 rank 只有部分行，而 LogitsProcessor 需要完整行
    def needs_final_all_gather(self) -> bool:
        """Whether the model output must gather the final layer's rows
        (replicated-row layouts whose attention legs reduce-scatter; a
        query-sharded model never scatters, and the logits processor gathers
        only the sampled rows of its shard, see :func:`gather_sampled_rows`)."""
        # 译：模型输出是否必须收集最后一层的行（复制行的布局中注意力的各段做 reduce-scatter；查询分片模型从不分散，
        #     而 logits processor 只收集其分片中要采样的行，见 :func:`gather_sampled_rows`）。
        return (
            not self.query_sharded
            and self.mapping.has_attn_tp
            and not self.use_all_reduce(self.is_moe)
        )

    def post_final_norm_comm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.needs_final_all_gather():
            return hidden_states, residual
        hidden_states = token_all_gather(
            hidden_states,
            group=self.mapping.attn.tp_group,
            scattered_num_tokens=self.attn_tp_group_scattered_num_tokens(ctx),
        )
        return hidden_states, residual

    # 下面是“all-reduce + 相加残差 + RMSNorm”融合的判断，以及层的四个交界点：
    #   input_reduce_norm（①）、post_attn_reduce_norm（⑤）、post_mlp_fused（⑧）、final_norm（主干末尾）
    # 融合的条件（should_fuse）：all-reduce 布局、有注意力 TP、开启了 --enable-allreduce-fusion、
    # 本次 token 数在 1 到 --comm-fusion-max-num-tokens 之间，且没有选择与融合冲突的数值选项。
    # 融合 kernel 需要一块预先分配、按最大 token 数确定大小的通信工作区，所以 token 太多时不融合。
    # ---- Fused allreduce+norm ----
    # 译：---- all-reduce 与 norm 的融合 ----

    def use_all_reduce_norm_fusion(self) -> bool:
        from tokenspeed.runtime.utils.env import global_server_args_dict

        # A query-sharded model's rows are never replicated at the
        # all-reduce boundary, so the fused all-reduce + norm has no place.
        # 译：查询分片模型的行在 all-reduce 边界上从不是复制的，所以融合的 all-reduce + norm 在那里没有用武之地。
        return (
            not self.query_sharded
            and self.use_all_reduce(self.is_moe)
            and self.mapping.has_attn_tp
            and global_server_args_dict.get("enable_allreduce_fusion", False)
        )

    def should_fuse(self, num_tokens: int) -> bool:
        """Whether this launch's fused all-reduce+norm kernel runs here.

        The trainer-order switches veto it at the point that relies on the
        veto, not only in ``resolve_numerics``: the unfused boundary norm needs
        the bf16 ``hidden + residual`` materialized first, and the slot-order
        MoE combine returns complete rows a fused all-reduce would sum
        ``tp_size`` times.
        """
        # 译：本次启动是否在这里运行融合的 all-reduce + norm kernel。
        #
        #     训练端顺序的开关在依赖这一否决的地方否决它，而不只是在 ``resolve_numerics`` 中：非融合的边界 norm 需要先
        #     把 bf16 的 ``hidden + residual`` 计算出来，而按槽位顺序的 MoE 合并返回的是完整的行，融合的 all-reduce 会把它们
        #     求和 ``tp_size`` 次。
        from tokenspeed.runtime.utils.env import global_server_args_dict

        if self.layer_boundary_norm == "unfused" or self.moe_combine_order == "slot":
            return False
        return (
            self.use_all_reduce_norm_fusion()
            and num_tokens > 0
            and num_tokens <= global_server_args_dict["comm_fusion_max_num_tokens"]
        )

    def _fused_add_norm(
        self,
        norm: torch.nn.Module,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the residual and normalize in one kernel (the fp32 sum feeds
        the norm), fused with the all-reduce when the launch allows it."""
        # 译：在一个 kernel 中加上残差并归一化（fp32 的和送入 norm），在启动条件允许时与 all-reduce 融合。
        # TP=1 时 should_fuse 为 False，走 else：普通的融合 add + RMSNorm（NVIDIA 上原地改写 hidden 和 residual）
        if self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = norm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                hidden_states,
                residual,
            )
        else:
            hidden_states, residual = norm(hidden_states, residual)
        return hidden_states, residual

    @staticmethod
    def _unfused_add_norm(
        norm: torch.nn.Module,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The trainer's layer boundary: the bf16 sum is materialized and is
        both the new residual and the norm's input."""
        # 译：训练端的层边界：计算出 bf16 的和，它既是新的残差，也是 norm 的输入。
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = norm(hidden_states)
        return hidden_states, residual

    # ① 一层的开头：补上上一层欠下的残差相加（以及推迟的 all-reduce），再做 input_layernorm。
    #   residual 为 None（第 0 层，未融合 embedding 的 all-reduce 时）：residual = hidden；hidden = norm(hidden)
    #   layer_boundary_norm="unfused"：训练端顺序（先求 bf16 的和，再归一化）
    #   默认：融合的 add + RMSNorm，可能同时融合 all-reduce
    # 返回 (norm 后的 hidden, 新的 residual)。
    def input_reduce_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ):
        """The norm that opens a physical layer, consuming the previous
        layer's output (``--layer-boundary-norm``)."""
        # 译：打开一个物理层的 norm，消费上一层的输出（``--layer-boundary-norm``）。
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif self.layer_boundary_norm == "unfused":
            hidden_states, residual = self._unfused_add_norm(
                self.input_layernorm, hidden_states, residual
            )
        else:
            hidden_states, residual = self._fused_add_norm(
                self.input_layernorm, hidden_states, residual
            )
        return hidden_states, residual

    # 物理层内部的 add + norm（不是层边界），LongCat 的第二个注意力分支使用
    def intra_layer_add_norm(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        """An add+norm inside a physical layer on ``input_layernorm`` (LongCat's
        second attention branch). Not a layer boundary, so it keeps the fused
        form under every ``--layer-boundary-norm``, as the trainer does."""
        # 译：物理层内部在 ``input_layernorm`` 上的 add+norm（LongCat 的第二个注意力分支）。它不是层边界，所以在每种
        #     ``--layer-boundary-norm`` 下都保持融合形式，与训练端一致。
        return self._fused_add_norm(self.input_layernorm, hidden_states, residual)

    # ⑤ 注意力之后：可融合时一个 kernel 完成 all-reduce + 相加残差 + post_attention_layernorm；
    # 否则先 post_attn_comm（all-reduce 或 reduce-scatter），再做融合的 add + RMSNorm。
    # 单卡（Qwen3 主线）：post_attn_comm 原样返回，然后 residual = hidden + residual；hidden = norm(residual)。
    def post_attn_reduce_norm(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual, *_ = (
                self.post_attn_layernorm.forward_with_allreduce_fusion(
                    self.mapping.attn.tp_rank,
                    self.mapping.attn.tp_group,
                    hidden_states,
                    residual,
                )
            )
        else:
            hidden_states, residual = self.post_attn_comm(hidden_states, residual, ctx)
            hidden_states, residual = self.post_attn_layernorm(hidden_states, residual)
        return hidden_states, residual

    # ⑧ MLP 之后：可融合时什么也不做——这次 all-reduce 推迟到下一层的 ①（或 final_norm），
    # 与那里的相加和 norm 一起完成；否则在这里做 post_mlp_comm。单卡时 post_mlp_comm 原样返回。
    def post_mlp_fused(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, ctx: ForwardContext
    ):
        if not self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual = self.post_mlp_comm(hidden_states, residual, ctx)
        return hidden_states, residual

    # 主干末尾的 final norm（由 BaseTransformerModel 调用最后一层的 CommManager）：
    #   与 ① 相同的三种方式（训练端顺序 / 融合 all-reduce / 普通 add + RMSNorm），
    #   不融合时如果最后一层是 RSAG 布局，还要 all-gather 回完整行。
    # 返回 (norm 后的 hidden, 相加后的残差 = 最后一层真正的输出)；空跑前向返回 (hidden, None)。
    def final_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        norm: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if ctx.forward_mode.is_idle():
            return hidden_states, None

        if self.layer_boundary_norm == "unfused":
            hidden_states, residual_out = self._unfused_add_norm(
                norm, hidden_states, residual
            )
            hidden_states, _ = self.post_final_norm_comm(hidden_states, residual, ctx)
        elif self.should_fuse(hidden_states.shape[0]):
            hidden_states, residual_out, *_ = norm.forward_with_allreduce_fusion(
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
                hidden_states,
                residual,
            )
        else:
            hidden_states, residual_out = norm(hidden_states, residual)
            hidden_states, _ = self.post_final_norm_comm(hidden_states, residual, ctx)

        return hidden_states, residual_out
