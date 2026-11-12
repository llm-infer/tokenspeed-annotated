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

"""What the data plane and the rank setup read off the C++ scheduler: its
config, its cache geometry and groups, the cache ops a plan carries and
their acknowledgments, and the block tables a forward reads."""

# 译：执行面和 rank 初始化从 C++ 调度器读取的东西：它的配置、cache 几何与 cache group、计划携带的 cache 操作及其确认，
#     以及一次前向读取的 block table。

# 执行面与 C++ 调度器（tokenspeed_scheduler，第 6 阶段）之间的数据转换。分四类：
#   1. 配置：make_config 构造 SchedulerConfig；scheduler_cache_geometry_from_pool / pool_to_cache_groups
#      把 arena 的契约（页数、cache group）告诉调度器
#   2. 运行方式：should_use_overlap_schedule、resolve_in_flight_depth（在途深度）
#   3. cache 操作：cache_ops_from_plan 把计划中的 cache 操作分成“GPU 拷贝执行器负责的”和“L3 存储负责的”两类；
#      cache_event_to_payload 把完成确认报告给控制面
#   4. 页表：packed_block_tables_from_forward_op 把调度器给出的每组 block table 一次性上传到 GPU（主线每次前向都调用）
# 另外还有启动后打印显存使用情况的 log_gpu_memory_summary。
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
from tokenspeed_scheduler import (
    Cache,
    CacheGroupConfig,
    SchedulerConfig,
    StorageConfig,
)

from tokenspeed.runtime.cache.transfer.ops import (
    CacheTransfer,
    HostTier,
    RestoreOp,
    SnapshotOp,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    require_positive_int,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
    cache_group_config,
    scheduler_role,
)

# The Host cache's ACKs, as the scheduler binds them: an L2 write-back or
# load-back ticket, a retraction image's snapshot store, a restore. The data
# plane reports them to the Rust control plane as ``(kind, op_id)`` payloads.
# 译：调度器绑定的 Host cache 确认（ACK）：L2 写回或读回的票据、抢占快照的存储完成、恢复完成。执行面把它们作为
#     ``(kind, op_id)`` 负载报告给 Rust 控制面。
_CACHE_EVENT_TYPES = {
    "WriteBackDoneEvent": Cache.WriteBackDoneEvent,
    "LoadBackDoneEvent": Cache.LoadBackDoneEvent,
    "SnapshotDoneEvent": Cache.SnapshotDoneEvent,
    "RestoreDoneEvent": Cache.RestoreDoneEvent,
}
# make_config's host_write_policy names; ServerArgs validates the flag.
# 译：make_config 的 host_write_policy 名字；由 ServerArgs 校验该参数。
_HOST_WRITE_POLICIES = {
    "write_through": SchedulerConfig.HostWritePolicy.WriteThrough,
    "write_back": SchedulerConfig.HostWritePolicy.WriteBack,
}


def engram_context_len(text_config) -> int:
    """Select caller-owned Engram inputs, not Qwen4's LCM-owned PLE state."""
    # 译：选择由调用方拥有的 Engram 输入，而不是 Qwen4 由 LCM 拥有的 PLE 状态。
    if not getattr(text_config, "engram_layer_ids", ()):
        return 0
    if getattr(text_config, "ngram_context_len", None) != 3:
        raise ValueError("Engram requires hf_text_config.ngram_context_len = 3")
    return text_config.ngram_context_len


# 调度器视角的 cache 几何：前缀粒度、设备上的页数（含保留的第 0 页）、可用页数、token 容量。
# 注意这里的“页”是 LCM 父块；单 cache group 的模型（Qwen3）中一个父块就是“所有层的一个 KV 页”。
@dataclass(frozen=True)
class SchedulerCacheGeometry:
    prefix_granularity: int
    num_device_pages: int
    num_usable_pages: int
    token_capacity: int


def scheduler_cache_geometry_from_pool(pool: Any) -> SchedulerCacheGeometry:
    """Project the arena's published contract onto scheduler page counts.

    The contract is the only source: the arena publishes it for every pool,
    so there is no pool-side copy to prefer and no server-args fallback to
    reconcile against.
    """
    # 译：把 arena 发布的契约投影成调度器的页数。
    #
    #     契约是唯一来源：arena 为每个池都发布它，所以不存在可以优先使用的池端副本，也没有需要协调的 server-args 回退值。
    contract = pool.arena.runtime_contract
    num_lcm_blocks = require_positive_int(
        "contract.num_lcm_blocks", contract.num_lcm_blocks
    )
    return SchedulerCacheGeometry(
        prefix_granularity=contract.prefix_granularity,
        # Parent 0 is reserved as the null LCM block.
        # 译：父块 0 保留为空 LCM 块。
        num_device_pages=num_lcm_blocks + 1,
        num_usable_pages=num_lcm_blocks,
        token_capacity=contract.token_capacity,
    )


# 构造 C++ 调度器的配置对象（启动时调用一次）。各参数的含义见下面的 docstring，与调度相关的细节在第 6、14 阶段。
def make_config(
    num_device_pages: int,
    max_scheduled_tokens: int,
    max_batch_size: int,
    prefix_granularity: int,
    num_host_pages: int,
    disable_l2_cache: bool,
    enable_l3_storage: bool,
    role: str,
    *,
    num_snapshot_pages: int,
    host_group_packing: Sequence[int],
    max_retracted_requests: int,
    storage_config: StorageConfig | None,
    host_write_policy: str,
    prefill_decode_interval: int,
    debug_force_retraction_interval: int = 0,
    enable_kv_cache_events: bool = False,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    disable_prefix_cache: bool = False,
    cache_groups: Sequence[CacheGroupConfig] | None = None,
    enable_mixed_prefill_decode: bool = False,
    enable_short_prefill_first: bool = False,
    prefix_replay_tokens: int = 0,
    state_checkpoint_in_forward: bool = False,
    demote_superseded_state_checkpoints: bool = False,
    drop_short_prompt_end_checkpoints: bool = False,
    max_kv_page_num: int = 0,
    checkpoint_retention_requests: int | None = None,
    checkpoint_retention_leaf_distance: int | None = None,
    prefill_chunk_yield_tokens: int = 0,
) -> SchedulerConfig:
    """Build the C++ scheduler's configuration.

    Args:
        num_device_pages: Device LCM blocks including the null page.
        max_scheduled_tokens: Per-step prefill token budget.
        enable_short_prefill_first: Opt in to remaining-work ordering with
            oldest-request protection; False preserves submission order.
        max_batch_size: Per-rank batch slots -- the most requests this rank
            runs at once (``--max-num-seqs`` over the attention-DP ranks).
        prefix_granularity: Prefix-hash page size in tokens.
        num_host_pages: Host L2 LCM blocks including the null page; 0 without
            the L2 tier.
        disable_l2_cache: No Host L2 tier.
        enable_l3_storage: An L3 store backs the L2 tier.
        role: ``server_args.disaggregation_mode``.
        num_snapshot_pages: The retraction snapshot pool's LCM blocks
            including the null page (``DeviceSpecs.num_snapshot_pages``).
            Required on every role: 1 -- the null page alone -- means no
            image can be taken, so a capacity-blocked round aborts the newest
            resident instead of suspending anyone; more enables suspend and
            restore on the fused and decode roles (the prefill role refuses a
            pool).
        host_group_packing: Virtual children per compact Host parent in
            scheduler group order; empty when both Host pools are absent.
        max_retracted_requests: Slot-state arena rows, hence the most
            requests suspended at once; positive exactly when a pool exists.
            A victim whose image does not fit the Host (L2 pins, pool, rows)
            is aborted by the scheduler instead of imaged.
        storage_config: Explicit asynchronous L3 budgets, or None without L3.
        host_write_policy: ``--kvstore-write-policy``: ``"write_through"``
            streams published entries to Host (at chunk ends, the prompt end
            and finish); ``"write_back"`` writes back what an admission or a
            restore evicts, in the same plan (``SchedulerConfig::Validate``
            refuses it where it cannot run). Required on every role; without a
            Host tier write_through has no effect, and the decode role's Host
            tier holds retraction images only.
        prefill_decode_interval: ``--prefill-decode-interval``: decode rounds
            owed to the running decodes after each round that schedules a
            prefill chunk, unless no decode can run, and the most rounds it
            holds any one prompt back in total; 0 is off. Fused role only.
        debug_force_retraction_interval: ``--debug-force-retraction-interval``:
            0 is off; every ``|N|`` plans the scheduler retracts the oldest
            quiescent Decoding (``N > 0``) or Prefilling (``N < 0``) request
            without capacity pressure. A test knob, never set in serving.
        enable_kv_cache_events: Publish KV cache events.
        decode_input_tokens: Verify width per decode step.
        overlap_schedule_depth: Dispatched-but-uncommitted plans allowed.
        disable_prefix_cache: No prefix matching at admission.
        cache_groups: The arena's scheduler group configs.
        enable_mixed_prefill_decode: Mixed prefill/decode batches.
        prefix_replay_tokens: Prompt tail recomputed after a prefix hit.
        state_checkpoint_in_forward:
            ``--enable-state-checkpoint-in-forward``: the prefill forward that
            spans a long prompt's last prefix-page boundary before its replay
            tail also writes the State checkpoint there, and no KV tail and no
            State checkpoint is published at a prompt end that checkpoint
            covers (prefill and fused roles) or at a fused finish past the
            prompt end. Requires prefix caching.
        demote_superseded_state_checkpoints:
            On with ``--enable-state-checkpoint-in-forward``: a finish that
            publishes an in-forward State checkpoint demotes the one its
            conversation published two turns back, unless another request
            shares it. Requires ``state_checkpoint_in_forward`` outside the
            decode role.
        drop_short_prompt_end_checkpoints:
            On a prefill server with ``--enable-state-checkpoint-in-forward``:
            when a prompt is too short for the in-forward checkpoint and
            ends inside a page, no KV tail and no State checkpoint is
            published at its end. Requires ``state_checkpoint_in_forward``
            and the prefill role.
        max_kv_page_num: Per-local-prefill-batch budget for the sum of
            ``ceil((cached prefix + current chunk) / prefix_granularity)``
            across requests. 0 leaves the budget unbounded. Derived from
            ``--chunk-prefill-kv-size``.
        checkpoint_retention_requests: ``--checkpoint-retention-requests``:
            under cache pressure a hybrid model's State checkpoint that a later
            request's checkpoint extends ranks as if last used this many new
            requests ago (0 ranks every cache entry by its own use). None keeps the
            scheduler's default, 16 times ``max_batch_size``, which is
            ``--max-num-seqs`` per attention-DP rank.
        checkpoint_retention_leaf_distance:
            ``--checkpoint-retention-leaf-distance``: checkpoints within this
            many later requests of the newest on their branch rank as it does. None keeps the
            scheduler's default, 0.
        prefill_chunk_yield_tokens:
            ``--prefill-chunk-yield-tokens``: the most tokens a prefill chunk
            that cannot finish its prompt gives up, page-aligned, so that
            bootstrapped prompts waiting behind it that fit whole run in the
            same forward. 0 is off; the prefill role only, at most half of
            ``max_scheduled_tokens`` (``SchedulerConfig::Validate`` refuses
            it elsewhere).
    """
    # 译：构造 C++ 调度器的配置。
    #
    #     参数：
    #         num_device_pages：设备上的 LCM 块数，包括空页。
    #         max_scheduled_tokens：每一步 prefill 的 token 预算。
    #         enable_short_prefill_first：选择按剩余工作量排序、同时保护最老请求的方式；False 保持提交顺序。
    #         max_batch_size：每个 rank 的 batch 槽位数——本 rank 同时运行的最多请求数（``--max-num-seqs`` 除以 attention-DP rank 数）。
    #         prefix_granularity：前缀哈希的页大小，单位 token。
    #         num_host_pages：Host L2 的 LCM 块数，包括空页；没有 L2 层时为 0。
    #         disable_l2_cache：没有 Host L2 层。
    #         enable_l3_storage：有 L3 存储支撑 L2 层。
    #         role：``server_args.disaggregation_mode``。
    #         num_snapshot_pages：抢占快照池的 LCM 块数，包括空页（``DeviceSpecs.num_snapshot_pages``）。每种角色都必须提供：
    #             1——只有空页——表示无法生成任何快照，所以一个因容量受阻的轮次会中止最新的驻留请求，而不是挂起某个请求；
    #             更多则在融合角色和 decode 角色上启用挂起与恢复（prefill 角色拒绝快照池）。
    #         host_group_packing：每个紧凑 Host 父块中的虚拟子块数，按调度器的组顺序；两个 Host 池都不存在时为空。
    #         max_retracted_requests：槽位状态 arena 的行数，也就是同时挂起的最多请求数；当且仅当存在快照池时为正数。
    #             快照放不进 Host（L2 固定、池、行数）的牺牲请求会被调度器中止，而不是生成快照。
    #         storage_config：显式的异步 L3 预算，没有 L3 时为 None。
    #         host_write_policy：``--kvstore-write-policy``：``"write_through"`` 把已发布的条目流式写到 Host（在块结束、prompt 结束
    #             和请求结束时）；``"write_back"`` 在同一个计划中写回被准入或恢复驱逐掉的内容（在不能运行的地方
    #             ``SchedulerConfig::Validate`` 会拒绝它）。每种角色都必须提供；没有 Host 层时 write_through 不起作用，
    #             decode 角色的 Host 层只存放抢占快照。
    #         prefill_decode_interval：``--prefill-decode-interval``：每个调度了 prefill 块的轮次之后，欠正在运行的 decode 的
    #             decode 轮数（除非没有 decode 可以运行），也是它总共最多压后任何一个 prompt 的轮数；0 表示关闭。只用于融合角色。
    #         debug_force_retraction_interval：``--debug-force-retraction-interval``：0 表示关闭；每 ``|N|`` 个计划，调度器在没有
    #             容量压力的情况下抢占最老的、处于静止状态的 Decoding 请求（``N > 0``）或 Prefilling 请求（``N < 0``）。
    #             这是测试用的开关，服务时从不设置。
    #         enable_kv_cache_events：发布 KV cache 事件。
    #         decode_input_tokens：每个 decode 步的验证宽度。
    #         overlap_schedule_depth：允许的已下发但未提交的计划数。
    #         disable_prefix_cache：准入时不做前缀匹配。
    #         cache_groups：arena 的调度器组配置。
    #         enable_mixed_prefill_decode：prefill/decode 混合批。
    #         prefix_replay_tokens：前缀命中后要重新计算的 prompt 尾部。
    #         state_checkpoint_in_forward：``--enable-state-checkpoint-in-forward``：跨越长 prompt 在其重放尾部之前的最后一个
    #             前缀页边界的 prefill 前向，也在那里写入 State 检查点；在该检查点覆盖的 prompt 结束处（prefill 和融合角色），
    #             或在融合角色越过 prompt 结束的请求结束处，不发布 KV 尾部和 State 检查点。需要前缀缓存。
    #         demote_superseded_state_checkpoints：在 ``--enable-state-checkpoint-in-forward`` 下开启：一个发布了前向内 State
    #             检查点的请求结束时，降级它的会话在两轮之前发布的那个检查点，除非另一个请求也在共享它。在 decode 角色
    #             之外需要 ``state_checkpoint_in_forward``。
    #         drop_short_prompt_end_checkpoints：在开启了 ``--enable-state-checkpoint-in-forward`` 的 prefill 服务器上：当一个
    #             prompt 短到无法做前向内检查点、并且在页内部结束时，不在它的结束处发布 KV 尾部和 State 检查点。需要
    #             ``state_checkpoint_in_forward`` 和 prefill 角色。
    #         max_kv_page_num：每个本地 prefill batch 中各请求 ``ceil((已缓存前缀 + 当前块) / prefix_granularity)`` 之和的预算。
    #             0 表示不限。由 ``--chunk-prefill-kv-size`` 推导。
    #         checkpoint_retention_requests：``--checkpoint-retention-requests``：在 cache 压力下，混合模型中被后续请求的检查点
    #             延伸的 State 检查点，按“最后一次使用是在这么多个新请求之前”来排序（0 表示每个 cache 条目按它自己的使用排序）。
    #             None 保持调度器的默认值，即 ``max_batch_size`` 的 16 倍，也就是每个 attention-DP rank 的 ``--max-num-seqs`` 的 16 倍。
    #         checkpoint_retention_leaf_distance：``--checkpoint-retention-leaf-distance``：在其分支上距离最新检查点不超过这么多
    #             后续请求的检查点，与最新的那个排序相同。None 保持调度器的默认值 0。
    #         prefill_chunk_yield_tokens：``--prefill-chunk-yield-tokens``：一个无法完成其 prompt 的 prefill 块最多让出多少 token
    #             （按页对齐），使排在它后面、已完成引导、能整个放下的 prompt 在同一次前向中运行。0 表示关闭；只用于 prefill
    #             角色，最多为 ``max_scheduled_tokens`` 的一半（其他地方 ``SchedulerConfig::Validate`` 会拒绝它）。
    if not 0 <= prefix_replay_tokens <= (1 << 31) - 1:
        raise ValueError(
            "prefix_replay_tokens must fit a non-negative int32; "
            f"got {prefix_replay_tokens}."
        )
    if num_snapshot_pages < 1:
        raise ValueError(
            "num_snapshot_pages must include the null page (1 = no pool); "
            f"got {num_snapshot_pages}."
        )
    if (num_snapshot_pages > 1) != (max_retracted_requests > 0):
        raise ValueError(
            "a retraction snapshot pool and its slot-state rows go together; got "
            f"num_snapshot_pages={num_snapshot_pages}, "
            f"max_retracted_requests={max_retracted_requests}."
        )
    if (storage_config is not None) != enable_l3_storage:
        raise ValueError("storage_config is required exactly when L3 is enabled")
    if prefill_chunk_yield_tokens < 0:
        raise ValueError(
            "prefill_chunk_yield_tokens must be non-negative; "
            f"got {prefill_chunk_yield_tokens}."
        )
    cfg = SchedulerConfig()
    cfg.num_device_pages = num_device_pages
    cfg.max_scheduled_tokens = max_scheduled_tokens
    if max_kv_page_num:
        cfg.max_kv_page_num = max_kv_page_num
    cfg.max_batch_size = max_batch_size
    cfg.prefix_granularity = prefix_granularity

    cfg.num_host_pages = num_host_pages
    cfg.num_snapshot_pages = num_snapshot_pages
    cfg.host_group_packing = list(host_group_packing)
    cfg.max_retracted_requests = max_retracted_requests
    cfg.debug_force_retraction_interval = debug_force_retraction_interval
    cfg.storage_config = storage_config
    cfg.enable_l3_storage = enable_l3_storage
    cfg.enable_kv_cache_events = enable_kv_cache_events

    cfg.role = scheduler_role(role)
    cfg.decode_input_tokens = decode_input_tokens
    cfg.overlap_schedule_depth = overlap_schedule_depth
    cfg.disable_prefix_cache = disable_prefix_cache
    cfg.prefix_replay_tokens = prefix_replay_tokens
    if state_checkpoint_in_forward:
        # Set only when on: off is the binding's default, so the runtime keeps
        # working with a scheduler wheel released before this field, where
        # turning the option on raises AttributeError here instead of being
        # ignored.
        # 译：只在开启时设置：关闭是绑定层的默认值，所以运行时可以继续配合在这个字段出现之前发布的调度器 wheel 工作，
        #     而在那种 wheel 上打开这个选项会在这里抛出 AttributeError，而不是被忽略。
        cfg.state_checkpoint_in_forward = True
    if demote_superseded_state_checkpoints:
        # Set only when on, as above.
        # 译：只在开启时设置，原因同上。
        cfg.demote_superseded_state_checkpoints = True
    if drop_short_prompt_end_checkpoints:
        # Likewise set only when on.
        # 译：同样只在开启时设置。
        cfg.drop_short_prompt_end_checkpoints = True
    if checkpoint_retention_requests is not None:
        # Set only when given, as above: an older scheduler wheel keeps its
        # plain LRU by default and refuses an explicit value.
        # 译：只在给出时设置，原因同上：较旧的调度器 wheel 默认保持普通的 LRU，并拒绝显式的值。
        cfg.checkpoint_retention_requests = checkpoint_retention_requests
    if checkpoint_retention_leaf_distance is not None:
        cfg.checkpoint_retention_leaf_distance = checkpoint_retention_leaf_distance
    if prefill_chunk_yield_tokens:
        # Set only when on, for the same reason.
        # 译：只在开启时设置，原因相同。
        cfg.prefill_chunk_yield_tokens = prefill_chunk_yield_tokens
    cfg.disable_l2_cache = disable_l2_cache
    cfg.host_write_policy = _HOST_WRITE_POLICIES[host_write_policy]

    cfg.enable_mixed_prefill_decode = enable_mixed_prefill_decode
    if prefill_decode_interval:
        # Set only when on, like state_checkpoint_in_forward: a scheduler
        # wheel without the field still runs the default.
        # 译：像 state_checkpoint_in_forward 一样只在开启时设置：没有这个字段的调度器 wheel 仍然运行默认行为。
        cfg.prefill_decode_interval = prefill_decode_interval
    if enable_short_prefill_first:
        # Older scheduler wheels retain FIFO and reject an explicit opt-in.
        # 译：较旧的调度器 wheel 保持 FIFO，并拒绝显式的开启。
        cfg.enable_short_prefill_first = True
    if cache_groups:
        cfg.cache_groups = list(cache_groups)
    return cfg


def pool_to_cache_groups(pool: Any) -> list:
    """Convert a cache's published contract into scheduler group configs."""
    # 译：把 cache 发布的契约转换为调度器的组配置。
    # The arena is the sole publisher, so there is exactly one source here --
    # no fallback to pool-side copies of the same specs.
    # 译：arena 是唯一的发布者，所以这里只有一个来源——不回退到池端对同一批 spec 的副本。
    contract = pool.arena.runtime_contract
    counts = contract.virtual_block_counts
    packing = contract.virtual_packing
    return [
        cache_group_config(
            spec,
            total_pages=counts[spec.group_id],
            cache_blocks_per_lcm_block=packing[spec.group_id],
        )
        for spec in contract.group_specs
    ]


def should_use_overlap_schedule(
    *,
    disable_overlap_schedule: bool,
    disaggregation_mode: str,
) -> bool:
    """Return whether the runtime runs the overlap schedule: a decode step
    dispatched before the previous one commits, with decode KV reserved for
    it (``overlap_schedule_depth``)."""
    # 译：返回运行时是否运行 overlap 调度：在上一步提交之前就下发一个 decode 步，并为它预留 decode 的 KV
    #     （``overlap_schedule_depth``）。

    if disable_overlap_schedule:
        return False
    if disaggregation_mode in ("prefill", "encode"):
        # The prefill role never decodes locally, so it has no decode step to
        # run ahead or reserve KV for; its chunks still run ahead of their
        # commits (resolve_in_flight_depth). Encode has no LM loop.
        # 译：prefill 角色从不在本地做 decode，所以没有可以提前运行或为其预留 KV 的 decode 步；它的块仍然在提交之前提前运行
        #     （resolve_in_flight_depth）。编码角色没有语言模型的循环。
        return False
    return True


def resolve_in_flight_depth(
    *,
    disable_overlap_schedule: bool,
    disaggregation_mode: str,
    pp_size: int,
) -> int:
    """Return the event loop's in-flight depth: how many dispatched forwards
    may await commit at once (``docs/design/event-loop.md``, Principle 4).

    ``pp_size`` under pipeline parallelism, where the prefill chunk pipeline
    keeps one forward per stage in flight. At PP=1, 0 with
    ``disable_overlap_schedule`` (each forward commits in the round that
    dispatched it) or on an encode rank, which has no LM loop; otherwise 1:
    the control plane plans the next round, commits the previous forward and
    polls transfers while the GPU runs the dispatched one. On a role that
    decodes that is the overlap schedule. The prefill role runs ahead without
    it (``should_use_overlap_schedule``): its chunks need no decode KV
    reserved, and the scheduler holds a prompt's remote decode until the
    final chunk's result has landed.

    Args:
        disable_overlap_schedule: ``--disable-overlap-schedule``.
        disaggregation_mode: ``--disaggregation-mode``.
        pp_size: Pipeline-parallel stages.

    Returns:
        The in-flight depth.
    """
    # 译：返回事件循环的在途深度：同时可以有多少个已下发的前向等待提交（``docs/design/event-loop.md``，原则 4）。
    #
    #     在流水线并行下为 ``pp_size``，此时 prefill 块流水线让每个 stage 都有一个前向在途。在 PP=1 时，开启
    #     ``disable_overlap_schedule`` 时为 0（每个前向在下发它的那一轮提交），编码 rank 也为 0（它没有语言模型循环）；
    #     否则为 1：控制面在 GPU 运行已下发的前向时，规划下一轮、提交上一次前向并轮询传输。在做 decode 的角色上，
    #     这就是 overlap 调度。prefill 角色不用它也会提前运行（``should_use_overlap_schedule``）：它的块不需要预留 decode KV，
    #     而调度器会把一个 prompt 的远程 decode 推迟到最后一个块的结果落地之后。
    #
    #     参数：
    #         disable_overlap_schedule：``--disable-overlap-schedule``。
    #         disaggregation_mode：``--disaggregation-mode``。
    #         pp_size：流水线并行的 stage 数。
    #
    #     返回：
    #         在途深度。
    if pp_size > 1:
        return pp_size
    if disable_overlap_schedule or disaggregation_mode == "encode":
        return 0
    return 1


def resolve_dspark_prefix_replay_tokens(
    *,
    speculative_algorithm: str | None,
    enable_prefix_caching: bool,
    enable_kvstore: bool,
    disaggregation_mode: str,
    draft_model_path_use_base: bool,
    draft_model_config: Any | None,
) -> int:
    """Resolve the prompt tail needed to rebuild DSpark runtime state.

    A same-checkpoint DSpark draft advertises the requirement through its
    draft ``ModelConfig``; DeepSeek V4.1 advertises zero because its windows
    are cache resident. Same-checkpoint DSpark configurations without that
    capability remain fail-closed. External generic DSpark configurations
    keep their existing scheduler behavior until they advertise an
    equivalent contract.
    """
    # 译：解析重建 DSpark 运行时状态所需的 prompt 尾部长度。
    #
    #     同一 checkpoint 的 DSpark 草稿通过它的草稿 ``ModelConfig`` 声明这一需求；DeepSeek V4.1 声明为零，因为它的窗口常驻于
    #     cache 中。没有这种能力的同一 checkpoint DSpark 配置保持拒绝。外部的通用 DSpark 配置保持其现有的调度器行为，
    #     直到它们声明一个等价的约定。

    if speculative_algorithm != "DSPARK":
        return 0
    if draft_model_config is None:
        raise ValueError("DSPARK requires a resolved draft model configuration.")

    replay_tokens = getattr(draft_model_config, "dspark_prefix_replay_tokens", None)
    if replay_tokens is None:
        if draft_model_path_use_base:
            raise ValueError(
                "DSPARK same-checkpoint decoding requires a draft model that "
                "advertises captured-context replay support."
            )
        return 0

    replay_tokens = int(replay_tokens)
    if not 0 <= replay_tokens <= (1 << 31) - 1:
        raise ValueError(
            "DSPARK captured-context replay requirement must fit a non-negative "
            f"int32; got {replay_tokens}."
        )
    if replay_tokens == 0:
        # The draft's context lives in the KV cache and follows the prefix.
        # 译：草稿模型的上下文在 KV cache 中，随前缀一起保存。
        return 0
    # A drafter-private context cannot be restored from the host tier, with or
    # without prefix reuse.
    # 译：草稿器私有的上下文无法从 Host 层恢复，无论是否复用前缀。
    if enable_kvstore:
        raise ValueError(
            "DSPARK captured-context replay does not support KVStore; "
            "use --disable-kvstore."
        )
    if not enable_prefix_caching:
        return 0
    if disaggregation_mode != "null":
        raise ValueError(
            "DSPARK captured-context replay does not support disaggregated "
            f"serving; got role {disaggregation_mode!r}."
        )
    return replay_tokens


def cache_event_to_payload(event) -> dict:
    """A cache ACK as the payload the host's data plane reports
    (``HostDataPlane.poll_cache_events``).

    ``(kind, op_id)`` identifies a completed Host copy.
    """
    # 译：一个 cache ACK，表示为 Host 的执行面所报告的负载（``HostDataPlane.poll_cache_events``）。
    #
    #     ``(kind, op_id)`` 标识一次已完成的 Host 拷贝。
    kind = type(event).__name__
    if kind not in _CACHE_EVENT_TYPES:
        raise ValueError(f"Unsupported cache event type: {kind}")
    payload = {
        "kind": kind,
        "op_id": int(event.op_id),
    }
    return payload


def _check_batch_rows(op, columns: Sequence[Sequence]) -> int:
    """The row count of a wire batch whose per-op columns must all agree."""
    # 译：一个 wire 批次的行数，它的各个按操作的列必须全部一致。
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError(
            f"ragged {type(op).__name__} batch: column lengths {sorted(lengths)}"
        )
    return lengths.pop()


def _wire_transfers(
    op_id: int,
    groups: Sequence[int],
    sources: Sequence[int],
    destinations: Sequence[int],
    hashes: Sequence[str] | None,
    offsets: Sequence[int] | None,
) -> tuple[CacheTransfer, ...]:
    """One op's rows of a wire batch as ``CacheTransfer`` values."""
    # 译：把 wire 批次中一个操作的各行转换为 ``CacheTransfer`` 值。
    if hashes is None:
        hashes = [""] * len(groups)
        offsets = [0] * len(groups)
    if not (
        len(groups) == len(sources) == len(destinations) == len(hashes) == len(offsets)
    ):
        raise ValueError(f"ragged cache operation {op_id}")
    return tuple(
        CacheTransfer(
            group_id=int(group),
            source_page=int(source),
            destination_page=int(destination),
            content_hash=str(content_hash),
            page_offset=int(page_offset),
        )
        for group, source, destination, content_hash, page_offset in zip(
            groups, sources, destinations, hashes, offsets
        )
    )


def _snapshot_batch_columns(op) -> tuple[Sequence, ...]:
    return (
        op.op_ids,
        op.request_ids,
        op.request_pool_indices,
        op.snapshot_slots,
        op.group_ids,
        op.src_pages,
        op.dst_pages,
    )


def snapshot_ops_from_wire(op) -> list[SnapshotOp]:
    """One ``SnapshotOp`` per request of a ``Cache.SnapshotOp`` batch."""
    # 译：``Cache.SnapshotOp`` 批次中每个请求一个 ``SnapshotOp``。
    rows = _check_batch_rows(op, _snapshot_batch_columns(op))
    return [
        SnapshotOp(
            op_id=int(op.op_ids[i]),
            request_id=str(op.request_ids[i]),
            request_pool_index=int(op.request_pool_indices[i]),
            snapshot_slot=int(op.snapshot_slots[i]),
            transfers=_wire_transfers(
                int(op.op_ids[i]),
                op.group_ids[i],
                op.src_pages[i],
                op.dst_pages[i],
                None,
                None,
            ),
        )
        for i in range(rows)
    ]


def restore_ops_from_wire(op) -> list[RestoreOp]:
    """One ``RestoreOp`` per request of a ``Cache.RestoreOp`` batch."""
    # 译：``Cache.RestoreOp`` 批次中每个请求一个 ``RestoreOp``。
    rows = _check_batch_rows(
        op,
        _snapshot_batch_columns(op)
        + (op.content_hashes, op.page_offsets, op.source_tiers),
    )
    ops = []
    for i in range(rows):
        transfers = _wire_transfers(
            int(op.op_ids[i]),
            op.group_ids[i],
            op.src_pages[i],
            op.dst_pages[i],
            op.content_hashes[i],
            op.page_offsets[i],
        )
        tiers = tuple(HostTier(int(tier)) for tier in op.source_tiers[i])
        if len(tiers) != len(transfers):
            raise ValueError(f"ragged cache operation {op.op_ids[i]}")
        ops.append(
            RestoreOp(
                op_id=int(op.op_ids[i]),
                request_id=str(op.request_ids[i]),
                request_pool_index=int(op.request_pool_indices[i]),
                snapshot_slot=int(op.snapshot_slots[i]),
                transfers=transfers,
                source_tier=tiers,
            )
        )
    return ops


def is_device_cache_op(operation) -> bool:
    """Whether an adapted operation belongs to the GPU copy executor."""
    # 译：一个转换后的操作是否属于 GPU 拷贝执行器。
    return isinstance(
        operation, (Cache.WriteBackOp, Cache.LoadBackOp, SnapshotOp, RestoreOp)
    )


class CacheOperations(NamedTuple):
    """One validated plan snapshot, partitioned by its execution owner.

    Each tuple keeps the plan's order. It defines no ordering between Host
    storage I/O and Device copies; tickets and completions express dependencies.
    """

    # 译：一个经过校验的计划快照，按其执行者划分。
    #
    #     每个元组保持计划中的顺序。它不在 Host 存储 I/O 与设备拷贝之间定义任何顺序；依赖关系由票据和完成通知表达。

    device: tuple[Cache.WriteBackOp | Cache.LoadBackOp | SnapshotOp | RestoreOp, ...]
    storage: tuple[
        Cache.StorageQueryOp
        | Cache.StorageGetOp
        | Cache.StoragePutOp
        | Cache.StorageCancelOp,
        ...,
    ]


def cache_ops_from_plan(execution_plan) -> CacheOperations:
    """Read and validate every cache op once before any owner submits work.

    L2 wire batches pass through; snapshot/restore batches become per-request
    operations. Host-only QUERY/GET/PUT/Cancel descriptors go to the storage
    tuple: the engine host's L3 executor runs them. Unknown kinds fail the
    whole conversion without partial submission.
    """
    # 译：在任何执行者提交工作之前，读取并校验每个 cache 操作一次。
    #
    #     L2 的 wire 批次原样通过；快照 / 恢复批次变成按请求的操作。只在 Host 上的 QUERY/GET/PUT/Cancel 描述符放进
    #     storage 元组：由引擎 Host 的 L3 执行器运行它们。未知的类型会让整个转换失败，不做部分提交。
    device: list = []
    storage: list = []
    for op in execution_plan.cache:
        if isinstance(op, (Cache.WriteBackOp, Cache.LoadBackOp)):
            device.append(op)
        elif isinstance(op, Cache.SnapshotOp):
            device.extend(snapshot_ops_from_wire(op))
        elif isinstance(op, Cache.RestoreOp):
            device.extend(restore_ops_from_wire(op))
        elif isinstance(
            op,
            (
                Cache.StorageQueryOp,
                Cache.StorageGetOp,
                Cache.StoragePutOp,
                Cache.StorageCancelOp,
            ),
        ):
            storage.append(op)
        else:
            raise TypeError(f"unsupported cache op kind: {type(op).__name__}")
    return CacheOperations(tuple(device), tuple(storage))


# 下面是主线每次前向都要调用的部分：把调度器给出的每个 cache group 的 block table 上传到 GPU。
#
# block table 的格式（每个 cache group 一张 [请求数, 列数] 的 int32 表）：
#   每个值是绝对页号；0 表示空洞页（例如滑动窗口已经释放的前部页，它们被导向保留的第 0 页）；
#   行尾不足的部分用 -1 填充（各请求的页数不同，表要补成矩形）
#   例（Qwen3，P=64，一个全历史组）：A 300 个 token 占 5 页，B 1001 个 token 占 16 页，列数 16
#     A 的行：[12,  3, 40,  7, 21, -1, -1, ..., -1]
#     B 的行：[ 9,  2, 31, ..., (共 16 个页号)]
#
# 上传的方式：调度器只给出每行中仍然有效的那一段（窗口）以及它在整行中的范围（spans），
# 所有组的窗口拼在一起，经一次 pinned 内存拷贝传到 GPU，再由一个 kernel 展开成完整的表
# （窗口前面补 0、后面补 -1），所有组的表共用同一块显存。
class PackedBlockTables(NamedTuple):
    """One batch's per-group block tables, uploaded once.

    Attributes:
        tables: Per-group int32 views into the one device storage.
        tables_cpu: The same tables as views into one pinned host copy, for
            planning that must not wait on the device; empty unless asked for.
    """

    # 译：一个 batch 的按组 block table，只上传一次。
    #
    #     属性：
    #         tables：各组的 int32 视图，指向同一块设备存储。
    #         tables_cpu：同样的表，作为指向同一份 pinned Host 副本的视图，供不能等待设备的规划使用；除非要求，否则为空。

    tables: dict[str, torch.Tensor]
    tables_cpu: dict[str, torch.Tensor]


def block_tables_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    num_reqs: int | None = None,
    expected_group_ids: tuple[str, ...] | None = None,
    max_page_id: int | None = None,
    max_page_ids: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """The device tables of :func:`packed_block_tables_from_forward_op`."""
    # 译：:func:`packed_block_tables_from_forward_op` 的设备端表。
    return packed_block_tables_from_forward_op(
        forward_op,
        device,
        host_tables=False,
        num_reqs=num_reqs,
        expected_group_ids=expected_group_ids,
        max_page_id=max_page_id,
        max_page_ids=max_page_ids,
    ).tables


def packed_block_tables_from_forward_op(
    forward_op: Any,
    device: "torch.device | str",
    *,
    host_tables: bool,
    num_reqs: int | None = None,
    expected_group_ids: tuple[str, ...] | None = None,
    max_page_id: int | None = None,
    max_page_ids: Mapping[str, int] | None = None,
) -> PackedBlockTables:
    """Bridge the per-group block tables to GPU int32 tensors: absolute
    page indices, null hole = 0 preserved, ragged-row padding -1. No
    base-offset companion -- the cache path never compacts.

    All groups ride ONE H2D copy and one kernel rebuilds them into ONE device
    storage; the returned per-group views share it, which is the
    precondition of the routers' one-launch fill of every group
    (``GroupTableStacks.fill``). Per-group uploads would fail its
    same-storage check and fall back to one launch per group. The copy
    carries the scheduler's compact row windows (``block_table_windows``)
    and their absolute spans (``block_table_spans``). The expansion fills
    omitted columns with null holes before the window and padding after it.
    With ``host_tables`` the tables are also rebuilt in pinned host memory,
    so a backend planning an extend from them reads them without a D2H sync.

    Structural contract checks run whenever bounds or expected groups are
    supplied. Every page ID is scanned against the bounds, and every row
    against its span, when they are supplied; at long contexts the scan
    costs more than the packing, so callers pass bounds only when they mean
    to scan (``CacheBatchMetadata.from_forward_op``'s ``validate_page_ids``).

    Args:
        forward_op: Scheduler forward operation exporting CPU NumPy row windows,
            absolute column counts, and row spans.
        device: Destination device for the packed tensor.
        host_tables: Also return the tables in host memory (``tables_cpu``).
        num_reqs: Optional expected row count for every group.
        expected_group_ids: Optional pack order and exact key set; the
            packed buffer and the returned views follow it.
        max_page_id: Optional common inclusive upper bound for page IDs.
        max_page_ids: Optional per-group inclusive upper bounds.

    Returns:
        Device and (with ``host_tables``) host per-group tensor views in
        ``expected_group_ids`` order when supplied, otherwise preserving
        producer order.

    Raises:
        ValueError: If strict contract validation or a row span fails before
            device transfer.
    """
    # 译：把按组的 block table 搬到 GPU 上的 int32 张量：绝对页号，保留空洞 = 0，不齐的行用 -1 填充。没有基址偏移的伴随量——
    #     cache 路径从不压缩。
    #
    #     所有组经由一次 H2D 拷贝，一个 kernel 把它们重建到同一块设备存储中；返回的按组视图共享这块存储，这是 router
    #     一次启动填充所有组（``GroupTableStacks.fill``）的前提。按组分别上传会通不过它的同一存储检查，退化为每组一次启动。
    #     这次拷贝携带调度器紧凑的行窗口（``block_table_windows``）及其绝对范围（``block_table_spans``）。展开时在窗口之前
    #     补空洞，在窗口之后补填充。开启 ``host_tables`` 时，这些表也会在 pinned Host 内存中重建，这样根据它们规划 extend
    #     的后端读取时不需要 D2H 同步。
    #
    #     只要提供了边界或期望的组，就运行结构上的契约检查。提供边界时，每个页 ID 都会对照边界扫描，每一行都会对照其范围扫描；
    #     在长上下文下扫描的开销超过打包本身，所以调用方只在确实要扫描时才传入边界（``CacheBatchMetadata.from_forward_op``
    #     的 ``validate_page_ids``）。
    #
    #     参数：
    #         forward_op：调度器的前向操作，导出 CPU 上的 NumPy 行窗口、绝对列数和行范围。
    #         device：打包后张量的目标设备。
    #         host_tables：同时返回 Host 内存中的表（``tables_cpu``）。
    #         num_reqs：可选，每个组期望的行数。
    #         expected_group_ids：可选，打包顺序以及确切的键集合；打包缓冲区和返回的视图都遵循它。
    #         max_page_id：可选，页 ID 的公共上界（含）。
    #         max_page_ids：可选，每个组的上界（含）。
    #
    #     返回：
    #         设备端以及（开启 ``host_tables`` 时）Host 端的按组张量视图；提供了 ``expected_group_ids`` 时按其顺序，
    #         否则保持生产者的顺序。
    #
    #     异常：
    #         ValueError：严格的契约校验或某一行的范围在设备传输之前失败。
    if max_page_id is not None and max_page_ids is not None:
        raise ValueError("pass max_page_id or max_page_ids, not both")
    strict_validation = (
        expected_group_ids is not None
        or max_page_id is not None
        or max_page_ids is not None
    )
    if strict_validation and num_reqs is not None:
        require_positive_int("num_reqs", num_reqs)
    if expected_group_ids is not None:
        seen_group_ids: set[str] = set()
        duplicate_group_ids: set[str] = set()
        for group_id in expected_group_ids:
            if group_id in seen_group_ids:
                duplicate_group_ids.add(group_id)
            seen_group_ids.add(group_id)
        if duplicate_group_ids:
            raise ValueError(
                f"expected_group_ids contains duplicates: {sorted(duplicate_group_ids)}"
            )
    window_items = [
        (str(key), array, columns)
        for key, (array, columns) in forward_op.block_table_windows().items()
    ]
    spans_by_id = {
        str(key): spans for key, spans in forward_op.block_table_spans().items()
    }
    if strict_validation:
        normalized_group_ids: set[str] = set()
        collided_ids: set[str] = set()
        for group_id, _, _ in window_items:
            if group_id in normalized_group_ids:
                collided_ids.add(group_id)
            normalized_group_ids.add(group_id)
        if collided_ids:
            raise ValueError(
                "cache group keys collide after string normalization: "
                f"{sorted(collided_ids)}"
            )
    if expected_group_ids is not None:
        windows_by_id = {item[0]: item for item in window_items}
        actual = set(windows_by_id)
        expected = set(expected_group_ids)
        if actual != expected:
            raise ValueError(
                f"cache group keys disagree: missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}"
            )
        ordered_items = [windows_by_id[group_id] for group_id in expected_group_ids]
    else:
        ordered_items = window_items
    if strict_validation:
        for group_id, arr, _ in ordered_items:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"cache group {group_id!r} must be a NumPy array")
            if arr.dtype != np.int32:
                raise ValueError(f"cache group {group_id!r} must use int32")
            if arr.ndim != 2:
                raise ValueError(f"cache group {group_id!r} has invalid shape")
            if arr.shape[0] == 0:
                raise ValueError(f"cache group {group_id!r} has zero rows")
            # rows-vs-num_reqs is checked once in the packing loop below.
            # 译：行数与 num_reqs 的比较在下面的打包循环中做一次。
            group_max_page_id = (
                max_page_ids.get(group_id) if max_page_ids is not None else max_page_id
            )
            if max_page_ids is not None and group_max_page_id is None:
                raise ValueError(f"max_page_ids is missing cache group {group_id!r}")
            if group_max_page_id is not None:
                if group_max_page_id < -1:
                    raise ValueError(
                        f"cache group {group_id!r} has an invalid page ID bound"
                    )
                invalid = (arr < -1) | (arr > group_max_page_id)
                if bool(invalid.any()):
                    raise ValueError(
                        f"cache group {group_id!r} contains a page ID outside "
                        f"-1..{group_max_page_id}"
                    )
    device = torch.device(device) if isinstance(device, str) else device
    out: dict[str, torch.Tensor] = {}
    out_cpu: dict[str, torch.Tensor] = {}
    # (key, window, absolute columns, offset in packed storage, row starts)
    # 译：(key, 窗口, 绝对列数, 在打包存储中的偏移, 各行起点)
    packable: list[tuple[str, np.ndarray, int, int, np.ndarray]] = []
    empty_groups: list[tuple[str, int]] = []
    total = 0
    for key, arr, columns in ordered_items:
        if num_reqs is not None and arr.shape[0] != num_reqs:
            raise ValueError(
                f"block_table_windows[{key}] has {arr.shape[0]} rows "
                f"but forward op reported num_reqs={num_reqs}"
            )
        if arr.shape[0] == 0:
            continue
        if type(columns) is not int or columns < 0 or arr.shape[1] > columns:
            raise ValueError(f"cache group {key!r} has invalid absolute columns")
        if strict_validation and columns == 0:
            raise ValueError(f"cache group {key!r} has zero width")
        if total + arr.shape[0] * columns >= 1 << 31:
            raise ValueError("block tables beyond 2**31 entries need int64 row fields")
        spans = spans_by_id.get(key)
        if (
            not isinstance(spans, np.ndarray)
            or spans.dtype != np.int32
            or spans.shape != (arr.shape[0], 2)
        ):
            raise ValueError(f"block_table_spans[{key}] must be int32 [rows, 2]")
        # spans 的每一行是 (released, pages)：前 released 列已释放（为空洞 0），共有 pages 列有效；
        # 窗口 arr 只包含其中最后 width 列，first 是窗口在整行中的起始列
        released, pages = spans[:, 0], spans[:, 1]
        if bool(((released < 0) | (released > pages) | (pages > columns)).any()):
            raise ValueError(f"block_table_spans[{key}] exceeds its table")
        width = arr.shape[1]
        if bool(((pages - released) > width).any()):
            raise ValueError(f"block_table_windows[{key}] omits live entries")
        first = np.maximum(pages.astype(np.int64) - width, 0)
        if max_page_id is not None or max_page_ids is not None:
            maximum = max_page_ids[key] if max_page_ids is not None else max_page_id
            if maximum < 0 and bool((released > 0).any()):
                raise ValueError(
                    f"cache group {key!r} contains a page ID outside -1..{maximum}"
                )
            column = first[:, None] + np.arange(width)[None, :]
            if bool(
                (arr[column < released[:, None]] != 0).any()
                or (arr[column >= pages[:, None]] != -1).any()
            ):
                raise ValueError(
                    f"cache group {key!r} holds entries outside its row spans"
                )
        if columns == 0:
            # Kept out of the pack: CacheBatchMetadata rejects a zero-width table.
            # 译：不放进打包：CacheBatchMetadata 会拒绝宽度为零的表。
            empty_groups.append((key, arr.shape[0]))
            continue
        packable.append((key, arr, columns, total, first))
        total += arr.shape[0] * columns
    for key, rows in empty_groups:
        out[key] = torch.empty((rows, 0), dtype=torch.int32, device=device)
        if host_tables:
            out_cpu[key] = torch.empty((rows, 0), dtype=torch.int32)
    if not packable:
        return PackedBlockTables(out, out_cpu)
    from tokenspeed_kernel.ops.metadata import (
        BLOCK_TABLE_ROW_FIELDS,
        expand_block_table_rows,
    )

    num_rows = sum(arr.shape[0] for _, arr, _, _, _ in packable)
    fields = num_rows * BLOCK_TABLE_ROW_FIELDS
    num_values = sum(arr.size for _, arr, _, _, _ in packable)
    # Fresh pinned stages per step (event-fenced; reuse races overlap).
    # arr is a read-only zero-copy view over the C++ buffer; np.copyto
    # reads it into our own writable pinned tensors (never writes back).
    # 译：每一步使用新的 pinned 暂存区（由 event 隔离；复用会与重叠的执行产生竞争）。
    #     arr 是 C++ 缓冲区上只读的零拷贝视图；np.copyto 把它读进我们自己的、可写的 pinned 张量（从不写回）。
    pin_memory = device.type in ("cuda", "npu")
    upload = torch.empty(fields + num_values, dtype=torch.int32, pin_memory=pin_memory)
    upload_np = upload.numpy()
    row_fields = upload_np[:fields].reshape(num_rows, BLOCK_TABLE_ROW_FIELDS)
    values = upload_np[fields:]
    row_at = value_at = 0
    for _, arr, columns, offset, first in packable:
        rows, width = arr.shape
        # (offset, width, start, end, source): expand_block_table_rows' fields.
        # 译：(offset, width, start, end, source)：expand_block_table_rows 的各字段。
        group_fields = row_fields[row_at : row_at + rows]
        group_fields[:, 0] = offset + np.arange(rows) * columns
        group_fields[:, 1] = columns
        group_fields[:, 2] = first
        group_fields[:, 3] = first + width
        group_fields[:, 4] = value_at + np.arange(rows) * width
        windows = values[value_at : value_at + rows * width].reshape(rows, width)
        np.copyto(windows, arr)
        row_at += rows
        value_at += rows * width
    # 一次异步 H2D 拷贝，然后在 GPU 上由 expand_block_table_rows 把所有组的窗口展开成完整的表
    upload = upload.to(device, non_blocking=True)
    packed = torch.empty(total, dtype=torch.int32, device=device)
    expand_block_table_rows(
        upload[:fields].view(num_rows, BLOCK_TABLE_ROW_FIELDS), upload[fields:], packed
    )
    for key, arr, columns, offset, _ in packable:
        shape = (arr.shape[0], columns)
        size = shape[0] * shape[1]
        out[key] = packed[offset : offset + size].view(shape)
    # 需要 Host 副本时（有 extend 的批次），在 CPU 上做同样的展开
    if host_tables:
        staged = torch.empty(total, dtype=torch.int32, pin_memory=pin_memory)
        staged_np = staged.numpy()
        for key, arr, columns, offset, first in packable:
            rows, width = arr.shape
            shape = (rows, columns)
            size = shape[0] * shape[1]
            host = staged_np[offset : offset + size].reshape(shape)
            if width == shape[1]:
                np.copyto(host, arr)
            else:
                for row, start in enumerate(first):
                    host[row, :start] = 0
                    host[row, start : start + width] = arr[row]
                    host[row, start + width :] = -1
            out_cpu[key] = staged[offset : offset + size].view(shape)
    return PackedBlockTables(out, out_cpu)


# 以下是启动后打印的显存使用汇总（按权重类别、KV cache、激活与 graph 等分项），与推理主线无关
def _classify_param(name: str) -> str:
    """Bucket a parameter/buffer name into a weight group for the memory
    summary. Names follow the Kimi-K3 / DeepSeek module layout."""
    # 译：把一个参数 / buffer 的名字归入显存汇总中的某一类权重。名字遵循 Kimi-K3 / DeepSeek 的模块布局。
    if ".engram." in name:
        return "engram_weights"
    if "self_attn" in name or ".attn." in name or "kv_a" in name or "q_a" in name:
        return "attention_weights"
    if (
        "experts" in name
        or "block_sparse_moe" in name
        or "routed_expert" in name
        or "gate" in name
    ) and "shared_experts" not in name:
        return "moe_weights"
    if (
        "mlp" in name
        or "shared_experts" in name
        or "down_proj" in name
        or "up_proj" in name
        or "gate_proj" in name
    ):
        return "dense_mlp_weights"
    return "other_weights"


def _kv_pool_bytes(*pools) -> int:
    """Best-effort total KV-buffer bytes across the given pools, deduped.

    Target and draft are two compute views of the ONE arena (draft KV lives
    in the target's arena), so counting both naively would double the KV.
    Dedupe by arena identity -- the allocation each view reports on.

    Arena-backed pools report actual device allocations, including residency
    workspace and excluding authoritative host history. Legacy pools may
    report nested numeric totals.
    """
    # 译：给定的这些池的 KV 缓冲区总字节数，尽力而为、去重。
    #
    #     目标和草稿是同一个 arena 的两个计算视图（草稿的 KV 位于目标的 arena 中），所以简单地两者都计会把 KV 算两遍。
    #     按 arena 的身份去重——即每个视图所报告的那个分配。
    #
    #     基于 arena 的池报告实际的设备分配，包括驻留工作区，不包括权威的 Host 历史。旧式的池可能报告嵌套的数值总和。
    from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena

    seen: set[int] = set()
    total = 0
    for pool in pools:
        if pool is None:
            continue
        arena = getattr(pool, "arena", pool)
        if id(arena) in seen:
            continue
        seen.add(id(arena))
        if isinstance(arena, CacheArena):
            total += arena.allocated_device_bytes
            continue
        getter = getattr(pool, "get_kv_size_bytes", None)
        if getter is None:
            continue
        try:
            result = getter()
        except Exception:
            continue

        def _sum(x):
            if isinstance(x, (int, float)):
                return int(x)
            if isinstance(x, (tuple, list)):
                return sum(_sum(e) for e in x)
            return 0

        total += _sum(result)
    return total


def log_gpu_memory_summary(
    model,
    gpu_id: int,
    rank: int,
    logger,
    draft_model=None,
    kv_pool=None,
    draft_kv_pool=None,
    device: str = "cuda",
) -> None:
    """Log a per-rank accelerator memory breakdown after cache allocation.

    Weight groups are summed from the model's parameters and buffers (deduped
    by storage pointer). A draft model (speculative decoding) is summed
    separately into its own row. KV cache, graph allocations, and non-torch
    allocations are derived from the selected device's allocator and driver
    views. Best-effort: never raises into startup.
    """
    # 译：在分配 cache 之后，记录每个 rank 的加速器显存明细。
    #
    #     权重分组由模型的参数和 buffer 求和得出（按存储指针去重）。草稿模型（投机解码）单独求和成自己的一行。KV cache、
    #     graph 的分配以及非 torch 的分配从所选设备的分配器和驱动的视角推导。尽力而为：从不把异常抛进启动流程。
    try:
        GB = 1024**3
        device_type = torch.device(device).type
        device_module = torch.get_device_module(device_type)
        groups = {
            "attention_weights": 0,
            "moe_weights": 0,
            "dense_mlp_weights": 0,
            "engram_weights": 0,
            "other_weights": 0,
        }
        seen: set[int] = set()

        def _accumulate(name, tensor, sink):
            if tensor is None or tensor.device.type != device_type:
                return
            ptr = tensor.data_ptr()
            if ptr in seen:
                return
            seen.add(ptr)
            sink[_classify_param(name)] += tensor.numel() * tensor.element_size()

        for n, p in model.named_parameters():
            _accumulate(n, p, groups)
        for n, b in model.named_buffers():
            _accumulate(n, b, groups)

        weights_total = sum(groups.values()) / GB

        # Draft model (MTP / DSpark) weights, deduped against target: shared
        # tensors (e.g. embed/lm_head lent from the target) are already in
        # ``seen`` and won't be double-counted.
        # 译：草稿模型（MTP / DSpark）的权重，与目标模型去重：共享的张量（例如从目标模型借来的 embed/lm_head）已经在
        #     ``seen`` 中，不会被重复计算。
        draft_total = 0
        if draft_model is not None:
            draft_sink = {k: 0 for k in groups}
            for n, p in draft_model.named_parameters():
                _accumulate(n, p, draft_sink)
            for n, b in draft_model.named_buffers():
                _accumulate(n, b, draft_sink)
            draft_total = sum(draft_sink.values())
        draft_gb = draft_total / GB

        free_bytes, total_bytes = device_module.mem_get_info(gpu_id)
        allocated = device_module.memory_allocated(gpu_id) / GB
        reserved = device_module.memory_reserved(gpu_id) / GB
        device_total = total_bytes / GB
        device_free = free_bytes / GB
        device_used = device_total - device_free
        # KV pool bytes read straight from the pool buffers. Draft KV lives in
        # the target's merged arena (its pool is a view), so dedupe to count once.
        # 译：直接从池的缓冲区读取 KV 池的字节数。草稿的 KV 位于目标模型合并后的 arena 中（它的池是一个视图），所以去重后只计一次。
        kv_cache_gb = _kv_pool_bytes(kv_pool, draft_kv_pool) / GB
        # Allocated beyond the classified target+draft weights and the KV pool is
        # activations + captured-graph private pools; non-torch is
        # context/NCCL/DeepEP.
        # 译：超出已分类的目标 + 草稿权重以及 KV 池的分配量，就是激活值 + 已捕获 graph 的私有池；非 torch 的部分是
        #     上下文/NCCL/DeepEP。
        activations_and_graphs = max(
            0.0, allocated - weights_total - draft_gb - kv_cache_gb
        )
        non_torch = max(0.0, device_used - reserved)

        rows = [
            ("Attention weights", groups["attention_weights"] / GB),
            ("MoE weights", groups["moe_weights"] / GB),
            ("Dense/MLP weights", groups["dense_mlp_weights"] / GB),
            ("Other weights (embed/head/norm)", groups["other_weights"] / GB),
        ]
        # Engram tables are only listed when resident on the device; with
        # --engram-host-table they live in host memory and are not counted.
        # 译：Engram 表只在常驻于设备上时列出；使用 --engram-host-table 时它们在 Host 内存中，不计入。
        if groups["engram_weights"]:
            rows.append(("Engram weights (tables/wkv)", groups["engram_weights"] / GB))
        if draft_model is not None:
            rows.append(("Draft model weights", draft_gb))
        rows += [
            ("KV cache", kv_cache_gb),
            ("Activations + device graphs", activations_and_graphs),
            ("Torch allocated (total)", allocated),
            ("Torch reserved (allocator pool)", reserved),
            ("Non-torch (context/collectives)", non_torch),
            ("Device used (driver view)", device_used),
            ("Device free", device_free),
            ("Device total", device_total),
        ]
        name_width = max(len(n) for n, _ in rows)
        sep = "+" + "-" * (name_width + 2) + "+" + "-" * 12 + "+"
        lines = [
            f"Device memory summary (rank {rank}, {device_type} {gpu_id}, GB):",
            sep,
            f"| {'Component'.ljust(name_width)} | {'GB'.rjust(10)} |",
            sep,
        ]
        for n, v in rows:
            lines.append(f"| {n.ljust(name_width)} | {v:10.2f} |")
        lines.append(sep)
        logger.info("\n".join(lines))
    except Exception:
        logger.warning("Failed to log GPU memory summary", exc_info=True)
