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

# KV 池（CachePool）的抽象基类，以及从 cache 计划中查找“哪一层的 KV 存在哪个 cache group”的辅助函数。
#
# 你在这里：注意力 prologue 写 KV 时通过 kv_write_target 取得写入目标；注意力后端读 KV 时通过
# get_key_buffer / get_value_buffer 取得本层的 KV 缓冲区（layers/paged_attention.py、backends/paged/mha.py）。
#
# 注意：KV cache 不是一个按 [2, L, 页数, 页大小, Hkv, D] 预分配的大张量。三层关系：
#   arena（CacheArena，kv_cache/arena.py）：一块（或几块）uint8 字节缓冲区，是唯一真正持有显存的对象
#   计划（CacheMemoryPlan，recipes/plan.py）：描述这块字节里有哪些“字段”（field），每个字段属于哪一层、
#     哪个 cache group、什么形状和 dtype、从哪个偏移开始、每页跨多少字节
#   KV 池（CachePool，本文件）：不持有显存，只是“某个模型的若干层”在 arena 上的视图，
#     把计划中属于这些层的字段整理成 kernel 方便索引的按层列表（k_buffer[layer]、v_buffer[layer]……）
# 这样目标模型和草稿模型（投机解码）可以是同一个 arena 上的两个 KV 池，甚至用不同的 dtype 解读同一块字节。
#
# 字段 id 的格式是 "layer.{层号}.{平面名}"，例如 "layer.3.k"、"layer.3.v"（层号是合并计划中的全局层号）。
#
# Qwen3-8B 只有一个全历史 cache group，每层两个字段 k、v，所以 KV 池里就是 36 个 K 缓冲区和 36 个 V 缓冲区，
# 每个形状为 [NP*P, 8, 128]（详见 kv_cache/mha.py）。
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

import torch
from tokenspeed_kernel.ops.attention.prologue import HeadKVCache, LatentKVCache

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
    cache_field_plane,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker

logger = get_colorful_logger(__name__)


# 把字段 id 拆成 (本视图内的局部层号, 平面名)；不在本视图层窗口内的字段返回 None。
# 例：草稿模型的视图从全局第 36 层开始（first_layer=36），"layer.37.k" -> (1, "k")
def _layer_plane(
    field_id: str, first_layer: int, num_layers: int
) -> tuple[int, str] | None:
    """Split a planned field id into this view's local layer id and plane.

    Returns None for fields outside the view's layer window.
    """
    # 译：把计划中的字段 id 拆成本视图的局部层号和平面名。
    #
    #     对于不在本视图层窗口内的字段返回 None。
    local_layer = cache_field_layer_id(field_id) - first_layer
    if not 0 <= local_layer < num_layers:
        return None
    return local_layer, cache_field_plane(field_id)


# 下面三个函数都从计划中“读回”层与 cache group 的对应关系（计划是唯一记录，不另外维护一份按层的表）：
#   derive_state_groups_by_layer：递归状态层（线性注意力）-> 其状态所在的 state 组（第 13 阶段）
#   derive_history_groups_by_layer：注意力层 -> 其 KV 所在的历史组（全历史或滑动窗口）
#   derive_paged_group_ids：本视图的层用到了哪些历史组；CacheGroupRouter 为每个组建一个叶子后端
# Qwen3-8B：36 层都映射到同一个全历史组。
def derive_state_groups_by_layer(
    arena: CacheArena,
    *,
    first_layer: int,
    num_layers: int,
    state_layer_ids: Iterable[int],
    state_field_suffixes: Iterable[str],
) -> dict[int, str]:
    """Map each recurrent layer to the state group holding its fields.

    The memory plan is the single record of which group a layer's fields were
    declared in, so the mapping is read back from the planned fields rather
    than carried as a parallel per-layer tuple.

    Args:
        arena: The cache arena whose plan and group specs to read.
        first_layer: This view's first layer in the merged plan.
        num_layers: Number of layers in this view's window.
        state_layer_ids: View-local ids of the recurrent (state) layers.
        state_field_suffixes: Field ID suffixes that make up the pool's
            recurrent state.

    Returns:
        View-local layer id -> state group id, one entry per state
        layer whose fields the plan declares inside this view's window.

    Raises:
        ValueError: a state layer's fields span more than one state group.
    """
    # 译：把每个递归层映射到持有其字段的 state 组。
    #
    #     内存计划是“一个层的字段声明在哪个组里”的唯一记录，所以这个映射从计划的字段中读回，而不是作为一个并行的
    #     按层元组另外携带。
    #
    #     参数：
    #         arena：要读取其计划和组 spec 的 cache arena。
    #         first_layer：本视图在合并计划中的第一层。
    #         num_layers：本视图窗口中的层数。
    #         state_layer_ids：递归（state）层在本视图中的局部 id。
    #         state_field_suffixes：组成该池递归状态的字段 ID 后缀。
    #
    #     返回：
    #         视图局部层号 -> state 组 id，计划在本视图窗口内声明了其字段的每个 state 层一项。
    #
    #     异常：
    #         ValueError：某个 state 层的字段跨越了不止一个 state 组。
    state_groups = {
        spec.group_id for spec in arena.cache_group_specs if spec.attn_kind == "state"
    }
    wanted = set(state_layer_ids)
    wanted_suffixes = set(state_field_suffixes)
    mapping: dict[int, str] = {}
    for field in arena.plan.fields:
        located = _layer_plane(field.field_id, first_layer, num_layers)
        if located is None:
            continue
        layer_id, suffix = located
        if layer_id not in wanted or suffix not in wanted_suffixes:
            continue
        if field.group_id not in state_groups:
            continue
        existing = mapping.setdefault(layer_id, field.group_id)
        if existing != field.group_id:
            raise ValueError(
                f"layer {layer_id} has state fields in more than one cache "
                f"group: {existing!r} and {field.group_id!r}"
            )
    return mapping


def derive_history_groups_by_layer(
    arena: CacheArena,
    *,
    first_layer: int,
    num_layers: int,
    kv_planes: Iterable[str],
) -> dict[int, str]:
    """Map each attention layer to the history group holding its KV.

    The counterpart of :func:`derive_state_groups_by_layer` for the paged
    side: read back from the planned fields, restricted to the planes this
    view's kernels write KV through (``layer_plane_bindings``), so a layer
    that also owns indexer or state planes in other groups still resolves to
    the one group its ``PagedAttention`` rides.

    Args:
        arena: The cache arena whose plan and group specs to read.
        first_layer: This view's first layer in the merged plan.
        num_layers: Number of layers in this view's window.
        kv_planes: Plane names of the KV fields (e.g. ``k``/``v`` or
            ``latent_kv``).

    Returns:
        View-local layer id -> history group id, one entry per layer
        whose KV planes the plan declares inside this view's window.

    Raises:
        ValueError: a layer's KV planes span more than one history group.
    """
    # 译：把每个注意力层映射到持有其 KV 的历史组。
    #
    #     这是 :func:`derive_state_groups_by_layer` 在分页一侧的对应函数：从计划的字段读回，只限于本视图的 kernel
    #     写 KV 所经过的平面（``layer_plane_bindings``），所以一个在其他组中还拥有 indexer 或 state 平面的层，
    #     仍然会解析到它的 ``PagedAttention`` 所使用的那一个组。
    #
    #     参数：
    #         arena：要读取其计划和组 spec 的 cache arena。
    #         first_layer：本视图在合并计划中的第一层。
    #         num_layers：本视图窗口中的层数。
    #         kv_planes：KV 字段的平面名（例如 ``k``/``v`` 或 ``latent_kv``）。
    #
    #     返回：
    #         视图局部层号 -> 历史组 id，计划在本视图窗口内声明了其 KV 平面的每个层一项。
    #
    #     异常：
    #         ValueError：某个层的 KV 平面跨越了不止一个历史组。
    history = {
        str(spec.group_id)
        for spec in arena.cache_group_specs
        if spec.attn_kind != "state"
    }
    wanted_planes = set(kv_planes)
    mapping: dict[int, str] = {}
    for field in arena.plan.fields:
        located = _layer_plane(field.field_id, first_layer, num_layers)
        if located is None:
            continue
        layer_id, plane = located
        if plane not in wanted_planes or str(field.group_id) not in history:
            continue
        existing = mapping.setdefault(layer_id, str(field.group_id))
        if existing != str(field.group_id):
            raise ValueError(
                f"layer {layer_id} has KV planes in more than one history "
                f"cache group: {existing!r} and {field.group_id!r}"
            )
    return mapping


def derive_paged_group_ids(
    arena: CacheArena, *, first_layer: int, num_layers: int
) -> tuple[str, ...]:
    """The history cache groups this view's layers deposit KV in.

    Read back from the planned fields like the state mapping above: a
    group counts when some per-layer field inside the view's layer window
    is declared in it. This is the group set a ``CacheGroupRouter`` builds
    one paged leaf for — a draft view over a shared arena sees only the
    groups its own layers use, never the target's whole set.

    Args:
        arena: The cache arena whose plan and group specs to read.
        first_layer: This view's first layer in the merged plan.
        num_layers: Number of layers in this view's window.

    Returns:
        Sorted group ids, possibly empty (a view with no paged layers).
    """
    # 译：本视图的层在其中存放 KV 的那些历史 cache group。
    #
    #     和上面的 state 映射一样从计划的字段读回：只要本视图层窗口内的某个按层字段声明在某个组中，这个组就算在内。
    #     这就是 ``CacheGroupRouter`` 为之各建一个分页叶子的组集合——共享 arena 上的草稿视图只看到它自己的层用到的组，
    #     绝不会看到目标模型的整个集合。
    #
    #     参数：
    #         arena：要读取其计划和组 spec 的 cache arena。
    #         first_layer：本视图在合并计划中的第一层。
    #         num_layers：本视图窗口中的层数。
    #
    #     返回：
    #         排好序的组 id，可能为空（没有分页层的视图）。
    history = {
        str(spec.group_id)
        for spec in arena.cache_group_specs
        if spec.attn_kind != "state"
    }
    found: set[str] = set()
    for field in arena.plan.fields:
        if _layer_plane(field.field_id, first_layer, num_layers) is None:
            continue
        if field.group_id in history:
            found.add(str(field.group_id))
    return tuple(sorted(found))


# KV 池基类。成员：
#   arena：底层的 CacheArena（显存、计划、调度器契约都归它）
#   dtype：kernel 读取这些字节时使用的 dtype（例如 bf16）
#   store_dtype：写入时使用的 dtype；FP8 时为 uint8（index_put 不支持 FP8）
#   _field_layer_offset：本视图的第 0 层在合并计划中是第几层（目标模型为 0，草稿模型接在后面）
#   layer_fields：(局部层号, 平面名) -> 字段 id
# 子类通过 layer_plane_bindings 声明“哪个平面放进哪个按层列表”，例如 MHA 的 {"k": "k_buffer", "v": "v_buffer"}。
class CachePool(ABC):
    """One model's typed layer window onto a shared cache arena.

    A pool owns no memory and no geometry: ``self.arena`` owns the
    allocation, the field views, the plan and the scheduler contract, and
    callers that want any of those ask the arena directly. What a pool
    adds is per-view: the dtype its kernels read these bytes as, where its
    layer window starts in the merged plan, and the per-layer buffers its
    kernels index. Target and draft are therefore two pools over one
    arena -- and may read it as different dtypes.
    """

    # 译：一个模型在共享 cache arena 上的、有类型的层窗口。
    #
    #     池不拥有显存，也不拥有几何：``self.arena`` 拥有分配、字段视图、计划和调度器契约，需要这些的调用方直接问 arena。
    #     池增加的是每个视图特有的东西：它的 kernel 把这些字节读作什么 dtype，它的层窗口在合并计划中从哪里开始，
    #     以及它的 kernel 索引的按层缓冲区。因此目标模型和草稿模型是同一个 arena 上的两个池——并且可以用不同的 dtype 读它。

    # Pools that alias recurrent-state bytes and KV in one buffer must
    # zero physical pages on reuse to avoid poisoned tails. Pure-attention
    # pools do not alias state, so reused pages need no sanitization.
    # 译：在一个缓冲区中同时存放递归状态字节和 KV 的池，在复用物理页时必须清零，以免读到被污染的尾部。
    #     纯注意力的池不与状态混用，所以复用的页不需要清理。
    requires_page_zeroing: bool = False

    # Whether every access a prefill graph replay makes to this view's layer
    # fields, in a captured segment or an eager break, goes through an
    # accessor that waits for the layer's Host load (``wait_for_layer`` on
    # ``layerwise_load_tracker``). Only then may a replay wait before each
    # captured segment for the layers its capture noted; otherwise it waits
    # for the whole Host load before its first segment. Set it only on a pool
    # whose accessors, and the code reaching them in a replay, were audited.
    # A subclass does not inherit it (see __init_subclass__).
    # 译：prefill graph 回放对本视图层字段的每一次访问（无论在捕获的段中还是在 eager 断点中）是否都经过一个会等待该层
    #     Host 加载的访问器（``layerwise_load_tracker`` 上的 ``wait_for_layer``）。只有这样，回放才可以在每个捕获段之前
    #     只等待该段捕获时记录的那些层；否则它要在第一个段之前等待整个 Host 加载完成。只在其访问器以及回放中到达这些
    #     访问器的代码都经过审查的池上设置它。子类不继承它（见 __init_subclass__）。
    layer_accesses_wait_for_load: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # The audit behind layer_accesses_wait_for_load covers one class: a
        # subclass adds or overrides accessors, so it declares it itself.
        # 译：layer_accesses_wait_for_load 背后的审查只覆盖一个类：子类会增加或覆盖访问器，所以要由它自己声明。
        if "layer_accesses_wait_for_load" not in cls.__dict__:
            cls.layer_accesses_wait_for_load = False

    def __init__(
        self,
        arena: CacheArena,
        dtype: torch.dtype,
        rank: int,
        *,
        field_layer_offset: int = 0,
    ):
        self.arena = arena
        self.dtype = dtype
        self.rank = rank
        self.layer_fields: dict[tuple[int, str], str] = {}
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            #  Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            # 译：存成 torch.uint8，因为 Tensor.index_put 没有为 torch.float8_e5m2 实现
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self._field_layer_offset = int(field_layer_offset)
        if self._field_layer_offset < 0:
            raise ValueError("field_layer_offset must be non-negative")
        # default state for optional layer-wise transfer control
        # 译：可选的逐层传输控制的默认状态
        # 逐层加载跟踪器：从 Host（L2 缓存）把前缀的 KV 读回 GPU 时，前向可以一层一层地等，
        # 而不必等全部加载完（第 14 阶段）。为 None 表示没有这种加载。
        self.layerwise_load_tracker = None
        logger.info(
            f"Initialized cache view over {arena.size:d} slots as {dtype!s}, layers "
            f"from {self._field_layer_offset:d}, rank {rank:d}",
        )

    @property
    def paged_group_ids(self) -> tuple[str, ...]:
        """History groups this view's layers write KV into (sorted);
        the router builds one paged leaf per id."""
        # 译：本视图的层写入 KV 的历史组（已排序）；router 为每个 id 构建一个分页叶子。
        return derive_paged_group_ids(
            self.arena,
            first_layer=self._field_layer_offset,
            num_layers=self.layer_num,
        )

    def history_group_by_layer(self) -> dict[int, str]:
        """View-local layer id -> the history group its KV planes
        ride; what ``bind_cache_groups`` stamps onto the model's
        ``PagedAttention`` layers."""
        # 译：视图局部层号 -> 其 KV 平面所在的历史组；``bind_cache_groups`` 把它标注到模型的 ``PagedAttention`` 层上。
        return derive_history_groups_by_layer(
            self.arena,
            first_layer=self._field_layer_offset,
            num_layers=self.layer_num,
            kv_planes=self.layer_plane_bindings,
        )

    def _field_layer_id(self, layer_id: int) -> int:
        """Map this compute view's local layer id into the merged plan.

        Callers pass the id their own model numbers the layer with, which for a
        draft view means ``0..num_draft_layers-1``. Reject anything outside the
        window: a global id offset a second time would silently address another
        model's planes.
        """
        # 译：把本计算视图的局部层号映射到合并计划中。
        #
        #     调用方传入的是它自己的模型给该层编的号，对草稿视图来说就是 ``0..num_draft_layers-1``。拒绝窗口之外的任何值：
        #     一个被第二次加上偏移的全局 id 会悄无声息地访问到另一个模型的平面。
        if not 0 <= layer_id < self.layer_num:
            raise ValueError(
                f"layer {layer_id} is outside this cache view's window of "
                f"{self.layer_num} layers (ids are local to the view)"
            )
        return self._field_layer_offset + layer_id

    @property
    def field_layer_range(self) -> range:
        """Global model-layer ids owned by this compute view.

        Cache-plan field ids use global layer numbers, while target and draft
        pools expose local layer numbers to their models.  Consumers that
        inspect the shared plan use this range to keep only fields belonging
        to their own view instead of accidentally including the adjacent
        target or draft window.
        """
        # 译：本计算视图拥有的全局模型层 id。
        #
        #     cache 计划的字段 id 使用全局层号，而目标模型和草稿模型的池向各自的模型暴露局部层号。检查共享计划的使用方
        #     用这个范围只保留属于自己视图的字段，避免意外包含相邻的目标模型或草稿模型窗口。

        return range(
            self._field_layer_offset,
            self._field_layer_offset + self.layer_num,
        )

    # Per-layer plane name -> the attribute holding its per-layer list.
    # Subclasses declare only the planes their kernels read; a plane they do
    # not name is not this view's concern.
    # 译：按层的平面名 -> 持有其按层列表的属性名。
    #     子类只声明它的 kernel 读取的平面；没有点名的平面与本视图无关。
    layer_plane_bindings: ClassVar[dict[str, str]] = {}

    # 启动时调用一次：遍历计划中的所有字段，把本视图层窗口内的字段放进对应的按层列表。
    # Qwen3-8B：self.k_buffer = [layer.0.k 的视图, ..., layer.35.k 的视图]，v_buffer 同理。
    # 没有对应字段的层（例如混合模型中的线性注意力层）在列表中为 None。
    def _bind_layer_planes(self) -> None:
        """Arrange this view's planned per-layer fields into kernel buffers.

        The plan names every field, its dtype, its shape and which layer it
        belongs to, and the arena already materialized every view in the
        shape it is addressed by. Walk the plan once, keep the fields inside
        this view's layer window, and file each under the attribute its
        kernels read.
        """
        # 译：把本视图计划中的按层字段整理成 kernel 使用的缓冲区。
        #
        #     计划给出了每个字段的名字、dtype、形状以及它属于哪一层，arena 也已经按每个视图被访问时的形状把它们实体化了。
        #     遍历一次计划，保留本视图层窗口内的字段，并把每个字段放到它的 kernel 读取的属性下。
        self.layer_fields.clear()
        lists = {
            attribute: [None] * self.layer_num
            for attribute in self.layer_plane_bindings.values()
        }
        for field in self.arena.plan.fields:
            located = _layer_plane(
                field.field_id, self._field_layer_offset, self.layer_num
            )
            if located is None:
                continue
            layer_id, plane = located
            self.layer_fields[(layer_id, plane)] = field.field_id
            attribute = self.layer_plane_bindings.get(plane)
            if attribute is None:
                continue
            lists[attribute][layer_id] = self.arena.compute_field(field.field_id)
        for attribute, values in lists.items():
            setattr(self, attribute, values)

    def register_layerwise_load_tracker(
        self, layerwise_load_tracker: LayerwiseLoadTracker
    ) -> None:
        self.layerwise_load_tracker = layerwise_load_tracker

    # Host 传输（L2 缓存、抢占快照，第 14 阶段）和 PD 传输（第 15 阶段）用的布局：本视图的字段在 arena 中的字节分布
    def cache_transfer_layout(self):
        """Return the transfer layout consumed by this compute view."""
        # 译：返回本计算视图使用的传输布局。
        from tokenspeed.runtime.cache.transfer.layout import (
            select_layer_fields,
        )

        try:
            field_ids, consumers = select_layer_fields(
                self.arena.plan.fields,
                first_layer=self._field_layer_offset,
                num_layers=self.layer_num,
            )
        except (AttributeError, IndexError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc
        return self._build_cache_transfer_layout(field_ids, consumers)

    def _build_cache_transfer_layout(self, field_ids, consumers):
        from tokenspeed.runtime.cache.transfer.layout import layout_from_lcm_plan

        if isinstance(self.arena, CacheArena) and self.arena.offload is not None:
            raise ValueError("KV offloading does not support L2 or snapshot transfers")

        local_group_ids = {
            field.group_id
            for field in self.arena.plan.fields
            if field.field_id in field_ids
        }
        scheduler_group_ids = tuple(
            spec.group_id
            for spec in self.arena.cache_group_specs
            if spec.group_id in local_group_ids
        )
        return layout_from_lcm_plan(
            self.arena.plan,
            self.arena.buffer,
            consumers=consumers,
            group_ids=scheduler_group_ids or None,
            field_ids=field_ids,
        )

    @torch.no_grad()
    def clear_kv_buffers(self) -> None:
        """Zero the shared cache arena after sleep/wake remaps its storage."""
        # 译：在 sleep/wake 重新映射其存储之后，把共享的 cache arena 清零。
        # The event loop visits both target and draft pools; both name the
        # same arena, and zeroing it twice is harmless.
        # 译：事件循环会访问目标和草稿两个池；它们指向同一个 arena，清零两次无害。
        self.arena.clear()

    # ------------------------------------------------------------------
    # What every cache view owes its kernels. Abstract, so a subclass that
    # forgets one fails at construction instead of at the first write.
    # ------------------------------------------------------------------
    # 译：------------------------------------------------------------------
    #     每个 cache 视图都必须为其 kernel 提供的东西。设为抽象方法，这样忘了实现某一个的子类会在构造时失败，
    #     而不是在第一次写入时失败。
    #     ------------------------------------------------------------------

    @abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """This layer's K plane, in the shape its kernels read."""
        # 译：本层的 K 平面，形状是其 kernel 读取的形状。

    @abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """This layer's V plane, in the shape its kernels read."""
        # 译：本层的 V 平面，形状是其 kernel 读取的形状。

    @abstractmethod
    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Both of this layer's planes at once."""
        # 译：一次取出本层的两个平面。

    @abstractmethod
    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """Scatter one forward pass's K/V into this layer's planes."""
        # 译：把一次前向的 K/V 散射写入本层的平面。

    # 注意力 prologue 写 KV 的目标：返回一个描述对象（本层的 K/V 缓冲区 + 每一行要写到的槽位 slots），
    # 由 prologue 的融合 kernel 完成写入（见 layers/paged_attention.py）。write_mask 用于 KVP 下跳过其他 rank 持有的行。
    def kv_write_target(
        self, layer_id: int, slots: torch.Tensor, write_mask: torch.Tensor | None
    ) -> HeadKVCache | LatentKVCache:
        """Where the attention prologue writes this layer's rows at ``slots``;
        ``write_mask`` skips the rows another KVP rank owns."""
        # 译：注意力 prologue 在 ``slots`` 处写入本层的行时的写入目标；``write_mask`` 跳过另一个 KVP rank 拥有的行。
        raise NotImplementedError(
            f"{type(self).__name__} serves no attention prologue writes"
        )
