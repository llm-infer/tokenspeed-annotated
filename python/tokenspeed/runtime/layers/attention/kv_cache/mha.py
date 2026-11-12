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

# MHA / GQA 模型的 KV 池（Qwen3、Llama 等）。
#
# 显存布局（Qwen3-8B，单卡，bf16，页大小 P，共 NP 个页）：
#   每层两个字段 layer.{i}.k、layer.{i}.v，规划的形状都是 [P, 8, 128]（一页内 P 个 token × 8 个 KV 头 × 128）。
#   每个字段的所有页在 arena 里连续存放，arena 把页维折叠掉，得到按 token 槽位索引的视图：
#     k_buffer[i] = layer.i.k 的视图，形状 [NP*P, 8, 128]；v_buffer[i] 同理
#   一个页、一层、K 或 V 占 P × 8 × 128 × 2 字节（P=64 时为 128 KiB）。
#
#   arena 的字节（单个全历史组，各平面依次排列）：
#     ┌──────────────┬──────────────┬──────────────┬──────────────┬─────┐
#     │ layer.0.k    │ layer.0.v    │ layer.1.k    │ layer.1.v    │ ... │
#     │ 页0 页1 ...  │ 页0 页1 ...  │ 页0 页1 ...  │ 页0 页1 ...  │     │
#     └──────────────┴──────────────┴──────────────┴──────────────┴─────┘
#   （平面的先后顺序由计划决定，这里只示意“每个平面内按页连续”。）
#
#   槽位（slot）= 页号 × P + 页内偏移，k_buffer[i][slot] 就是某个 token 在第 i 层的 K，形状 [8, 128]。
#   例（P=64）：B 的历史有 1000 个 token，占 16 个页（1000 = 15 × 64 + 40）；如果它的页表是
#     [9, 2, 31, ...]，那么 B 的第 70 个 token 在第 1 个逻辑页、偏移 6，物理页 2，槽位 = 2 × 64 + 6 = 134。
#
# 主线的写入不走 set_kv_buffer，而是由注意力 prologue 通过 kv_write_target 在融合 kernel 里写入；
# set_kv_buffer 留给单独写 KV 的路径使用。
from __future__ import annotations

from typing import ClassVar

import torch
from tokenspeed_kernel.ops.attention.prologue import (
    HeadKVCache,
    MXFP8Scales,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    quantize_store_kv_mxfp8,
    store_kv_cache,
)

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    MXFP8_KV_SCALE_TILE_TOKENS,
    MXFP8_SCALE_BLOCK_SIZE,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


GB = 1024 * 1024 * 1024


# MHA 的 KV 按头存储，从不在 KVP 下切分，所以 prologue 写入每一行，不需要掩码
def _head_rows_are_never_masked(write_mask: torch.Tensor | None) -> None:
    if write_mask is not None:
        raise ValueError("head caches are never sharded; the prologue writes every row")


# MHA / GQA 的 KV 池。参数：
#   head_num：本 rank 的 KV 头数（Qwen3-8B、TP=1 时为 8）；head_dim：128；layer_num：本视图的层数（36）
#   layer_kv_head_counts：各层 KV 头数不同的模型（异构，如 Inkling）用；Qwen3 为 None（各层相同）
#     头数少的层把同样的字节解读成更多的行：分配时按最大头数，[行, 大头数, D] 重新 reshape 成 [更多行, 小头数, D]
#   kv_alloc_head_count：分配宽度对应的（TP 切分前的）头数，用于上面的换算
class MHATokenToKVPool(CachePool):
    def __init__(
        self,
        arena: CacheArena,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        rank: int,
        *,
        layer_kv_head_counts: tuple[int, ...] | None = None,
        kv_alloc_head_count: int | None = None,
        field_layer_offset: int = 0,
    ):
        super().__init__(
            arena,
            dtype,
            rank,
            field_layer_offset=field_layer_offset,
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        # Fewer-head layers reinterpret the allocation width as more rows.
        # 译：头数较少的层把分配的宽度重新解读为更多的行。
        self._layer_kv_head_counts = (
            tuple(int(h) for h in layer_kv_head_counts)
            if layer_kv_head_counts
            else None
        )
        # Pre-TP head count ``head_num`` is the per-rank shard of — the view
        # normalization base. Falling back to max(counts) is only correct
        # when the pool's own layers include an alloc-width layer: an
        # all-narrow pool (Inkling MTP draft with full-attention-only
        # depths, alloc'd at the config max) would silently collapse the
        # reinterpretation and write narrow rows into wide strides (the
        # #65/1.82-accept GQA-stride corruption).
        # 译：``head_num`` 是 TP 之前的头数在每个 rank 上的分片——视图归一化的基准。只有当池自己的层中包含一个按分配宽度的层时，
        #     回退到 max(counts) 才是正确的：一个全部是窄层的池（Inkling 的 MTP 草稿，只有全注意力深度，按配置的最大值分配）
        #     会悄无声息地取消这种重新解读，把窄行写进宽的步长里（#65/1.82-accept 的 GQA 步长损坏问题）。
        self._kv_alloc_head_count = (
            int(kv_alloc_head_count) if kv_alloc_head_count else None
        )
        self._bind_layer_planes()

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. K size: {k_size / GB:.2f} GB, V size: "
            f"{v_size / GB:.2f} GB.",
        )

    # 本池读取的平面：k 平面放进 self.k_buffer，v 平面放进 self.v_buffer（由基类的 _bind_layer_planes 填充）
    layer_plane_bindings: ClassVar[dict[str, str]] = {
        "k": "k_buffer",
        "v": "v_buffer",
    }

    def get_kv_size_bytes(self):
        # Different layer views may share an address through the memory plan.
        # Count each physical region once, independent of Python identity.
        # 译：不同的层视图可能通过内存计划共享同一个地址。每个物理区域只计一次，与 Python 对象的身份无关。
        k_caches = {t.data_ptr(): t for t in self.k_buffer if t is not None}
        v_caches = {t.data_ptr(): t for t in self.v_buffer if t is not None}
        k_size_bytes = sum(t.nbytes for t in k_caches.values())
        v_size_bytes = sum(t.nbytes for t in v_caches.values())
        return k_size_bytes, v_size_bytes

    def _layer_row_view(self, buf: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Per-layer token-row view over one byte-uniform cache field.

        A layer serving fewer heads than the field's planned head count
        reinterprets the same bytes as proportionally more rows of
        ``heads_l`` heads; a field planned at the layer's own head count is
        served as is.
        """
        # 译：一个字节统一的 cache 字段上，按层的 token 行视图。
        #
        #     服务的头数少于字段规划头数的层，把同样的字节按比例重新解读为更多的、每行 ``heads_l`` 个头的行；
        #     按该层自己的头数规划的字段原样提供。
        if self._layer_kv_head_counts is None:
            return buf
        heads_l = self._layer_heads_per_rank(layer_id)
        if heads_l == self.head_num:
            return buf
        return buf.reshape(-1, heads_l, self.head_dim)

    def _layer_heads_per_rank(self, layer_id: int) -> int:
        counts = self._layer_kv_head_counts
        if counts is None:
            # Uniform head counts (hetero off): every layer serves head_num.
            # 译：头数统一（不异构）：每层都服务 head_num 个头。
            return self.head_num
        served = counts[layer_id]
        # head_num is the per-rank shard of the ALLOCATION width (the config
        # max); scale the pre-TP served count proportionally. max(counts) is
        # only a valid stand-in when some layer serves the alloc width — an
        # all-narrow pool must still reinterpret every layer.
        # 译：head_num 是分配宽度（配置中的最大值）在每个 rank 上的分片；按比例缩放 TP 之前的服务头数。只有当某一层
        #     服务的是分配宽度时，max(counts) 才是有效的替代——全部是窄层的池仍然必须重新解读每一层。
        alloc = self._kv_alloc_head_count or max(counts)
        return max(1, self.head_num * served // alloc)

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        # 译：供内部引用使用
        buf = self.k_buffer[layer_id]
        if buf is None:
            raise ValueError(f"layer {layer_id} is a state layer; it has no KV buffer")
        # FP8 存储时 k_buffer 是 uint8，读取时 view 回 FP8 dtype
        if self.store_dtype != self.dtype:
            buf = buf.view(self.dtype)
        return self._layer_row_view(buf, layer_id)

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        # 译：注意：get_key_buffer 挂接了逐层 KV cache 加载的同步
        #     它只应由注意力后端使用，不应用于获取信息
        #     get_value_buffer 和 get_kv_buffer 同理
        # 从 Host 逐层加载前缀 KV 时，先等本层加载完成再交出缓冲区
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        # 译：供内部引用使用
        buf = self.v_buffer[layer_id]
        if buf is None:
            raise ValueError(f"layer {layer_id} is a state layer; it has no KV buffer")
        if self.store_dtype != self.dtype:
            buf = buf.view(self.dtype)
        return self._layer_row_view(buf, layer_id)

    def get_value_buffer(self, layer_id: int):
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def kv_write_target(
        self, layer_id: int, slots: torch.Tensor, write_mask: torch.Tensor | None
    ) -> HeadKVCache:
        """Where the attention prologue writes this layer's K/V rows."""
        # 译：注意力 prologue 写入本层 K/V 行的目标。
        _head_rows_are_never_masked(write_mask)
        k_cache, v_cache = self.get_kv_buffer(layer_id)
        return HeadKVCache(
            k_cache=k_cache,
            v_cache=v_cache,
            scales=None,
            slots=slots,
        )

    # 单独写入 KV：cache_k、cache_v 为 [行数, Hkv, D]，loc 为每行的槽位；用 Triton kernel store_kv_cache 散射写入。
    # dtype 不同时先转换（例如 bf16 -> FP8），FP8 再 view 成 uint8 写入。
    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        # Wait for the layer's Host load: a tail hit loads the page these
        # rows land in whole.
        # 译：等待该层的 Host 加载：前缀尾部命中时会把这些行所在的页整页加载进来。
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)
        # Locs are in per-layer view rows: the store must target the same view get_key_buffer serves
        # 译：loc 是按层视图中的行号：写入必须指向 get_key_buffer 提供的同一个视图
        store_kv_cache(
            cache_k,
            cache_v,
            self._layer_row_view(self.k_buffer[layer_id], layer_id),
            self._layer_row_view(self.v_buffer[layer_id], layer_id),
            loc,
        )


# MXFP8 格式的 KV 池：数据为 FP8，每 32 个元素一个 UE8M0 scale。scale 存放在 FA4 blockscaled kernel
# 要求的按页交错布局中。主线（bf16）不使用。
class MHATokenToKVPoolMXFP8(MHATokenToKVPool):
    """MHA KV pool storing MXFP8 block-scaled FP8 (data + UE8M0 scales).

    Data buffers hold float8_e4m3fn; scale buffers hold one float8_e8m0fnu
    per 32 elements of head_dim. ``set_kv_buffer`` quantizes K/V rows and
    stores their scales, as the attention prologue does through
    ``kv_write_target``.

    Scales live in the page-major interleaved layout the FA4 blockscaled
    kernel consumes -- ``(num_pages, heads, page_tokens // 128, 32, sf, sf)``
    BlockScaledBasicChunk atoms written by ``quantize_store_kv_mxfp8`` -- the
    one shape ``plan.mxfp8_kv_scale_fields`` declares.

    When the plan aliases fields, data and scale views remain layer-local
    Python objects while addressing the same physical bytes.
    """

    # 译：存储 MXFP8 分块缩放 FP8（数据 + UE8M0 scale）的 MHA KV 池。
    #
    #     数据缓冲区存放 float8_e4m3fn；scale 缓冲区在 head_dim 上每 32 个元素存放一个 float8_e8m0fnu。``set_kv_buffer``
    #     量化 K/V 行并存储它们的 scale，就像注意力 prologue 通过 ``kv_write_target`` 所做的那样。
    #
    #     scale 采用 FA4 blockscaled kernel 使用的按页为主、交错的布局——由 ``quantize_store_kv_mxfp8`` 写入的
    #     ``(num_pages, heads, page_tokens // 128, 32, sf, sf)`` BlockScaledBasicChunk 原子——这是
    #     ``plan.mxfp8_kv_scale_fields`` 声明的唯一形状。
    #
    #     当计划让字段共用地址时，数据视图和 scale 视图仍然是按层的 Python 对象，但访问的是同一份物理字节。

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **MHATokenToKVPool.layer_plane_bindings,
        "k_scale": "k_scale_buffer",
        "v_scale": "v_scale_buffer",
    }

    def _bind_layer_planes(self) -> None:
        if self.head_dim % MXFP8_SCALE_BLOCK_SIZE:
            raise ValueError(
                f"MXFP8 head_dim must be divisible by {MXFP8_SCALE_BLOCK_SIZE}"
            )
        # These writes go through dtype-aware kernels, so the fp8 view the
        # plan hands out is the one to keep for input reinterpretation too.
        # 译：这些写入经过能感知 dtype 的 kernel，所以计划提供的 fp8 视图也正是用来重新解读输入时要保留的视图。
        self.store_dtype = torch.float8_e4m3fn
        super()._bind_layer_planes()
        for layer_id in range(self.layer_num):
            for plane in (self.k_scale_buffer[layer_id], self.v_scale_buffer[layer_id]):
                if plane is not None:
                    self._check_scale_plane(plane, layer_id)

    def _check_scale_plane(self, plane: torch.Tensor, layer_id: int) -> None:
        """Reject a scale plane the layer's interleaved view cannot factor."""
        # 译：拒绝一个该层的交错视图无法分解的 scale 平面。
        sf_dim = self.head_dim // MXFP8_SCALE_BLOCK_SIZE
        heads_l = self._layer_heads_per_rank(layer_id)
        if (
            plane.ndim != 6
            or plane.shape[3:] != (32, sf_dim, sf_dim)
            or (plane.shape[1] * plane.shape[2]) % heads_l
        ):
            raise ValueError(
                f"layer {layer_id} mxfp8 scale plane {tuple(plane.shape)} is not "
                f"the interleaved (pages, heads, tiles, 32, {sf_dim}, {sf_dim}) "
                f"layout for {heads_l} served heads"
            )

    def _layer_page_tokens(self, layer_id: int) -> int:
        """Tokens one page id spans in this layer's view.

        The plan fixes how many head columns and 128-token tiles a page
        holds; a layer serving fewer heads than its planned columns folds
        the spare columns into further token rows of the same page.
        """
        # 译：在本层视图中一个页 id 跨越的 token 数。
        #
        #     计划确定一个页容纳多少个头列和多少个 128-token 的 tile；服务的头数少于规划列数的层，把多余的列折叠成同一页中
        #     更多的 token 行。
        planned_heads, planned_tiles = self.k_scale_buffer[layer_id].shape[1:3]
        heads_l = self._layer_heads_per_rank(layer_id)
        return planned_heads * planned_tiles * MXFP8_KV_SCALE_TILE_TOKENS // heads_l

    def _layer_scale_view(self, buf: torch.Tensor, layer_id: int) -> torch.Tensor:
        """(num_ids, heads_l, k_l, 32, sf, sf) view over a layer's SF slots
        (the paged interleaved layout the blockscaled kernels consume)."""
        # 译：一个层的 SF 槽位上的 (num_ids, heads_l, k_l, 32, sf, sf) 视图（blockscaled kernel 使用的分页交错布局）。
        heads_l = self._layer_heads_per_rank(layer_id)
        k_l = self._layer_page_tokens(layer_id) // MXFP8_KV_SCALE_TILE_TOKENS
        sf_dim = self.head_dim // MXFP8_SCALE_BLOCK_SIZE
        return buf.view(buf.shape[0], heads_l, k_l, 32, sf_dim, sf_dim)

    def get_kv_scale_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(k_scale, v_scale) views for the blockscaled attention kernel."""
        # 译：供 blockscaled 注意力 kernel 使用的 (k_scale, v_scale) 视图。
        k_sf = self.k_scale_buffer[layer_id]
        v_sf = self.v_scale_buffer[layer_id]
        if k_sf is None or v_sf is None:
            raise ValueError(f"layer {layer_id} is a state layer; it has no KV scales")
        return (
            self._layer_scale_view(k_sf, layer_id),
            self._layer_scale_view(v_sf, layer_id),
        )

    def kv_write_target(
        self, layer_id: int, slots: torch.Tensor, write_mask: torch.Tensor | None
    ) -> HeadKVCache:
        """Where the attention prologue writes this layer's K/V rows and scales."""
        # 译：注意力 prologue 写入本层 K/V 行和 scale 的目标。
        _head_rows_are_never_masked(write_mask)
        k_cache, v_cache = self.get_kv_buffer(layer_id)
        return HeadKVCache(
            k_cache=k_cache,
            v_cache=v_cache,
            scales=MXFP8Scales(
                k=self.k_scale_buffer[layer_id],
                v=self.v_scale_buffer[layer_id],
                page_tokens=self._layer_page_tokens(layer_id),
            ),
            slots=slots,
        )

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """Quantize K/V rows to MXFP8 and scatter them with their block scales."""
        # 译：把 K/V 行量化为 MXFP8，并连同它们的分块 scale 一起散射写入。
        layer_id = layer.layer_id
        # Wait for the layer's Host load: a tail hit loads the page these
        # rows land in whole.
        # 译：等待该层的 Host 加载：前缀尾部命中时会把这些行所在的页整页加载进来。
        if self.layerwise_load_tracker is not None:
            self.layerwise_load_tracker.wait_for_layer(layer_id)
        # Locs are per-layer view rows: target the served view
        # 译：loc 是按层视图中的行号：写入指向所服务的视图
        k_rows = self._layer_row_view(self.k_buffer[layer_id], layer_id)
        v_rows = self._layer_row_view(self.v_buffer[layer_id], layer_id)
        quantize_store_kv_mxfp8(
            cache_k.view(-1, *k_rows.shape[1:]),
            cache_v.view(-1, *v_rows.shape[1:]),
            k_rows,
            v_rows,
            self.k_scale_buffer[layer_id],
            self.v_scale_buffer[layer_id],
            loc,
            page_tokens=self._layer_page_tokens(layer_id),
        )

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        for sf in {
            buffer.data_ptr(): buffer for buffer in self.k_scale_buffer
        }.values():
            k_size += sf.numel() * sf.element_size()
        for sf in {
            buffer.data_ptr(): buffer for buffer in self.v_scale_buffer
        }.values():
            v_size += sf.numel() * sf.element_size()
        return k_size, v_size
