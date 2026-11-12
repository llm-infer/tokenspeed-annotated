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

"""Per-kernel page geometry, in one place.

``kernel_page_size`` is the token span of one attention-kernel page — a property of the kernel implementation, not of the scheduler.
"""

# 译：按 kernel 区分的页几何，集中在一处。
#
#     ``kernel_page_size`` 是一个注意力 kernel 页覆盖的 token 跨度——它是 kernel 实现的属性，而不是调度器的属性。

# 各注意力 kernel 要求（或默认使用）的页大小，单位是 token。
#
# 注意：调度器管理 KV 的单位是 block（block_granularity 个 token），kernel 读 KV 的单位是 page
# （kernel_page_size 个 token），两者可以不同。例如调度器按 128 个 token 一个 block 分配，而某个 kernel
# 只能按 64 个 token 一页去读，此时一个 block 对应 2 个 kernel 页。这个换算由 CacheGroupRouter 完成
# （backends/paged/router.py）。这里只记录 kernel 一侧的约束。
#
# 主线（Qwen3 + MHA 后端）不在本表中：MHA 后端的页大小由启动时的 cache 规划决定（见 backends/paged/mha.py）。
# FlashMLA decode kernels read the KV cache with a fixed 64-token page stride.
# 译：FlashMLA 的 decode kernel 以固定的 64 个 token 的页步长读取 KV cache。
FLASH_MLA_PAGE_SIZE = 64

# DeepGEMM fp8_paged_mqa_logits lays out sparse-decode KV as 64-token pages
# (block-split FP8 payload + FP32 scales).
# 译：DeepGEMM 的 fp8_paged_mqa_logits 把稀疏 decode 的 KV 排布成 64 个 token 一页
#     （分块的 FP8 数据 + FP32 scale）。
DSA_SPARSE_PAGE_SIZE = 64

# V4.1 FlatKV has 64-row global pages: raw spans are 128 (ratio 2) and
# 64 (ratio 1). SWA remains a separate 64-token page address domain.
# 译：V4.1 的 FlatKV 采用 64 行的全局页：原始跨度为 128（比例 2）和 64（比例 1）。
#     SWA 仍然是一个独立的、64 个 token 一页的地址空间。
DEEPSEEK_V41_SWA_PAGE_SIZE = 64
DEEPSEEK_V41_GLOBAL_ROWS = 64

# trtllm-gen paged MLA kernels support exactly these page sizes.
# 译：trtllm-gen 的分页 MLA kernel 恰好支持这些页大小。
TRTLLM_MLA_SUPPORTED_PAGE_SIZES = (32, 64)

# tokenspeed-mla (CuteDSL) paged decode supports exactly these page sizes.
# 译：tokenspeed-mla（CuteDSL）的分页 decode 恰好支持这些页大小。
TOKENSPEED_MLA_SUPPORTED_PAGE_SIZES = (32, 64)

# Flexible paged kernel: a chosen default, not a hardware constant.
# 译：灵活的分页 kernel：这是选定的默认值，不是硬件常量。
MLA_PAGE_SIZE = 64

# The MSA sparse-attention kernels register exactly one supported page size
# (see tokenspeed_kernel/ops/attention/msa.py: page_size {128}); this is a
# kernel constraint, not a chosen default.
# 译：MSA 稀疏注意力 kernel 只注册了一种支持的页大小（见 tokenspeed_kernel/ops/attention/msa.py：page_size {128}）；
#     这是 kernel 的约束，不是选定的默认值。
MSA_PAGE_SIZE = 128
TRTLLM_MHA_PAGE_SIZE = 64

# Constrained kernels: default within the supported set.
# 译：受约束的 kernel：在支持的集合中选一个默认值。
TRTLLM_MLA_DEFAULT_PAGE_SIZE = 64
TOKENSPEED_MLA_DEFAULT_PAGE_SIZE = 64
