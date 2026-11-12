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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import torch


# 把 kernel 页表展开成“每个 token 一个 KV 槽位编号”的一维数组。
# 槽位编号 = 页号 × 页大小 + 页内偏移，按请求依次排列，每个请求只取前 seq_lens[i] 个位置。
# 用于需要把整段历史 KV 按 token 收集出来的场景（稀疏注意力 DSA 的 prefill、GLM-5 等，第 12 阶段），
# 主线的 MHA 后端不使用它。
#
# 例：页大小 4，请求 0 的页表 [7, 3]、长度 6；请求 1 的页表 [5]、长度 2
#   -> [28, 29, 30, 31, 12, 13, 20, 21]
#      （请求 0：页 7 的 4 个槽位 + 页 3 的前 2 个；请求 1：页 5 的前 2 个）
#
# 两种实现：
#   已知总 token 数 num_tokens 时：直接为每个 token 算出所属请求和请求内偏移，再查页表；
#     输出大小已知，不需要 GPU 与 CPU 同步
#   未知时：先对每个请求按 max_seq_len 展开，再用布尔掩码挑出有效位置；布尔索引的输出大小
#     要等 GPU 算完才知道，会引起一次 GPU -> CPU 同步
def build_prefill_kv_workspace_slots(
    *,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    device: torch.device,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """Flatten a kernel page table into per-token KV cache slot ids.

    Args:
        page_table: ``[num_reqs, num_pages]`` kernel page table whose entries
            index pages of ``page_size`` tokens.
        seq_lens: ``[num_reqs]`` number of valid tokens per request.
        max_seq_len: Upper bound of ``seq_lens``; fixes the gather width.
        page_size: Tokens covered by one kernel page.
        device: Device the slot tensor is built on.
        num_tokens: Exact sum of ``seq_lens`` when already known on the host.
            Providing it avoids the device synchronization otherwise required by
            boolean indexing to determine the packed output size.

    Returns:
        1-D int64 tensor of cache slot ids, request-major, covering exactly
        the first ``seq_lens[i]`` positions of each request.
    """
    # 译：把 kernel 页表展平成每个 token 的 KV cache 槽位编号。
    #
    #     参数：
    #         page_table：``[num_reqs, num_pages]`` 的 kernel 页表，其中的项是覆盖 ``page_size`` 个 token 的页的下标。
    #         seq_lens：``[num_reqs]``，每个请求的有效 token 数。
    #         max_seq_len：``seq_lens`` 的上界；决定收集的宽度。
    #         page_size：一个 kernel 页覆盖的 token 数。
    #         device：构建槽位张量的设备。
    #         num_tokens：如果在主机上已经知道，则为 ``seq_lens`` 的准确总和。提供它可以避免布尔索引为了确定
    #             打包后的输出大小而必须进行的设备同步。
    #
    #     返回：
    #         1 维 int64 的 cache 槽位编号张量，按请求为主序排列，恰好覆盖每个请求的前 ``seq_lens[i]`` 个位置。
    if num_tokens is not None:
        seq_lens_i64 = seq_lens.to(device=device, dtype=torch.int64)
        request_ids = torch.repeat_interleave(
            torch.arange(seq_lens.numel(), dtype=torch.int64, device=device),
            seq_lens_i64,
            output_size=int(num_tokens),
        )
        seq_starts = torch.zeros(
            seq_lens.numel() + 1,
            dtype=torch.int64,
            device=device,
        )
        torch.cumsum(seq_lens_i64, dim=0, out=seq_starts[1:])
        local_offsets = torch.arange(
            int(num_tokens), dtype=torch.int64, device=device
        ) - seq_starts.index_select(0, request_ids)
        pages = page_table.to(device=device, dtype=torch.int64)[
            request_ids,
            torch.div(local_offsets, int(page_size), rounding_mode="floor"),
        ]
        return (pages * int(page_size) + local_offsets % int(page_size)).contiguous()

    local_offsets = torch.arange(
        int(max_seq_len),
        dtype=torch.int64,
        device=device,
    )
    page_offsets = torch.div(
        local_offsets,
        int(page_size),
        rounding_mode="floor",
    )
    block_offsets = local_offsets % int(page_size)
    pages = page_table.to(device=device, dtype=torch.int64).index_select(
        1,
        page_offsets,
    )
    slots = pages * int(page_size) + block_offsets
    valid = local_offsets.unsqueeze(0) < seq_lens.to(
        device=device,
        dtype=torch.int64,
    ).unsqueeze(1)
    return slots[valid].contiguous()
