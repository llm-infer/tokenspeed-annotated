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

import torch

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.utils import get_available_gpu_memory

# 启动时 KV cache 显存预算的计算（由 layers/attention/registry.py 在创建 KV 池时调用，容量规划详见第 14 阶段）。
#
# 整体思路（与“先加载权重，再把剩余显存分给 KV cache”的做法相同，多了几项扣除）：
#   可用于 cache 的显存 = 当前空闲显存
#                       - 启动后还要分配的部分（post_profile_bytes，例如 EPD 的接收池）
#                       - 总显存 × (1 - gpu_memory_utilization)（留给激活值和碎片的余量）
#                       - CUDA Graph 预留（reserve_cache_budget）
# 多卡时各 rank 取最小值，保证所有 rank 的 KV 池大小一致。
# --- Page-based memory profiling ---
# 译：--- 基于页的显存测量 ---


def profile_available_cache_memory_bytes(
    attn_config: AttnConfig,
    gpu_id: int,
    tp_size: int,
    gpu_memory_utilization: float,
    total_gpu_memory: int,
    post_profile_bytes: int,
    world_group=None,
) -> int:
    """The cache budget before any CUDA-graph reserve, the same on every rank.

    Free memory less the utilization headroom and ``post_profile_bytes``, what
    the node allocates after the cache is sized (the EPD receive pool); each rank
    leaves its own out before the cross-rank minimum.
    """
    # 译：在扣除任何 CUDA Graph 预留之前的 cache 预算，在每个 rank 上都相同。
    #
    #     空闲显存减去利用率余量以及 ``post_profile_bytes``（节点在确定 cache 大小之后才分配的部分，即 EPD 接收池）；
    #     每个 rank 先扣掉自己的这部分，再取跨 rank 的最小值。
    # 跨 rank 取最小值用的是 CPU 上的 gloo 通信组（此时还没有建立 GPU 上的数据通信）
    cpu_group = (
        pg_manager.get_process_group("gloo", world_group)
        if world_group is not None
        else None
    )
    # get_available_gpu_memory 的单位是 GiB，所以下面先把字节数换算成 GiB 再相减，最后再换回字节
    available_gpu_memory = get_available_gpu_memory(attn_config.device, gpu_id)
    available_gpu_memory -= post_profile_bytes / (1 << 30)
    if tp_size > 1:
        # Exact for any byte count, so the minimum never rounds above a rank's free memory.
        # 译：对任意字节数都是精确的，所以取到的最小值不会因舍入而超过某个 rank 的空闲显存。
        tensor = torch.tensor(available_gpu_memory, dtype=torch.float64)
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
        available_gpu_memory = tensor.item()
    cache_memory = available_gpu_memory - total_gpu_memory * (
        1 - gpu_memory_utilization
    )
    return int(cache_memory * (1 << 30))


def reserve_cache_budget(profiled_cache_bytes: int, graph_reserve_bytes: int) -> int:
    """The profiled cache budget less the CUDA-graph reserve.

    The utilization headroom was already left out of the profile; it funds
    activations and fragmentation, and the reserve for the graphs and, on
    CUDA, what startup kept resident comes on top of it. A boot the reserve
    leaves with no cache fails here.
    """
    # 译：测得的 cache 预算减去 CUDA Graph 的预留。
    #
    #     利用率余量已经在测量时扣除了；它用来容纳激活值和碎片，而为 graph 预留的部分，以及在 CUDA 上启动阶段
    #     常驻的部分，是在它之上另外扣除的。预留之后没有 cache 空间可用时，启动在这里失败。
    cache_memory = profiled_cache_bytes - graph_reserve_bytes
    if graph_reserve_bytes and cache_memory <= 0:
        raise ValueError(
            f"no cache budget left: the memory profile left "
            f"{profiled_cache_bytes / (1 << 30):.2f} GiB after the utilization "
            f"headroom and {graph_reserve_bytes / (1 << 30):.2f} GiB is reserved "
            "for the CUDA graphs and, on CUDA, what startup kept resident; "
            "raise --gpu-memory-utilization, or re-run with "
            "--disable-cudagraph-memory-reserve to size the cache from free "
            "memory instead"
        )
    return cache_memory


def check_probe_cache_fits(probe_cache_bytes: int, profiled_cache_bytes: int) -> None:
    """Refuse a probe arena the memory profile cannot hold.

    A probe sizes its arena by block count, not by budget, so nothing else
    stops it from asking the device for more than it has. It is the smallest
    arena the family runs on: a serving arena would not fit either.
    """
    # 译：拒绝一个显存测量结果容纳不下的探测 arena。
    #
    #     探测按 block 数而不是按预算确定 arena 的大小，所以没有别的东西能阻止它向设备申请超过其拥有量的显存。
    #     它是该模型族能运行的最小 arena：如果它都放不下，用于服务的 arena 也放不下。
    if probe_cache_bytes <= profiled_cache_bytes:
        return
    raise ValueError(
        f"the KV cache does not fit: its smallest arena needs "
        f"{probe_cache_bytes / (1 << 30):.2f} GiB but the memory profile left "
        f"{profiled_cache_bytes / (1 << 30):.2f} GiB after the weights and the "
        "utilization headroom; add GPUs (a larger --tensor-parallel-size) or "
        "raise --gpu-memory-utilization"
    )
