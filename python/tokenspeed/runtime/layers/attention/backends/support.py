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

"""Static CUDA-graph capability declarations shared by every backend node."""

# 译：所有后端节点共用的、静态的 CUDA Graph 能力声明。

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed_kernel.selection import KernelPlan

logger = get_colorful_logger(__name__)


# 每个注意力后端类声明自己能否被 CUDA Graph 捕获（第 7 阶段）：
#   decode_graph：能否参与 decode 的整步 graph
#   prefill_graph：能否参与可断开的 prefill graph
# 启动时把目标模型和草稿模型的所有后端（包括 router 下的各个叶子后端）的声明取“与”，
# 只要有一个后端不支持，对应的 graph 就整体关闭（resolve_cuda_graph_support）。
# 这个决定必须在所有 rank 上一致，所以只能由各 rank 相同的静态配置决定。
@dataclass(frozen=True)
class CudaGraphSupport:
    """Per-backend-class CUDA-graph capability declaration.

    Rank-uniform by construction: declarations are class attributes or are
    derived at construction from replicated configuration (such as the leaves
    planned from static geometry, ``from_plans``), resolved identically on
    every rank at startup (event-loop.md requires graph decisions to derive
    from replicated state). ``decode_graph=False``
    disables capture/replay of the whole-step decode graph — the unified
    refresh still serves eager decode, so ``init_cuda_graph_state`` and
    ``refresh_decode_metadata`` stay mandatory. ``prefill_graph=False``
    disables the breakable prefill (extend) graph. Static "never works"
    declarations only — and they are the ONLY escape: a prefill capture
    failure at runtime is fatal, so a family that cannot capture must say
    so here rather than rely on a degrade path.
    """

    # 译：按后端类给出的 CUDA Graph 能力声明。
    #
    #     按构造就是各 rank 一致的：声明是类属性，或者在构造时从复制的配置推导出来（例如由静态几何规划出的叶子，
    #     ``from_plans``），在每个 rank 的启动阶段以相同方式解析（event-loop.md 要求 graph 相关的决定必须来自复制的状态）。
    #     ``decode_graph=False`` 关闭整步 decode graph 的捕获 / 回放——统一的刷新仍然服务于 eager 的 decode，所以
    #     ``init_cuda_graph_state`` 和 ``refresh_decode_metadata`` 仍然是必需的。``prefill_graph=False`` 关闭可断开的
    #     prefill（extend）graph。只能是静态的“永远不行”的声明——并且它们是唯一的出路：运行时 prefill 捕获失败是致命的，
    #     所以无法捕获的系列必须在这里声明，而不是依赖某条降级路径。

    decode_graph: bool = True
    prefill_graph: bool = True

    @classmethod
    def from_plans(
        cls, *, prefill: KernelPlan | None, decode: KernelPlan | None
    ) -> CudaGraphSupport:
        """Support of a backend whose attention leaves are planned at startup.

        A planned leaf is recorded into a graph only if it declares the
        registry's ``cuda_graph`` feature. An axis without a plan keeps the
        default.
        """
        # 译：注意力叶子在启动时规划好的后端的支持情况。
        #
        #     规划好的叶子只有声明了注册表的 ``cuda_graph`` 特性时才会被记录进 graph。没有规划的那一维保持默认值。
        return cls(
            decode_graph=decode is None or "cuda_graph" in decode.spec.features,
            prefill_graph=prefill is None or "cuda_graph" in prefill.spec.features,
        )

    def __and__(self, other: CudaGraphSupport) -> CudaGraphSupport:
        return CudaGraphSupport(
            decode_graph=self.decode_graph and other.decode_graph,
            prefill_graph=self.prefill_graph and other.prefill_graph,
        )


def resolve_cuda_graph_support(*backends) -> CudaGraphSupport:
    """AND-compose ``cuda_graph_support`` over ``backends`` and their
    ``child_backends()`` trees, logging every backend class that lowers an
    axis.

    Args:
        backends: Root attention backends; ``None`` entries are skipped. Pass
            the target AND the draft — the decode graph records the whole
            step, drafter loop included.

    Returns:
        The composed support: an axis is False iff any backend in any tree
        declares it False.
    """
    # 译：对 ``backends`` 及其 ``child_backends()`` 树上的 ``cuda_graph_support`` 做“与”组合，并记录每个降低了某一维的后端类。
    #
    #     参数：
    #         backends：根注意力后端；``None`` 项会被跳过。目标模型和草稿模型都要传入——decode graph 记录的是整步，
    #             包括草稿器的循环。
    #
    #     返回：
    #         组合后的支持情况：当且仅当某棵树中的某个后端声明某一维为 False 时，该维为 False。
    resolved = CudaGraphSupport()
    stack = [backend for backend in backends if backend is not None]
    while stack:
        backend = stack.pop()
        declared = backend.cuda_graph_support
        if not declared.decode_graph:
            logger.info(f"Decode CUDA graphs disabled by {type(backend).__name__!s}")
        if not declared.prefill_graph:
            logger.info(f"Prefill CUDA graphs disabled by {type(backend).__name__!s}")
        resolved = resolved & declared
        stack.extend(backend.child_backends())
    return resolved


# 树状投机解码（第 11 阶段）的能力声明，用法与上面相同：遍历所有后端，有一个不支持就在启动时报错
@dataclass(frozen=True)
class TreeSupport:
    """One backend node's draft-tree capability (--speculative-eagle-topk > 1).

    Each field is ``None`` when this node can take part, otherwise the reason
    it cannot. A node answers for itself only; ``resolve_tree_support`` walks
    ``child_backends()`` so a composite never has to forward the question.
    """

    # 译：一个后端节点在草稿树（--speculative-eagle-topk > 1）方面的能力。
    #
    #     当该节点能参与时，对应字段为 ``None``，否则为它不能参与的原因。节点只回答它自己；``resolve_tree_support``
    #     遍历 ``child_backends()``，所以组合节点不必转发这个问题。

    verify_blocker: str | None
    draft_blocker: str | None


def resolve_tree_support(verify_root, draft_root) -> None:
    """Refuse draft trees unless every node under both roots supports its part.

    Args:
        verify_root: The target's attention backend; every node must verify trees.
        draft_root: The drafter's attention backend; every node must draft trees.

    Raises:
        NotImplementedError: Listing every blocking node and its reason.
    """
    # 译：除非两个根下的每个节点都支持各自的部分，否则拒绝草稿树。
    #
    #     参数：
    #         verify_root：目标模型的注意力后端；每个节点都必须能验证树。
    #         draft_root：草稿器的注意力后端；每个节点都必须能起草树。
    #
    #     异常：
    #         NotImplementedError：列出每个造成阻碍的节点及其原因。
    blockers: list[str] = []
    for root, part in ((verify_root, "verify"), (draft_root, "draft")):
        stack = [root]
        while stack:
            backend = stack.pop()
            support = backend.tree_support()
            blocker = (
                support.verify_blocker if part == "verify" else support.draft_blocker
            )
            if blocker is not None:
                blockers.append(f"{part}: {blocker}")
            stack.extend(backend.child_backends())
    if blockers:
        raise NotImplementedError(
            "draft trees (--speculative-eagle-topk > 1) are not supported here: "
            + "; ".join(blockers)
        )
