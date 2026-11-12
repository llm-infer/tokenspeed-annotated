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

"""Small tensor helpers shared across runtime layers."""

# 译：各运行时层共用的小型张量辅助函数。

from __future__ import annotations

import torch


# 两个小工具，各自只有一行：
#   concat：沿最后一维拼接两个张量（EAGLE3 草稿头用它拼接 embedding 和目标模型的 hidden）
#   fp8_cast_contiguous：转成 FP8（E4M3 格式）并保证内存连续（TRT-LLM 注意力后端用它准备 FP8 输入）
def concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concatenate two tensors along the last dimension."""
    # 译：沿最后一维拼接两个张量。
    return torch.cat([a, b], dim=-1)


def fp8_cast_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Cast to FP8 E4M3 and return a contiguous tensor."""
    # 译：转换为 FP8 E4M3 并返回内存连续的张量。
    return x.to(torch.float8_e4m3fn).contiguous()
