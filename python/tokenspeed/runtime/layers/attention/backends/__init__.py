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

# 导入各注意力后端模块。每个模块在导入时调用 register_backend(名字, 支持的注意力架构, 类)，
# 把自己登记到 layers/attention/registry.py 的后端注册表中；启动时再按配置从表中选出后端。
# 部分后端只在特定硬件上导入：FlashMLA 需要 Hopper 及以上，TRT-LLM / tokenspeed-mla 需要 NVIDIA。
# 主线（Qwen3）使用的 MHA 后端来自下面的 paged/mha.py。
# ruff: noqa: E402,F401
# Import all backend modules to trigger register_backend() calls.
# 译：导入所有后端模块，以触发 register_backend() 调用。
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

from tokenspeed.runtime.layers.attention.backends.specific import (  # noqa: F401
    deepseek_v41,
)

if platform.is_hopper_plus:
    from tokenspeed.runtime.layers.attention.backends.paged import (  # noqa: F401
        flashmla,
    )

if platform.is_nvidia:
    from tokenspeed.runtime.layers.attention.backends.paged import trtllm  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends.paged import (
        trtllm_mla,
    )  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends.paged import (  # noqa: F401
        tokenspeed_mla,
    )

from tokenspeed.runtime.layers.attention.backends.paged import ascend_dsa  # noqa: F401
from tokenspeed.runtime.layers.attention.backends.paged import ascend_mla  # noqa: F401
from tokenspeed.runtime.layers.attention.backends.paged import dsa  # noqa: F401
from tokenspeed.runtime.layers.attention.backends.paged import mha  # noqa: F401
from tokenspeed.runtime.layers.attention.backends.paged import mla  # noqa: F401
from tokenspeed.runtime.layers.attention.backends.paged import msa  # noqa: F401
from tokenspeed.runtime.layers.attention.backends.paged import qsa  # noqa: F401
