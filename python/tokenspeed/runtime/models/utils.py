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

"""Helpers shared across runtime model implementations."""

# 译：各个运行时模型实现共用的辅助函数。


# 各模型共用的小工具。
#
# validate_attention_partition：检查注意力头能否按 TP 切分，不能时在构建模型时就报错。
#   Q 头必须被 tp_size 整除；KV 头多于或等于 tp_size 时也要被整除，
#   KV 头少于 tp_size 时反过来要求 tp_size 被 KV 头数整除（每个 KV 头复制到若干 rank）。
#   例：32 个 Q 头、8 个 KV 头：TP=2、4、8 可以；TP=16 也可以（每个 KV 头复制到 2 个 rank）；TP=3 不行。
def validate_attention_partition(
    total_num_heads: int,
    total_num_kv_heads: int,
    tp_size: int,
) -> None:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}.")
    if total_num_heads % tp_size != 0:
        raise ValueError(
            f"num_attention_heads={total_num_heads} must be divisible by tp_size={tp_size}."
        )
    if total_num_kv_heads <= 0:
        raise ValueError(
            f"num_key_value_heads must be positive, got {total_num_kv_heads}."
        )
    if total_num_kv_heads >= tp_size:
        if total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads={total_num_kv_heads} must be divisible by tp_size={tp_size}."
            )
    elif tp_size % total_num_kv_heads != 0:
        raise ValueError(
            f"tp_size={tp_size} must be divisible by num_key_value_heads={total_num_kv_heads}."
        )
