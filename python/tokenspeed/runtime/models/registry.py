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

"""Model registry helpers for resolving runtime model entry classes."""

# 译：用于解析运行时模型入口类的模型注册表辅助代码。

import importlib
import pkgutil
from collections.abc import Set
from dataclasses import dataclass, field
from functools import lru_cache

import torch.nn as nn

from tokenspeed.runtime.plugins.registry import (
    registered_architectures,
    registered_model,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

# 模型注册表：HF 配置里的架构名（config.architectures，例如 "Qwen3ForCausalLM"）-> 模型类。
#
# 你在这里：启动时 model_loader/utils.py 的 get_model_architecture 调用
#   ModelRegistry.resolve_model_cls(architectures)，得到要构建的模型类（第 8 阶段）。
#
# 注册表的两个来源：
#   1. 仓库内的模型：import_model_classes 导入 tokenspeed.runtime.models 下的每个模块，
#      收集模块级变量 EntryClass（一个类或类的列表），以“类名”作为架构名登记。
#      所以模型类的名字必须与 HF 配置里的架构名一致（Qwen3ForCausalLM）。
#   2. 插件注册的模型（plugins/registry.py，第 19 阶段）：外部包可以注册新架构；
#      如果要替换仓库内已有的同名架构，必须显式声明 override=True，否则报错。


# 注册表本体。models 字段保存仓库内的“架构名 -> 模型类”。
@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    # 译：以 model_arch（架构名）为键
    models: dict[str, type[nn.Module]] = field(default_factory=dict)

    # 所有支持的架构名：仓库内的 + 插件注册的
    def get_supported_archs(self) -> Set[str]:
        return self.models.keys() | registered_architectures()

    def _raise_for_unsupported(self, architectures: list[str]):
        all_supported_archs = self.get_supported_archs()

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    # 查找一个架构名对应的模型类，找不到返回 None。插件注册的优先，
    # 但插件与仓库内模型同名而又没有声明 override=True 时报错，避免无意中被替换。
    def _try_load_model_cls(self, model_arch: str) -> type[nn.Module] | None:
        registered = registered_model(model_arch)
        in_tree = self.models.get(model_arch)
        if registered is None:
            return in_tree
        if in_tree is not None and not registered.override:
            raise ValueError(
                f"Plugin model {registered.cls.__name__} registers architecture "
                f"{model_arch!r}, which the in-tree {in_tree.__name__} already "
                "serves; register it with override=True to replace it"
            )
        return registered.cls

    # 统一成列表；HF 配置里 architectures 通常是列表，也兼容单个字符串
    def _normalize_archs(
        self,
        architectures: str | list[str],
    ) -> list[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        return architectures

    # 入口：按顺序尝试 architectures 中的每个架构名，返回第一个能找到的 (模型类, 架构名)；
    # 都找不到时报错，并列出所有支持的架构。
    def resolve_model_cls(
        self,
        architectures: str | list[str],
    ) -> tuple[type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)


# 导入 models 包下的所有模块，收集 EntryClass，返回 {类名: 类}。
# 任何一个模块导入失败都会直接报错（而不是跳过），避免某个模型悄悄不可用。
# 同名的模型类出现两次也报错。@lru_cache 保证整个进程只扫描一次。
@lru_cache
def import_model_classes():
    """Import model modules and collect their ``EntryClass`` exports."""
    # 译：导入各个模型模块，并收集它们导出的 ``EntryClass``。
    model_arch_name_to_cls = {}
    package_name = "tokenspeed.runtime.models"
    package = importlib.import_module(package_name)

    for _, name, _ in pkgutil.iter_modules(package.__path__, package_name + "."):

        try:
            module = importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"Failed to import model module {name}.") from exc
        if hasattr(module, "EntryClass"):
            entry = module.EntryClass
            if isinstance(
                entry, list
            ):  # To support multiple model classes in one module
                # 译：支持一个模块中有多个模型类
                for tmp in entry:
                    if tmp.__name__ in model_arch_name_to_cls:
                        raise ValueError(
                            f"Duplicated model implementation for {tmp.__name__}"
                        )
                    model_arch_name_to_cls[tmp.__name__] = tmp
            else:
                if entry.__name__ in model_arch_name_to_cls:
                    raise ValueError(
                        f"Duplicated model implementation for {entry.__name__}"
                    )
                model_arch_name_to_cls[entry.__name__] = entry

    return model_arch_name_to_cls


# 全局唯一的注册表实例；导入本模块时就会扫描全部模型模块
ModelRegistry = _ModelRegistry(import_model_classes())
