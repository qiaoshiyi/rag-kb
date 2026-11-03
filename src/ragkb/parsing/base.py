"""解析器协议与注册表。

新增格式只需写一个 BaseParser 子类并用 @register 装饰，
切分、索引、检索链路完全不用改动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from ragkb.models.document import Document


class ParseError(RuntimeError):
    """无法解析该文档时抛出。调用方应记录到 failed 表并继续处理其他文档。"""


class BaseParser(ABC):
    source_type: ClassVar[str] = ""
    suffixes: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def parse(self, path: Path, meta: dict[str, Any] | None = None) -> Document:
        """把文件解析成 IR。实现必须保证不抛未捕获异常之外的东西。"""

    def new_document(self, path: Path, meta: dict[str, Any] | None = None) -> Document:
        merged: dict[str, Any] = {
            "title": path.stem,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
        }
        if meta:
            merged.update(meta)
        return Document(source_path=str(path), source_type=self.source_type, meta=merged)


_REGISTRY: dict[str, BaseParser] = {}


def register(cls: type[BaseParser]) -> type[BaseParser]:
    instance = cls()
    if not instance.suffixes:
        raise ValueError(f"{cls.__name__} 未声明 suffixes")
    for suffix in instance.suffixes:
        _REGISTRY[suffix.lower()] = instance
    return cls


def get_parser(path: str | Path) -> BaseParser:
    suffix = Path(path).suffix.lower()
    try:
        return _REGISTRY[suffix]
    except KeyError:
        raise ParseError(
            f"不支持的文件类型 {suffix!r}，当前支持：{sorted(_REGISTRY)}"
        ) from None


def parse_document(path: str | Path, meta: dict[str, Any] | None = None) -> Document:
    p = Path(path)
    if not p.exists():
        raise ParseError(f"文件不存在：{p}")
    return get_parser(p).parse(p, meta)


def supported_suffixes() -> list[str]:
    return sorted(_REGISTRY)
