"""向量化抽象。

真实实现用 bge-m3（稠密 + 稀疏一体，中文友好、可私有化）；
测试与离线开发用 HashingEmbedder 替身，不依赖 torch，也不需要下载模型。

接口只暴露 dense + sparse 两路，bge-m3 的 colbert 多向量暂不使用——
它带来的精度提升不足以抵消存储与检索成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass(slots=True)
class SparseVector:
    """词法权重向量。indices 是词表 id，values 是权重，两者等长。"""

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.indices)


@dataclass(slots=True)
class Embedding:
    dense: list[float]
    sparse: SparseVector | None = None


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]: ...

    def embed_query(self, text: str) -> Embedding: ...
