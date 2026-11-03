"""向量化入口。

backend:
- "bge-m3"  本地模型，需 pip install 'ragkb[embed]'（有 GPU 时首选）
- "api"     百炼 / DashScope 的 embedding 接口，无需 GPU，需 DASHSCOPE_API_KEY
- "hashing" 确定性替身，仅用于测试与离线开发
- "auto"    优先本地 bge-m3，其次 API，最后回退替身
"""

from __future__ import annotations

import warnings

from ragkb.embedding.base import Embedder, Embedding, SparseVector
from ragkb.embedding.hashing import HashingEmbedder

__all__ = ["Embedder", "Embedding", "SparseVector", "HashingEmbedder", "build_embedder"]


def build_embedder(backend: str = "auto", **kwargs) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder(**kwargs)

    if backend in {"bge-m3", "bge_m3"}:
        from ragkb.embedding.bge_m3 import BGEM3Embedder

        return BGEM3Embedder(**kwargs)

    if backend == "api":
        from ragkb.embedding.openai_compat import OpenAICompatEmbedder

        return OpenAICompatEmbedder(**kwargs)

    if backend == "auto":
        try:
            from ragkb.embedding.bge_m3 import BGEM3Embedder

            return BGEM3Embedder(**kwargs)
        except Exception:
            pass
        try:
            from ragkb.embedding.openai_compat import OpenAICompatEmbedder

            return OpenAICompatEmbedder(**kwargs)
        except Exception as exc:
            warnings.warn(
                f"本地模型与 embedding API 均不可用（{exc}），回退到 HashingEmbedder。"
                "该替身只能验证链路逻辑，不能用于评估检索质量。",
                RuntimeWarning,
                stacklevel=2,
            )
            return HashingEmbedder()

    raise ValueError(f"未知的 embedder backend: {backend!r}")
