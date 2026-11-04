"""OpenAI 兼容的向量化实现（阿里云百炼 / 硅基流动 / 任意兼容端点）。

稠密向量走 API，稀疏向量走本地词法编码 —— qwen embedding 只返回 dense，
缺了稀疏侧，型号、编号、专有名词这类查询会明显漏召。

好处是不需要本地 GPU 和 torch，适合先跑通质量基线；
代价是每次索引都要联网，且数据出内网。

端点默认是百炼，换别家只要覆盖两个环境变量：

    export KB_EMBED_BASE_URL=https://api.siliconflow.cn/v1
    export KB_EMBED_API_KEY=sk-xxx
    export KB_EMBED_MODEL=BAAI/bge-m3      # 维度由服务端返回决定，不用手填
"""

from __future__ import annotations

import os
from typing import Sequence

from ragkb.embedding.base import Embedding
from ragkb.embedding.sparse import SparseEncoder

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-text-embedding-flash"


class OpenAICompatEmbedder:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 10,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError("需要 openai SDK：pip install openai") from exc

        key = (
            api_key
            or os.environ.get("KB_EMBED_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if not key:
            raise RuntimeError(
                "未找到向量化的 API key（KB_EMBED_API_KEY 或 DASHSCOPE_API_KEY）"
            )

        self._client = OpenAI(
            api_key=key,
            base_url=(
                base_url
                or os.environ.get("KB_EMBED_BASE_URL")
                or os.environ.get("DASHSCOPE_BASE_URL")
                or DEFAULT_BASE_URL
            ),
            timeout=timeout,
        )
        self._model = model or os.environ.get("KB_EMBED_MODEL") or DEFAULT_MODEL
        self._batch_size = batch_size
        self._sparse = SparseEncoder()
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            # 维度只能靠一次探测拿到，建 collection 之前必须先调
            self._dim = len(self._embed_dense(["dimension probe"])[0])
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        # 先累积文档频率，再编码，这样稀疏侧带 IDF；只 observe 一次避免重复计数
        self._sparse.observe(texts)
        dense_vectors = self._embed_dense(list(texts))
        return [
            Embedding(dense=vector, sparse=self._sparse.encode(text))
            for text, vector in zip(texts, dense_vectors)
        ]

    def embed_query(self, text: str) -> Embedding:
        return Embedding(dense=self._embed_dense([text])[0], sparse=self._sparse.encode(text))

    def _embed_dense(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])
        return vectors
