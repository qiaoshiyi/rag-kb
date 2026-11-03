"""bge-m3 真实实现（需 FlagEmbedding + torch）。

一次编码同时产出稠密与稀疏权重，省掉独立的 BM25 服务——
这是选型时选 bge-m3 而非纯 dense 模型的主要理由。
"""

from __future__ import annotations

from typing import Any, Sequence

from ragkb.embedding.base import Embedding, SparseVector


class BGEM3Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        use_fp16: bool = True,
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "bge-m3 需要 FlagEmbedding，请先安装：pip install 'ragkb[embed]'"
            ) from exc

        kwargs: dict[str, Any] = {"use_fp16": use_fp16}
        if device:
            kwargs["device"] = device

        self._model = BGEM3FlagModel(model_name, **kwargs)
        self._batch_size = batch_size
        self._max_length = max_length
        self._dim = 1024  # bge-m3 稠密向量固定 1024 维

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> Embedding:
        # bge-m3 的 query 与 passage 共用同一编码器，无需指令前缀
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []

        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_rows = output["dense_vecs"]
        lexical_list = output.get("lexical_weights") or [{} for _ in texts]

        results: list[Embedding] = []
        for row, lexical in zip(dense_rows, lexical_list):
            dense = row.tolist() if hasattr(row, "tolist") else list(row)
            sparse = SparseVector(
                indices=[int(token) for token in lexical.keys()],
                values=[float(weight) for weight in lexical.values()],
            )
            results.append(Embedding(dense=dense, sparse=sparse))
        return results
