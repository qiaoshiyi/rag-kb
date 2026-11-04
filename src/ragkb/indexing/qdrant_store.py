"""Qdrant 索引层。

一个 collection 内同时存稠密与稀疏向量：
- dense 走语义相似
- sparse 走词法匹配（型号、编号、专有名词）
- payload 里带 acl_tags，权限过滤在检索时下推，不做事后过滤

point id 用 uuid5(chunk_id) 而不是自增 id，
这样重复入库同一份文档是幂等的，不会产生重复点。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from ragkb.chunking.hierarchy import Chunk
from ragkb.embedding.base import Embedding, SparseVector

DENSE = "dense"
SPARSE = "sparse"

_BATCH = 128


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@dataclass(slots=True)
class ScoredHit:
    chunk_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))


class QdrantStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str = "kb",
        dim: int = 1024,
        with_sparse: bool = True,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim
        self.with_sparse = with_sparse

    # ---------- collection ----------

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure(self, recreate: bool = False) -> None:
        if recreate and self.exists():
            self.client.delete_collection(self.collection)

        if self.exists():
            return

        sparse_config = (
            {SPARSE: models.SparseVectorParams()} if self.with_sparse else None
        )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config=sparse_config,
        )

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def count_documents(self) -> int:
        """不同 doc_id 的数量。

        加载预建索引时启动流程跳过了建索引，文档数只能从已有 payload 里数出来，
        否则 /health 会报 0 篇、前端弹「空索引」横幅，实际上库是满的。
        """
        seen: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            for point in points:
                doc_id = (point.payload or {}).get("doc_id")
                if doc_id:
                    seen.add(str(doc_id))
            if offset is None:
                break
        return len(seen)

    # ---------- 写入 ----------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Embedding],
        doc_meta: dict[str, Any] | None = None,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量必须一致")

        points: list[models.PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            vector: dict[str, Any] = {DENSE: emb.dense}
            if self.with_sparse and emb.sparse:
                vector[SPARSE] = models.SparseVector(
                    indices=emb.sparse.indices, values=emb.sparse.values
                )
            payload = chunk.to_payload(doc_meta)
            points.append(
                models.PointStruct(id=point_id(chunk.chunk_id), vector=vector, payload=payload)
            )

        for start in range(0, len(points), _BATCH):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + _BATCH],
                wait=True,
            )
        return len(points)

    def delete_by_doc(self, doc_id: str) -> None:
        """增量更新：文档变更时先删旧块，再写新块，避免残留过期内容。"""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    # ---------- 检索 ----------

    def search_dense(
        self, vector: Sequence[float], limit: int, flt: models.Filter | None = None
    ) -> list[ScoredHit]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            using=DENSE,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [self._to_hit(p) for p in response.points]

    def search_sparse(
        self, sparse: SparseVector, limit: int, flt: models.Filter | None = None
    ) -> list[ScoredHit]:
        if not sparse:
            return []
        response = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=sparse.indices, values=sparse.values),
            using=SPARSE,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [self._to_hit(p) for p in response.points]

    def fetch(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """按 chunk_id 批量取 payload，用于命中子块后回填父块。"""
        if not chunk_ids:
            return {}
        ids = [point_id(cid) for cid in chunk_ids]
        records = self.client.retrieve(
            collection_name=self.collection, ids=ids, with_payload=True
        )
        out: dict[str, dict[str, Any]] = {}
        for record in records:
            payload = record.payload or {}
            cid = payload.get("chunk_id")
            if cid:
                out[str(cid)] = payload
        return out

    @staticmethod
    def _to_hit(point: Any) -> ScoredHit:
        payload = point.payload or {}
        return ScoredHit(
            chunk_id=str(payload.get("chunk_id", point.id)),
            score=float(point.score),
            payload=payload,
        )

    # ---------- 便捷构造 ----------


def acl_filter(tags: Sequence[str]) -> models.Filter:
    """权限过滤在检索时下推。绝不能在生成后再删结果——那时模型已经看过内容了。"""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="acl_tags", match=models.MatchAny(any=list(tags))
            )
        ]
    )


def doc_filter(doc_ids: Sequence[str]) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))
        ]
    )
