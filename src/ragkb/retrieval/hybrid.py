"""混合召回与 RRF 融合。

RRF（Reciprocal Rank Fusion）不需要归一化两路分数，只依赖名次：
    score = Σ  weight_i / (k + rank_i)
稠密分是余弦相似度、稀疏分是词权重内积，量纲完全不同，直接加权求和必然失衡。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from ragkb.indexing.qdrant_store import ScoredHit


@dataclass(slots=True)
class Hit:
    chunk_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    raw_score: float = 0.0
    """两路原始分的较大值。只能用于横向比较，不能当绝对阈值——
    稠密是 0~1 余弦，稀疏是无上界的内积，两者不可比。"""
    rerank_score: float | None = None
    source_scores: dict[str, float] = field(default_factory=dict)
    """按来源记录的原始分，如 {"dense": 0.62, "sparse": 5.3}。拒答阈值只看 dense。"""

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    @property
    def section_path(self) -> str:
        return str(self.payload.get("section_path", ""))

    @property
    def doc_id(self) -> str:
        return str(self.payload.get("doc_id", ""))

    @property
    def parent_id(self) -> str | None:
        pid = self.payload.get("parent_id")
        return str(pid) if pid else None


def rrf_fuse(
    runs: Sequence[Sequence[ScoredHit]],
    k: int = 60,
    names: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    if names is None:
        names = [f"run{i}" for i in range(len(runs))]
    if weights is None:
        weights = [1.0] * len(runs)

    fused: dict[str, float] = defaultdict(float)
    payloads: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[str]] = defaultdict(list)
    raw: dict[str, float] = {}
    per_source: dict[str, dict[str, float]] = defaultdict(dict)

    for name, weight, run in zip(names, weights, runs):
        for rank, hit in enumerate(run, start=1):
            fused[hit.chunk_id] += weight / (k + rank)
            payloads.setdefault(hit.chunk_id, hit.payload)
            sources[hit.chunk_id].append(name)
            raw[hit.chunk_id] = max(raw.get(hit.chunk_id, 0.0), hit.score)
            per_source[hit.chunk_id][name] = hit.score

    hits = [
        Hit(
            chunk_id=cid,
            score=score,
            payload=payloads[cid],
            sources=tuple(sources[cid]),
            raw_score=raw[cid],
            source_scores=per_source[cid],
        )
        for cid, score in fused.items()
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits
