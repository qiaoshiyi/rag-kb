"""检索编排：双路召回 → RRF → 重排 → 拒答 → 父块回填。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import models

from ragkb.chunking.hierarchy import Chunk
from ragkb.embedding.base import Embedder
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.models.document import Document
from ragkb.retrieval.hybrid import Hit, rrf_fuse
from ragkb.retrieval.rerank import Reranker


@dataclass(slots=True)
class RetrievalConfig:
    fetch_k: int = 30
    candidate_k: int = 20
    top_n: int = 6
    min_score: float | None = None
    """重排后的拒答阈值，默认关闭。

    这个值必须在自己的验证集上标定后才能启用：取正样本最低分与负样本最高分的中间值。
    跨 reranker 复用同一个数字会误杀——lexical 重排与 bge-reranker 的分数分布完全不同，
    实测 0.3 就足以把「出差住酒店能报多少」这种有答案的问题拒掉。
    不标定就启用，比不拒答更糟：生成层还有一道拒答兜底，误杀则是直接丢答案。
    """
    min_raw_score: float | None = None
    """未重排时的拒答阈值，只作用于稠密余弦分（0~1）。

    默认关闭：稀疏内积没有上界，RRF 分只有 0.01 量级，都不是可解释的绝对分。
    没有 cross-encoder 时宁可让下游 prompt 兜底，也不要用一个假阈值乱拒。
    """
    rrf_k: int = 60
    rrf_weights: tuple[float, float] = (1.0, 1.0)
    path_weights: tuple[float, ...] = (1.0, 0.7)
    """双路召回时每条查询路径的权重：改写问题 1.0、原问题 0.7。

    原问题那一路是兜底——改写模型一旦跑偏，它多少还能捞回相关片段。
    权重压在 1.0 以下是因为追问原文（"那海外呢"）本身信息量太低，
    与改写结果等权会把噪声一起放大。
    """
    use_sparse: bool = True
    use_rerank: bool = True
    expand_parents: bool = True


@dataclass(slots=True)
class RetrievalResult:
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


def _build_filter(extra: models.Filter | None) -> models.Filter:
    """检索时永远排除父块：父块只用于回填上下文，不参与相似度竞争。"""
    must_not = [
        models.FieldCondition(key="kind", match=models.MatchValue(value="parent"))
    ]
    must = list(extra.must) if extra is not None and extra.must else []
    return models.Filter(must=must, must_not=must_not)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: QdrantStore,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    # ---------- 写入 ----------

    def index_document(
        self,
        doc: Document,
        chunks: list[Chunk],
        doc_meta: dict[str, Any] | None = None,
        replace: bool = True,
    ) -> int:
        """文档级入库。replace=True 时先按 doc_id 删旧块，实现增量更新。"""
        meta = dict(doc_meta or {})
        meta.setdefault("title", doc.title)
        meta.setdefault("source_type", doc.source_type)
        meta.setdefault("filename", doc.meta.get("filename", ""))
        # 解析阶段传入的部门、权限、版本要一路带到 payload，二期加 ACL 不必重建索引
        for key in ("dept", "acl_tags", "version"):
            if key in doc.meta and key not in meta:
                meta[key] = doc.meta[key]

        doc_id = chunks[0].doc_id if chunks else doc.sha256[:16]
        if replace:
            self.store.delete_by_doc(doc_id)

        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        return self.store.upsert_chunks(chunks, embeddings, meta)

    # ---------- 检索 ----------

    def search(
        self, query: str, flt: models.Filter | None = None, config: RetrievalConfig | None = None
    ) -> RetrievalResult:
        return self.search_multi([query], flt=flt, config=config)

    def search_multi(
        self,
        queries: list[str],
        flt: models.Filter | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        """多查询召回。queries[0] 为主查询：重排以它为基准。

        多轮场景传 [改写问题, 原问题]：改写那路负责精确命中，
        原问题那路防止改写跑偏，两路 RRF 融合后统一重排。
        重排必须只用主查询——把两个问题拼起来喂给 cross-encoder 会稀释判定。
        """
        cfg = config or self.config
        unique: list[str] = []
        for q in queries:
            if q and q.strip() and q not in unique:
                unique.append(q)
        if not unique:
            raise ValueError("queries 不能为空")

        query_filter = _build_filter(flt)
        runs: list[list[Any]] = []
        names: list[str] = []
        weights: list[float] = []
        dense_total = 0
        sparse_total = 0

        for path_index, text in enumerate(unique):
            path_weight = (
                cfg.path_weights[path_index]
                if path_index < len(cfg.path_weights)
                else cfg.path_weights[-1]
            )
            embedding = self.embedder.embed_query(text)

            suffix = "" if len(unique) == 1 else str(path_index)
            dense_run = self.store.search_dense(embedding.dense, cfg.fetch_k, query_filter)
            dense_total += len(dense_run)
            runs.append(dense_run)
            names.append(f"dense{suffix}")
            weights.append(path_weight * cfg.rrf_weights[0])

            if cfg.use_sparse and embedding.sparse:
                sparse_run = self.store.search_sparse(embedding.sparse, cfg.fetch_k, query_filter)
                sparse_total += len(sparse_run)
                runs.append(sparse_run)
                names.append(f"sparse{suffix}")
                weights.append(path_weight * cfg.rrf_weights[1])

        fused = rrf_fuse(runs, k=cfg.rrf_k, names=names, weights=weights)
        candidates = fused[: cfg.candidate_k]

        debug: dict[str, Any] = {
            "queries": unique,
            "paths": len(unique),
            "dense_hits": dense_total,
            "sparse_hits": sparse_total,
            "fused": len(fused),
            "candidates": len(candidates),
            "reranked": False,
        }

        if cfg.use_rerank and self.reranker is not None and candidates:
            candidates = self.reranker.rerank(unique[0], candidates)
            debug["reranked"] = True

        top = candidates[: cfg.top_n]
        debug["top_score"] = round(top[0].score, 4) if top else 0.0

        if not top:
            return RetrievalResult(refused=True, reason="no_candidates", debug=debug)

        # 阈值必须和当前评分量纲对齐：
        # 重排后看交叉编码器分（0~1），未重排只看稠密余弦（稀疏内积无上界，不能当绝对分）
        reranked = bool(debug["reranked"])
        if reranked:
            gate_score: float | None = top[0].score
            threshold: float | None = cfg.min_score
        else:
            gate_score = next(
                (v for k, v in top[0].source_scores.items() if k.startswith("dense")), None
            )
            threshold = cfg.min_raw_score

        debug["gate"] = {
            "score": round(gate_score, 4) if gate_score is not None else None,
            "threshold": threshold,
        }

        if gate_score is not None and threshold is not None and gate_score < threshold:
            return RetrievalResult(
                refused=True,
                reason=f"below_threshold:{gate_score:.3f}<{threshold}",
                debug=debug,
            )

        if cfg.expand_parents:
            self._attach_parents(top)

        return RetrievalResult(hits=top, debug=debug)

    def _attach_parents(self, hits: list[Hit]) -> None:
        parent_ids = sorted({pid for pid in (h.parent_id for h in hits) if pid})
        if not parent_ids:
            return

        parents = self.store.fetch(parent_ids)
        for hit in hits:
            pid = hit.parent_id
            if pid and pid in parents:
                hit.payload["parent_text"] = parents[pid].get("text", "")
                hit.payload["parent_section"] = parents[pid].get("section_path", "")
