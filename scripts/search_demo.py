"""端到端检索演示：解析 → 切分 → 双向量索引 → 混合召回 → RRF → 重排 → 父块回填。

    python scripts/search_demo.py
    python scripts/search_demo.py --embedder bge-m3        # 装好 FlagEmbedding 之后

默认用 HashingEmbedder（字面匹配替身），只能证明链路通，
不反映检索质量；真实效果必须用 bge-m3 + 金标集评估。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ragkb.chunking import split_document
from ragkb.embedding import build_embedder
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.parsing import parse_document
from ragkb.pipeline.ingest import iter_documents
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import build_reranker

QUERIES = [
    "住宿费标准是多少",
    "年假有几天",
    "十万元以上的采购谁审批",
    "账号密码多久更换一次",
]


def preview(text: str, width: int = 70) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width] + "…"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="demo")
    parser.add_argument("--embedder", default="hashing", choices=["hashing", "bge-m3", "auto"])
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    embedder = build_embedder(args.embedder)
    reranker = build_reranker("lexical" if args.embedder == "hashing" else "auto")

    client = QdrantClient(":memory:")
    store = QdrantStore(client, collection="kb", dim=embedder.dim)
    store.ensure(recreate=True)

    retriever = Retriever(
        embedder,
        store,
        reranker=reranker,
        config=RetrievalConfig(top_n=args.top_n),
    )

    print(f"embedder = {type(embedder).__name__} (dim={embedder.dim})")
    print(f"reranker = {type(reranker).__name__}\n")

    total = 0
    for path in iter_documents(args.path):
        doc = parse_document(path)
        chunks = split_document(doc)
        written = retriever.index_document(doc, chunks)
        total += written
        print(f"  索引 {path.name:<22} {len(chunks):>3} 块   ({doc.source_type})")
    print(f"  合计 {total} 个点（父块 + 子块）\n")

    for query in QUERIES:
        result = retriever.search(query)
        d = result.debug
        print(f"查询：{query}")
        print(
            f"  召回 dense {d['dense_hits']:>2} / sparse {d['sparse_hits']:>2}"
            f" → 融合 {d['fused']:>2} → 重排 {d['candidates']:>2} → 取 {len(result.hits)}"
        )

        if result.refused:
            print(f"  ✗ 拒答（{result.reason}）\n")
            continue

        for rank, hit in enumerate(result.hits, start=1):
            parent = hit.payload.get("parent_text", "")
            print(f"  {rank}. [{hit.score:.3f}] {hit.section_path or '(无标题)'}")
            print(f"     {preview(hit.text)}")
            if parent:
                print(f"     ↳ 父块 {len(parent)} 字：{preview(parent, 60)}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
