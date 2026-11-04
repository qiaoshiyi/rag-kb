"""预建 Qdrant 本地索引，供部署时直接加载。

为什么要有这个：冷启动最贵的部分是「解析 12 份文档 + 逐块向量化」，
本地要 12 秒，放到 0.1 CPU 的免费实例上要一分钟以上。把这步挪到构建阶段，
运行时就只剩打开一个 3 MB 的目录。

    python scripts/build_seed_index.py                       # 默认 corpus_full + hashing
    python scripts/build_seed_index.py --backend api         # 用真实 embedding（需 API key）

**构建时的 --backend 必须和运行时的 KB_EMBEDDER 一致**，否则维度对不上，检索直接报错。
会同时写一份 seed_meta.json 记下用的是哪个后端，方便对账。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from ragkb.chunking import split_document  # noqa: E402
from ragkb.embedding import build_embedder  # noqa: E402
from ragkb.indexing.qdrant_store import QdrantStore  # noqa: E402
from ragkb.parsing import ParseError, parse_document  # noqa: E402
from ragkb.pipeline.ingest import iter_documents  # noqa: E402
from ragkb.retrieval import RetrievalConfig, Retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="预建 Qdrant 本地索引")
    parser.add_argument("--docs", default="corpus_full", help="语料目录")
    parser.add_argument("--out", default="deploy/qdrant_seed", help="输出目录")
    parser.add_argument(
        "--backend",
        default="hashing",
        choices=["hashing", "api", "bge-m3"],
        help="向量后端；必须与运行时 KB_EMBEDDER 一致",
    )
    parser.add_argument("--keep", action="store_true", help="保留已有输出目录（增量写入）")
    args = parser.parse_args()

    docs = (ROOT / args.docs).resolve()
    if not docs.is_dir():
        raise SystemExit(f"语料目录不存在：{docs}")

    out = (ROOT / args.out).resolve()
    if out.exists() and not args.keep:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    embedder = build_embedder(args.backend)
    client = QdrantClient(path=str(out))
    try:
        store = QdrantStore(client, collection="kb", dim=embedder.dim)
        store.ensure(recreate=True)
        retriever = Retriever(
            embedder, store, reranker=None, config=RetrievalConfig(min_score=None)
        )

        ok = failed = 0
        for path in iter_documents(docs):
            try:
                doc = parse_document(path)
                retriever.index_document(doc, split_document(doc))
                ok += 1
            except ParseError as exc:
                failed += 1
                print(f"  跳过 {path.name}: {exc}", file=sys.stderr)

        points = store.count()
        doc_count = store.count_documents()
        elapsed = time.time() - t0

        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        meta = {
            "docs_dir": docs.name,
            "backend": args.backend,
            "dim": embedder.dim,
            "documents": doc_count,
            "points": points,
            "failed": failed,
            "build_seconds": round(elapsed, 1),
            "size_mb": round(size / 1048576, 2),
        }
        (out / "seed_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"索引已写入 {out}")
        print(
            f"  {doc_count} 篇 / {points} 块 / {meta['size_mb']} MB / "
            f"{elapsed:.1f} 秒（{args.backend}, dim={embedder.dim}）"
        )
        if failed:
            print(f"  有 {failed} 篇解析失败，已跳过", file=sys.stderr)
        return 0 if points else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
