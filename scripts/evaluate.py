"""检索质量评测与消融对比。

    python scripts/evaluate.py                    # 完整链路
    python scripts/evaluate.py --ablations        # 加上消融对比表

消融对比回答一个具体问题：混合检索和重排各自贡献了多少。
没有这组数字，所有"优化"都只是感觉。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from qdrant_client import QdrantClient

from ragkb.chunking import split_document
from ragkb.embedding import build_embedder
from ragkb.evaluation import evaluate_retrieval, load_jsonl
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.parsing import parse_document
from ragkb.pipeline.ingest import iter_documents
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import build_reranker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = ROOT / "demo"
DEFAULT_GOLDEN = ROOT / "eval" / "golden_set.jsonl"


def build_retriever(
    docs: Path,
    embedder_name: str,
    config: RetrievalConfig,
    use_rerank: bool,
    reranker_name: str = "auto",
) -> Retriever:
    embedder = build_embedder(embedder_name)
    client = QdrantClient(":memory:")
    store = QdrantStore(client, collection="kb", dim=embedder.dim)
    store.ensure(recreate=True)

    reranker = None
    if use_rerank:
        # 替身 embedder 没必要消耗 API 调用
        name = "lexical" if (reranker_name == "auto" and embedder_name == "hashing") else reranker_name
        reranker = build_reranker(name)

    retriever = Retriever(embedder, store, reranker=reranker, config=config)

    for path in iter_documents(docs):
        doc = parse_document(path)
        retriever.index_document(doc, split_document(doc))

    return retriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--embedder", default="hashing", choices=["hashing", "api", "bge-m3", "auto"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="只走向量召回、不融合不重排，用于直接比较不同 embedder 的语义能力",
    )
    parser.add_argument(
        "--reranker",
        default="auto",
        choices=["auto", "lexical", "dashscope", "bge"],
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="拒答阈值，取 eval/thresholds.json 里标定出的值；不传则不拒答",
    )
    args = parser.parse_args()

    items = load_jsonl(args.golden)
    print(f"金标集 {len(items)} 条 · embedder={args.embedder} · k={args.k}\n")

    base = RetrievalConfig(min_score=args.min_score)
    if args.dense_only:
        base = replace(base, use_sparse=False, use_rerank=False)
    retriever = build_retriever(
        Path(args.docs),
        args.embedder,
        base,
        use_rerank=not args.dense_only,
        reranker_name=args.reranker,
    )
    report = evaluate_retrieval(retriever, items, k=args.k)

    print(f"完整链路：{report.summary()}")
    failures = report.failures()
    if failures:
        print(f"\n失败 {len(failures)} 条：")
        for item in failures:
            print(f"  ✗ {item.question}")
            print(
                f"    命中={item.hit} rank={item.rank} "
                f"缺关键词={item.missing_keywords} 拒答={item.refused}"
            )
    else:
        print("\n全部通过。")

    if args.ablations:
        variants = {
            "仅稠密向量": (replace(base, use_sparse=False, use_rerank=False), False),
            "稠密+稀疏（无重排）": (replace(base, use_sparse=True, use_rerank=False), False),
            "稠密+稀疏+重排": (replace(base, use_sparse=True, use_rerank=True), True),
        }
        print("\n消融对比：")
        print(f"  {'配置':<20}{'hit':>8}{'MRR':>8}{'keyword':>9}")
        reports: dict[str, object] = {}
        for label, (config, use_rerank) in variants.items():
            r = build_retriever(
                Path(args.docs),
                args.embedder,
                config,
                use_rerank=use_rerank,
                reranker_name=args.reranker,
            )
            rep = evaluate_retrieval(r, items, k=args.k)
            reports[label] = rep
            print(
                f"  {label:<20}{rep.hit_rate:>8.3f}{rep.mrr:>8.3f}"
                f"{rep.keyword_coverage:>9.3f}"
            )

        # 结论从这一轮的数字里推出来，不写死。写死的话换一套语料就会变成误导。
        if args.reranker == "lexical":
            note = "重排用的是字面替身，对语义改述的查询有害，这一行只作对照。"
        elif args.embedder == "hashing":
            note = "替身 embedder 只按字面重叠打分，结论仅反映链路，不代表语义能力。"
        else:
            dense = reports["仅稠密向量"].mrr
            hybrid = reports["稠密+稀疏（无重排）"].mrr
            rerank = reports["稠密+稀疏+重排"].mrr
            if abs(hybrid - dense) < 5e-4:
                note = (
                    f"稀疏侧与纯稠密持平（{dense:.3f}）：这批问题都是自然语言描述，"
                    "没有型号、编号、条款号这类需要精确匹配的 token，它的价值测不出来。"
                )
            elif hybrid > dense:
                note = f"稀疏侧把 MRR 从 {dense:.3f} 提到 {hybrid:.3f}，重排再加到 {rerank:.3f}。"
            else:
                note = (
                    f"稀疏侧反而把 MRR 从 {dense:.3f} 拉到 {hybrid:.3f}——"
                    "在语义问法上它引入的是噪声，不是补充。"
                )
        print(f"\n  注：{note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
