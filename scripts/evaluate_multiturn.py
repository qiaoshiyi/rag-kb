"""多轮检索对比评测：裸问 / 拼历史 / 改写+双路。

只回答一个问题：多轮改写到底值不值得做。
三个策略跑同一个索引、同一个重排器、同一套指标，差别只在「用什么查询去召回」。

    export DASHSCOPE_API_KEY=sk-xxx
    python scripts/make_corpus.py                                  # 先生成 16 篇评测语料
    python scripts/evaluate_multiturn.py --embedder api --reranker dashscope
    python scripts/evaluate_multiturn.py --rewriter heuristic      # 不调模型

用 corpus/ 而不是 demo/：8 篇小语料里任意一块都能进前 5，改写的差异测不出来。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ragkb.chunking import split_document
from ragkb.embedding import build_embedder
from ragkb.evaluation import evaluate_retrieval, load_jsonl
from ragkb.evaluation.dataset import GoldenItem
from ragkb.generation.base import Message
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.multiturn import build_rewriter, prepare_query
from ragkb.parsing import parse_document
from ragkb.pipeline.ingest import iter_documents
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import build_reranker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = ROOT / "corpus"
DEFAULT_GOLDEN = ROOT / "eval" / "multiturn_set.jsonl"


def build_retriever(
    docs: Path, embedder_name: str, reranker_name: str, min_score: float | None
) -> Retriever:
    embedder = build_embedder(embedder_name)
    store = QdrantStore(QdrantClient(":memory:"), collection="kb", dim=embedder.dim)
    store.ensure(recreate=True)

    name = "lexical" if (reranker_name == "auto" and embedder_name == "hashing") else reranker_name
    retriever = Retriever(
        embedder,
        store,
        reranker=build_reranker(name),
        config=RetrievalConfig(min_score=min_score),
    )
    for path in iter_documents(docs):
        doc = parse_document(path)
        retriever.index_document(doc, split_document(doc))
    return retriever


def to_messages(history: list[dict[str, str]]) -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in history]


def prev_user(history: list[dict[str, str]]) -> str:
    for message in reversed(history):
        if message["role"] == "user":
            return message["content"]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--embedder", default="api", choices=["hashing", "api", "bge-m3", "auto"])
    parser.add_argument("--reranker", default="dashscope", choices=["auto", "lexical", "dashscope", "bge"])
    parser.add_argument("--rewriter", default="llm", choices=["llm", "heuristic", "none"])
    parser.add_argument("--rewrite-model", default=os.environ.get("KB_REWRITE_MODEL"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--show-rewrites", action="store_true")
    args = parser.parse_args()

    items = load_jsonl(args.golden)
    if not items:
        print("金标集为空")
        return 1

    print(
        f"多轮金标集 {len(items)} 条 · embedder={args.embedder} · "
        f"reranker={args.reranker} · rewriter={args.rewriter} · k={args.k}\n"
    )

    retriever = build_retriever(Path(args.docs), args.embedder, args.reranker, args.min_score)
    rewriter = build_rewriter(args.rewriter, model=args.rewrite_model)

    def raw(item: GoldenItem) -> list[str]:
        return [item.question]

    def concat(item: GoldenItem) -> list[str]:
        # 朴素做法：把上一轮问题直接拼在当前问题前面
        previous = prev_user(item.history)
        return [f"{previous} {item.question}".strip() if previous else item.question]

    def rewritten(item: GoldenItem) -> list[str]:
        prepared = prepare_query(
            item.question,
            to_messages(item.history),
            rewriter=rewriter,
            embedder=retriever.embedder,
        )
        return prepared.queries()

    strategies = {
        "裸问（不处理）": raw,
        "拼接上一轮": concat,
        f"改写+双路（{args.rewriter}）": rewritten,
    }

    reports = {}
    header = f"  {'策略':<24}{'hit':>7}{'MRR':>7}{'kw':>7}{'span':>7}{'spanMRR':>9}{'拒答':>7}"
    print(header)
    for label, fn in strategies.items():
        report = evaluate_retrieval(retriever, items, k=args.k, queries_for=fn)
        reports[label] = report
        print(
            f"  {label:<24}{report.hit_rate:>7.3f}{report.mrr:>7.3f}"
            f"{report.keyword_coverage:>7.3f}{report.span_rate:>7.3f}"
            f"{report.span_mrr:>9.3f}{report.refusal_rate:>7.3f}"
        )

    # 逐条对比答案片段排名，看清差异具体落在哪几条
    print(f"\n  逐条答案片段排名（越靠前越好，— 表示未进前 {args.k}）：")
    print(f"  {'追问':<22}{'裸问':>8}{'拼历史':>8}{'改写':>8}")
    labels = list(strategies)
    for index, item in enumerate(items):
        cells = []
        for label in labels:
            rank = reports[label].items[index].span_rank
            cells.append(f"{rank}" if rank else "—")
        print(f"  {item.question:<22}{cells[0]:>8}{cells[1]:>8}{cells[2]:>8}")

    # ---- 改写本身的质量 ----
    prepared_all = []
    for item in items:
        prepared = prepare_query(
            item.question,
            to_messages(item.history),
            rewriter=rewriter,
            embedder=retriever.embedder,
        )
        prepared_all.append((item, prepared))

    anchored = [p for item, p in prepared_all if item.must_contain]
    anchor_ok = [
        p for item, p in prepared_all if item.must_contain and all(t in p.rewritten for t in item.must_contain)
    ]
    resets = [(item, p) for item, p in prepared_all if item.expect_reset]
    reset_ok = [(item, p) for item, p in resets if p.reset]
    changed = [p for _item, p in prepared_all if p.changed]

    print("\n改写质量：")
    if anchored:
        print(f"  上下文锚点补齐率：{len(anchor_ok)}/{len(anchored)} = {len(anchor_ok)/len(anchored):.3f}")
    if resets:
        print(f"  话题跳转识别率：{len(reset_ok)}/{len(resets)} = {len(reset_ok)/len(resets):.3f}")
    print(f"  实际改写比例：{len(changed)}/{len(prepared_all)}")

    if args.show_rewrites:
        print("\n逐条改写结果：")
        for item, prepared in prepared_all:
            flag = "✓" if not item.must_contain or all(t in prepared.rewritten for t in item.must_contain) else "✗"
            print(f"  {flag} {item.question}")
            print(f"      → {prepared.rewritten}")
            print(f"      reason={prepared.reason} sim={prepared.similarity}")
            if item.standalone:
                print(f"      参考：{item.standalone}")

    ours = reports[f"改写+双路（{args.rewriter}）"]
    base = reports["裸问（不处理）"]
    print("\n结论：")
    print(
        f"  改写把 hit@{args.k} 从 {base.hit_rate:.3f} 提到 {ours.hit_rate:.3f}"
        f"（{(ours.hit_rate - base.hit_rate) * 100:+.1f} 个百分点），"
        f"MRR 从 {base.mrr:.3f} 到 {ours.mrr:.3f}"
    )

    failures = ours.failures()
    if failures:
        print(f"\n改写后仍失败 {len(failures)} 条：")
        for result in failures:
            print(f"  ✗ {result.question} 命中={result.hit} 缺词={result.missing_keywords}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
