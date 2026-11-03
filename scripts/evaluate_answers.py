"""回答质量评测：带引用生成 + LLM 判分。

    python scripts/make_corpus.py
    export DASHSCOPE_API_KEY=sk-xxx
    python scripts/evaluate_answers.py --min-score 0.771 --model deepseek-v4.1-flash

指标只看趋势，不要拿去追绝对值——判分模型自己也有噪声。
它真正的作用是：改完 prompt / 换模型 / 调阈值之后，能立刻看出有没有变差。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ragkb.chunking import split_document
from ragkb.embedding import build_embedder
from ragkb.evaluation import evaluate_answers, load_jsonl
from ragkb.evaluation.answer_eval import LLMJudge
from ragkb.generation import QAService
from ragkb.generation.compat import build_llm
from ragkb.generation.mock import MockLLM
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.multiturn import build_rewriter
from ragkb.parsing import parse_document
from ragkb.pipeline.ingest import iter_documents
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import build_reranker

ROOT = Path(__file__).resolve().parent.parent


class CountingLLM:
    """数调用次数。阈值到底省了多少钱，只有数出来才说得清。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0

    def complete(self, messages, temperature=0.1, max_tokens=1024) -> str:
        self.calls += 1
        return self._inner.complete(messages, temperature=temperature, max_tokens=max_tokens)

    def stream(self, messages, temperature=0.1, max_tokens=1024):
        self.calls += 1
        yield from self._inner.stream(messages, temperature=temperature, max_tokens=max_tokens)
DEFAULT_DOCS = ROOT / "corpus"
DEFAULT_GOLDEN = ROOT / "eval" / "answer_set.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--embedder", default="api", choices=["hashing", "api", "bge-m3", "auto"])
    parser.add_argument("--reranker", default="dashscope", choices=["auto", "lexical", "dashscope", "bge"])
    parser.add_argument("--llm", default="auto", choices=["auto", "mock", "dashscope", "deepseek"])
    parser.add_argument("--model", default=os.environ.get("KB_MODEL"))
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--rewriter", default="none", choices=["none", "heuristic", "llm"])
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于快速验证")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    items = load_jsonl(args.golden)
    if args.limit:
        items = items[: args.limit]
    print(
        f"回答集 {len(items)} 条（其中应拒答 {sum(1 for i in items if i.expect_refusal)} 条）· "
        f"embedder={args.embedder} · llm={args.llm}\n"
    )

    embedder = build_embedder(args.embedder)
    store = QdrantStore(QdrantClient(":memory:"), collection="kb", dim=embedder.dim)
    store.ensure(recreate=True)

    name = "lexical" if (args.reranker == "auto" and args.embedder == "hashing") else args.reranker
    retriever = Retriever(
        embedder,
        store,
        reranker=build_reranker(name),
        config=RetrievalConfig(min_score=args.min_score),
    )
    for path in iter_documents(Path(args.docs)):
        doc = parse_document(path)
        retriever.index_document(doc, split_document(doc))

    base_llm = None if args.llm == "mock" else build_llm(
        args.llm, **({"model": args.model} if args.model else {})
    )
    if base_llm is None:
        base_llm = MockLLM()
        print("注意：用的是替补 LLM（抽取式），下面的 faithful / coverage 无参考意义\n")
    llm = CountingLLM(base_llm)

    rewriter = None if args.rewriter == "none" else build_rewriter(args.rewriter, model=args.model)

    qa = QAService(retriever, llm, rewriter=rewriter)
    judge = None if isinstance(base_llm, MockLLM) else LLMJudge(llm)

    report = evaluate_answers(qa, items, judge=judge)

    print(f"回答 {report.answered} 条 · 拒答 {report.refused} 条 · 模型调用 {llm.calls} 次")
    print(f"阈值：{'关闭' if args.min_score is None else args.min_score}\n")
    print(report.summary())

    if args.verbose:
        print("\n逐条：")
        print(f"  {'问题':<24}{'拒答':>6}{'含事实':>7}{'引用':>6}{'有据':>6}{'覆盖':>6}")
        for item in report.items:
            def fmt(value, flag=False):
                if value is None:
                    return "—"
                return ("✓" if value else "✗") if flag else str(value)

            print(
                f"  {item.question[:22]:<24}{fmt(item.refused, True):>6}"
                f"{fmt(item.keyword_in_answer, True):>7}{fmt(item.citation_ok, True):>6}"
                f"{fmt(item.faithful, True):>6}{fmt(item.coverage):>6}"
            )
            if item.judge_error:
                print(f"      判分失败：{item.judge_error}")
            elif item.faithful is False and item.verdict_reason:
                print(f"      判分意见：{item.verdict_reason}")

    partial = [
        item
        for item in report.items
        if item.coverage is not None and 0 < item.coverage < 1 and not item.refused
    ]
    if partial:
        print(f"\n部分覆盖 {len(partial)} 条（答到了但少了限定，看判分意见）：")
        for item in partial:
            print(f"  ~ {item.question}")
            print(f"     答：{item.answer[:60]}")
            if item.verdict_reason:
                print(f"     判：{item.verdict_reason}")

    failures = report.failures()
    if failures:
        print(f"\n有问题 {len(failures)} 条：")
        for item in failures:
            flags = []
            if not item.refusal_correct:
                flags.append("该拒未拒" if item.expect_refusal else "该答未答")
            if item.faithful is False:
                flags.append("无依据")
            if (item.coverage or 0) < 0.5 and not item.refused:
                flags.append("没答到要点")
            if item.keyword_in_answer is False:
                flags.append("答案缺事实")
            print(f"  ✗ {item.question}")
            print(f"     {', '.join(flags)}  |  {item.answer[:60]}")
    else:
        print("\n全部通过。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
