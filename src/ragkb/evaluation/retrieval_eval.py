"""检索评测指标（无需 LLM）。

hit@k          期望文档是否落在前 k 条
MRR@k          第一个命中的倒数排名，衡量排序质量
keyword@k      答案关键词是否出现在命中块里——这是 context_recall 的廉价代理，
               能区分「没召回对文档」和「召回对了但切分把答案切碎了」
span@k         含答案关键词的**那一段**是否落在前 k 条，以及它排第几

为什么要 span 指标：文档级 hit 在语料小的时候几乎恒为 1，
一篇文章被切成的任意一块进入前 5 都算命中，差异全被抹平。
按「含答案的块排第几」来看，才能反映多轮改写到底有没有把精确片段提上来。
refusal_rate   不该拒答的问题被拒答的比例
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from ragkb.evaluation import text_match
from ragkb.evaluation.dataset import GoldenItem
from ragkb.retrieval.pipeline import Retriever


@dataclass(slots=True)
class ItemResult:
    question: str
    hit: bool
    rank: int | None
    keyword_ok: bool
    refused: bool
    top1_score: float
    matched_files: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    span_rank: int | None = None
    """含全部答案关键词的块里，排最前的那个的名次。"""
    queries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalReport:
    k: int
    total: int
    hit_rate: float
    mrr: float
    keyword_coverage: float
    refusal_rate: float
    avg_top1_score: float
    span_rate: float = 0.0
    span_mrr: float = 0.0
    items: list[ItemResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"hit@{self.k}={self.hit_rate:.3f}  MRR@{self.k}={self.mrr:.3f}  "
            f"keyword@{self.k}={self.keyword_coverage:.3f}  "
            f"span@{self.k}={self.span_rate:.3f}  spanMRR={self.span_mrr:.3f}  "
            f"refusal={self.refusal_rate:.3f}"
        )

    def failures(self) -> list[ItemResult]:
        return [i for i in self.items if not (i.hit and i.keyword_ok)]


def _collect_text(hits: Sequence) -> str:
    parts = []
    for hit in hits:
        parts.append(str(hit.payload.get("parent_text") or hit.text))
    return "\n".join(parts)


def evaluate_retrieval(
    retriever: Retriever,
    items: Sequence[GoldenItem],
    k: int = 5,
    queries_for: Callable[[GoldenItem], list[str]] | None = None,
) -> EvalReport:
    """queries_for 决定每条用例用哪些查询去召回。

    单轮评测不传，直接用问题本身；多轮评测传不同策略（裸问 / 拼历史 / 改写双路），
    指标口径完全一致，否则对比没有意义。
    """
    results: list[ItemResult] = []

    for item in items:
        queries = queries_for(item) if queries_for else [item.question]
        outcome = retriever.search_multi(queries)
        hits = outcome.hits[:k]

        rank: int | None = None
        matched: list[str] = []
        for position, hit in enumerate(hits, start=1):
            filename = str(hit.payload.get("filename", ""))
            if filename in item.expected_files:
                rank = position if rank is None else rank
                matched.append(filename)

        combined = _collect_text(hits)
        # 数字写法差异（30 分钟 / 三十分钟）不应该判成没答到
        missing = [kw for kw in item.keywords if not text_match.contains(combined, kw)]

        # 答案片段排名：第一个含全部关键词的块排第几
        span_rank: int | None = None
        for position, hit in enumerate(hits, start=1):
            text = str(hit.payload.get("parent_text") or hit.text)
            if item.keywords and text_match.contains_all(text, item.keywords):
                span_rank = position
                break

        results.append(
            ItemResult(
                question=item.question,
                hit=rank is not None,
                rank=rank,
                keyword_ok=not missing,
                refused=outcome.refused,
                top1_score=hits[0].score if hits else 0.0,
                matched_files=matched,
                missing_keywords=missing,
                span_rank=span_rank,
                queries=list(queries),
            )
        )

    total = len(results)
    if total == 0:
        return EvalReport(k, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    hit_rate = sum(1 for r in results if r.hit) / total
    mrr = sum((1.0 / r.rank) for r in results if r.rank) / total
    keyword_cov = sum(1 for r in results if r.keyword_ok) / total
    refusal = sum(1 for r in results if r.refused) / total
    avg_score = sum(r.top1_score for r in results) / total

    scored = [r for r in results if r.span_rank is not None]
    span_rate = len(scored) / total
    span_mrr = sum(1.0 / r.span_rank for r in scored) / total if scored else 0.0

    return EvalReport(
        k=k,
        total=total,
        hit_rate=hit_rate,
        mrr=mrr,
        keyword_coverage=keyword_cov,
        refusal_rate=refusal,
        avg_top1_score=avg_score,
        span_rate=span_rate,
        span_mrr=span_mrr,
        items=results,
    )
