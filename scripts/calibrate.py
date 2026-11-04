"""拒答阈值标定：用正负样本的分数分布决定 min_score。

    export DASHSCOPE_API_KEY=sk-xxx
    python scripts/calibrate.py --embedder api --reranker dashscope
    python scripts/calibrate.py --reuse            # 复用上次采到的分数，不调 API

输出 eval/thresholds.json，服务层和评测脚本都会读它。
没有这组数字之前，min_score 一直是关闭状态——那不是严谨，是缺一张表。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ragkb.chunking import split_document
from ragkb.embedding import build_embedder
from ragkb.evaluation import evaluate_at, load_jsonl, sweep
from ragkb.evaluation.calibration import apply_safety_margin, load, save
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.parsing import parse_document
from ragkb.pipeline.ingest import iter_documents
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import build_reranker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = ROOT / "demo"
DEFAULT_POSITIVE = [
    ROOT / "eval" / "golden_set.jsonl",
    ROOT / "eval" / "golden_set_hard.jsonl",
]
DEFAULT_NEGATIVE = ROOT / "eval" / "negative_set.jsonl"
DEFAULT_OUT = ROOT / "eval" / "thresholds.json"


def build_retriever(docs: Path, embedder_name: str, reranker_name: str) -> Retriever:
    embedder = build_embedder(embedder_name)
    store = QdrantStore(QdrantClient(":memory:"), collection="kb", dim=embedder.dim)
    store.ensure(recreate=True)

    name = "lexical" if (reranker_name == "auto" and embedder_name == "hashing") else reranker_name
    reranker = build_reranker(name)
    # 标定时必须关掉阈值，否则拿不到被拒样本的真实分数
    retriever = Retriever(
        embedder,
        store,
        reranker=reranker,
        config=RetrievalConfig(min_score=None, min_raw_score=None),
    )
    for path in iter_documents(docs):
        doc = parse_document(path)
        retriever.index_document(doc, split_document(doc))
    return retriever


def collect(retriever: Retriever, questions: list[str]) -> list[float]:
    return [float(retriever.search(q).debug.get("top_score") or 0.0) for q in questions]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    parser.add_argument("--positive", action="append", default=None, help="可多次指定")
    parser.add_argument("--negative", default=str(DEFAULT_NEGATIVE))
    parser.add_argument("--embedder", default="api", choices=["hashing", "api", "bge-m3", "auto"])
    parser.add_argument(
        "--reranker",
        default="dashscope",
        choices=["auto", "lexical", "dashscope", "bge", "siliconflow"],
    )
    parser.add_argument(
        "--cost-refusal",
        type=float,
        default=2.0,
        help="误杀相对漏放的代价倍数，默认 2（误杀直接丢答案，漏放还有生成层兜底）",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.05,
        help="把阈值往「少误杀」方向下移的边距，默认 0.05；设 0 则直接用扫描最优点",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="复用 --out 文件里已采到的分数，只重算阈值（调 cost/margin 时用）",
    )
    args = parser.parse_args()

    positive_paths = args.positive or [str(p) for p in DEFAULT_POSITIVE]
    questions: list[str] = []
    for p in positive_paths:
        questions.extend(item.question for item in load_jsonl(p))
    negatives = [item.question for item in load_jsonl(args.negative)]

    print(
        f"正样本 {len(questions)} 条 · 负样本 {len(negatives)} 条 · "
        f"embedder={args.embedder} · reranker={args.reranker}\n"
    )

    if args.reuse:
        cached = load(args.out)
        pos_scores = list(cached["positives"])
        neg_scores = list(cached["negatives"])
        print("复用已采分数（未调用模型）\n")
    else:
        retriever = build_retriever(Path(args.docs), args.embedder, args.reranker)
        pos_scores = collect(retriever, questions)
        neg_scores = collect(retriever, negatives)

    if args.verbose:
        print("正样本分数（降序）：")
        for q, s in sorted(zip(questions, pos_scores), key=lambda x: -x[1]):
            print(f"  {s:.4f}  {q}")
        print("\n负样本分数（降序）：")
        for q, s in sorted(zip(negatives, neg_scores), key=lambda x: -x[1]):
            print(f"  {s:.4f}  {q}")
        print()

    raw = sweep(pos_scores, neg_scores, cost_ratio=args.cost_refusal)
    cal = apply_safety_margin(raw, args.safety_margin)

    print(f"正样本区间 [{min(pos_scores):.3f}, {max(pos_scores):.3f}] · "
          f"负样本区间 [{min(neg_scores):.3f}, {max(neg_scores):.3f}]")
    if cal.overlap > 0:
        if cal.overlap >= 0.10:
            print(f"两类完全可分，间隔 {cal.overlap:.3f}，阈值有充足余量。")
        else:
            print(f"两类完全可分但间隔仅 {cal.overlap:.3f}——阈值夹在窄缝里，换一批文档就要重标。")
    else:
        print(f"两类有重叠，宽度 {abs(cal.overlap):.3f}——不存在零误杀又零漏放的阈值，只能按代价取舍。")

    print(f"\n扫描最优点：{raw.summary()}")
    if abs(cal.threshold - raw.threshold) > 1e-6:
        drop = raw.threshold - cal.threshold
        print(
            f"安全边距后：{cal.summary()}\n"
            f"  阈值下移 {drop:.3f}（{raw.threshold:.3f} → {cal.threshold:.3f}），"
            f"换来对未见正样本的余量；多放进来 {cal.false_accept - raw.false_accept:.3f} 的负样本，"
            "由生成层拒答兜底。"
        )
    else:
        print(f"最终阈值：{cal.threshold:.3f}")

    print("\n候选阈值对比：")
    print(f"  {'阈值':<8}{'误杀':>8}{'漏放':>8}{'加权代价':>10}")
    marks = {round(cal.threshold, 3): "  ← 选定"}
    for t in sorted({0.3, 0.5, 0.6, 0.7, round(raw.threshold, 3), round(cal.threshold, 3)}):
        fr, fa = evaluate_at(t, pos_scores, neg_scores)
        cost = args.cost_refusal * fr + fa
        print(f"  {t:<8.3f}{fr:>8.3f}{fa:>8.3f}{cost:>10.3f}{marks.get(t, '')}")

    path = save(
        cal,
        args.out,
        meta={
            "embedder": args.embedder,
            "reranker": args.reranker,
            "positives": len(pos_scores),
            "negatives": len(neg_scores),
            "docs": str(args.docs),
            "safety_margin": args.safety_margin,
        },
    )
    try:
        shown = path.resolve().relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"\n已写入 {shown}")
    print(f"启用方式：KB_MIN_SCORE={cal.threshold:.2f} python -m ragkb.api.app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
