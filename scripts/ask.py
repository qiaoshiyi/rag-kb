"""单次问答演示。

    python scripts/ask.py "年假有几天"
    python scripts/ask.py "年假有几天" --stream
    python scripts/ask.py "年假有几天" --llm dashscope      # 需 DASHSCOPE_API_KEY
    python scripts/ask.py "年假有几天" --llm deepseek       # 需 DEEPSEEK_API_KEY

没有 API key 时自动回退 MockLLM（抽取式作答），
它只验证 prompt 组装与引用映射，不代表真实回答质量。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ragkb.chunking import split_document
from ragkb.embedding import build_embedder
from ragkb.generation import QAService, to_sse
from ragkb.generation.compat import build_llm
from ragkb.generation.mock import MockLLM
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.parsing import parse_document
from ragkb.pipeline.ingest import iter_documents
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import build_reranker

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="年假有几天")
    parser.add_argument("--docs", default=str(ROOT / "demo"))
    parser.add_argument("--embedder", default="hashing", choices=["hashing", "api", "bge-m3", "auto"])
    parser.add_argument("--llm", default="auto", choices=["auto", "mock", "deepseek", "dashscope"])
    parser.add_argument("--model", default=None, help="覆盖预设模型，如 qwen3-max / deepseek-reasoner")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="拒答阈值，取 eval/thresholds*.json 里标定出的值；不传则不拒答",
    )
    args = parser.parse_args()

    embedder = build_embedder(args.embedder)
    client = QdrantClient(":memory:")
    store = QdrantStore(client, collection="kb", dim=embedder.dim)
    store.ensure(recreate=True)

    reranker = build_reranker("lexical" if args.embedder == "hashing" else "auto")
    retriever = Retriever(
        embedder,
        store,
        reranker=reranker,
        config=RetrievalConfig(min_score=args.min_score),
    )

    for path in iter_documents(args.docs):
        doc = parse_document(path)
        retriever.index_document(doc, split_document(doc))

    llm = None if args.llm == "mock" else build_llm(args.llm, **({"model": args.model} if args.model else {}))
    llm_name = f"{type(llm).__name__}" if llm else "MockLLM"
    if llm is None:
        llm = MockLLM()

    print(f"问题：{args.question}")
    print(f"模型：{llm_name}")
    if isinstance(llm, MockLLM):
        print("      （替身按字面重叠抽取，仅验证链路；表格类的跨行推理它答不了）")
    print()

    service = QAService(retriever, llm)

    if not args.stream:
        answer = service.answer(args.question)
        print(f"答案：{answer.text}\n")
        if answer.refused:
            print(f"（已拒答：{answer.reason}）")
        elif answer.uncited:
            print("（警告：答案未标注任何引用）")
        if answer.citations:
            print("引用：")
            for citation in answer.citations:
                page = f" 第 {citation.page} 页" if citation.page else ""
                print(f"  [{citation.index}] {citation.filename} / {citation.section_path}{page}")
        return 0

    for event in service.stream(args.question):
        if event["event"] == "retrieval":
            debug = event["data"].get("debug", {})
            print(
                f"[检索] dense {debug.get('dense_hits', 0)} / sparse {debug.get('sparse_hits', 0)}"
                f" → 融合 {debug.get('fused', 0)} → 候选 {debug.get('candidates', 0)}"
            )
        elif event["event"] == "citations":
            print(f"[来源] 共 {len(event['data'])} 条")
        elif event["event"] == "delta":
            print(event["data"]["text"], end="", flush=True)
        elif event["event"] == "done":
            print("\n")
            data = event["data"]
            print(f"[完成] refused={data.get('refused')} uncited={data.get('uncited')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
