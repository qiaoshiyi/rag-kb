"""多轮对话演示：连续追问，看清每一轮到底拿什么去检索。

    python scripts/make_corpus.py
    export DASHSCOPE_API_KEY=sk-xxx
    python scripts/chat_demo.py                      # 真实作答
    python scripts/chat_demo.py --llm mock            # 只看链路，不消耗调用

重点观察两列：
    改写后   —— 实际送进检索的独立问题（追问时会补全）
    检索     —— 走的是单路还是「改写 + 原问题」双路
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ragkb.chunking import split_document
from ragkb.embedding import build_embedder
from ragkb.generation import QAService
from ragkb.generation.compat import build_llm
from ragkb.generation.mock import MockLLM
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.multiturn import Conversation, HeuristicRewriter, LLMRewriter
from ragkb.parsing import parse_document
from ragkb.pipeline.ingest import iter_documents
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import build_reranker

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = [
    "国内差旅住宿费一线城市上限是多少？",
    "另一个档次呢？",          # 零内容追问：必须补出「住宿费」
    "那海外呢？",              # 同主题跨文档：住宿费 → 海外标准
    "单个项目预算多少要公开招标？",  # 话题跳转：应当清空历史
    "那立项金额两百万呢？",      # 跳转后的新话题追问
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=str(ROOT / "corpus"))
    parser.add_argument("--embedder", default="api", choices=["hashing", "api", "bge-m3", "auto"])
    parser.add_argument("--reranker", default="dashscope", choices=["auto", "lexical", "dashscope", "bge"])
    parser.add_argument("--llm", default="auto", choices=["auto", "mock", "dashscope", "deepseek"])
    parser.add_argument("--model", default=os.environ.get("KB_MODEL"))
    parser.add_argument("--rewriter", default="llm", choices=["llm", "heuristic", "none"])
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    embedder = build_embedder(args.embedder)
    store = QdrantStore(QdrantClient(":memory:"), collection="kb", dim=embedder.dim)
    store.ensure(recreate=True)

    reranker_name = "lexical" if args.embedder == "hashing" else args.reranker
    retriever = Retriever(
        embedder,
        store,
        reranker=build_reranker(reranker_name),
        config=RetrievalConfig(min_score=args.min_score),
    )
    for path in iter_documents(Path(args.docs)):
        doc = parse_document(path)
        retriever.index_document(doc, split_document(doc))

    llm = None if args.llm == "mock" else build_llm(
        args.llm, **({"model": args.model} if args.model else {})
    )
    if llm is None:
        llm = MockLLM()

    if args.rewriter == "none":
        rewriter = None
    elif args.rewriter == "heuristic":
        rewriter = HeuristicRewriter()
    elif llm is not None and not isinstance(llm, MockLLM):
        rewriter = LLMRewriter(llm)
    else:
        rewriter = HeuristicRewriter()

    service = QAService(retriever, llm, rewriter=rewriter)
    conversation = Conversation()

    print(f"语料 {len(list(iter_documents(Path(args.docs))))} 篇 / {store.count()} 点")
    print(f"作答模型 {type(llm).__name__} · 改写器 {type(rewriter).__name__ if rewriter else '无'}")
    print("=" * 72)

    for turn, question in enumerate(SCRIPT, start=1):
        answer = service.answer(question, history=conversation.turns)
        info = answer.debug.get("rewrite", {})
        paths = answer.debug.get("paths", 1)

        print(f"\n[{turn}] 用户：{question}")
        if info.get("changed"):
            print(f"    改写后：{info['rewritten']}   （相似度 {info['similarity']}）")
        else:
            print(f"    改写后：未改写（{info.get('reason')}）")
        print(f"    检索：{'改写 + 原问题双路' if paths == 2 else '单路'}")

        if answer.refused:
            print(f"    回答：{answer.text}  [{answer.reason}]")
        else:
            print(f"    回答：{answer.text}")
            for citation in answer.citations:
                page = f" 第 {citation.page} 页" if citation.page else ""
                print(f"           [{citation.index}] {citation.filename} / {citation.section_path}{page}")

        if info.get("reset"):
            print("    ⤶ 话题已跳转，清空历史")
            conversation.reset()
        conversation.record(question, answer.text)

    print("\n" + "=" * 72)
    print("说明：第 2 轮如果不改写，「另一个档次呢」本身没有任何可检索的内容；")
    print("      第 3 轮靠改写把「住宿费」带过去，才能跨到海外那篇文档；")
    print("      第 4 轮是自足问题，改写器不动它，并按相似度判定话题跳转。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
