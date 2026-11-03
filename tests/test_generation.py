from __future__ import annotations

import pytest

from ragkb.chunking import split_document
from ragkb.embedding.hashing import HashingEmbedder
from ragkb.evaluation import GoldenItem, evaluate_retrieval
from ragkb.generation import QAService, REFUSAL_TEXT, parse_citations
from ragkb.generation.base import Message
from ragkb.generation.compat import build_llm
from ragkb.generation.mock import MockLLM
from ragkb.generation.prompt import Citation, build_context
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.parsing import parse_document
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.hybrid import Hit
from ragkb.retrieval.rerank import LexicalReranker

DIM = 256


def build_service(md_file, **config_kwargs):
    client = pytest.importorskip("qdrant_client").QdrantClient(":memory:")
    embedder = HashingEmbedder(dim=DIM)
    store = QdrantStore(client, collection="gen", dim=DIM)
    store.ensure(recreate=True)

    config = RetrievalConfig(min_score=0.0, **config_kwargs)
    retriever = Retriever(embedder, store, reranker=LexicalReranker(), config=config)

    doc = parse_document(md_file)
    retriever.index_document(doc, split_document(doc))
    return QAService(retriever, MockLLM())


# ---------- prompt 组装 ----------


def test_build_context_formats_source_line():
    hits = [
        Hit(
            chunk_id="c1",
            score=0.5,
            payload={"filename": "a.md", "section_path": "X 章", "page": 3, "text": "内容一"},
        ),
        Hit(chunk_id="c2", score=0.4, payload={"filename": "b.pdf", "text": "内容二"}),
    ]
    context, citations = build_context(hits)

    assert "资料 1（来源：a.md / X 章 / 第 3 页）" in context
    assert "资料 2（来源：b.pdf）" in context
    assert [c.index for c in citations] == [1, 2]


def test_build_context_prefers_parent_text():
    hit = Hit(
        chunk_id="c1",
        score=0.5,
        payload={"filename": "a.md", "text": "子块内容", "parent_text": "父块完整内容"},
    )
    context, _ = build_context([hit])
    assert "父块完整内容" in context
    assert "子块内容" not in context


def test_parse_citations_dedup_and_keep_order():
    citations = [
        Citation(1, "c1", "d1", "标题", "a.md", "章节", None, "片段"),
        Citation(2, "c2", "d2", "标题", "b.md", "章节", None, "片段"),
    ]
    used = parse_citations("结论甲[2]，补充[1]，再提一次[2]", citations)

    assert [c.index for c in used] == [2, 1]


def test_parse_citations_ignores_out_of_range():
    citations = [Citation(1, "c1", "d1", "t", "a.md", "s", None, "x")]
    assert parse_citations("答案[9]", citations) == []


# ---------- 问答 ----------


def test_answer_contains_citation(md_file):
    service = build_service(md_file)
    answer = service.answer("年假有几天")

    assert not answer.refused
    assert answer.citations
    assert not answer.uncited
    assert "[1]" in answer.text


def test_generation_layer_refusal(md_file):
    """检索给到了候选，但资料里没有依据 —— 由模型侧拒答。"""
    service = build_service(md_file)
    answer = service.answer("逆回购利率对流动性的影响")

    assert answer.refused
    assert answer.text == REFUSAL_TEXT
    assert answer.reason.startswith("generation:")


def test_retrieval_layer_refusal(md_file):
    service = build_service(md_file, use_rerank=False, min_raw_score=0.99)
    answer = service.answer("年假有几天")

    assert answer.refused
    assert answer.reason.startswith("retrieval:")
    assert answer.hits == []


def test_uncited_answer_is_flagged(md_file):
    class NoCiteLLM:
        def complete(self, messages, temperature=0.1, max_tokens=1024) -> str:
            return "年假是五天。"

        def stream(self, messages, temperature=0.1, max_tokens=1024):
            yield "年假是五天。"

    service = build_service(md_file)
    service.llm = NoCiteLLM()

    answer = service.answer("年假有几天")

    assert answer.uncited is True
    assert answer.refused is False
    assert answer.citations == []


def test_stream_event_sequence(md_file):
    service = build_service(md_file)
    events = list(service.stream("年假有几天"))
    names = [e["event"] for e in events]

    assert names[0] == "retrieval"
    assert "citations" in names
    assert "delta" in names
    assert names[-1] == "done"

    done = events[-1]["data"]
    assert done["text"]
    assert done["refused"] is False
    assert len(done["citations"]) >= 1


def test_stream_refusal_short_circuits(md_file):
    service = build_service(md_file, use_rerank=False, min_raw_score=0.99)
    events = list(service.stream("年假有几天"))

    assert events[0]["data"]["refused"] is True
    assert events[-1]["event"] == "done"
    assert "citations" not in [e["event"] for e in events]


def test_history_is_trimmed(md_file):
    service = build_service(md_file)
    history = [Message("user", f"历史 {i}") for i in range(10)]

    service.answer("年假有几天", history=history)
    last_call = service.llm.calls[-1]

    # 1 条 system + 最近 6 条历史 + 1 条当前提问
    assert len(last_call) == 8


# ---------- LLM 装配 ----------


def test_build_llm_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    assert build_llm("auto") is None


def test_build_llm_invalid_provider():
    with pytest.raises(ValueError):
        build_llm("no-such-provider")


# ---------- 评测指标 ----------


def test_evaluate_retrieval_metrics(md_file):
    service = build_service(md_file)
    items = [
        GoldenItem("年假有几天", ["员工手册.md"], ["5-15"]),
        GoldenItem("并购重组审批流程", ["不存在.docx"], []),
    ]

    report = evaluate_retrieval(service.retriever, items, k=5)

    assert report.total == 2
    assert report.hit_rate == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)  # 一条命中排第 1，一条未命中


def test_evaluate_empty_dataset(md_file):
    service = build_service(md_file)
    report = evaluate_retrieval(service.retriever, [], k=5)
    assert report.total == 0
    assert report.hit_rate == 0.0


# ---------- 空回答兜底 ----------


class _ScriptedLLM:
    """按脚本返回，记录每次的 token 预算。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.budgets: list[int] = []

    def complete(self, messages, temperature=0.1, max_tokens=1024) -> str:
        self.budgets.append(max_tokens)
        return self.replies.pop(0) if self.replies else ""

    def stream(self, messages, temperature=0.1, max_tokens=1024):
        yield self.complete(messages, temperature, max_tokens)


def test_empty_generation_retries_with_larger_budget(md_file):
    """推理型模型偶尔返回空 content，必须放大预算重试，而不是把空串交给用户。"""
    qa = build_service(md_file)
    qa.llm = _ScriptedLLM(["", "年假 5-15 天[1]。"])
    answer = qa.answer("年假有几天")

    assert answer.text.startswith("年假")
    assert qa.llm.budgets == [1024, 2048]


def test_empty_generation_falls_back_to_refusal(md_file):
    """重试仍为空时宁可拒答，也不要返回空白回答。"""
    qa = build_service(md_file)
    qa.llm = _ScriptedLLM(["", "   "])
    answer = qa.answer("年假有几天")

    assert answer.text == REFUSAL_TEXT
    assert answer.refused
