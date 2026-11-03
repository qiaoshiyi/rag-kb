"""多轮改写与会话状态测试（不调模型）。"""

from __future__ import annotations

import pytest

from ragkb.chunking import split_document
from ragkb.embedding.hashing import HashingEmbedder
from ragkb.generation import QAService
from ragkb.generation.base import Message
from ragkb.generation.mock import MockLLM
from ragkb.indexing.qdrant_store import QdrantStore
from ragkb.multiturn import (
    Conversation,
    HeuristicRewriter,
    LLMRewriter,
    NoopRewriter,
    build_rewriter,
    is_follow_up,
    prepare_query,
)
from ragkb.parsing import parse_document
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import LexicalReranker

# ---------- 追问识别 ----------


@pytest.mark.parametrize(
    "question",
    [
        "那海外呢？",
        "那这个怎么算？",
        "它包含哪些内容？",
        "还有别的吗？",
        "提前多久申请？",
        "飞机呢？",
    ],
)
def test_follow_up_detected(question):
    assert is_follow_up(question)


@pytest.mark.parametrize(
    "question",
    [
        "一线城市差旅住宿费每晚上限是多少？",
        "供应商准入需要满足哪些条件？",
        "员工离职要提前三十天提交书面申请吗？",
    ],
)
def test_self_contained_not_flagged(question):
    assert not is_follow_up(question)


# ---------- 启发式改写 ----------


def test_heuristic_concat_carries_topic():
    history = [Message("user", "一线城市差旅住宿费每晚上限是多少？")]
    result = HeuristicRewriter().rewrite("那二线城市呢？", history)

    assert result.changed
    assert "二线城市" in result.question
    assert "住宿费" in result.question


def test_heuristic_leaves_self_contained_alone():
    history = [Message("user", "年假有几天？")]
    result = HeuristicRewriter().rewrite("供应商准入需要满足哪些条件？", history)

    assert not result.changed
    assert result.reason == "self_contained"


def test_heuristic_without_history_is_noop():
    result = HeuristicRewriter().rewrite("那海外呢？", [])
    assert not result.changed
    assert result.reason == "no_history"


def test_noop_rewriter_never_changes():
    result = NoopRewriter().rewrite("那海外呢？", [Message("user", "住宿费多少？")])
    assert not result.changed


# ---------- LLM 改写（用假模型）----------


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[Message]] = []

    def complete(self, messages, temperature=0.1, max_tokens=1024) -> str:
        self.calls.append(list(messages))
        return self.reply

    def stream(self, messages, temperature=0.1, max_tokens=1024):
        yield self.complete(messages)


def test_llm_rewriter_cleans_output():
    llm = FakeLLM('“差旅住宿费中二线城市的上限是多少？”')
    rewriter = LLMRewriter(llm)
    result = rewriter.rewrite("那二线城市呢？", [Message("user", "一线城市住宿费上限是多少？")])

    assert result.changed
    assert result.question == "差旅住宿费中二线城市的上限是多少？"
    # system + 历史 + 当前问题
    assert len(llm.calls[0]) == 3
    assert llm.calls[0][0].role == "system"


def test_llm_rewriter_rejects_runaway_answers():
    llm = FakeLLM("这个问题可以从以下几个方面理解：第一，你要问的是住宿费……" * 6)
    rewriter = LLMRewriter(llm)
    result = rewriter.rewrite("那二线城市呢？", [Message("user", "住宿费多少？")])

    assert not result.changed
    assert result.reason == "too_long"


def test_llm_rewriter_skips_self_contained_questions():
    llm = FakeLLM("不该被调用")
    rewriter = LLMRewriter(llm)
    result = rewriter.rewrite(
        "供应商准入需要满足哪些条件？", [Message("user", "年假有几天？")]
    )

    assert not result.changed
    assert llm.calls == []  # 省一次调用，也不会把正确问法改坏


def test_llm_rewriter_survives_llm_error():
    class BrokenLLM:
        def complete(self, messages, **kwargs):
            raise RuntimeError("timeout")

        def stream(self, messages, **kwargs):
            raise RuntimeError("timeout")

    result = LLMRewriter(BrokenLLM()).rewrite(
        "那海外呢？", [Message("user", "住宿费多少？")]
    )
    assert not result.changed
    assert result.reason.startswith("llm_error")


def test_build_rewriter_backends(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert isinstance(build_rewriter("none"), NoopRewriter)
    assert isinstance(build_rewriter("heuristic"), HeuristicRewriter)
    with pytest.warns(RuntimeWarning):
        assert isinstance(build_rewriter("llm"), HeuristicRewriter)


# ---------- 决策层 ----------


def test_prepare_without_history_returns_original():
    prepared = prepare_query("那海外呢？", [], rewriter=HeuristicRewriter())
    assert prepared.rewritten == prepared.original
    assert prepared.reason == "no_history"


def test_prepare_detects_topic_shift():
    history = [Message("user", "年假有几天？")]
    prepared = prepare_query(
        "单个项目预算多少要公开招标？",
        history,
        rewriter=HeuristicRewriter(),
    )
    assert not prepared.changed
    assert prepared.reset is True


class _FixedEmbedder:
    """按预设返回定长向量，用来精确控制相似度，避免测试依赖真实模型。"""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping
        self.dim = 2

    def embed_query(self, text: str):
        from ragkb.embedding.base import Embedding

        return Embedding(dense=self._mapping.get(text, [0.0, 1.0]))


def test_topic_shift_survives_paraphrase():
    """LLM 把自足问题润色了一遍（changed=True），也不能因此漏判跳转。"""

    class Paraphrase:
        def rewrite(self, question, history):
            from ragkb.multiturn.rewrite import RewriteResult

            return RewriteResult(
                question="员工离职需要提前多少天提交书面申请？",
                changed=True,
                reason="llm",
            )

    embedder = _FixedEmbedder(
        {
            "员工离职需要提前多少天提交书面申请？": [1.0, 0.0],
            "密码多久换一次？": [0.4, 0.9],  # 余弦 ≈ 0.41，高于 guard、低于跳转阈值
        }
    )
    history = [Message("user", "密码多久换一次？")]
    prepared = prepare_query(
        "离职要提前几天申请？", history, rewriter=Paraphrase(), embedder=embedder
    )

    assert prepared.changed
    assert prepared.similarity == pytest.approx(0.406, abs=0.01)
    assert prepared.reset is True


def test_guard_discards_off_topic_rewrite():
    class OffTopic:
        def rewrite(self, question, history):
            from ragkb.multiturn.rewrite import RewriteResult

            return RewriteResult(question="完全不相关的另一个话题", changed=True, reason="fake")

    history = [Message("user", "供应商分级标准是什么？")]
    prepared = prepare_query("那最低那级呢？", history, rewriter=OffTopic())

    assert prepared.rewritten == prepared.original
    assert prepared.reason.startswith("guard_off_topic")


def test_dual_path_only_when_rewritten():
    history = [Message("user", "一线城市住宿费上限是多少？")]
    prepared = prepare_query("那二线城市呢？", history, rewriter=HeuristicRewriter())
    assert len(prepared.queries()) == 2
    assert prepared.queries()[0] == prepared.rewritten

    single = prepare_query("供应商准入条件？", history, rewriter=HeuristicRewriter())
    assert single.queries() == ["供应商准入条件？"]


def test_conversation_records_and_trims():
    conv = Conversation(rewriter=HeuristicRewriter(), max_turns=2)
    for i in range(3):
        conv.record(f"问题{i}", f"回答{i}")

    assert len(conv.turns) == 4  # 2 轮 × (user + assistant)
    assert conv.turns[-1].content == "回答2"


def test_conversation_reset_on_topic_shift():
    conv = Conversation(rewriter=HeuristicRewriter())
    conv.record("年假有几天？", "5-15 天。")
    prepared = conv.prepare("单个项目预算多少要公开招标？")

    assert prepared.reset
    assert conv.turns == []  # 历史已清，下一轮不会被旧话题污染


# ---------- 双路召回 ----------


@pytest.fixture()
def retriever(tmp_path):
    pytest.importorskip("qdrant_client")
    embedder = HashingEmbedder(dim=128)
    store = QdrantStore(
        pytest.importorskip("qdrant_client").QdrantClient(":memory:"), collection="mt", dim=128
    )
    store.ensure(recreate=True)
    r = Retriever(
        embedder,
        store,
        reranker=LexicalReranker(),
        config=RetrievalConfig(min_score=None, min_raw_score=None),
    )
    content = (
        "# 差旅标准\n\n## 一、一线城市\n\n一线城市住宿费每晚上限 600 元。\n\n"
        "## 二、二线城市\n\n二线城市住宿费每晚上限 400 元。\n"
    )
    path = tmp_path / "差旅标准.md"
    path.write_text(content, encoding="utf-8")
    doc = parse_document(path)
    r.index_document(doc, split_document(doc))
    return r


def test_search_multi_uses_both_paths(retriever):
    result = retriever.search_multi(["二线城市住宿费上限", "那二线城市呢"])
    assert result.debug["paths"] == 2
    assert result.hits
    sources = {name for hit in result.hits for name in hit.sources}
    assert any(name.startswith("dense") for name in sources)


def test_search_single_keeps_legacy_names(retriever):
    result = retriever.search("二线城市住宿费上限")
    sources = {name for hit in result.hits for name in hit.sources}
    assert "dense" in sources and "sparse" in sources


def test_search_multi_dedupes_identical_queries(retriever):
    result = retriever.search_multi(["同一句话", "同一句话"])
    assert result.debug["paths"] == 1


# ---------- 端到端 ----------


def test_answer_exposes_rewrite_info(retriever):
    service = QAService(retriever, MockLLM(), rewriter=HeuristicRewriter())
    history = [Message("user", "一线城市住宿费上限是多少？"), Message("assistant", "600 元。")]

    answer = service.answer("那二线城市呢？", history=history)

    assert answer.rewritten != "那二线城市呢？"
    assert "住宿费" in answer.rewritten
    assert answer.debug["rewrite"]["changed"] is True
    assert answer.debug["paths"] == 2


def test_answer_without_rewriter_keeps_single_path(retriever):
    service = QAService(retriever, MockLLM())
    history = [Message("user", "一线城市住宿费上限是多少？")]

    answer = service.answer("那二线城市呢？", history=history)

    assert answer.rewritten == "那二线城市呢？"
    assert answer.debug["paths"] == 1
