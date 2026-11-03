"""回答评测的判分解析与聚合测试（不发网络请求）。"""

from __future__ import annotations

import pytest

from ragkb.evaluation.answer_eval import (
    JudgeVerdict,
    LLMJudge,
    build_judge_messages,
    parse_verdict,
)
from ragkb.generation.base import Message


def test_parse_plain_json():
    verdict = parse_verdict('{"supported": true, "coverage": 1, "unsupported_claims": [], "reason": "ok"}')
    assert verdict.supported is True
    assert verdict.coverage == 1.0
    assert verdict.reason == "ok"


def test_parse_strips_code_fence_and_chatter():
    raw = '判定如下：\n```json\n{"supported": false, "coverage": 0.5, "unsupported_claims": ["多说了三年"]}\n```\n以上。'
    verdict = parse_verdict(raw)
    assert verdict.supported is False
    assert verdict.coverage == 0.5
    assert verdict.unsupported_claims == ["多说了三年"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"supported": true, "coverage": 0.73}', 0.5),   # 伪精度就近归位
        ('{"supported": true, "coverage": 0.9}', 1.0),
        ('{"supported": true, "coverage": 0.2}', 0.0),
        ('{"supported": true}', 0.0),                      # 缺字段按 0
    ],
)
def test_coverage_only_three_levels(raw, expected):
    assert parse_verdict(raw).coverage == expected


def test_parse_accepts_string_claims():
    verdict = parse_verdict('{"supported": false, "coverage": 1, "unsupported_claims": "编了一个数字"}')
    assert verdict.unsupported_claims == ["编了一个数字"]


@pytest.mark.parametrize("raw", ["", "抱歉，我无法判断", "{不是 json}", "{}"])
def test_parse_rejects_garbage_or_empty_payload(raw):
    if raw == "{}":
        # 空对象不算错，但必须给出保守结论
        assert parse_verdict(raw).supported is False
        return
    with pytest.raises(ValueError):
        parse_verdict(raw)


def test_judge_messages_contain_all_parts():
    messages = build_judge_messages("问题？", "答案", "资料正文", "参考答案")
    assert messages[0].role == "system"
    body = messages[1].content
    assert "问题？" in body and "答案" in body and "资料正文" in body and "参考答案" in body


class ScriptedLLM:
    """按脚本依次返回，用于测重试与解析。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[list[Message]] = []
        self.budgets: list[int] = []

    def complete(self, messages, temperature=0.1, max_tokens=1024) -> str:
        self.calls.append(list(messages))
        self.budgets.append(max_tokens)
        return self.replies[len(self.calls) - 1]

    def stream(self, messages, temperature=0.1, max_tokens=1024):
        yield self.complete(messages)


def test_judge_retries_when_reasoning_eats_budget():
    llm = ScriptedLLM(["", '{"supported": true, "coverage": 1}'])
    verdict = LLMJudge(llm).judge("问", "答", "资料")

    assert verdict.supported is True
    assert len(llm.calls) == 2
    assert llm.budgets[1] > llm.budgets[0]  # 重试必须放大预算，否则还会是空


def test_judge_raises_when_always_empty():
    llm = ScriptedLLM(["", "  "])
    with pytest.raises(ValueError, match="预算"):
        LLMJudge(llm).judge("问", "答", "资料")
