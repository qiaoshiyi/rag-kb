"""回答质量评测（LLM 判分）。

检索指标能说明「资料找没找到」，说明不了「答得对不对」。这一层补的就是后者。

四条指标，两条不用模型、两条要：

| 指标 | 计算方式 | 为什么这样设计 |
|---|---|---|
| answer_keyword | 关键词是否出现在答案正文 | 确定性。检索命中但答案没说出事实，是最常见的"看着对其实没答" |
| citation_ok | 被引用的块里是否真有答案关键词 | 确定性。引用错位比不引用更危险，读者会信 |
| faithful | LLM 判「答案的每个结论在资料里有无依据」 | 只能判分，但能被测试锁死解析逻辑 |
| coverage | LLM 判「参考答案的要点覆盖了多少」 | 0/0.5/1 三档，避免伪精度 |

另有一条 refusal_accuracy：该拒的拒了、该答的答了。它同时衡量两层拒答——
检索层阈值与生成层 prompt，任何一层失效都会掉分。

判分本身有噪声：同一个答案，模型两次判分可能不一致。所以这里的数字只能看**趋势**，
不能当成绝对值去追。评测的目的是发现"改坏了"，不是拿一个 0.87 去交差。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ragkb.evaluation import text_match
from ragkb.evaluation.dataset import GoldenItem
from ragkb.generation.answer import Answer
from ragkb.generation.base import LLM, Message
from ragkb.generation.prompt import REFUSAL_TEXT

JUDGE_SYSTEM = """你是 RAG 问答系统的评测员。给你一个问题、系统回答、参考资料和参考答案，请判定两件事。

判定 1 —— 是否有依据（supported）：
- 答案里每一个结论都要能在参考资料中找到明确依据。
- 数字、比例、期限、审批人必须与资料一致，近似或推断都不算有依据。
- 只要存在一处资料里没有的断言，supported 就为 false，并把那句话放进 unsupported_claims。
- 拒答（回答"资料中未找到相关内容"）视为 supported = true，此时 unsupported_claims 为空。

判定 2 —— 是否有用（coverage）：
- 参考答案里的事实要点被答案覆盖了多少，只能是 0、0.5、1 三个值之一。
- 1 表示要点齐了；0.5 表示答了但少了关键限定（如漏了适用条件）；0 表示没答到。

只输出一行 JSON，不要任何解释或代码块标记：
{"supported": true, "coverage": 1, "unsupported_claims": [], "reason": "一句话"}"""

_RE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass(slots=True)
class JudgeVerdict:
    supported: bool
    coverage: float
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "coverage": self.coverage,
            "unsupported_claims": self.unsupported_claims,
            "reason": self.reason,
        }


def build_judge_messages(
    question: str,
    answer: str,
    context: str,
    reference: str,
    max_context_chars: int = 4000,
) -> list[Message]:
    body = context[:max_context_chars]
    return [
        Message(role="system", content=JUDGE_SYSTEM),
        Message(
            role="user",
            content=(
                f"问题：{question}\n\n"
                f"参考资料：\n{body}\n\n"
                f"系统回答：{answer}\n\n"
                f"参考答案：{reference or '（无）'}"
            ),
        ),
    ]


def parse_verdict(raw: str) -> JudgeVerdict:
    """模型输出不可靠，这里必须容错：代码块、前后废话、缺字段都要能扛。"""
    text = (raw or "").strip()
    if not text:
        raise ValueError("判分结果为空")

    fenced = _RE_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"判分结果不是 JSON：{text[:120]}")
    payload = json.loads(text[start : end + 1])

    coverage = float(payload.get("coverage", 0))
    # 只接受三档，其他值就近归位，避免模型给出 0.73 这种伪精度
    coverage = min((0.0, 0.5, 1.0), key=lambda v: abs(v - coverage))
    claims = payload.get("unsupported_claims") or []
    if isinstance(claims, str):
        claims = [claims]

    return JudgeVerdict(
        supported=bool(payload.get("supported", False)),
        coverage=coverage,
        unsupported_claims=[str(c) for c in claims if str(c).strip()],
        reason=str(payload.get("reason", ""))[:200],
    )


class LLMJudge:
    def __init__(self, llm: LLM, max_context_chars: int = 4000) -> None:
        self._llm = llm
        self._max_context_chars = max_context_chars

    def judge(
        self,
        question: str,
        answer: str,
        context: str,
        reference: str = "",
    ) -> JudgeVerdict:
        messages = build_judge_messages(
            question, answer, context, reference, self._max_context_chars
        )
        # 推理型模型会先把 token 花在 reasoning 上，预算给小了 content 直接是空的。
        # 实测 400 tokens 时有一半判分返回空串，所以这里递进重试一次。
        for budget in (1024, 2048):
            raw = self._llm.complete(messages, temperature=0.0, max_tokens=budget)
            if raw.strip():
                return parse_verdict(raw)
        raise ValueError("判分结果为空：模型的输出预算被推理过程耗尽")


@dataclass(slots=True)
class AnswerItemResult:
    question: str
    answer: str
    refused: bool
    reason: str = ""
    expect_refusal: bool = False
    keyword_in_answer: bool | None = None
    """确定性：答案正文里有没有出现预期的关键事实。"""
    citation_ok: bool | None = None
    """确定性：被引用的块里有没有这些事实。"""
    cited: int = 0
    faithful: bool | None = None
    coverage: float | None = None
    verdict_reason: str = ""
    judge_error: str = ""

    @property
    def refusal_correct(self) -> bool:
        return self.refused == self.expect_refusal


@dataclass(slots=True)
class AnswerReport:
    total: int
    answered: int
    refused: int
    refusal_accuracy: float
    faithfulness: float
    coverage: float
    answer_keyword_rate: float
    citation_rate: float
    items: list[AnswerItemResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"拒答准确率={self.refusal_accuracy:.3f}  faithful={self.faithfulness:.3f}  "
            f"coverage={self.coverage:.3f}  答案含事实={self.answer_keyword_rate:.3f}  "
            f"引用对位={self.citation_rate:.3f}"
        )

    def failures(self) -> list[AnswerItemResult]:
        bad: list[AnswerItemResult] = []
        for item in self.items:
            if not item.refusal_correct:
                bad.append(item)
            elif not item.expect_refusal and (
                item.faithful is False or (item.coverage or 0) < 0.5 or item.keyword_in_answer is False
            ):
                bad.append(item)
            elif item.judge_error:
                bad.append(item)
        return bad


def _context_of(answer: Answer) -> str:
    """只被判分答案真正引用过的资料，且必须带来源行。

    带来源不是为了好看：少了「来源：国内差旅费用标准.md」，模型会把答案里的
    「国内差旅」当成凭空添加的限定语，直接判无依据——实测踩过。
    判分永远要用被判定方能看到的那个上下文，否则判的是另一个任务。
    """
    by_id = {hit.chunk_id: hit for hit in answer.hits}

    blocks: list[str] = []
    for citation in answer.citations:
        hit = by_id.get(citation.chunk_id)
        text = (
            str(hit.payload.get("parent_text") or hit.text) if hit is not None else citation.snippet
        )
        source = citation.filename
        if citation.section_path:
            source = f"{source} / {citation.section_path}"
        blocks.append(f"资料（来源：{source}）：\n{text}")

    if blocks:
        return "\n\n".join(blocks)

    # 一个引用都没标时（uncited 坏味道），退回全部命中，但仍带来源
    for hit in answer.hits:
        source = str(hit.payload.get("filename", "")) or hit.doc_id
        section = hit.section_path
        if section:
            source = f"{source} / {section}"
        text = str(hit.payload.get("parent_text") or hit.text)
        blocks.append(f"资料（来源：{source}）：\n{text}")
    return "\n\n".join(blocks)


def _cited_texts(answer: Answer) -> list[str]:
    by_id = {hit.chunk_id: str(hit.payload.get("parent_text") or hit.text) for hit in answer.hits}
    return [by_id.get(c.chunk_id, c.snippet) for c in answer.citations]


def evaluate_answers(
    qa,
    items: Sequence[GoldenItem],
    judge: LLMJudge | None = None,
    flt: Any = None,
) -> AnswerReport:
    results: list[AnswerItemResult] = []

    for item in items:
        answer = qa.answer(item.question, flt=flt)
        result = AnswerItemResult(
            question=item.question,
            answer=answer.text,
            refused=answer.refused,
            reason=answer.reason,
            expect_refusal=item.expect_refusal,
        )

        if item.expect_refusal:
            # 该拒答的用例没有"知识点"，只判拒没拒
            results.append(result)
            continue

        if item.keywords:
            result.keyword_in_answer = text_match.contains_all(answer.text, item.keywords)
            cited_texts = _cited_texts(answer) if not answer.refused else []
            result.citation_ok = any(
                text_match.contains_all(text, item.keywords) for text in cited_texts
            ) if cited_texts else False

        result.cited = len(answer.citations)

        if answer.refused or not answer.text or answer.text == REFUSAL_TEXT:
            # 拒答不用判 faithful，但要点覆盖按 0 计
            result.coverage = 0.0
            results.append(result)
            continue

        if judge is not None:
            try:
                verdict = judge.judge(
                    item.question, answer.text, _context_of(answer), item.reference
                )
            except Exception as exc:  # pragma: no cover - 网络/解析失败路径
                result.judge_error = str(exc)[:160]
            else:
                result.faithful = verdict.supported
                result.coverage = verdict.coverage
                result.verdict_reason = verdict.reason

        results.append(result)

    return _aggregate(results)


def _aggregate(results: list[AnswerItemResult]) -> AnswerReport:
    total = len(results)
    if total == 0:
        return AnswerReport(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    refused = sum(1 for r in results if r.refused)
    answered = total - refused
    refusal_ok = sum(1 for r in results if r.refusal_correct) / total

    judged = [r for r in results if r.faithful is not None]
    faithfulness = sum(1 for r in judged if r.faithful) / len(judged) if judged else 0.0
    coverage = (
        sum(float(r.coverage or 0) for r in judged) / len(judged) if judged else 0.0
    )

    keyword_items = [r for r in results if r.keyword_in_answer is not None]
    keyword_rate = (
        sum(1 for r in keyword_items if r.keyword_in_answer) / len(keyword_items)
        if keyword_items
        else 0.0
    )
    cited_items = [r for r in results if r.citation_ok is not None]
    citation_rate = (
        sum(1 for r in cited_items if r.citation_ok) / len(cited_items) if cited_items else 0.0
    )

    return AnswerReport(
        total=total,
        answered=answered,
        refused=refused,
        refusal_accuracy=refusal_ok,
        faithfulness=faithfulness,
        coverage=coverage,
        answer_keyword_rate=keyword_rate,
        citation_rate=citation_rate,
        items=results,
    )
