"""问答编排：检索 → 组装 prompt → 生成 → 引用映射。

拒答分两层：
- 检索层：重排最高分低于阈值，直接回「资料中未找到」，不浪费一次 LLM 调用
- 生成层：模型判断资料不足以回答，输出同样的拒答语
两层的 reason 都要记下来，否则线上无法区分「没检索到」和「检索到了但答不出」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from ragkb.generation.base import LLM, Message
from ragkb.generation.prompt import (
    REFUSAL_TEXT,
    Citation,
    build_context,
    build_messages,
    has_citation,
    parse_citations,
)
from ragkb.multiturn.session import PreparedQuery, prepare_query
from ragkb.retrieval.hybrid import Hit
from ragkb.retrieval.pipeline import Retriever


@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    uncited: bool = False
    """有内容但一个引用都没标，说明 prompt 约束失效，需要盯的坏味道。"""
    rewritten: str = ""
    """多轮场景下实际用于检索的独立问题，便于前端展示与排查。"""
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "refused": self.refused,
            "reason": self.reason,
            "uncited": self.uncited,
            "rewritten": self.rewritten,
            "citations": [c.to_dict() for c in self.citations],
            "debug": self.debug,
        }


class QAService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        max_history: int = 6,
        rewriter=None,
        guard_threshold: float = 0.30,
        shift_threshold: float = 0.50,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_history = max_history
        self.rewriter = rewriter
        self.guard_threshold = guard_threshold
        self.shift_threshold = shift_threshold

    def _generate(self, messages: list[Message]) -> str:
        """生成回答，空内容要重试并兜底。

        实测踩过：推理型模型（deepseek-v4.1-flash）在跨文档题上把 1024 token
        预算全用在校验推理上，content 返回空字符串。如果不兜底，用户会看到
        一个空回答——比拒答更糟，因为前端无法判断是出错还是没内容。
        """
        for budget in (1024, 2048):
            text = self.llm.complete(messages, max_tokens=budget).strip()
            if text:
                return text
        return REFUSAL_TEXT

    def prepare(self, question: str, history: Any = None) -> PreparedQuery:
        return prepare_query(
            question,
            self._trim(history),
            rewriter=self.rewriter,
            embedder=self.retriever.embedder if self.rewriter is not None else None,
            guard_threshold=self.guard_threshold,
            shift_threshold=self.shift_threshold,
        )

    def answer(
        self,
        question: str,
        flt: Any = None,
        history: Sequence[Message] | None = None,
    ) -> Answer:
        prepared = self.prepare(question, history)
        result = self.retriever.search_multi(prepared.queries(), flt=flt)
        debug = {**result.debug, "rewrite": prepared.to_dict()}

        if result.refused:
            return Answer(
                text=REFUSAL_TEXT,
                refused=True,
                reason=f"retrieval:{result.reason}",
                rewritten=prepared.rewritten,
                debug=debug,
            )

        _, citations = build_context(result.hits)
        messages = build_messages(
            _for_prompt(prepared), result.hits, self._trim(history)
        )
        text = self._generate(messages)

        used = parse_citations(text, citations)
        refused = REFUSAL_TEXT in text and not used
        return Answer(
            text=text,
            citations=used,
            hits=result.hits,
            refused=refused,
            reason="generation:insufficient_context" if refused else "",
            uncited=bool(text) and not used and not refused,
            rewritten=prepared.rewritten,
            debug=debug,
        )

    def stream(
        self,
        question: str,
        flt: Any = None,
        history: Sequence[Message] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """产出结构化事件，外层转成 SSE 即可。"""
        prepared = self.prepare(question, history)
        result = self.retriever.search_multi(prepared.queries(), flt=flt)
        debug = {**result.debug, "rewrite": prepared.to_dict()}

        if result.refused:
            yield {
                "event": "retrieval",
                "data": {
                    "refused": True,
                    "reason": result.reason,
                    "rewritten": prepared.rewritten,
                },
            }
            yield {"event": "delta", "data": {"text": REFUSAL_TEXT}}
            yield {
                "event": "done",
                "data": {
                    "text": REFUSAL_TEXT,
                    "refused": True,
                    "citations": [],
                    "rewritten": prepared.rewritten,
                },
            }
            return

        _, citations = build_context(result.hits)
        yield {
            "event": "retrieval",
            "data": {
                "refused": False,
                "debug": debug,
                "rewritten": prepared.rewritten,
            },
        }
        yield {"event": "citations", "data": [c.to_dict() for c in citations]}

        messages = build_messages(_for_prompt(prepared), result.hits, self._trim(history))
        buffer: list[str] = []
        for delta in self.llm.stream(messages):
            buffer.append(delta)
            yield {"event": "delta", "data": {"text": delta}}

        text = "".join(buffer).strip()
        if not text:
            # 推理型模型可能把预算全花在思考上，content 直接返回空。
            # 流已经吐完了，只能补一次非流式重试，仍空则按拒答处理。
            text = self._generate(messages)
            yield {"event": "delta", "data": {"text": text}}
        used = parse_citations(text, citations)
        yield {
            "event": "done",
            "data": {
                "text": text,
                "refused": REFUSAL_TEXT in text and not used,
                "uncited": bool(text) and not used,
                "citations": [c.to_dict() for c in used],
                "rewritten": prepared.rewritten,
            },
        }

    def _trim(self, history: Sequence[Message] | None) -> list[Message] | None:
        if not history:
            return None
        return list(history)[-self.max_history :]


def _for_prompt(prepared: PreparedQuery) -> str:
    """给模型的问题：保留用户原话，改写结果作为补充说明。

    直接把原话换成改写结果看似更干净，但改写一旦错了，
    模型连用户原本问什么都不知道，错误无法被察觉。
    """
    if not prepared.changed:
        return prepared.original
    return f"{prepared.original}\n（如果表述不完整，可理解为：{prepared.rewritten}）"


def to_sse(event: dict[str, Any]) -> str:
    payload = json.dumps(event["data"], ensure_ascii=False)
    return f"event: {event['event']}\ndata: {payload}\n\n"
