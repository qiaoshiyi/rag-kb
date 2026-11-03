"""会话状态：改写 + 双路召回的决策入口。

两件事在这里决定：

1. **这一轮要不要改写**。自足问题不改写——白花一次调用，还可能把本来正确的问法改坏。
2. **改写结果可不可信**。改写模型编造实体是最难排查的失效：检索稳定跑偏、
   答案看起来还很合理。用「改写结果与上一轮话题的相似度」做兜底，
   低于阈值就丢弃改写、退回原问题单路检索。

话题跳转也在这里处理：用户换话题后，若历史不清，下一轮改写会把旧话题带进来。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence

from ragkb.generation.base import Message
from ragkb.multiturn.rewrite import (
    NoopRewriter,
    QueryRewriter,
    RewriteResult,
    is_follow_up,
)


def _bigrams(text: str) -> set[str]:
    chars = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9]", "", text)
    if len(chars) < 2:
        return {chars} if chars else set()
    return {a + b for a, b in zip(chars, chars[1:])}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


@dataclass(slots=True)
class PreparedQuery:
    original: str
    rewritten: str
    is_follow_up: bool = False
    reset: bool = False
    """本轮之后是否应清空历史（话题跳转）。"""
    reason: str = ""
    similarity: float | None = None

    @property
    def changed(self) -> bool:
        return self.rewritten.strip() != self.original.strip()

    def queries(self, include_original: bool = True) -> list[str]:
        """双路召回的查询列表：改写结果在前（重排以它为主），原问题兜底。"""
        if not self.changed:
            return [self.original]
        return [self.rewritten, self.original] if include_original else [self.rewritten]

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "changed": self.changed,
            "is_follow_up": self.is_follow_up,
            "reset": self.reset,
            "reason": self.reason,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
        }


def _recent_user(history: Sequence[Message], lookback: int = 2) -> list[str]:
    """最近几条用户提问，最新的在前。

    只看最后一条是不够的：如果上一轮本身就是省略式追问（「那餐补呢」），
    它没有可比的语义内容，任何改写都会被判成离题。
    取最近两三轮的最大相似度，才是「有没有接上这段对话」的正确度量。
    """
    found: list[str] = []
    for message in reversed(list(history)):
        if message.role == "user" and message.content.strip():
            found.append(message.content)
            if len(found) >= lookback:
                break
    return found


def prepare_query(
    question: str,
    history: Sequence[Message] | None,
    rewriter: QueryRewriter | None = None,
    embedder=None,
    guard_threshold: float = 0.30,
    shift_threshold: float = 0.50,
) -> PreparedQuery:
    """决定本轮检索用哪些查询。

    guard_threshold 管「改写是否可信」，shift_threshold 管「话题是否已经换了」。
    两个阈值都是在本项目的 12 条多轮集上标出来的：

        真实追问与上一轮的相似度   0.59 ~ 0.90
        话题跳转与上一轮的相似度   0.30 ~ 0.41

    取 0.50 落在两个分布之间；再低会把跳转当成追问，再高会误清同话题的追问。
    换 embedder 或换语料都要重标，这两个数字不是通用常数。
    """
    history = list(history or [])
    follow_up = is_follow_up(question)
    recent = _recent_user(history)

    if rewriter is None or not history:
        return PreparedQuery(
            original=question, rewritten=question, is_follow_up=follow_up, reason="no_history"
        )

    result: RewriteResult = rewriter.rewrite(question, history)
    similarity: float | None = None

    if recent:
        similarity = max(_similarity(result.question, text, embedder) for text in recent)

    # 改了但改到别的话题上去了 → 丢弃改写，单路检索原问题
    if result.changed and similarity is not None and similarity < guard_threshold:
        return PreparedQuery(
            original=question,
            rewritten=question,
            is_follow_up=follow_up,
            reason=f"guard_off_topic:{similarity:.2f}",
            similarity=similarity,
        )

    # 判话题跳转只看相似度，不看「有没有被改写」。
    # 两个反例都踩过：
    #   1) 「离职要提前几天申请」是自足问题，但短且含疑问词，被启发式判成追问；
    #   2) 同一句被 LLM 顺手润色成「员工离职需要提前多少天提交申请」，
    #      于是 changed=True 就把跳转判定绕过去了。
    # 真正的判据是「结束后的问法与这段历史像不像」——润色会保持相似度低，补全会拉高。
    reset = False
    if similarity is not None:
        reset = similarity < shift_threshold

    return PreparedQuery(
        original=question,
        rewritten=result.question,
        is_follow_up=follow_up or result.changed,
        reset=reset,
        reason=result.reason or ("topic_shift" if reset else ""),
        similarity=similarity,
    )


def _similarity(text: str, other: str, embedder) -> float:
    if embedder is not None:
        try:
            a = embedder.embed_query(text).dense
            b = embedder.embed_query(other).dense
            return _cosine(a, b)
        except Exception:  # pragma: no cover - 网络失败时退回字面
            pass
    return _jaccard(text, other)


@dataclass
class Conversation:
    """有状态会话。HTTP 层是客户端传 history，这里给本地/CLI 用。"""

    rewriter: QueryRewriter = field(default_factory=NoopRewriter)
    embedder: object | None = None
    max_turns: int = 6
    guard_threshold: float = 0.30
    shift_threshold: float = 0.50
    turns: list[Message] = field(default_factory=list)

    def prepare(self, question: str) -> PreparedQuery:
        prepared = prepare_query(
            question,
            self.turns,
            rewriter=None if isinstance(self.rewriter, NoopRewriter) else self.rewriter,
            embedder=self.embedder,
            guard_threshold=self.guard_threshold,
            shift_threshold=self.shift_threshold,
        )
        if prepared.reset:
            self.turns.clear()
        return prepared

    def record(self, question: str, answer: str) -> None:
        self.turns.append(Message(role="user", content=question))
        if answer:
            self.turns.append(Message(role="assistant", content=answer))
        keep = self.max_turns * 2
        if len(self.turns) > keep:
            self.turns = self.turns[-keep:]

    def reset(self) -> None:
        self.turns.clear()
