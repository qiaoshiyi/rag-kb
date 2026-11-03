"""查询改写：把「那海外呢」还原成可独立检索的问题。

多轮问答的召回失败几乎都发生在这一步：用户追问天生不完整，
直接拿原文去检索，向量和 BM25 都没有可匹配的内容。

两种实现：
- LLMRewriter     真实改写，负责指代消解与省略补全，质量好但多一次模型调用
- HeuristicRewriter  字面拼接，无模型依赖，用于离线测试与降级

共同约束：**只补全对话里已经出现的信息，不得引入新实体**。
改写模型一旦编造实体，比不改写更糟——检索会稳定地跑偏。
所以 session 层还有一道 guard：改写结果与上一轮话题的相似度过低就丢弃。
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from ragkb.generation.base import LLM, Message


@dataclass(slots=True)
class RewriteResult:
    question: str
    changed: bool = False
    reason: str = ""

    @property
    def is_rewritten(self) -> bool:
        return self.changed and self.question.strip() != ""


@runtime_checkable
class QueryRewriter(Protocol):
    def rewrite(self, question: str, history: Sequence[Message]) -> RewriteResult: ...


class NoopRewriter:
    """不做改写。用于对照实验：证明多轮改写到底带来了多少收益。"""

    def rewrite(self, question: str, history: Sequence[Message]) -> RewriteResult:
        return RewriteResult(question=question, changed=False, reason="noop")


# ---------------------------------------------------------------- 启发式

_PRONOUN_HEAD = re.compile(
    r"^(?:那么|那|这个|那个|这些|那些|这样|那样|它|他们|她们|它们|该|此|这|其|"
    r"还有|另外|以及|如果|要是|换成|至于|同样)"
)
_TAIL_PARTICLE = re.compile(r"(?:呢|吗|么|吧)?[？?。.]?$")
_INTERROGATIVE = re.compile(
    r"(?:是多少|是几天|有几天|几个|几天|几位|多久|多长时间|多少|谁|什么|怎么|如何|哪些|哪个|哪些人|是否|可不可以|能不能|要不要|需不需要)"
)
_PUNCT = re.compile(r"[\s，。、？！；：\"'（）()\[\]【】]+")


def _compact(text: str) -> str:
    return _PUNCT.sub("", text)


def is_follow_up(question: str) -> bool:
    """判断这个问题脱离上下文是否还成立。

    三类信号：指代/转折开头、极短且带疑问词、以「呢」结尾的省略问。
    """
    q = question.strip()
    if not q:
        return False
    if _PRONOUN_HEAD.search(q):
        return True
    if re.search(r"呢[？?]?$", q):
        return True
    stripped = _compact(q)
    return len(stripped) <= 10 and bool(_INTERROGATIVE.search(q))


def _topic_of(question: str) -> str:
    """从上一轮问题里抽话题短语：去掉疑问尾和疑问词，留下实体与限定语。"""
    text = _TAIL_PARTICLE.sub("", question.strip())
    text = _INTERROGATIVE.sub(" ", text)
    text = re.sub(r"[？?。！!，,、；;：:]+", " ", text)
    return " ".join(part for part in text.split() if part)


class HeuristicRewriter:
    """字面拼接式改写，不调用模型。

    效果不如 LLM，但完全确定、零成本，能把「多轮比单轮强」这件事在离线环境下测出来。
    它做的事很有限：把上一轮的话题短语和本轮的替换焦点拼在一起。
    真实改写要处理「它指的是哪个它」，那是 LLMRewriter 的活。
    """

    def __init__(self, max_turns: int = 4) -> None:
        self._max_turns = max_turns

    def rewrite(self, question: str, history: Sequence[Message]) -> RewriteResult:
        previous = [m.content for m in history if m.role == "user"]
        if not previous:
            return RewriteResult(question=question, changed=False, reason="no_history")

        prev = previous[-1]
        if not is_follow_up(question):
            return RewriteResult(question=question, changed=False, reason="self_contained")

        focus = _PRONOUN_HEAD.sub("", question.strip())
        focus = re.sub(r"呢[？?]?$", "", focus).strip("，, ")
        topic = _topic_of(prev)

        if not focus:
            focus = topic
        elif topic:
            focus = f"{focus} {topic}"

        if not focus or focus == question.strip():
            return RewriteResult(question=question, changed=False, reason="no_change")

        return RewriteResult(question=focus, changed=True, reason="heuristic_concat")


# ---------------------------------------------------------------- LLM

_SYSTEM = """你是企业知识库检索系统的查询改写器。用户正在追问，问题里有指代或省略。

你的任务：把追问改写成一句**可以独立检索**的完整问题。

规则：
1. 只使用历史对话中出现过的信息，绝对不要引入历史里没有的实体、数字或条件。
2. 必须保留用户原话里的关键术语、数字、专有名词。
3. 若当前问题本身已经完整、不需要依赖上下文，原样返回。
4. 只输出改写后的问题，一行，不要解释，不要加引号或前缀。"""


class LLMRewriter:
    """用对话模型做改写。建议用小模型（改写是高频、低难度任务）。"""

    def __init__(
        self,
        llm: LLM,
        max_turns: int = 3,
        max_growth: float = 2.5,
        max_extra_chars: int = 60,
    ) -> None:
        self._llm = llm
        self._max_turns = max_turns
        self._max_growth = max_growth
        self._max_extra_chars = max_extra_chars

    def rewrite(self, question: str, history: Sequence[Message]) -> RewriteResult:
        turns = self._recent(history)
        if not turns:
            return RewriteResult(question=question, changed=False, reason="no_history")
        if not is_follow_up(question):
            # 自足问题不必花一次调用；也避免模型「顺手」改掉本来正确的问法
            return RewriteResult(question=question, changed=False, reason="self_contained")

        messages = [Message(role="system", content=_SYSTEM)]
        messages.extend(turns)
        messages.append(
            Message(role="user", content=f"当前问题：{question}\n请输出改写后的问题：")
        )

        try:
            raw = self._llm.complete(messages)
        except Exception as exc:  # pragma: no cover - 网络异常路径
            return RewriteResult(question=question, changed=False, reason=f"llm_error:{exc}")

        candidate = _clean(raw)
        if not candidate or candidate == question.strip():
            return RewriteResult(question=question, changed=False, reason="unchanged")

        if len(candidate) > len(question) * self._max_growth + self._max_extra_chars:
            # 长出一大截基本是模型在编解释，不能拿去检索
            return RewriteResult(question=question, changed=False, reason="too_long")

        return RewriteResult(question=candidate, changed=True, reason="llm")

    def _recent(self, history: Sequence[Message]) -> list[Message]:
        usable = [m for m in history if m.role in {"user", "assistant"} and m.content.strip()]
        return usable[-self._max_turns * 2 :]


def _clean(raw: str) -> str:
    """模型输出清洗：去引号、去前缀、只取第一行。"""
    text = (raw or "").strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    text = re.sub(r"^(?:改写后的问题|改写结果|问题)[:：]\s*", "", text)
    return text.strip().strip("\"'“”‘’ ")


def build_rewriter(
    backend: str = "auto", model: str | None = None, **kwargs
) -> QueryRewriter:
    if backend in {"none", "noop", "off"}:
        return NoopRewriter()
    if backend == "heuristic":
        return HeuristicRewriter(**kwargs)
    if backend in {"llm", "auto"}:
        try:
            from ragkb.generation.compat import build_llm

            llm = build_llm("auto", **({"model": model} if model else {}))
            if llm is not None:
                return LLMRewriter(llm, **kwargs)
        except Exception:
            if backend == "llm":
                raise
        if backend == "llm":
            warnings.warn(
                "没有可用的对话模型（未设置 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY），"
                "改写降级为字面拼接的 HeuristicRewriter",
                RuntimeWarning,
                stacklevel=2,
            )
        return HeuristicRewriter(max_turns=kwargs.get("max_turns", 4))
    raise ValueError(f"未知的 rewriter backend: {backend!r}")
