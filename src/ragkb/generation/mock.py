"""确定性 LLM 替身。

策略：从 prompt 的资料块里挑与问题词面重叠最高的句子，拼成答案并标注引用编号。

它的作用是让 prompt 组装、引用映射、流式输出、拒答判定这些**工程链路**可测，
完全不反映真实生成质量。接真实模型时换成 DeepSeekLLM 即可，其余代码不动。
"""

from __future__ import annotations

import re
from typing import Iterator, Sequence

from ragkb.generation.base import Message
from ragkb.generation.prompt import REFUSAL_TEXT

_RE_BLOCK = re.compile(
    r"资料 (\d+)（来源：([^）]*)）：\n(.*?)(?=\n\n资料 \d+（来源：|\n\n问题：|\Z)", re.S
)
_RE_QUESTION = re.compile(r"问题：(.*)\Z", re.S)
_RE_TOKEN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")
_RE_SPLIT = re.compile(r"[。；！？\n]")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _RE_TOKEN.findall(text)}


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for piece in _RE_SPLIT.split(text):
        piece = piece.strip()
        if not piece or set(piece) <= {"-", "|", " "}:
            continue
        out.append(piece)
    return out


class MockLLM:
    def __init__(
        self, chunk_size: int = 12, max_sentences: int = 2, min_overlap: int = 2
    ) -> None:
        self._chunk_size = chunk_size
        self._max_sentences = max_sentences
        # 只重合一个虚词不算有依据，否则几乎什么问题都能"答"出来
        self._min_overlap = min_overlap
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        return "".join(self.stream(messages, temperature, max_tokens))

    def stream(
        self,
        messages: Sequence[Message],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        self.calls.append(list(messages))
        text = self._compose(messages)
        for start in range(0, len(text), self._chunk_size):
            yield text[start : start + self._chunk_size]

    def _compose(self, messages: Sequence[Message]) -> str:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        question_match = _RE_QUESTION.search(user)
        question = question_match.group(1).strip() if question_match else user
        blocks = _RE_BLOCK.findall(user)
        if not blocks:
            return REFUSAL_TEXT

        query_tokens = _tokens(question)
        if not query_tokens:
            return REFUSAL_TEXT

        scored: list[tuple[int, str, str]] = []
        for index, _source, text in blocks:
            for sentence in _sentences(text):
                overlap = len(query_tokens & _tokens(sentence))
                if overlap >= self._min_overlap:
                    scored.append((overlap, index, sentence))

        if not scored:
            return REFUSAL_TEXT

        scored.sort(key=lambda item: (-item[0], len(item[2])))
        # 只取与最高重叠同档的句子，避免把表头之类的弱相关句也塞进答案
        best_overlap = scored[0][0]
        picked = [item for item in scored if item[0] == best_overlap][: self._max_sentences]
        parts = [f"{sentence}[{index}]" for _overlap, index, sentence in picked]
        return "；".join(parts) + "。"
