"""Prompt 组装与引用解析。

上下文格式固定为「资料 N（来源：文件 / 章节 / 页码）：正文」，
这样 MockLLM 与真实模型看到的是同一种结构，引用编号也能直接映射回 chunk。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ragkb.generation.base import Message
from ragkb.retrieval.hybrid import Hit

REFUSAL_TEXT = "资料中未找到相关内容。"

SYSTEM_PROMPT = """你是企业知识库问答助手。必须遵守以下规则：

1. 只依据「参考资料」作答，不得使用资料之外的任何知识，不得推测。
2. 每个结论句末尾用 [编号] 标注来源，编号对应资料的序号，例如 [1]、[2]。
3. 如果参考资料中没有能回答问题的依据，直接输出「资料中未找到相关内容。」，不要解释、不要猜测。
4. 不得编造数字、日期、人名、条款编号。引用原文时保持措辞一致。
5. 用简洁的中文直接作答，不要复述问题，不要罗列资料原文。
"""

_RE_CITE = re.compile(r"\[(\d+)\]")


@dataclass(slots=True)
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    title: str
    filename: str
    section_path: str
    page: int | None
    snippet: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "filename": self.filename,
            "section_path": self.section_path,
            "page": self.page,
            "snippet": self.snippet,
        }


def build_context(hits: Sequence[Hit]) -> tuple[str, list[Citation]]:
    """优先用父块做上下文：子块负责命中，父块负责给足语义。"""
    blocks: list[str] = []
    citations: list[Citation] = []

    for index, hit in enumerate(hits, start=1):
        text = str(hit.payload.get("parent_text") or hit.text)
        filename = str(
            hit.payload.get("filename") or hit.payload.get("title") or hit.doc_id
        )
        section = hit.section_path
        page = hit.payload.get("page")

        source = filename
        if section:
            source = f"{source} / {section}"
        if page:
            source = f"{source} / 第 {page} 页"

        blocks.append(f"资料 {index}（来源：{source}）：\n{text}")
        citations.append(
            Citation(
                index=index,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                title=str(hit.payload.get("title", "")),
                filename=filename,
                section_path=section,
                page=page if isinstance(page, int) else None,
                snippet=" ".join(text.split())[:160],
            )
        )

    return "\n\n".join(blocks), citations


def build_messages(
    question: str,
    hits: Sequence[Hit],
    history: Sequence[Message] | None = None,
) -> list[Message]:
    context, _ = build_context(hits)
    messages: list[Message] = [Message(role="system", content=SYSTEM_PROMPT)]
    if history:
        # 只带最近 3 轮，避免历史把检索片段挤掉
        messages.extend(list(history)[-6:])
    messages.append(
        Message(role="user", content=f"参考资料：\n\n{context}\n\n问题：{question}")
    )
    return messages


def parse_citations(text: str, citations: Sequence[Citation]) -> list[Citation]:
    """按出现顺序返回被真正引用的资料，用于前端渲染来源卡片。"""
    by_index = {c.index: c for c in citations}
    ordered: list[Citation] = []
    for match in _RE_CITE.finditer(text):
        idx = int(match.group(1))
        citation = by_index.get(idx)
        if citation is not None and citation not in ordered:
            ordered.append(citation)
    return ordered


def has_citation(text: str) -> bool:
    return bool(_RE_CITE.search(text))
