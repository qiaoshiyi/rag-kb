"""文档块模型：内容只写一份，格式由渲染器决定。

这里刻意不引入 markdown 作为中间格式——txt 需要空格对齐、pdf 需要按页排版、
docx 需要真表格与标题样式，用 markdown 当中转会把三者都压扁成一种表达。
"""

from __future__ import annotations

from typing import Any

Block = tuple[Any, ...]


def h1(text: str) -> Block:
    """章 / 一级标题。"""
    return ("h1", text)


def h2(text: str) -> Block:
    """节 / 条。"""
    return ("h2", text)


def h3(text: str) -> Block:
    """目 / 款。"""
    return ("h3", text)


def p(text: str) -> Block:
    return ("p", text)


def li(*items: str) -> Block:
    """无序列表。以两个空格开头的项自动降一级。"""
    return ("li", list(items))


def oli(*items: str) -> Block:
    """有序列表，渲染时自动编号。"""
    return ("oli", list(items))


def tb(headers: list[str], rows: list[list[str]], caption: str = "") -> Block:
    return ("table", headers, rows, caption)


def code(text: str, lang: str = "") -> Block:
    return ("code", text.strip("\n"), lang)


def quote(text: str) -> Block:
    return ("quote", text)


def sign(*lines: str) -> Block:
    """落款：右对齐的签发单位与日期。"""
    return ("sign", list(lines))


def pb() -> Block:
    """强制分页（仅 pdf / docx 生效）。"""
    return ("pagebreak",)


def doc(
    *,
    filename: str,
    fmt: str,
    title: str,
    meta: dict[str, str],
    blocks: list[Block],
    encoding: str = "utf-8",
    revisions: list[list[str]] | None = None,
    subtitle: str = "",
) -> dict[str, Any]:
    return {
        "filename": filename,
        "fmt": fmt,
        "title": title,
        "subtitle": subtitle,
        "meta": meta,
        "blocks": blocks,
        "encoding": encoding,
        "revisions": revisions or [],
    }
