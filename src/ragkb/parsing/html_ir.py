"""HTML → IR。

docx 经 mammoth 转成语义 HTML 后走这里；markdown 里内嵌的 HTML 也能复用。
用标准库 HTMLParser，不引入 bs4/lxml。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

from ragkb.models.document import Block

_RE_HEADING = re.compile(r"h([1-6])")


class _HtmlToBlocks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []

        self._buf: list[str] = []
        self._mode: str | None = None
        self._level: int | None = None

        self._in_table = False
        self._rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    # ---------- 事件 ----------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._reset()
            self._in_table = True
            self._rows = []
            return
        if self._in_table:
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th"):
                self._cell = []
            return

        m = _RE_HEADING.fullmatch(tag)
        if m:
            self._reset()
            self._mode = "title"
            self._level = int(m.group(1))
        elif tag == "p":
            self._reset()
            self._mode = "para"
        elif tag == "li":
            self._reset()
            self._mode = "list"
        elif tag == "pre":
            self._reset()
            self._mode = "code"
        elif tag == "br" and self._mode:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._flush_table()
            return
        if self._in_table:
            if tag == "tr" and self._row is not None:
                self._rows.append(self._row)
                self._row = None
            elif tag in ("td", "th") and self._cell is not None:
                if self._row is not None:
                    self._row.append(_clean_cell("".join(self._cell)))
                self._cell = None
            return

        if _RE_HEADING.fullmatch(tag) and self._mode == "title":
            self._emit("title")
        elif tag == "p" and self._mode == "para":
            self._emit("para")
        elif tag == "li" and self._mode == "list":
            self._emit("list")
        elif tag == "pre" and self._mode == "code":
            self._emit("code", strip=False)

    def handle_data(self, data: str) -> None:
        if self._in_table:
            if self._cell is not None:
                self._cell.append(data)
            return
        if self._mode:
            self._buf.append(data)

    def close(self) -> None:  # pragma: no cover - 兜底未闭合标签
        super().close()
        if self._mode in ("para", "list", "code"):
            self._emit(self._mode)  # type: ignore[arg-type]
        if self._in_table:
            self._flush_table()

    # ---------- 内部 ----------

    def _reset(self) -> None:
        """遇到新的块级标签，丢弃上一个未闭合的缓冲。"""
        self._buf = []
        self._mode = None
        self._level = None

    def _emit(self, kind: str, strip: bool = True) -> None:
        text = "".join(self._buf)
        text = text.strip() if strip else text.strip("\n")
        if text.strip():
            block = Block(type=kind, text=text)  # type: ignore[arg-type]
            if kind == "title":
                block.level = self._level or 1
            self.blocks.append(block)
        self._reset()

    def _flush_table(self) -> None:
        rows = [r for r in self._rows if any(c.strip() for c in r)]
        if rows:
            width = max(len(r) for r in rows)
            lines: list[str] = []
            for idx, row in enumerate(rows):
                padded = row + [""] * (width - len(row))
                lines.append("| " + " | ".join(padded) + " |")
                if idx == 0:
                    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
            self.blocks.append(Block(type="table", text="\n".join(lines)))
        self._in_table = False
        self._rows = []
        self._row = None
        self._cell = None


def _clean_cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|").strip()


def html_to_blocks(html: str) -> list[Block]:
    parser = _HtmlToBlocks()
    parser.feed(html)
    parser.close()
    return parser.blocks
