"""Markdown 解析。

四类文档里 md 的解析质量最高：标题层级原生存在，直接就是父子块的骨架。
表格与代码块这里选择「按 token.map 回切原文」而不是重建，避免行内语法被拍平。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from ragkb.models.document import Block, Document
from ragkb.parsing.base import BaseParser, register

# frontmatter 用 --- 包住，不剥离的话会被当成水平线或段落
_RE_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_RE_FM_TITLE = re.compile(r"^title\s*:\s*(.+)$", re.M)


def strip_frontmatter(text: str) -> tuple[str, dict[str, Any]]:
    m = _RE_FRONTMATTER.match(text)
    if not m:
        return text, {}
    head = m.group(1)
    meta: dict[str, Any] = {}
    tm = _RE_FM_TITLE.search(head)
    if tm:
        meta["title"] = tm.group(1).strip().strip("\"'")
    return text[m.end() :], meta


def parse_markdown(text: str) -> list[Block]:
    md = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")
    tokens = md.parse(text)
    lines = text.splitlines()
    blocks: list[Block] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok.type == "heading_open" and i + 1 < len(tokens):
            inline = tokens[i + 1]
            try:
                level = int(tok.tag[1])
            except (ValueError, IndexError):
                level = 1
            blocks.append(Block(type="title", text=inline.content, level=level))
            i += 3  # heading_open / inline / heading_close
            continue

        if tok.type in ("fence", "code_block"):
            blocks.append(
                Block(type="code", text=tok.content.rstrip("\n"), meta={"lang": tok.info.strip()})
            )
            i += 1
            continue

        if tok.type == "table_open":
            # table_close 的 map 为 None，范围只能从 table_open 取
            start, end = tok.map if tok.map else (None, None)
            if start is not None and end is not None:
                blocks.append(Block(type="table", text="\n".join(lines[start:end])))
            while i < len(tokens) and tokens[i].type != "table_close":
                i += 1
            i += 1
            continue

        if tok.type == "inline" and tok.content.strip():
            blocks.append(Block(type="para", text=tok.content))

        i += 1

    return blocks


@register
class MarkdownParser(BaseParser):
    source_type = "md"
    suffixes = (".md", ".markdown")

    def parse(self, path: Path, meta: dict[str, Any] | None = None) -> Document:
        text = path.read_text(encoding="utf-8", errors="replace")
        body, fm_meta = strip_frontmatter(text)

        doc = self.new_document(path, meta)
        doc.blocks = parse_markdown(body)

        # 标题优先级：frontmatter > 首个 H1 > 文件名
        if fm_meta.get("title"):
            doc.meta["title"] = fm_meta["title"]
        else:
            for b in doc.blocks:
                if b.type == "title" and (b.level or 9) == 1:
                    doc.meta["title"] = b.text
                    break

        return doc
