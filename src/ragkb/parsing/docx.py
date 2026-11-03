"""docx 解析。

为什么不用 python-docx：
- document.paragraphs 只含 body 顶层段落，表格内段落、文本框、页眉页脚取不到
- 标题层级要靠 p.style.name 判断，中文模板返回的是「标题 1」不是「Heading 1」
- Word 的自动编号（1.1 / 1.2）不在 text 里，直接导致标题层级断裂

mammoth 直接产出语义 HTML（h1~h6 / table / ul），这些问题一并消失，
而且和 markdown 共用同一套 IR 解析。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mammoth

from ragkb.models.document import Document
from ragkb.parsing.base import BaseParser, ParseError, register
from ragkb.parsing.html_ir import html_to_blocks


@register
class DocxParser(BaseParser):
    source_type = "docx"
    suffixes = (".docx",)

    def parse(self, path: Path, meta: dict[str, Any] | None = None) -> Document:
        if path.suffix.lower() == ".doc":
            raise ParseError("不支持旧版 .doc，请先另存为 .docx")

        try:
            with path.open("rb") as f:
                result = mammoth.convert_to_html(f)
        except Exception as exc:  # mammoth 抛的异常类型不稳定
            raise ParseError(f"docx 解析失败：{path.name} → {exc}") from exc

        doc = self.new_document(path, meta)
        doc.blocks = html_to_blocks(result.value)

        warnings = [getattr(m, "message", str(m)) for m in result.messages]
        if warnings:
            doc.meta["warnings"] = warnings

        for b in doc.blocks:
            if b.type == "title" and (b.level or 9) == 1:
                doc.meta["title"] = b.text
                break

        return doc
