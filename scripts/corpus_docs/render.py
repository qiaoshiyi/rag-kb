"""四套渲染器：把统一块模型写成 txt / md / docx / pdf。

每种格式都按它的"本土习惯"排，而不是把同一段 markdown 转四遍：
- txt：空格对齐的表、全角缩进、无任何标记语法（来自老系统导出的样子）
- md ：frontmatter + GFM 表 + 围栏代码
- docx：封面页、修订记录、Heading 样式、真表格、页眉页脚域
- pdf ：A4 按页排版、公文式标题块、带边框表格、页码"第 X 页 共 Y 页"
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Iterable

Block = tuple[Any, ...]


# --------------------------------------------------------------------------- #
# 公共
# --------------------------------------------------------------------------- #

def _dwidth(text: str) -> int:
    """显示宽度： CJK 全角算 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WFA" else 1 for ch in text)


def _wrap(text: str, limit: int) -> list[str]:
    """按显示宽度折行，尽量不在标点前断。"""
    out: list[str] = []
    line, width = "", 0
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in "WFA" else 1
        if width + w > limit and line:
            # 行首避头标点：把上一行最后一个字一起带下来
            if ch in "，。、；：）】》”’！？%":
                out.append(line[:-1])
                line, width = line[-1] + ch, _dwidth(line[-1] + ch)
                continue
            out.append(line)
            line, width = ch, w
        else:
            line += ch
            width += w
    if line:
        out.append(line)
    return out


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _dwidth(text))


# --------------------------------------------------------------------------- #
# txt
# --------------------------------------------------------------------------- #

_TXT_WRAP = 68


def render_txt(d: dict[str, Any]) -> bytes:
    meta = d["meta"]
    lines: list[str] = [
        "瀚宇智能科技股份有限公司",
        d["title"],
        "",
        f"文档编号：{meta.get('文档编号', '')}　　　　版本号：{meta.get('版本', '')}",
        f"生效日期：{meta.get('生效日期', '')}　　　责任部门：{meta.get('责任部门', '')}",
        f"密　　级：{meta.get('密级', '')}　　　　　适用对象：{meta.get('适用对象', '')}",
        "─" * _TXT_WRAP,
        "",
    ]

    for block in d["blocks"]:
        tag = block[0]
        if tag == "h1":
            lines += ["", block[1], ""]
        elif tag == "h2":
            lines += ["", block[1], ""]
        elif tag == "h3":
            lines += ["", block[1], ""]
        elif tag == "p":
            # 只有首行缩进；续行顶格，否则折行会被读成新段落
            segs = _wrap(block[1], _TXT_WRAP - 4)
            for i, seg in enumerate(segs):
                lines.append("　　" + seg if i == 0 else seg)
            lines.append("")
        elif tag in {"li", "oli"}:
            for i, item in enumerate(block[1], start=1):
                nested = item.startswith("  ")
                text = item.strip()
                bullet = f"{i}. " if tag == "oli" else "· "
                prefix = "　　　　" if nested else "　　"
                head = prefix + bullet
                # 续行与正文文字左对齐，悬挂缩进
                for j, seg in enumerate(_wrap(text, _TXT_WRAP - _dwidth(head))):
                    lines.append(head + seg if j == 0 else " " * _dwidth(head) + seg)
            lines.append("")
        elif tag == "table":
            headers, rows, caption = block[1], block[2], block[3]
            if caption:
                lines += ["", "　　" + caption]
            ncol = len(headers)
            widths = []
            for c in range(ncol):
                w = _dwidth(headers[c])
                for r in rows:
                    if c < len(r):
                        w = max(w, _dwidth(r[c]))
                widths.append(w)
            lines.append("　　" + "  ".join(_pad(headers[c], widths[c]) for c in range(ncol)))
            lines.append("　　" + "  ".join("─" * widths[c] for c in range(ncol)))
            for r in rows:
                cells = [r[c] if c < len(r) else "" for c in range(ncol)]
                # 单元格过长时单独折行，避免整张表被拉坏
                over = any(_dwidth(x) > widths[c] + 8 for c, x in enumerate(cells))
                if over:
                    for c, cell in enumerate(cells):
                        lines.append(f"　　{headers[c]}：{cell}")
                else:
                    lines.append("　　" + "  ".join(_pad(cells[c], widths[c]) for c in range(ncol)))
            lines.append("")
        elif tag == "code":
            lines += ["　　" + ln for ln in block[1].splitlines()]
            lines.append("")
        elif tag == "quote":
            lines += ["　　" + seg for seg in _wrap("【说明】" + block[1], _TXT_WRAP)]
            lines.append("")
        elif tag == "sign":
            lines.append("")
            for ln in block[1]:
                lines.append("　" * 0 + ln.rjust(_TXT_WRAP - 4))
            lines.append("")
        elif tag == "pagebreak":
            lines += ["", "（本页以下为附件）", ""]

    if d["revisions"]:
        lines += ["", "文件修订记录", ""]
        head = ["版本", "修订日期", "修订内容", "修订人"]
        widths = [_dwidth(h) for h in head]
        for r in d["revisions"]:
            for c in range(4):
                widths[c] = max(widths[c], _dwidth(r[c] if c < len(r) else ""))
        lines.append("　　" + "  ".join(_pad(head[c], widths[c]) for c in range(4)))
        for r in d["revisions"]:
            lines.append("　　" + "  ".join(_pad(r[c] if c < len(r) else "", widths[c]) for c in range(4)))

    text = "\n".join(lines).rstrip() + "\n"
    for enc in (d.get("encoding") or "utf-8", "gb18030"):
        try:
            return text.encode(enc)
        except UnicodeEncodeError:
            continue
    return text.encode("utf-8")


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #

def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_md(d: dict[str, Any]) -> str:
    meta = d["meta"]
    fm = [
        "---",
        f"title: {d['title']}",
        f"doc_id: {meta.get('文档编号', '')}",
        f"version: {meta.get('版本', '')}",
        f"effective: {meta.get('生效日期', '')}",
        f"owner: {meta.get('责任部门', '')}",
        f"classification: {meta.get('密级', '')}",
        f"scope: {meta.get('适用对象', '')}",
        "---",
        "",
        f"# {d['title']}",
        "",
        f"> 文档编号 {meta.get('文档编号', '')} · 版本 {meta.get('版本', '')} · "
        f"{meta.get('责任部门', '')} · {meta.get('密级', '')}级",
        "",
    ]

    body: list[str] = []
    for block in d["blocks"]:
        tag = block[0]
        if tag == "h1":
            body += ["", f"## {block[1]}", ""]
        elif tag == "h2":
            body += ["", f"### {block[1]}", ""]
        elif tag == "h3":
            body += ["", f"#### {block[1]}", ""]
        elif tag == "p":
            body += [block[1], ""]
        elif tag == "li":
            for item in block[1]:
                indent = "  " if item.startswith("  ") else ""
                body.append(f"{indent}- {item.strip()}")
            body.append("")
        elif tag == "oli":
            for i, item in enumerate(block[1], start=1):
                indent = "   " if item.startswith("  ") else ""
                body.append(f"{indent}{i}. {item.strip()}")
            body.append("")
        elif tag == "table":
            headers, rows, caption = block[1], block[2], block[3]
            if caption:
                body += ["", f"**{caption}**", ""]
            body.append("| " + " | ".join(_md_cell(h) for h in headers) + " |")
            body.append("| " + " | ".join("---" for _ in headers) + " |")
            for r in rows:
                cells = [r[c] if c < len(r) else "" for c in range(len(headers))]
                body.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
            body.append("")
        elif tag == "code":
            lang = block[2] if len(block) > 2 else ""
            body += ["", f"```{lang}", *block[1].splitlines(), "```", ""]
        elif tag == "quote":
            body += [f"> {ln}" for ln in block[1].splitlines()] + [""]
        elif tag == "sign":
            body.append("")
            body += [f"<p align=\"right\">{ln}</p>" for ln in block[1]]
            body.append("")
        elif tag == "pagebreak":
            body += ["", "---", ""]

    if d["revisions"]:
        body += ["", "## 文件修订记录", ""]
        body.append("| 版本 | 修订日期 | 修订内容 | 修订人 |")
        body.append("| --- | --- | --- | --- |")
        for r in d["revisions"]:
            cells = [r[c] if c < len(r) else "" for c in range(4)]
            body.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
        body.append("")

    return "\n".join(fm + body).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# docx
# --------------------------------------------------------------------------- #

def render_docx(d: dict[str, Any], path: Path) -> None:
    from docx import Document as Docx
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    BODY_FONT = "宋体"
    HEAD_FONT = "微软雅黑"

    docx = Docx()
    sec = docx.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.6)
    sec.top_margin, sec.bottom_margin = Cm(2.4), Cm(2.2)

    def set_font(run, name: str, size: float, bold: bool = False, color=None) -> None:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = name
        run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
        if color is not None:
            run.font.color.rgb = color

    normal = docx.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    normal.paragraph_format.line_spacing = 1.45
    normal.paragraph_format.space_after = Pt(4)

    for level, size in ((1, 15), (2, 12.5), (3, 11)):
        style = docx.styles[f"Heading {level}"]
        style.font.name = HEAD_FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0x1F, 0x28, 0x37)
        style.element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)
        style.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.line_spacing = 1.3

    # ---- 封面 ----
    for _ in range(3):
        docx.add_paragraph()
    cover = docx.add_paragraph()
    cover.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(cover.add_run("瀚宇智能科技股份有限公司"), HEAD_FONT, 15, color=RGBColor(0x33, 0x41, 0x55))

    title_p = docx.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(title_p.add_run(d["title"]), HEAD_FONT, 26, bold=True)

    if d.get("subtitle"):
        sub = docx.add_paragraph()
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(sub.add_run(d["subtitle"]), BODY_FONT, 11, color=RGBColor(0x6B, 0x72, 0x80))

    docx.add_paragraph()
    info = docx.add_table(rows=0, cols=2)
    info.style = "Table Grid"
    info.alignment = WD_TABLE_ALIGNMENT.CENTER
    for key in ("文档编号", "版本", "生效日期", "责任部门", "密级", "适用对象"):
        row = info.add_row()
        row.cells[0].width = Cm(4.2)
        row.cells[1].width = Cm(9)
        c0 = row.cells[0].paragraphs[0]
        c0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(c0.add_run(key), BODY_FONT, 10, bold=True)
        set_font(row.cells[1].paragraphs[0].add_run(d["meta"].get(key, "")), BODY_FONT, 10)

    docx.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---- 页眉页脚 ----
    header_p = sec.header.paragraphs[0]
    header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    set_font(
        header_p.add_run(f"{d['meta'].get('文档编号', '')}　{d['meta'].get('密级', '')}"),
        BODY_FONT,
        9,
        color=RGBColor(0x8A, 0x8A, 0x8A),
    )
    footer_p = sec.footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(footer_p.add_run("第 "), BODY_FONT, 9, color=RGBColor(0x8A, 0x8A, 0x8A))
    for instr in ("PAGE", "NUMPAGES"):
        run = footer_p.add_run()
        set_font(run, BODY_FONT, 9, color=RGBColor(0x8A, 0x8A, 0x8A))
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr_el = OxmlElement("w:instrText")
        instr_el.set(qn("xml:space"), "preserve")
        instr_el.text = f" {instr} "
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        run._r.append(begin)
        run._r.append(instr_el)
        run._r.append(end)
        set_font(
            footer_p.add_run(" 共 " if instr == "PAGE" else " 页"),
            BODY_FONT,
            9,
            color=RGBColor(0x8A, 0x8A, 0x8A),
        )

    # ---- 修订记录 ----
    if d["revisions"]:
        docx.add_heading("文件修订记录", level=1)
        _docx_table(docx, ["版本", "修订日期", "修订内容", "修订人"], d["revisions"], set_font)

    # ---- 正文 ----
    def shade(cell, hex_color: str) -> None:
        el = OxmlElement("w:shd")
        el.set(qn("w:fill"), hex_color)
        cell._tc.get_or_add_tcPr().append(el)

    for block in d["blocks"]:
        tag = block[0]
        if tag in {"h1", "h2", "h3"}:
            docx.add_heading(block[1], level=int(tag[1]))
        elif tag == "p":
            para = docx.add_paragraph(block[1])
            para.paragraph_format.first_line_indent = Pt(21)
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif tag in {"li", "oli"}:
            for i, item in enumerate(block[1], start=1):
                nested = item.startswith("  ")
                text = f"{i}. {item.strip()}" if tag == "oli" else f"· {item.strip()}"
                para = docx.add_paragraph(style="List Bullet" if tag == "li" else None)
                if tag == "oli":
                    para = docx.add_paragraph()
                para.paragraph_format.left_indent = Pt(42 if nested else 21)
                para.paragraph_format.space_after = Pt(2)
                set_font(para.add_run(text), BODY_FONT, 10.5)
        elif tag == "table":
            headers, rows, caption = block[1], block[2], block[3]
            if caption:
                cap = docx.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                set_font(cap.add_run(caption), BODY_FONT, 9, bold=True,
                         color=RGBColor(0x4B, 0x55, 0x63))
            table = _docx_table(docx, headers, rows, set_font)
            for cell in table.rows[0].cells:
                shade(cell, "EDF1F5")
        elif tag == "code":
            para = docx.add_paragraph()
            para.paragraph_format.left_indent = Pt(21)
            run = set_font(para.add_run(block[1]), "Consolas", 9)
            run.font.color.rgb = RGBColor(0x2D, 0x33, 0x3B)
        elif tag == "quote":
            para = docx.add_paragraph()
            para.paragraph_format.left_indent = Pt(21)
            set_font(para.add_run("【说明】" + block[1]), BODY_FONT, 10, color=RGBColor(0x55, 0x5F, 0x6D))
        elif tag == "sign":
            for ln in block[1]:
                para = docx.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                set_font(para.add_run(ln), BODY_FONT, 11)
        elif tag == "pagebreak":
            docx.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    props = docx.core_properties
    props.title = d["title"]
    props.subject = d["meta"].get("文档编号", "")
    props.author = "瀚宇智能科技股份有限公司"
    props.category = d["meta"].get("责任部门", "")
    props.comments = f"{d['meta'].get('密级', '')}级 · {d['meta'].get('版本', '')}"

    docx.save(path)


def _docx_table(docx, headers: list[str], rows: list[list[str]], set_font):
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.shared import Pt

    table = docx.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(para.add_run(h), "微软雅黑", 9.5, bold=True)
    for r in rows:
        row = table.add_row()
        for i in range(len(headers)):
            cell = row.cells[i]
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            set_font(para.add_run(r[i] if i < len(r) else ""), "宋体", 9.5)
    docx.add_paragraph()
    return table


# --------------------------------------------------------------------------- #
# pdf
# --------------------------------------------------------------------------- #

PAGE_W, PAGE_H = 595.0, 842.0
M_LEFT = M_RIGHT = 72.0
BODY_W = PAGE_W - M_LEFT - M_RIGHT
BODY_TOP = 92.0
BODY_BOTTOM = 776.0

BODY_SIZE = 10.5
LINE_H = 16.6
H1_SIZE, H2_SIZE, H3_SIZE = 17.0, 13.8, 12.2
TBL_SIZE = 8.8
TBL_LH = 12.4
PAD = 4.6

# 内置 CJK 字体（china-s）把 ASCII 也按全角排，"HY-SEC-001" 会被拉成 "H Y - S E C"，
# 而且没有粗体变体。所以优先嵌入系统里的宋体/黑体，取不到再退回内置字体。
_FONT_CANDIDATES = {
    "body": (
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ),
    "head": (
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ),
}


def _resolve_font(kind: str):
    """返回 (字体文件路径 或 None, Font 对象, 资源名)。"""
    import pymupdf

    for candidate in _FONT_CANDIDATES[kind]:
        if Path(candidate).exists():
            try:
                return candidate, pymupdf.Font(fontfile=candidate), f"F-{kind}"
            except Exception:
                continue
    return None, pymupdf.Font("china-s"), "china-s"


class _Pdf:
    def __init__(self, d: dict[str, Any]):
        import pymupdf

        self.pymupdf = pymupdf
        self.d = d
        self.doc = pymupdf.open()
        self.body_file, self.body_font, self.body_name = _resolve_font("body")
        self.head_file, self.head_font, self.head_name = _resolve_font("head")
        self.page = None
        self.y = BODY_TOP

    # -- 基础 ------------------------------------------------------------
    def text(self, x: float, y: float, s: str, size: float, *, bold: bool = False,
             color=(0, 0, 0), align: str = "left", width: float = 0.0) -> None:
        font = self.head_font if bold else self.body_font
        name = self.head_name if bold else self.body_name
        if align == "center":
            x = x + (width - font.text_length(s, fontsize=size)) / 2
        elif align == "right":
            x = x + width - font.text_length(s, fontsize=size)
        self.page.insert_text((x, y), s, fontsize=size, fontname=name, color=color)
        if bold and name == "china-s":
            # 退回内置字体时没有粗体，用 0.3pt 双描凑合
            self.page.insert_text((x + 0.3, y), s, fontsize=size, fontname=name, color=color)

    def wrap(self, s: str, size: float, width: float) -> list[str]:
        lines, cur = [], ""
        for ch in s:
            if self.body_font.text_length(cur + ch, fontsize=size) > width and cur:
                if ch in "，。、；：）】》”’！？%":
                    lines.append(cur[:-1])
                    cur = cur[-1] + ch
                    continue
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        lines.append(cur)
        return lines

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        if self.body_file:
            self.page.insert_font(fontname=self.body_name, fontfile=self.body_file)
        if self.head_file:
            self.page.insert_font(fontname=self.head_name, fontfile=self.head_file)
        self.y = BODY_TOP
        meta = self.d["meta"]
        self.text(M_LEFT, 58, meta.get("文档编号", ""), 8.5, color=(0.45, 0.45, 0.45))
        self.text(PAGE_W - M_RIGHT - 120, 58, meta.get("密级", ""), 8.5,
                  color=(0.45, 0.45, 0.45), align="right", width=120)
        self.page.draw_line((M_LEFT, 66), (PAGE_W - M_RIGHT, 66),
                            color=(0.72, 0.72, 0.72), width=0.6)

    def need(self, h: float) -> None:
        if self.y + h > BODY_BOTTOM:
            self.new_page()

    # -- 块 --------------------------------------------------------------
    def title_block(self) -> None:
        self.y = 120
        self.text(M_LEFT, self.y, self.d["title"], 21, bold=True,
                  align="center", width=BODY_W, color=(0.09, 0.12, 0.18))
        self.y += 34
        sub = self.d.get("subtitle") or "瀚宇智能科技股份有限公司"
        self.text(M_LEFT, self.y, sub, 10.5, align="center", width=BODY_W,
                  color=(0.35, 0.38, 0.44))
        self.y += 18
        meta = self.d["meta"]
        line = (f"文档编号 {meta.get('文档编号', '')}　　版本 {meta.get('版本', '')}　　"
                f"责任部门 {meta.get('责任部门', '')}　　密级 {meta.get('密级', '')}")
        self.text(M_LEFT, self.y, line, 9, align="center", width=BODY_W,
                  color=(0.42, 0.45, 0.50))
        self.y += 16
        self.page.draw_line((M_LEFT, self.y), (PAGE_W - M_RIGHT, self.y),
                            color=(0.60, 0.63, 0.68), width=0.9)
        self.y += 26

    def heading(self, text: str, size: float, before: float) -> None:
        self.need(before + size * 1.5)
        self.y += before
        self.text(M_LEFT, self.y, text, size, bold=True, color=(0.10, 0.14, 0.20))
        self.y += size * 1.45

    def paragraph(self, text: str) -> None:
        indent = BODY_SIZE * 2
        lines = self.wrap(text, BODY_SIZE, BODY_W - indent)
        for i, ln in enumerate(lines):
            self.need(LINE_H)
            self.text(M_LEFT + (indent if i == 0 else 0), self.y, ln, BODY_SIZE,
                      color=(0.13, 0.15, 0.19))
            self.y += LINE_H
        self.y += 3.5

    def item(self, text: str, marker: str, nested: bool = False) -> None:
        left = M_LEFT + (26 if nested else 14)
        width = BODY_W - (left - M_LEFT)
        lines = self.wrap(text, BODY_SIZE, width - 12)
        for i, ln in enumerate(lines):
            self.need(LINE_H)
            if i == 0:
                self.text(left - 12, self.y, marker, BODY_SIZE, color=(0.30, 0.33, 0.38))
            self.text(left, self.y, ln, BODY_SIZE, color=(0.13, 0.15, 0.19))
            self.y += LINE_H
        self.y += 2

    def _alloc(self, natural: list[float]) -> list[float]:
        """按内容分配列宽，并给每列设下限——否则窄列会把「数据类别」挤成竖排。"""
        n = len(natural)
        min_w = BODY_W / (n * 1.55)
        widths = [0.0] * n
        fixed = [False] * n
        remaining = BODY_W
        while True:
            pool = [i for i in range(n) if not fixed[i]]
            if not pool:
                break
            total = sum(natural[i] for i in pool)
            if total <= 0:
                for i in pool:
                    widths[i] = remaining / len(pool)
                break
            touched = False
            for i in pool:
                w = remaining * natural[i] / total
                if w < min_w:
                    widths[i] = min_w
                    fixed[i] = True
                    remaining -= min_w
                    touched = True
            if not touched:
                for i in pool:
                    widths[i] = remaining * natural[i] / total
                break
        return widths

    def table(self, headers: list[str], rows: list[list[str]], caption: str) -> None:
        ncol = len(headers)
        natural = []
        for i in range(ncol):
            w = self.body_font.text_length(headers[i], fontsize=TBL_SIZE)
            for r in rows:
                w = max(w, self.body_font.text_length(r[i] if i < len(r) else "", fontsize=TBL_SIZE))
            natural.append(max(w, 24))
        widths = self._alloc(natural)

        def row_height(cells: list[str]) -> tuple[list[list[str]], float]:
            wrapped = [self.wrap(cells[i], TBL_SIZE, widths[i] - 2 * PAD) for i in range(ncol)]
            return wrapped, max(len(w) for w in wrapped) * TBL_LH + 2 * PAD

        def draw_row(cells: list[str], bold: bool = False, fill=None) -> None:
            wrapped, h = row_height(cells)
            if fill is not None:
                self.page.draw_rect(
                    self.pymupdf.Rect(M_LEFT, self.y, M_LEFT + BODY_W, self.y + h),
                    color=None, fill=fill,
                )
            x = M_LEFT
            for i in range(ncol):
                self.page.draw_rect(
                    self.pymupdf.Rect(x, self.y, x + widths[i], self.y + h),
                    color=(0.72, 0.75, 0.79), width=0.5,
                )
                for j, ln in enumerate(wrapped[i]):
                    self.text(x + PAD, self.y + PAD + TBL_SIZE + j * TBL_LH, ln, TBL_SIZE,
                              bold=bold, color=(0.12, 0.14, 0.18))
                x += widths[i]
            self.y += h

        # 表题必须跟表在一起：先算高度再决定分页，否则表题会被留在上一页页尾
        head_h = row_height(headers)[1]
        first_h = row_height([rows[0][i] if i < len(rows[0]) else "" for i in range(ncol)])[1] if rows else 0
        self.need((20 if caption else 8) + head_h + first_h + 6)

        if caption:
            self.text(M_LEFT, self.y, caption, 9, align="center", width=BODY_W,
                      color=(0.30, 0.34, 0.40))
            self.y += 13
        draw_row(headers, bold=True, fill=(0.929, 0.945, 0.961))
        for r in rows:
            cells = [r[i] if i < len(r) else "" for i in range(ncol)]
            h = row_height(cells)[1]
            if self.y + h > BODY_BOTTOM:
                self.new_page()
                draw_row(headers, bold=True, fill=(0.929, 0.945, 0.961))
            draw_row(cells)
        self.y += 8

    def code_block(self, text: str) -> None:
        lines = text.splitlines()
        h = len(lines) * 12.6 + 12
        self.need(h + 4)
        rect = self.pymupdf.Rect(M_LEFT + 12, self.y, M_LEFT + BODY_W - 12, self.y + h)
        self.page.draw_rect(rect, color=(0.85, 0.87, 0.90), fill=(0.965, 0.970, 0.976))
        y = self.y + 16
        for ln in lines:
            self.text(M_LEFT + 22, y, ln, 9, color=(0.20, 0.24, 0.30))
            y += 12.6
        self.y += h + 8

    def signature(self, lines: Iterable[str]) -> None:
        self.need(24 * len(list(lines)) + 10)
        self.y += 14
        for ln in lines:
            self.text(M_LEFT, self.y, ln, 11, align="right", width=BODY_W,
                      color=(0.15, 0.18, 0.22))
            self.y += 22

    def footers(self) -> None:
        total = self.doc.page_count
        for i, page in enumerate(self.doc):
            if self.body_file:
                page.insert_font(fontname=self.body_name, fontfile=self.body_file)
            label = f"第 {i + 1} 页 共 {total} 页"
            page.insert_text(
                (PAGE_W / 2 - 34, PAGE_H - 40), label,
                fontsize=8.5, fontname=self.body_name, color=(0.45, 0.47, 0.52),
            )


def render_pdf(d: dict[str, Any], path: Path) -> None:
    writer = _Pdf(d)
    writer.new_page()
    writer.title_block()

    for block in d["blocks"]:
        tag = block[0]
        if tag == "h1":
            writer.heading(block[1], H1_SIZE, 14)
        elif tag == "h2":
            writer.heading(block[1], H2_SIZE, 10)
        elif tag == "h3":
            writer.heading(block[1], H3_SIZE, 6)
        elif tag == "p":
            writer.paragraph(block[1])
        elif tag == "li":
            for item in block[1]:
                writer.item(item.strip(), "·", nested=item.startswith("  "))
            writer.y += 3
        elif tag == "oli":
            for i, item in enumerate(block[1], start=1):
                writer.item(item.strip(), f"{i}.", nested=item.startswith("  "))
            writer.y += 3
        elif tag == "table":
            writer.table(block[1], block[2], block[3])
        elif tag == "code":
            writer.code_block(block[1])
        elif tag == "quote":
            writer.paragraph("【说明】" + block[1])
        elif tag == "sign":
            writer.signature(block[1])
        elif tag == "pagebreak":
            writer.new_page()

    if d["revisions"]:
        writer.heading("文件修订记录", H2_SIZE, 16)
        writer.table(["版本", "修订日期", "修订内容", "修订人"], d["revisions"], "")

    writer.footers()
    writer.doc.set_metadata({
        "title": d["title"],
        "author": "瀚宇智能科技股份有限公司",
        "subject": d["meta"].get("文档编号", ""),
        "keywords": f"{d['meta'].get('密级', '')},{d['meta'].get('责任部门', '')}",
        "creator": "瀚宇科技文档中心",
    })
    try:
        # 嵌入的整套中文字体有一二十兆，子集化后回到几百 KB
        writer.doc.subset_fonts()
    except Exception:
        pass
    writer.doc.save(path, garbage=4, deflate=True)
    writer.doc.close()


# --------------------------------------------------------------------------- #
# 分发
# --------------------------------------------------------------------------- #

def render(d: dict[str, Any], out_root: Path) -> Path:
    folder = out_root / d["fmt"]
    folder.mkdir(parents=True, exist_ok=True)
    name = d["filename"]
    if not name.lower().endswith(f".{d['fmt']}"):
        name = f"{name}.{d['fmt']}"
    path = folder / name

    if d["fmt"] == "txt":
        path.write_bytes(render_txt(d))
    elif d["fmt"] == "md":
        path.write_text(render_md(d), encoding="utf-8")
    elif d["fmt"] == "docx":
        render_docx(d, path)
    elif d["fmt"] == "pdf":
        render_pdf(d, path)
    else:
        raise ValueError(f"未知格式：{d['fmt']}")
    return path
