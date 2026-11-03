"""PDF 解析（双通道）。

通道判断：前若干页平均字符数 < 50 视为扫描件。
- 文字层通道：PyMuPDF 直取，毫秒级，字符无识别误差
- OCR 通道：MinerU 子进程，1~3 秒/页，仅扫描件才走

一份 1000 页电子版 PDF 全量走 MinerU 要半小时以上，双通道只要几十秒。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pymupdf

from ragkb.models.document import Block, Document
from ragkb.parsing.base import BaseParser, ParseError, register

# 前 N 页平均字符数低于该阈值，且图片覆盖率高，才判为扫描件
_SCAN_SAMPLE_PAGES = 5
_SCAN_CHAR_THRESHOLD = 50
_SCAN_IMAGE_RATIO = 0.3

# 字号相对正文基准字号的倍率 → 标题层级
_LEVEL_RATIOS = ((1.6, 1), (1.3, 2), (1.15, 3))


def _image_area_ratio(page: "pymupdf.Page") -> float:
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    covered = 0.0
    try:
        infos = page.get_image_info()
    except Exception:
        return 0.0
    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        covered += abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return min(covered / page_area, 1.0)


def looks_scanned(pdf: "pymupdf.Document", sample_pages: int = _SCAN_SAMPLE_PAGES) -> bool:
    """判断是否需要 OCR。

    只看平均字符数会误判：封面页、单行通知这类短文档字数极少但确实是电子版。
    所以再加一道条件——文字少「且」整页被图片占据，才认定为扫描件。
    """
    if pdf.page_count == 0:
        return True

    n = min(pdf.page_count, sample_pages)
    chars = 0
    img_ratio = 0.0
    for i in range(n):
        page = pdf[i]
        chars += len(page.get_text().strip())
        img_ratio += _image_area_ratio(page)

    avg_chars = chars / n
    avg_img = img_ratio / n

    if avg_chars >= _SCAN_CHAR_THRESHOLD:
        return False
    return avg_img >= _SCAN_IMAGE_RATIO


def _join_lines(lines: Iterable[str]) -> str:
    out = ""
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if not out:
            out = s
        elif out[-1].isascii() and out[-1].isalnum() and s[0].isascii() and s[0].isalnum():
            out += " " + s
        else:
            out += s
    return out


def _base_size(lines: list[tuple[str, float]]) -> float:
    """正文基准字号。

    按「字号覆盖的字符数」加权，而不是统计出现频次：
    正文段落总是占据最大篇幅，标题再大也只有寥寥几行。
    短文档（一页三行）用频次会直接失效，加权法依然成立。
    """
    weight: dict[float, int] = {}
    for text, size in lines:
        if size < 4:
            continue
        weight[size] = weight.get(size, 0) + len(text)
    if not weight:
        return 0.0
    return max(weight.items(), key=lambda kv: kv[1])[0]


def _heading_level(size: float, base: float) -> int | None:
    if base <= 0:
        return None
    ratio = size / base
    for threshold, level in _LEVEL_RATIOS:
        if ratio >= threshold:
            return level
    return None


def _lines_to_blocks(lines: list[tuple[str, float]], base: float, page_no: int) -> list[Block]:
    out: list[Block] = []
    buf: list[str] = []

    def flush() -> None:
        text = _join_lines(buf)
        if text:
            out.append(Block(type="para", text=text, page=page_no))
        buf.clear()

    for text, size in lines:
        level = _heading_level(size, base)
        if level is not None and len(text) <= 60:
            flush()
            out.append(Block(type="title", text=text, level=level, page=page_no))
        else:
            buf.append(text)
    flush()
    return out


def _table_to_markdown(rows: list[list[Any]]) -> str:
    cleaned = [
        [("" if c is None else str(c)).replace("\n", " ").replace("|", "\\|").strip() for c in row]
        for row in rows
    ]
    cleaned = [r for r in cleaned if any(c for c in r)]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    lines: list[str] = []
    for idx, row in enumerate(cleaned):
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if idx == 0:
            lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    return "\n".join(lines)


def _extract_tables(page: "pymupdf.Page", page_no: int) -> list[Block]:
    """用 PyMuPDF 内置表格检测。

    若某类文档表格还原不准（合并单元格、无边框表），换 pdfplumber
    的 table extraction 在这条分支里替代即可（需自行安装），IR 不受影响。
    """
    blocks: list[Block] = []
    try:
        finder = page.find_tables()
    except Exception:
        return blocks

    for idx, table in enumerate(getattr(finder, "tables", []) or []):
        try:
            md = _table_to_markdown(table.extract())
        except Exception:
            continue
        if md:
            blocks.append(Block(type="table", text=md, page=page_no, meta={"table_index": idx}))
    return blocks


def _extract_text_layer(pdf: "pymupdf.Document", with_tables: bool = True) -> list[Block]:
    pages: list[tuple[int, list[list[tuple[str, float]]]]] = []
    all_lines: list[tuple[str, float]] = []

    for pno in range(pdf.page_count):
        page = pdf[pno]
        data = page.get_text("dict")
        page_blocks: list[list[tuple[str, float]]] = []

        for raw_block in data.get("blocks", []):
            if raw_block.get("type") != 0:  # 0 = 文本块，1 = 图像
                continue
            lines: list[tuple[str, float]] = []
            for line in raw_block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                size = round(max((float(s.get("size", 0)) for s in spans), default=0.0), 1)
                lines.append((text, size))
                all_lines.append((text, size))
            if lines:
                page_blocks.append(lines)
        pages.append((pno + 1, page_blocks))

    base = _base_size(all_lines)
    blocks: list[Block] = []
    for page_no, page_blocks in pages:
        for lines in page_blocks:
            blocks.extend(_lines_to_blocks(lines, base, page_no))
        if with_tables:
            blocks.extend(_extract_tables(pdf[page_no - 1], page_no))
    return blocks


def _extract_with_mineru(path: Path, timeout: int = 1800) -> list[Block]:
    """调用 MinerU CLI 处理扫描件，再按 markdown 回读。

    走子进程而不是 Python API：MinerU 各版本内部接口变动频繁，CLI 最稳。
    """
    from ragkb.parsing.markdown import parse_markdown

    try:
        subprocess.run(["mineru", "--version"], capture_output=True, check=True, timeout=60)
    except (FileNotFoundError, subprocess.SubprocessError):
        raise ParseError(
            "检测到扫描件（无文字层），需要 OCR 能力。请安装：pip install 'ragkb[ocr]'"
        ) from None

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        proc = subprocess.run(
            ["mineru", "-p", str(path), "-o", str(out_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise ParseError(f"MinerU 处理失败：{proc.stderr[:500]}")

        md_files = sorted(out_dir.rglob("*.md"), key=lambda p: p.stat().st_size, reverse=True)
        if not md_files:
            raise ParseError("MinerU 未产出 markdown，请检查 OCR 依赖与模型是否就绪")
        return parse_markdown(md_files[0].read_text(encoding="utf-8", errors="replace"))


@register
class PdfParser(BaseParser):
    source_type = "pdf"
    suffixes = (".pdf",)

    def __init__(self, with_tables: bool = True) -> None:
        self.with_tables = with_tables

    def parse(self, path: Path, meta: dict[str, Any] | None = None) -> Document:
        doc = self.new_document(path, meta)

        try:
            pdf = pymupdf.open(path)
        except Exception as exc:
            raise ParseError(f"无法打开 PDF：{path.name} → {exc}") from exc

        try:
            if pdf.needs_pass:
                raise ParseError(f"PDF 已加密，需先解密：{path.name}")

            doc.meta["pages"] = pdf.page_count
            if looks_scanned(pdf):
                doc.meta["channel"] = "ocr"
                doc.blocks = _extract_with_mineru(path)
            else:
                doc.meta["channel"] = "text"
                doc.blocks = _extract_text_layer(pdf, with_tables=self.with_tables)
        finally:
            pdf.close()

        for b in doc.blocks:
            if b.type == "title" and (b.level or 9) == 1:
                doc.meta["title"] = b.text
                break

        return doc
