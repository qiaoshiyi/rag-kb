"""txt 解析。

两个真实的坑：
1. 编码。国内 txt 大量是 GBK/GB18030，直接用 utf-8 读会整篇乱码且不报错。
   这里按 utf-8 → gb18030 → charset-normalizer 的顺序尝试，前两者用 strict 校验。
2. 无结构。纯文本没有标题概念，靠行首编号模式提取「伪标题」，
   否则切分只能滑窗，父子块关系无从建立。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ragkb.models.document import Block, Document
from ragkb.parsing.base import BaseParser, register

# 第X章 / 第X节 / 第X篇 / 第X部分
_RE_CHAPTER = re.compile(r"^第\s*[一二三四五六七八九十百千零〇\d]+\s*[章节篇部]")
# 第X条：制度文件里最主力的层级。不识别它，整章会被压成一个块，
# 检索时「第几条」这类精确问法基本召回不到
_RE_ARTICLE = re.compile(r"^第\s*[一二三四五六七八九十百千零〇\d]+\s*条")
# 一、 二、  （中文顿号）
_RE_CN_ORDINAL = re.compile(r"^[一二三四五六七八九十]+\s*[、.]\s*")
# 1.  1.2  1.2.3（点分层级）
_RE_NUM_DOT = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})[\s、.]\s*")
# （一） (一)
_RE_PAREN_CN = re.compile(r"^[（(][一二三四五六七八九十]+[）)]\s*")

# 标题行长度上限：超过这个长度基本是正文里恰好以编号开头
_MAX_TITLE_CHARS = 40


def decode_text(raw: bytes) -> tuple[str, str]:
    """返回 (文本, 实际使用的编码)。

    先严格试 utf-8-sig，再试 gb18030，最后交给 charset-normalizer 兜底。
    顺序不能反：charset-normalizer 对短文本和纯中文容易误判。
    """
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue

    from charset_normalizer import from_bytes

    best = from_bytes(raw).best()
    if best is not None:
        return str(best), best.encoding
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def match_heading(line: str) -> int | None:
    """识别伪标题，返回推断出的层级；不是标题返回 None。"""
    s = line.strip()
    if not s or len(s) > _MAX_TITLE_CHARS:
        return None
    # 标题不会以句读符号收尾。少了这一条，「第一条　为规范……制定本制度。」
    # 这种正文（以编号开头且不足 40 字）会被当成标题，切分层级立刻错乱
    if s[-1] in "。！？；：，、""''":
        return None

    if _RE_CHAPTER.match(s):
        return 1
    if _RE_ARTICLE.match(s):
        return 2
    if _RE_PAREN_CN.match(s):
        return 3
    if _RE_CN_ORDINAL.match(s):
        return 2

    m = _RE_NUM_DOT.match(s)
    if m:
        depth = m.group(1).count(".")
        # "1. 总则" 视作二级（一级给了"第X章"），"1.1" 三级，"1.1.1" 四级
        return min(2 + depth, 6)
    return None


def _join_soft_lines(lines: list[str]) -> str:
    """合并段内软换行。中文直接拼接，两侧都是 ASCII 时补一个空格。"""
    out = ""
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if not out:
            out = s
        elif out[-1].isascii() and out[-1].isalnum() and s[0].isascii() and s[0].isalnum():
            out += " " + s
        else:
            out += s
    return out


def split_plain_text(text: str) -> list[Block]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks: list[Block] = []

    # 有空行 → 空行是段边界；没有空行 → 每个非空行自成一段
    if re.search(r"\n\s*\n", text):
        chunks = re.split(r"\n\s*\n", text)
    else:
        chunks = text.split("\n")

    for chunk in chunks:
        lines = [ln for ln in chunk.split("\n") if ln.strip()]
        if not lines:
            continue

        # 块首行是标题 → 单独成 title 块，其余行仍作正文
        level = match_heading(lines[0])
        if level is not None:
            blocks.append(Block(type="title", text=lines[0].strip(), level=level))
            lines = lines[1:]

        body = _join_soft_lines(lines)
        if body:
            blocks.append(Block(type="para", text=body))

    # 首行孤立的短句通常是文档标题（如「员工报销管理制度」）。
    # 只在文档本身存在编号标题时才提升，否则整篇无结构，首行当标题没有意义。
    if blocks and blocks[0].type == "para":
        first = blocks[0].text
        has_titles = any(b.is_title for b in blocks)
        if has_titles and len(first) <= 30 and not re.search(r"[。，；：！？,;:!?]", first):
            blocks[0] = Block(type="title", text=first, level=1)

    return blocks


@register
class TxtParser(BaseParser):
    source_type = "txt"
    suffixes = (".txt",)

    def parse(self, path: Path, meta: dict[str, Any] | None = None) -> Document:
        raw = path.read_bytes()
        text, encoding = decode_text(raw)

        doc = self.new_document(path, meta)
        doc.meta["encoding"] = encoding
        doc.blocks = split_plain_text(text)

        for block in doc.blocks:
            if block.type == "title" and (block.level or 9) == 1:
                doc.meta["title"] = block.text
                break

        return doc
