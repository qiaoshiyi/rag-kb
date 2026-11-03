from __future__ import annotations

import pytest

from ragkb.parsing import ParseError, parse_document, supported_suffixes


def test_supported_suffixes():
    assert set(supported_suffixes()) == {".txt", ".md", ".markdown", ".pdf", ".docx"}


def test_unsupported_suffix_raises(tmp_path):
    path = tmp_path / "表格.xlsx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ParseError, match="不支持的文件类型"):
        parse_document(path)


# ---------- txt ----------


def test_txt_gbk_decodes_without_mojibake(gbk_txt):
    doc = parse_document(gbk_txt)
    assert doc.meta["encoding"] in {"gb18030", "gbk"}
    text = doc.to_markdown()
    assert "差旅管理制度" in text
    assert "\ufffd" not in text  # 无替换字符


def test_txt_utf8_first_try(utf8_txt):
    doc = parse_document(utf8_txt)
    assert doc.meta["encoding"] == "utf-8-sig"
    assert "差旅管理制度" in doc.to_markdown()


def test_txt_pseudo_headings(gbk_txt):
    doc = parse_document(gbk_txt)
    titles = [(b.level, b.text) for b in doc.iter_blocks("title")]
    assert (1, "第一章 总则") in titles
    assert (1, "第二章 报销标准") in titles
    assert (2, "一、国内差旅") in titles
    assert (3, "1.1 住宿费标准") in titles
    assert (3, "1.2 交通费标准") in titles


def test_txt_body_not_mistaken_as_heading(gbk_txt):
    doc = parse_document(gbk_txt)
    for block in doc.iter_blocks("title"):
        assert len(block.text) <= 40


@pytest.mark.parametrize(
    "line,level",
    [
        ("第一章　总则", 1),
        ("第二条　适用范围", 2),
        ("（一）打卡方式", 3),
        ("三、违规处理", 2),
    ],
)
def test_txt_article_level_heading(line, level):
    """「第X条」是制度文件最主力的层级，必须识别，否则整章被压成一个块。"""
    from ragkb.parsing.txt import match_heading

    assert match_heading(line) == level


@pytest.mark.parametrize(
    "line",
    [
        # 以编号开头但其实是正文：靠「标题不以句读收尾」拦下
        "第一条　为规范公司考勤管理，保障正常经营秩序，维护员工合法权益，制定本制度。",
        "第三条规定的审批流程，由部门负责人执行，财务部复核。",
    ],
)
def test_txt_numbered_body_stays_body(line):
    from ragkb.parsing.txt import match_heading

    assert match_heading(line) is None


@pytest.mark.parametrize(
    "line,level",
    [
        ("第一章　总则", 1),
        ("第二条　适用范围", 2),
        ("（一）打卡方式", 3),
        ("三、违规处理", 2),
    ],
)
def test_txt_article_level_heading(line, level):
    """「第X条」是制度文件最主力的层级，必须识别，否则整章被压成一个块。"""
    from ragkb.parsing.txt import match_heading

    assert match_heading(line) == level


@pytest.mark.parametrize(
    "line",
    [
        # 以编号开头但其实是正文：靠「标题不以句读收尾」拦下
        "第一条　为规范公司考勤管理，保障正常经营秩序，维护员工合法权益，制定本制度。",
        "第三条规定的审批流程，由部门负责人执行，财务部复核。",
    ],
)
def test_txt_numbered_body_stays_body(line):
    from ragkb.parsing.txt import match_heading

    assert match_heading(line) is None


# ---------- markdown ----------


def test_md_frontmatter_title(md_file):
    doc = parse_document(md_file)
    assert doc.meta["title"] == "员工手册"


def test_md_structure(md_file):
    doc = parse_document(md_file)
    titles = [(b.level, b.text) for b in doc.iter_blocks("title")]
    assert (1, "员工手册") in titles
    assert (2, "一、入职") in titles
    assert (2, "二、假期") in titles


def test_md_table_kept_as_markdown(md_file):
    doc = parse_document(md_file)
    tables = [b for b in doc.blocks if b.type == "table"]
    assert tables, "表格应被保留为独立块"
    assert "| 年假 |" in tables[0].text
    assert "| --- |" in tables[0].text  # 分隔行存在，说明结构完整


def test_md_empty_document(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")
    doc = parse_document(path)
    assert doc.blocks == []


# ---------- docx ----------


def test_docx_headings(sample_docx):
    doc = parse_document(sample_docx)
    titles = [(b.level, b.text) for b in doc.iter_blocks("title")]
    assert (1, "采购管理办法") in titles
    assert (2, "一、采购申请") in titles
    assert (2, "二、验收") in titles


def test_docx_table_extracted(sample_docx):
    doc = parse_document(sample_docx)
    tables = [b for b in doc.blocks if b.type == "table"]
    assert tables
    assert "审批人" in tables[0].text
    assert "部门经理" in tables[0].text


def test_docx_paragraphs(sample_docx):
    doc = parse_document(sample_docx)
    paras = [b.text for b in doc.iter_blocks("para")]
    assert any("规范采购流程" in p for p in paras)


# ---------- pdf ----------


def test_pdf_text_layer_channel(text_pdf):
    doc = parse_document(text_pdf)
    assert doc.meta["channel"] == "text"
    assert doc.meta["pages"] == 1
    assert doc.char_count > 20
    assert "信息安全管理制度" in doc.to_markdown()


def test_pdf_font_size_infers_title(text_pdf):
    doc = parse_document(text_pdf)
    titles = [b.text for b in doc.iter_blocks("title")]
    # 24pt 与 16pt 的文本应被识别为标题，11pt 正文不算
    assert any("信息安全管理制度" in t for t in titles)
    assert all("制定本制度" not in t for t in titles)


def test_pdf_page_number_recorded(text_pdf):
    doc = parse_document(text_pdf)
    assert all(b.page == 1 for b in doc.blocks)


def test_scanned_pdf_requires_ocr(scanned_pdf, monkeypatch):
    """无文字层时应走 OCR 通道；未装 OCR 依赖要给出明确指引而不是静默返回空。"""
    from ragkb.parsing import pdf as pdf_module

    def _raise(*args, **kwargs):
        raise FileNotFoundError("mineru not installed")

    monkeypatch.setattr(pdf_module.subprocess, "run", _raise)

    with pytest.raises(ParseError) as exc:
        parse_document(scanned_pdf)
    assert "OCR" in str(exc.value)
