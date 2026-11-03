"""四类文档的自包含样例。

不依赖仓库里的二进制夹具，测试时现场生成，避免二进制文件进版本库。
"""

from __future__ import annotations

import pytest

TXT_CONTENT = """差旅管理制度

第一章 总则

第一条 为规范公司差旅行为，结合公司实际情况制定本制度。

第二条 本制度适用于公司全体在职员工。

第二章 报销标准

一、国内差旅

1.1 住宿费标准

一线城市每晚不超过 600 元，二线城市不超过 400 元，其他城市不超过 300 元。

1.2 交通费标准

高铁二等座据实报销，飞机经济舱需提前三天申请。

二、海外差旅

海外差旅住宿费按地区分级，A 类地区每晚不超过 1800 元。
"""

MD_CONTENT = """---
title: 员工手册
---

# 员工手册

## 一、入职

新员工需在入职当天提交身份证复印件与学历证明。

## 二、假期

各类假期的天数与薪资规定如下表。

| 假期类型 | 天数 | 是否带薪 |
| --- | --- | --- |
| 年假 | 5-15 | 是 |
| 事假 | 不限 | 否 |

## 三、离职

离职需提前三十天提交书面申请。
"""


@pytest.fixture
def gbk_txt(tmp_path):
    path = tmp_path / "差旅管理制度.txt"
    path.write_bytes(TXT_CONTENT.encode("gbk"))
    return path


@pytest.fixture
def utf8_txt(tmp_path):
    path = tmp_path / "utf8制度.txt"
    path.write_bytes(TXT_CONTENT.encode("utf-8"))
    return path


@pytest.fixture
def md_file(tmp_path):
    path = tmp_path / "员工手册.md"
    path.write_text(MD_CONTENT, encoding="utf-8")
    return path


@pytest.fixture
def sample_docx(tmp_path):
    docx = pytest.importorskip("docx", reason="需要 python-docx 生成样例")
    doc = docx.Document()
    doc.add_heading("采购管理办法", level=1)
    doc.add_paragraph("为规范采购流程、控制采购成本，制定本办法。")
    doc.add_heading("一、采购申请", level=2)
    doc.add_paragraph("需求部门填写采购申请单，注明用途与预算。")

    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "金额区间"
    table.cell(0, 1).text = "审批人"
    table.cell(1, 0).text = "10 万以下"
    table.cell(1, 1).text = "部门经理"

    doc.add_heading("二、验收", level=2)
    doc.add_paragraph("到货后三个工作日内完成验收并入库。")

    path = tmp_path / "采购管理办法.docx"
    doc.save(path)
    return path


@pytest.fixture
def text_pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "信息安全管理制度", fontsize=24, fontname="china-s")
    page.insert_text((72, 150), "第一章 总则", fontsize=16, fontname="china-s")
    page.insert_text(
        (72, 190),
        "为保障公司信息系统与数据资产安全，制定本制度。",
        fontsize=11,
        fontname="china-s",
    )
    path = tmp_path / "信息安全管理制度.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def scanned_pdf(tmp_path):
    """无文字层：整页是一张位图，get_text() 返回空。"""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 160), 0)
    pix.set_rect(pix.irect, (210, 210, 210))
    page.insert_image(page.rect, pixmap=pix)

    path = tmp_path / "扫描件.pdf"
    doc.save(path)
    doc.close()
    return path
