"""生成 demo/ 下的四类样本文档，便于本地验证解析效果。

    python scripts/make_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent.parent / "demo"

TXT = """员工报销管理制度

第一章 总则

第一条 为规范费用报销流程，制定本制度。

第二条 本制度适用于全体在职员工。

第二章 报销标准

一、差旅报销

1.1 住宿费

一线城市每晚不超过 600 元，二线城市不超过 400 元。

1.2 交通费

高铁二等座据实报销，飞机经济舱需提前三天审批。

二、招待费

单次招待费用不得超过 2000 元，且需提前报备。
"""

MD = """---
title: 员工手册
---

# 员工手册

## 一、入职流程

新员工需在入职当天提交身份证复印件与学历证明。

## 二、假期规定

| 假期类型 | 天数 | 是否带薪 |
| --- | --- | --- |
| 年假 | 5-15 天 | 是 |
| 事假 | 不限 | 否 |
| 病假 | 按医嘱 | 部分 |

## 三、离职流程

离职需提前三十天提交书面申请，交接完成后办理手续。
"""


def make_txt() -> None:
    # GBK 编码，模拟从老系统导出的制度文件
    (OUT / "报销管理制度.txt").write_bytes(TXT.encode("gbk"))


def make_md() -> None:
    (OUT / "员工手册.md").write_text(MD, encoding="utf-8")


def make_docx() -> None:
    try:
        import docx
    except ImportError:
        print("跳过 docx：未安装 python-docx（pip install -e '.[dev]'）")
        return

    doc = docx.Document()
    doc.add_heading("采购管理办法", level=1)
    doc.add_paragraph("为规范采购流程、控制采购成本，制定本办法。")
    doc.add_heading("一、采购申请", level=2)
    doc.add_paragraph("需求部门填写采购申请单，注明用途、数量与预算。")

    table = doc.add_table(rows=3, cols=2)
    rows = [("金额区间", "审批人"), ("10 万以下", "部门经理"), ("10 万以上", "总经理")]
    for r, (a, b) in enumerate(rows):
        table.cell(r, 0).text = a
        table.cell(r, 1).text = b

    doc.add_heading("二、验收入库", level=2)
    doc.add_paragraph("到货后三个工作日内完成验收并办理入库。")
    doc.save(OUT / "采购管理办法.docx")


def make_pdf() -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    lines = [
        (90, "信息安全管理制度", 24),
        (150, "第一章 总则", 16),
        (185, "为保障公司信息系统与数据资产安全，依据国家有关法律法规，制定本制度。", 11),
        (210, "全体员工在使用公司信息系统时，应当遵守本制度的规定。", 11),
        (250, "第二章 账号管理", 16),
        (285, "账号密码长度不得少于十二位，且每九十天更换一次。", 11),
    ]
    for y, text, size in lines:
        page.insert_text((72, y), text, fontsize=size, fontname="china-s")
    pdf.save(OUT / "信息安全管理制度.pdf")
    pdf.close()


ATTENDANCE_MD = """# 考勤管理制度

## 一、工作时间

公司实行弹性工作制，核心工作时间为上午十点至下午四点，每日工时不少于八小时。

## 二、迟到与早退

每月累计迟到超过三次，扣减当月全勤奖；迟于中午十二点到达视为旷工半天。

## 三、加班管理

加班需提前在考勤系统提交申请。工作日加班按 1.5 倍计算调休，法定节假日加班按 3 倍计算。
"""

VPN_TXT = """远程办公管理规定

第一章 申请流程

员工远程办公需提前一天在 OA 系统提交申请，经直属主管批准后生效。

第二章 安全要求

远程办公必须通过公司 VPN 接入内网，禁止使用公共无线网络处理秘密级及以上数据。

VPN 账号仅限本人使用，不得转借他人，违反者按信息安全违规处理。
"""


def make_attendance_md() -> None:
    (OUT / "考勤管理制度.md").write_text(ATTENDANCE_MD, encoding="utf-8")


def make_vpn_txt() -> None:
    (OUT / "远程办公管理规定.txt").write_bytes(VPN_TXT.encode("gbk"))


def make_supplier_docx() -> None:
    try:
        import docx
    except ImportError:
        print("跳过 docx：未安装 python-docx")
        return

    doc = docx.Document()
    doc.add_heading("供应商管理办法", level=1)
    doc.add_paragraph("为规范供应商准入、分级与考核，制定本办法。")
    doc.add_heading("一、准入条件", level=2)
    doc.add_paragraph("供应商需依法成立满两年，且近三年无重大违法记录。")
    doc.add_heading("二、分级管理", level=2)
    doc.add_paragraph("按年度采购额将供应商分为三级，对应不同结算账期。")

    table = doc.add_table(rows=4, cols=3)
    rows = [
        ("供应商等级", "年度采购额", "结算账期"),
        ("A 级", "500 万元以上", "60 天"),
        ("B 级", "100 万至 500 万元", "45 天"),
        ("C 级", "100 万元以下", "30 天"),
    ]
    for r, (a, b, c) in enumerate(rows):
        table.cell(r, 0).text = a
        table.cell(r, 1).text = b
        table.cell(r, 2).text = c

    doc.save(OUT / "供应商管理办法.docx")


def make_data_pdf() -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    lines = [
        (90, "数据分级分类规范", 24),
        (150, "第一章 分级标准", 16),
        (185, "公司数据按敏感程度分为公开、内部、秘密、机密四级。", 11),
        (210, "公开级数据可对外发布，内部级数据仅限员工访问。", 11),
        (250, "第二章 访问控制", 16),
        (285, "秘密级数据仅限部门负责人及以上人员访问。", 11),
        (310, "机密级数据的访问需经总裁审批，并留存审批记录三年。", 11),
    ]
    for y, text, size in lines:
        page.insert_text((72, y), text, fontsize=size, fontname="china-s")
    pdf.save(OUT / "数据分级分类规范.pdf")
    pdf.close()


def main() -> int:
    OUT.mkdir(exist_ok=True)
    make_txt()
    make_md()
    make_docx()
    make_pdf()
    make_attendance_md()
    make_vpn_txt()
    make_supplier_docx()
    make_data_pdf()
    print("已生成：", sorted(p.name for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
