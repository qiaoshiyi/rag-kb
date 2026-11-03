"""企业知识库语料集：12 份完整制度文件，四类格式各三份。"""

from __future__ import annotations

from .docs_docx import DOCS as DOCX_DOCS
from .docs_md import DOCS as MD_DOCS
from .docs_pdf import DOCS as PDF_DOCS
from .docs_txt import DOCS as TXT_DOCS

ALL = [*TXT_DOCS, *MD_DOCS, *DOCX_DOCS, *PDF_DOCS]

__all__ = ["ALL", "TXT_DOCS", "MD_DOCS", "DOCX_DOCS", "PDF_DOCS"]
