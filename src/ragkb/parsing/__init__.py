"""解析层入口。导入本包即完成四类适配器注册。"""

from ragkb.parsing.base import (
    BaseParser,
    ParseError,
    get_parser,
    parse_document,
    register,
    supported_suffixes,
)

# 触发注册。顺序无关，但 html_ir 需先于 docx 可用
from ragkb.parsing import html_ir  # noqa: F401
from ragkb.parsing import txt  # noqa: F401
from ragkb.parsing import markdown  # noqa: F401
from ragkb.parsing import pdf  # noqa: F401
from ragkb.parsing import docx  # noqa: F401

__all__ = [
    "BaseParser",
    "ParseError",
    "get_parser",
    "parse_document",
    "register",
    "supported_suffixes",
]
