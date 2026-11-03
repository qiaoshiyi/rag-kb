"""生成层：把检索结果组装成带引用的答案。"""

from ragkb.generation.answer import REFUSAL_TEXT, Answer, QAService, to_sse
from ragkb.generation.base import LLM, Message
from ragkb.generation.mock import MockLLM
from ragkb.generation.prompt import Citation, build_context, build_messages, parse_citations

__all__ = [
    "LLM",
    "Message",
    "MockLLM",
    "Citation",
    "build_context",
    "build_messages",
    "parse_citations",
    "Answer",
    "QAService",
    "REFUSAL_TEXT",
    "to_sse",
]
