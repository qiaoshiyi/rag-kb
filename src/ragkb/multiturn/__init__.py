"""多轮对话：查询改写、话题跳转检测、双路召回决策。

多轮不是「多传几轮 history」这么简单。核心问题在检索侧：
追问天生不完整，「那海外呢」单独拿去检索必然失败，任何向量库都救不回来。
"""

from ragkb.multiturn.rewrite import (
    HeuristicRewriter,
    LLMRewriter,
    NoopRewriter,
    QueryRewriter,
    RewriteResult,
    build_rewriter,
    is_follow_up,
)
from ragkb.multiturn.session import (
    Conversation,
    PreparedQuery,
    prepare_query,
)

__all__ = [
    "QueryRewriter",
    "RewriteResult",
    "NoopRewriter",
    "HeuristicRewriter",
    "LLMRewriter",
    "build_rewriter",
    "is_follow_up",
    "PreparedQuery",
    "prepare_query",
    "Conversation",
]
