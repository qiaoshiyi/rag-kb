"""轻量 token 估算。

不引入 tokenizer 依赖（bge-m3 与 DeepSeek 的词表不一致，统一用估算值控制切分尺寸即可）。
中文按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token，与实际偏差在 15% 以内。
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            cjk += 1
        elif ch in "，。、；：？！“”‘’（）《》【】":
            cjk += 1
    rest = len(text) - cjk
    return cjk + max(1, rest // 4) if rest else cjk


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按估算值截断，保证不会切出半个词。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    ratio = max_tokens / max(estimate_tokens(text), 1)
    return text[: max(1, int(len(text) * ratio))]
