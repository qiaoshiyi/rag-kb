"""关键词匹配的归一化：中文数字 ↔ 阿拉伯数字、空白差异。

为什么需要：
金标里写 `30 分钟`，模型答「三十分钟」，直接子串匹配就判 miss，
于是「答案含事实」这个确定性指标开始说谎——同一份答案，只是写法不同。
这类偏差不是模型的问题，是匹配器太字面。所以比较前双方都归一化：
中文数字转阿拉伯、去掉所有空白，短横线/波浪号统一。
"""

from __future__ import annotations

import re

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_BIG = {"万": 10_000, "亿": 100_000_000}

# 连续的中文数字串。刻意不含「万/亿」：把它们展开会引入新的不对称，
# 「五十万」变成 500000 而「50 万」只是 50万，反而匹配不上。
_CN_NUM = re.compile(r"[零〇一二两三四五六七八九十百千]{2,}")
_SPACE = re.compile(r"\s+")


def cn_to_int(text: str) -> int | None:
    """把「三十」「十五」「一百五十八」「两千」转成整数；无法解析返回 None。"""
    if not text:
        return None
    total, section, number = 0, 0, 0
    for ch in text:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            section += (number or 1) * unit
            number = 0
        elif ch in _CN_BIG:
            section += number
            total += (section or 1) * _CN_BIG[ch]
            section, number = 0, 0
        else:
            return None
    return total + section + number


def _replace_cn(match: re.Match[str]) -> str:
    raw = match.group(0)
    # 单个「十」这类太容易误伤（「十分重要」），只处理能解析成两位以上的串
    value = cn_to_int(raw)
    if value is None or value < 10:
        return raw
    return str(value)


def normalize(text: str) -> str:
    """归一化后用于关键词包含判断。"""
    text = _CN_NUM.sub(_replace_cn, text)
    text = text.replace("　", " ")
    text = _SPACE.sub("", text)
    return text


def contains(haystack: str, needle: str) -> bool:
    """needle 是否出现在 haystack 中，忽略数字写法与空白差异。"""
    if not needle:
        return True
    return normalize(needle) in normalize(haystack)


def contains_all(haystack: str, needles: list[str] | tuple[str, ...]) -> bool:
    return all(contains(haystack, n) for n in needles)
