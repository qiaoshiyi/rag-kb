"""词法稀疏编码。

把文本转成 sparse vector（term hash → tf 权重），不需要分词器和语料统计。

两个用途：
1. HashingEmbedder 的稀疏侧（测试替身）
2. API embedding 的稀疏侧——qwen embedding 只返回稠密向量，
   混合检索的另一半得靠本地补上，否则模型型号、编号这类查询会漏
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Sequence

from ragkb.embedding.base import SparseVector

_RE_WORD = re.compile(r"[a-zA-Z0-9_]+")
_RE_CJK = re.compile(r"[\u4e00-\u9fff]")

DEFAULT_SPACE = 1 << 20


def hash_token(token: str, mod: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % mod


def sparse_tokens(text: str) -> list[str]:
    """单字 + 整词。刻意不用 2-gram：稀疏侧负责精确匹配，模糊交给稠密侧。"""
    tokens = _RE_CJK.findall(text)
    tokens.extend(w.lower() for w in _RE_WORD.findall(text))
    return tokens


def build_sparse(text: str, space: int = DEFAULT_SPACE) -> SparseVector:
    """无状态版本：只有 TF，没有 IDF。给 HashingEmbedder 这类替身用。"""
    weights: dict[int, float] = {}
    for token, count in Counter(sparse_tokens(text)).items():
        idx = hash_token(token, space)
        weights[idx] = max(weights.get(idx, 0.0), 1.0 + math.log(count))
    return SparseVector(indices=list(weights), values=list(weights.values()))


class SparseEncoder:
    """带 IDF 的词法编码器（简化版 BM25）。

    为什么必须有 IDF：只用词频时，"的 / 是 / 在"这类虚词权重和"住宿费"一样高，
    在语义改述的查询上会引入大量噪声，实测反而把 MRR 从 1.000 拉到 0.900。
    IDF 靠 observe() 累积，索引完所有文档后再查询即可。
    """

    def __init__(self, space: int = DEFAULT_SPACE) -> None:
        self._space = space
        self._df: Counter[str] = Counter()
        self._n = 0

    @property
    def documents_seen(self) -> int:
        return self._n

    def observe(self, texts: Sequence[str]) -> None:
        """在编码前统计文档频率。重复调用同一个文档会重复计数，索引时注意只调一次。"""
        for text in texts:
            for token in set(sparse_tokens(text)):
                self._df[token] += 1
        self._n += len(texts)

    def encode(self, text: str) -> SparseVector:
        weights: dict[int, float] = {}
        for token, count in Counter(sparse_tokens(text)).items():
            df = self._df.get(token, 0)
            idf = math.log(1.0 + self._n / (1.0 + df)) if self._n else 1.0
            weight = (1.0 + math.log(count)) * idf
            idx = hash_token(token, self._space)
            weights[idx] = max(weights.get(idx, 0.0), weight)
        return SparseVector(indices=list(weights), values=list(weights.values()))
