"""确定性哈希向量化 —— 仅用于测试与离线开发。

它没有任何语义理解能力，只按字面重叠给分。
用途是验证索引写入、混合召回、RRF 融合、父子块回填这些**逻辑**是否正确，
不能用来评估检索质量（那必须用真模型 + 金标集）。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from ragkb.embedding.base import Embedding
from ragkb.embedding.sparse import build_sparse, hash_token

_RE_WORD = re.compile(r"[a-zA-Z0-9_]+")
_RE_CJK = re.compile(r"[\u4e00-\u9fff]")


def _dense_tokens(text: str) -> list[str]:
    """稠密侧用字级 2-gram，模拟“语义相近”的模糊匹配。"""
    chars = _RE_CJK.findall(text)
    grams = [a + b for a, b in zip(chars, chars[1:])]
    grams.extend(w.lower() for w in _RE_WORD.findall(text))
    return grams


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正数")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> Embedding:
        return self._embed(text)

    def _embed(self, text: str) -> Embedding:
        dense = [0.0] * self._dim
        for token, count in Counter(_dense_tokens(text)).items():
            dense[hash_token(token, self._dim)] += 1.0 + math.log(count)

        norm = math.sqrt(sum(v * v for v in dense))
        if norm > 0:
            dense = [v / norm for v in dense]

        return Embedding(dense=dense, sparse=build_sparse(text))
