"""重排。

召回 30~50 条后用 cross-encoder 精排到 5~8 条，
这是整条链路里性价比最高的一步——通常比换更大的生成模型更有效。

bge-reranker 输出经 sigmoid 归一化到 0~1，可直接当拒答阈值用。
"""

from __future__ import annotations

import math
import os
import re
from typing import Protocol, Sequence, runtime_checkable

from ragkb.retrieval.hybrid import Hit

_RE_TOKEN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")

DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
DEFAULT_RERANK_MODEL = "qwen3.7-text-rerank"

SILICONFLOW_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
SILICONFLOW_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _RE_TOKEN.findall(text)}


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


class LexicalReranker:
    """替身重排器：按查询词在候选文本中的覆盖率打分。

    只用于测试链路与阈值逻辑，不能替代 cross-encoder。
    """

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return list(hits)

        scored: list[Hit] = []
        for hit in hits:
            doc_tokens = _tokens(hit.text)
            overlap = len(query_tokens & doc_tokens) / len(query_tokens)
            # 加一点长度惩罚，避免长文本天然占优
            penalty = 1.0 / (1.0 + math.log1p(len(hit.text) / 400.0))
            item = Hit(
                chunk_id=hit.chunk_id,
                score=overlap * penalty,
                payload=hit.payload,
                sources=hit.sources,
                raw_score=hit.raw_score,
                rerank_score=overlap * penalty,
                source_scores=hit.source_scores,
            )
            scored.append(item)

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        max_length: int = 512,
    ) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "bge-reranker 需要 FlagEmbedding，请先安装：pip install 'ragkb[embed]'"
            ) from exc

        self._model = FlagReranker(model_name, use_fp16=use_fp16)
        self._max_length = max_length

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []

        pairs = [[query, hit.text] for hit in hits]
        scores = self._model.compute_score(pairs, normalize=True, max_length=self._max_length)
        if isinstance(scores, (int, float)):
            scores = [scores]

        rescored: list[Hit] = []
        for hit, score in zip(hits, scores):
            value = float(score)
            rescored.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    score=value,
                    payload=hit.payload,
                    sources=hit.sources,
                    raw_score=hit.raw_score,
                    rerank_score=value,
                    source_scores=hit.source_scores,
                )
            )
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored


def _apply_rerank_scores(hits: Sequence[Hit], items: Sequence[dict]) -> list[Hit]:
    """把接口返回的 [{index, relevance_score}] 落回 Hit 列表并重新排序。

    各家供应商的差别只在「请求怎么发」「结果放在哪个字段」，
    打分与排序逻辑完全一样，所以抽出来，避免两套代码各自漂移。
    """
    rescored: list[Hit] = []
    for item in items:
        index = int(item.get("index", -1))
        if not 0 <= index < len(hits):
            continue
        base = hits[index]
        score = float(item.get("relevance_score", 0.0))
        rescored.append(
            Hit(
                chunk_id=base.chunk_id,
                score=score,
                payload=base.payload,
                sources=base.sources,
                raw_score=base.raw_score,
                rerank_score=score,
                source_scores=base.source_scores,
            )
        )

    rescored.sort(key=lambda h: h.score, reverse=True)
    return rescored


class DashScopeReranker:
    """qwen3.7-text-rerank，走 DashScope 原生端点（不是 OpenAI 兼容路径）。

    返回 relevance_score（0~1），量纲干净，可直接用于拒答阈值。
    相比 lexical 替身，它真的理解「出差住酒店能报多少」和「住宿费标准」是一回事。
    """

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        api_key: str | None = None,
        url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY，无法调用重排接口")
        self._key = key
        self._model = model
        self._url = url or os.environ.get("DASHSCOPE_RERANK_URL", DASHSCOPE_RERANK_URL)
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError("需要 httpx：pip install httpx") from exc

        payload = {
            "model": self._model,
            "input": {"query": query, "documents": [hit.text for hit in hits]},
            "parameters": {"return_documents": False, "top_n": len(hits)},
        }
        response = httpx.post(
            self._url,
            json=payload,
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        results = response.json().get("output", {}).get("results", [])
        if not results:
            return list(hits)
        return _apply_rerank_scores(hits, results) or list(hits)


class SiliconFlowReranker:
    """BAAI/bge-reranker-v2-m3，走硅基流动的 /v1/rerank（Cohere 风格）。

    和 DashScope 的差别只有两处：请求体是扁平的，结果直接在顶层 results 里。
    选它是因为有免费额度且国内可直连——公开 demo 不适合挂自己的付费 key。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        url: str | None = None,
        timeout: float = 90.0,
        retries: int = 3,
    ) -> None:
        key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        if not key:
            raise RuntimeError("未找到 SILICONFLOW_API_KEY，无法调用重排接口")
        self._key = key
        self._model = (
            model or os.environ.get("SILICONFLOW_RERANK_MODEL") or SILICONFLOW_RERANK_MODEL
        )
        self._url = (
            url or os.environ.get("SILICONFLOW_RERANK_URL") or SILICONFLOW_RERANK_URL
        )
        self._timeout = timeout
        self._retries = max(1, retries)

    @property
    def model(self) -> str:
        return self._model

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError("需要 httpx：pip install httpx") from exc

        payload = {
            "model": self._model,
            "query": query,
            "documents": [hit.text for hit in hits],
            "top_n": len(hits),
            "return_documents": False,
        }
        # 8B 级重排在高峰期偶发读超时（实测标定时遇到过一次），
        # 单次失败就把整轮标定打断不划算，退避重试两次比抛错更有用。
        last_error: Exception | None = None
        for attempt in range(self._retries):
            try:
                response = httpx.post(
                    self._url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._key}"},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                if not results:
                    return list(hits)
                return _apply_rerank_scores(hits, results) or list(hits)
            except Exception as exc:  # noqa: BLE001 - 网络/配额/超时都走同一条退路
                last_error = exc
                if attempt + 1 < self._retries:
                    time.sleep(1.5 * (attempt + 1))

        warnings.warn(
            f"重排失败（{self._model}，{self._retries} 次）：{last_error}；本次退化为 RRF 顺序",
            RuntimeWarning,
        )
        return list(hits)


def build_reranker(backend: str = "auto", **kwargs) -> Reranker:
    if backend == "lexical":
        return LexicalReranker()

    if backend in {"bge", "bge-reranker"}:
        return BGEReranker(**kwargs)

    if backend in {"dashscope", "qwen", "qwen-rerank"}:
        return DashScopeReranker(**kwargs)

    if backend in {"siliconflow", "silicon", "sf"}:
        return SiliconFlowReranker(**kwargs)

    if backend == "auto":
        try:
            return BGEReranker(**kwargs)
        except Exception:
            pass
        if os.environ.get("SILICONFLOW_API_KEY"):
            try:
                return SiliconFlowReranker(**kwargs)
            except Exception:
                pass
        if os.environ.get("DASHSCOPE_API_KEY"):
            try:
                return DashScopeReranker(**kwargs)
            except Exception:
                pass
        return LexicalReranker()

    raise ValueError(f"未知的 reranker backend: {backend!r}")
