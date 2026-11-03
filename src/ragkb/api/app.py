"""FastAPI 服务层。

    export DASHSCOPE_API_KEY=sk-xxx
    python -m ragkb.api.app                     # 默认索引 demo/，用 API embedding

环境变量：
    KB_DOCS      待索引目录（默认 demo）
    KB_EMBEDDER  api / bge-m3 / hashing（默认 api，无 key 时自动降级 hashing）
    KB_RERANKER  auto / dashscope / bge / lexical / none
    KB_LLM       auto / dashscope / deepseek / mock
    KB_MODEL     覆盖 LLM 模型名
    KB_MIN_SCORE 标定后的拒答阈值（由 scripts/calibrate.py 产出），不设则不拒答
    KB_REWRITER  llm / heuristic / none（默认 llm，无 key 时降级 heuristic）
    KB_REWRITE_MODEL 改写模型名（建议小模型），不设则用 provider 默认
    QDRANT_URL   不设则用内存模式
"""

from __future__ import annotations

import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from ragkb.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    HealthResponse,
    HitOut,
    SearchRequest,
    SearchResponse,
)
from ragkb.chunking import split_document
from ragkb.embedding import build_embedder
from ragkb.generation import QAService, to_sse
from ragkb.generation.base import Message
from ragkb.generation.compat import build_llm
from ragkb.generation.mock import MockLLM
from ragkb.indexing.qdrant_store import QdrantStore, acl_filter
from ragkb.multiturn import build_rewriter
from ragkb.parsing import ParseError, parse_document
from ragkb.pipeline.ingest import iter_documents
from ragkb.retrieval import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import build_reranker


WEB_DIR = Path(__file__).resolve().parents[3] / "web"

# 候选示例问题。前端把它们当"点一下就能看到完整效果"的入口，
# 所以**必须是当前索引真答得出的**：写死的列表一旦和语料错位，
# 用户点第一个示例就吃一记拒答，会误判成系统坏了。
# 启动时逐条试检索，只把没被拒的透出去（见 _pick_examples）。
EXAMPLE_CANDIDATES = (
    "年假天数按工龄怎么分档？",
    "二类城市每天餐补多少钱？",
    "出差回来多久内要提交报销单？",
    "迟到超过三次每次扣多少钱？",
    "二线城市住宿费每晚上限是多少？",
    "远程办公要提前多久申请？",
    "50 万元以上的采购找谁审批？",
    "乘坐飞机出差对舱位和审批有什么要求？",
)


def _make_client():
    from qdrant_client import QdrantClient

    url = os.environ.get("QDRANT_URL")
    return QdrantClient(url=url) if url else QdrantClient(":memory:")


def _min_score() -> float | None:
    """没标定过就不开阈值。拍一个数上去只会静默误杀。"""
    raw = os.environ.get("KB_MIN_SCORE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"KB_MIN_SCORE={raw!r} 不是数字，已忽略", RuntimeWarning)
        return None


def _build_rewriter(llm_name: str):
    """改写默认开启：多轮下不改写，召回必然失败。

    没有真实模型时降到 heuristic——它是字面拼接，质量一般，
    但至少让「那二线城市呢」这类追问不至于拿一堆虚词去检索。
    """
    backend = os.environ.get("KB_REWRITER", "llm").strip().lower()
    if backend in {"none", "off", "noop"}:
        return None
    if backend == "heuristic":
        return build_rewriter("heuristic")
    try:
        from ragkb.generation.compat import build_llm

        llm = build_llm(
            llm_name if llm_name != "mock" else "auto",
            **({"model": model} if os.environ.get("KB_REWRITE_MODEL") else {}),
        )
        if llm is None:
            return build_rewriter("heuristic")
        from ragkb.multiturn import LLMRewriter

        return LLMRewriter(llm)
    except Exception as exc:
        warnings.warn(f"改写模型不可用（{exc}），回退 heuristic", RuntimeWarning)
        return build_rewriter("heuristic")


def _build_embedder(backend: str):
    try:
        return build_embedder(backend)
    except Exception as exc:
        warnings.warn(f"embedder {backend} 不可用（{exc}），降级为 hashing 替身", RuntimeWarning)
        return build_embedder("hashing")


def _pick_examples(retriever, limit: int = 4) -> list[str]:
    """只保留当前索引能检索到的示例问题。

    每条示例照最省的方式跑一遍完整检索（双路 + 重排 + 阈值），拒答的丢掉。
    代价是启动时多几次 embedding/rerank 调用，换来的是"示例永远可点"。
    """
    raw = os.environ.get("KB_EXAMPLES", "").strip()
    if raw.lower() in {"off", "none", "no"}:
        return []
    candidates = [q.strip() for q in raw.split(",") if q.strip()] if raw else list(EXAMPLE_CANDIDATES)

    picked: list[str] = []
    for question in candidates:
        try:
            if retriever.search(question).refused:
                continue
        except Exception as exc:  # 探测失败不能拖垮启动
            warnings.warn(f"示例问题探测失败（{question}）：{exc}", RuntimeWarning)
            continue
        picked.append(question)
        if len(picked) >= limit:
            break
    return picked


def create_app(
    docs_dir: str | Path | None = None,
    embedder_backend: str | None = None,
    reranker_backend: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    index_on_start: bool = True,
) -> FastAPI:
    docs = Path(docs_dir or os.environ.get("KB_DOCS", "demo"))
    emb_name = embedder_backend or os.environ.get("KB_EMBEDDER", "api")
    rer_name = reranker_backend or os.environ.get("KB_RERANKER", "auto")
    llm_name = llm_provider or os.environ.get("KB_LLM", "auto")
    model = llm_model or os.environ.get("KB_MODEL")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        def bootstrap() -> dict[str, Any]:
            embedder = _build_embedder(emb_name)
            store = QdrantStore(_make_client(), collection="kb", dim=embedder.dim)
            store.ensure(recreate=True)

            reranker = None
            if rer_name != "none":
                try:
                    reranker = build_reranker(rer_name)
                except Exception as exc:
                    warnings.warn(f"reranker {rer_name} 不可用（{exc}），回退 lexical", RuntimeWarning)
                    reranker = build_reranker("lexical")

            retriever = Retriever(
                embedder,
                store,
                reranker=reranker,
                config=RetrievalConfig(min_score=_min_score()),
            )

            indexed = failed = 0
            if index_on_start:
                for path in iter_documents(docs):
                    try:
                        doc = parse_document(path)
                        retriever.index_document(doc, split_document(doc))
                        indexed += 1
                    except ParseError:
                        failed += 1
                if indexed == 0:
                    warnings.warn(
                        f"索引目录 {docs} 里没有可解析的文档，服务起来了但一问就拒答。",
                        RuntimeWarning,
                    )

            return {
                "retriever": retriever,
                "embedder": type(embedder).__name__,
                "reranker": type(reranker).__name__ if reranker else "none",
                "documents": indexed,
                "failed": failed,
                "examples": _pick_examples(retriever) if indexed else [],
            }

        state = await run_in_threadpool(bootstrap)

        llm = None if llm_name == "mock" else build_llm(llm_name, **({"model": model} if model else {}))
        if llm is None:
            llm = MockLLM()

        rewriter = _build_rewriter(llm_name)
        app.state.retriever = state["retriever"]
        app.state.qa = QAService(state["retriever"], llm, rewriter=rewriter)
        app.state.info = {
            "documents": state["documents"],
            "failed_documents": state["failed"],
            "embedder": state["embedder"],
            "reranker": state["reranker"],
            "llm": type(llm).__name__,
            "min_score": _min_score(),
            "rewriter": type(rewriter).__name__,
            "docs_dir": docs.resolve().name if docs.exists() else str(docs),
            "collection": "kb",
            "examples": state["examples"],
        }
        if os.environ.get("KB_BANNER"):
            info = app.state.info
            print(
                f"[rag-kb] 索引 {info['docs_dir']} · {info['documents']} 篇 · "
                f"{state['retriever'].store.count()} 块 · "
                f"阈值 {info['min_score'] if info['min_score'] is not None else '关闭'} · "
                f"示例 {len(info['examples'])} 条",
                flush=True,
            )
        yield

    app = FastAPI(title="rag-kb", version="0.1.0", lifespan=lifespan)

    # 前端要能从 file:// 直接打开调试，所以本地开发放开跨域。
    # 生产环境务必收窄到具体域名。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        page = WEB_DIR / "index.html"
        if not page.exists():  # pragma: no cover - 缺文件时的兜底
            raise FileNotFoundError(f"前端文件缺失：{page}")
        return FileResponse(page, media_type="text/html; charset=utf-8")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        info = app.state.info
        return HealthResponse(
            **info,
            points=app.state.retriever.store.count(),
        )

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        flt = acl_filter(req.acl_tags) if req.acl_tags else None
        result = app.state.retriever.search(req.query, flt=flt)

        hits = [
            HitOut(
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                text=str(hit.payload.get("parent_text") or hit.text),
                section_path=hit.section_path,
                filename=str(hit.payload.get("filename", "")),
                page=hit.payload.get("page"),
                score=round(hit.score, 4),
                sources=list(hit.sources),
            )
            for hit in result.hits[: req.top_k]
        ]
        return SearchResponse(
            refused=result.refused, reason=result.reason, hits=hits, debug=result.debug
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        history = _to_history(req)
        flt = acl_filter(req.acl_tags) if req.acl_tags else None
        answer = app.state.qa.answer(req.query, flt=flt, history=history)
        return AskResponse(
            text=answer.text,
            rewritten=answer.rewritten,
            refused=answer.refused,
            reason=answer.reason,
            uncited=answer.uncited,
            citations=[CitationOut(**c.to_dict()) for c in answer.citations],
            debug=answer.debug,
        )

    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest):
        history = _to_history(req)
        flt = acl_filter(req.acl_tags) if req.acl_tags else None

        def generate():
            for event in app.state.qa.stream(req.query, flt=flt, history=history):
                yield to_sse(event)

        return StreamingResponse(
            iterate_in_threadpool(generate()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _to_history(req: AskRequest) -> list[Message] | None:
    if not req.history:
        return None
    return [Message(role=m.role, content=m.content) for m in req.history]


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "ragkb.api.app:app",
        host=os.environ.get("KB_HOST", "127.0.0.1"),
        port=int(os.environ.get("KB_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
