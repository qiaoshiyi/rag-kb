"""HTTP 接口的数据契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    acl_tags: list[str] | None = Field(
        default=None, description="不传则不过滤；传了就在检索时下推为 payload filter"
    )


class HitOut(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    section_path: str
    filename: str
    page: int | None = None
    score: float
    sources: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    refused: bool = False
    reason: str = ""
    hits: list[HitOut] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    history: list[MessageIn] | None = None
    acl_tags: list[str] | None = None


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    title: str
    filename: str
    section_path: str
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    text: str
    rewritten: str = ""
    """多轮追问实际用于检索的独立问题；与 query 相同表示本轮未改写。"""
    refused: bool = False
    reason: str = ""
    uncited: bool = False
    citations: list[CitationOut] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    documents: int
    points: int
    embedder: str
    reranker: str
    llm: str
    failed_documents: int = 0
    min_score: float | None = None
    """已标定的拒答阈值；None 表示未启用。"""
    rewriter: str = "none"
    docs_dir: str = ""
    """当前索引的语料目录名。页面顶部要显示它——问错语料是最高频的"全屏拒答"原因。"""
    collection: str = ""
    examples: list[str] = Field(default_factory=list)
    """已用当前索引实测验证过的示例问题；空表示一个都没通过，前端不应展示示例。"""
