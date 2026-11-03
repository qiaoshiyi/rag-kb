from __future__ import annotations

import json

import pytest

from ragkb.api.app import create_app


@pytest.fixture
def client(md_file):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    app = create_app(
        docs_dir=md_file.parent,
        embedder_backend="hashing",
        reranker_backend="lexical",
        llm_provider="mock",
    )
    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_health_reports_index_stats(client):
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents"] >= 1
    assert body["points"] > 0
    assert body["embedder"] == "HashingEmbedder"
    assert body["llm"] == "MockLLM"
    assert body["docs_dir"], "页面要靠它显示当前索引的是哪批文档"


def test_examples_are_actually_answerable(client):
    """示例问题必须是当前索引真检索得到的。

    写死示例 + 换语料 = 用户点第一个示例就吃一记拒答，然后以为系统坏了。
    """
    body = client.get("/health").json()
    assert body["examples"], "有可答问题时应该给出示例"

    for question in body["examples"]:
        hit = client.post("/search", json={"query": question}).json()
        assert hit["refused"] is False, f"示例问题答不出来：{question}"


def test_examples_empty_when_index_empty(tmp_path):
    """空索引下宁可不给示例，也不要摆一排点了必拒答的问题。"""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    app = create_app(
        docs_dir=tmp_path,
        embedder_backend="hashing",
        reranker_backend="lexical",
        llm_provider="mock",
    )
    with fastapi_testclient.TestClient(app) as c:
        body = c.get("/health").json()

    assert body["documents"] == 0
    assert body["examples"] == []


def test_search_returns_hits(client):
    resp = client.post("/search", json={"query": "年假有几天", "top_k": 3})
    assert resp.status_code == 200

    body = resp.json()
    assert body["refused"] is False
    assert body["hits"]
    assert len(body["hits"]) <= 3
    assert body["hits"][0]["section_path"]
    assert body["hits"][0]["filename"]


def test_ask_returns_citations(client):
    resp = client.post("/ask", json={"query": "年假有几天"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["text"]
    assert body["citations"], "答案必须能回溯到来源"
    assert body["citations"][0]["filename"].endswith(".md")
    assert body["uncited"] is False


def test_ask_refuses_off_topic(client):
    resp = client.post("/ask", json={"query": "逆回购利率对流动性的影响"})
    body = resp.json()

    assert body["refused"] is True
    assert body["citations"] == []


def test_ask_stream_emits_sse(client):
    with client.stream(
        "POST", "/ask/stream", json={"query": "年假有几天"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    events = [
        line.removeprefix("event: ")
        for line in raw.splitlines()
        if line.startswith("event: ")
    ]
    assert events[0] == "retrieval"
    assert "citations" in events
    assert "delta" in events
    assert events[-1] == "done"

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]
    assert payloads[-1]["text"]


def test_search_validates_empty_query(client):
    resp = client.post("/search", json={"query": ""})
    assert resp.status_code == 422


def test_search_top_k_bounds(client):
    resp = client.post("/search", json={"query": "年假", "top_k": 99})
    assert resp.status_code == 422


def test_acl_filter_applied(client):
    """没有 acl_tags 的文档，在带过滤条件的查询里应该被排除。"""
    resp = client.post(
        "/search", json={"query": "年假有几天", "acl_tags": ["hr"]}
    )
    body = resp.json()
    assert body["hits"] == []


def test_index_serves_console(client):
    """首页必须能直接打开：它是这个项目的演示入口。"""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    body = res.text
    assert "企业知识库问答" in body
    assert "/ask/stream" in body  # 前端确实在调流式接口


def test_cors_allows_local_preview(client):
    """允许从 file:// 打开页面调试（本地开发放开，生产要收窄）。"""
    res = client.options(
        "/ask/stream",
        headers={
            "Origin": "null",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "*"
