from __future__ import annotations

import pytest

from ragkb.chunking import split_document
from ragkb.embedding.hashing import HashingEmbedder
from ragkb.indexing.qdrant_store import QdrantStore, ScoredHit, acl_filter
from ragkb.parsing import parse_document
from ragkb.retrieval.hybrid import rrf_fuse
from ragkb.retrieval.pipeline import RetrievalConfig, Retriever
from ragkb.retrieval.rerank import LexicalReranker

DIM = 256


@pytest.fixture
def store():
    qdrant_client = pytest.importorskip("qdrant_client")
    client = qdrant_client.QdrantClient(":memory:")
    s = QdrantStore(client, collection="test", dim=DIM)
    s.ensure(recreate=True)
    return s


def make_retriever(store, min_score: float = 0.0, use_rerank: bool = True) -> Retriever:
    return Retriever(
        embedder=HashingEmbedder(dim=DIM),
        store=store,
        reranker=LexicalReranker() if use_rerank else None,
        config=RetrievalConfig(min_score=min_score, use_rerank=use_rerank),
    )


@pytest.fixture
def retriever(store):
    return make_retriever(store)


# ---------- RRF 纯逻辑 ----------


def test_rrf_puts_mutual_hits_first():
    run_a = [ScoredHit("a", 0.9), ScoredHit("b", 0.8), ScoredHit("c", 0.7)]
    run_b = [ScoredHit("c", 0.9), ScoredHit("d", 0.8), ScoredHit("a", 0.1)]

    fused = rrf_fuse([run_a, run_b], k=60, names=["dense", "sparse"])

    assert {h.chunk_id for h in fused[:2]} == {"a", "c"}
    assert fused[0].sources == ("dense", "sparse") or fused[0].sources == ("sparse", "dense")


def test_rrf_records_single_path_source():
    run_a = [ScoredHit("a", 0.9)]
    run_b = [ScoredHit("b", 0.9)]

    fused = rrf_fuse([run_a, run_b], names=["dense", "sparse"])
    by_id = {h.chunk_id: h for h in fused}
    assert by_id["a"].sources == ("dense",)
    assert by_id["b"].sources == ("sparse",)


def test_rrf_weight_shifts_order():
    run_a = [ScoredHit("a", 0.9)]
    run_b = [ScoredHit("b", 0.9)]

    balanced = rrf_fuse([run_a, run_b], weights=[1.0, 1.0])
    assert balanced[0].score == pytest.approx(balanced[1].score, abs=1e-9)

    dense_heavy = rrf_fuse([run_a, run_b], weights=[3.0, 1.0])
    assert dense_heavy[0].chunk_id == "a"


def test_rrf_empty_runs():
    assert rrf_fuse([[], []]) == []


# ---------- 索引 ----------


def test_index_counts(md_file, retriever):
    doc = parse_document(md_file)
    chunks = split_document(doc)

    written = retriever.index_document(doc, chunks)

    assert written == len(chunks)
    assert retriever.store.count() == len(chunks)


def test_upsert_is_idempotent(md_file, retriever):
    doc = parse_document(md_file)
    chunks = split_document(doc)

    retriever.index_document(doc, chunks)
    first = retriever.store.count()
    retriever.index_document(doc, chunks)

    assert retriever.store.count() == first, "重复入库同一文档不应产生重复点"


def test_delete_by_doc(md_file, retriever):
    doc = parse_document(md_file)
    chunks = split_document(doc)
    doc_id = chunks[0].doc_id

    retriever.index_document(doc, chunks)
    assert retriever.store.count() > 0

    retriever.store.delete_by_doc(doc_id)
    assert retriever.store.count() == 0


# ---------- 检索 ----------


def test_search_hits_relevant_section(md_file, retriever):
    doc = parse_document(md_file)
    retriever.index_document(doc, split_document(doc))

    result = retriever.search("年假有几天")

    assert not result.refused
    assert result.hits
    assert any("假期" in h.section_path for h in result.hits[:2])


def test_both_recall_paths_contribute(md_file, retriever):
    doc = parse_document(md_file)
    retriever.index_document(doc, split_document(doc))

    result = retriever.search("入职需要提交什么材料")
    sources = {name for hit in result.hits for name in hit.sources}

    assert "dense" in sources
    assert "sparse" in sources


def test_parent_blocks_never_returned_as_hits(md_file, retriever):
    doc = parse_document(md_file)
    retriever.index_document(doc, split_document(doc))

    result = retriever.search("离职要提前多久")

    assert result.hits
    assert all(h.payload.get("kind") == "child" for h in result.hits)


def test_parent_expansion_attaches_context(md_file, retriever):
    doc = parse_document(md_file)
    retriever.index_document(doc, split_document(doc))

    result = retriever.search("假期天数规定")
    hit = result.hits[0]

    assert hit.payload.get("parent_text")
    assert len(hit.payload["parent_text"]) >= len(hit.text)


def test_refuse_irrelevant_query(md_file, store):
    retriever = make_retriever(store, min_score=0.3)
    doc = parse_document(md_file)
    retriever.index_document(doc, split_document(doc))

    result = retriever.search("kubernetes horizontal pod autoscaler")

    assert result.refused
    assert result.hits == []
    assert "threshold" in result.reason or result.reason == "no_candidates"


def test_debug_carries_stage_counts(md_file, retriever):
    doc = parse_document(md_file)
    retriever.index_document(doc, split_document(doc))

    debug = retriever.search("入职流程").debug

    assert debug["dense_hits"] > 0
    assert debug["sparse_hits"] > 0
    assert debug["fused"] >= debug["candidates"]
    assert debug["reranked"] is True


# ---------- ACL ----------


def test_acl_filter_blocks_unauthorized_docs(tmp_path, store):
    hr_doc = tmp_path / "hr.md"
    hr_doc.write_text(
        "# 薪酬制度\n\n## 一、工资结构\n\n基本工资按月发放，绩效工资按季度考核。\n",
        encoding="utf-8",
    )
    tech_doc = tmp_path / "tech.md"
    tech_doc.write_text(
        "# 技术规范\n\n## 一、代码评审\n\n所有合并请求需两名评审人同意。\n",
        encoding="utf-8",
    )

    retriever = make_retriever(store)

    for path, tags in ((hr_doc, ["hr", "all"]), (tech_doc, ["tech", "all"])):
        doc = parse_document(path, meta={"acl_tags": tags})
        retriever.index_document(doc, split_document(doc))

    result = retriever.search("工资 绩效", flt=acl_filter(["hr"]))

    assert result.hits
    assert all("hr" in hit.payload.get("acl_tags", []) for hit in result.hits)
    assert all("技术规范" not in hit.text for hit in result.hits)


def test_without_filter_both_docs_visible(tmp_path, store):
    retriever = make_retriever(store)
    for name, body, tags in (
        ("a.md", "# 薪酬制度\n\n## 一、工资\n\n基本工资按月发放。\n", ["hr"]),
        ("b.md", "# 技术规范\n\n## 一、评审\n\n工资相关数据不得外传。\n", ["tech"]),
    ):
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        doc = parse_document(path, meta={"acl_tags": tags})
        retriever.index_document(doc, split_document(doc))

    result = retriever.search("工资")
    doc_ids = {hit.doc_id for hit in result.hits}
    assert len(doc_ids) >= 2
