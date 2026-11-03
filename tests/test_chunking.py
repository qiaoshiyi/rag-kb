from __future__ import annotations

from ragkb.chunking import ChunkConfig, split_document, split_parents_first
from ragkb.parsing import parse_document


def test_parent_child_links(md_file):
    doc = parse_document(md_file)
    chunks = split_document(doc)
    parents, children = split_parents_first(chunks)

    assert parents and children
    parent_ids = {c.chunk_id for c in parents}
    for child in children:
        assert child.parent_id in parent_ids, "每个子块都必须挂到一个存在的父块上"


def test_section_path_reflects_headings(md_file):
    doc = parse_document(md_file)
    chunks = split_document(doc)
    paths = {c.section_path for c in chunks}
    assert any("员工手册" in p and "一、入职" in p for p in paths)


def test_order_is_monotonic(md_file):
    doc = parse_document(md_file)
    chunks = split_document(doc)
    orders = [c.order for c in chunks]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)


def test_overlap_defaults_by_type():
    cfg = ChunkConfig()
    assert cfg.overlap_for("txt") == 0.20
    assert cfg.overlap_for("pdf") == 0.15
    assert cfg.overlap_for("md") == 0.10
    assert cfg.overlap_for("unknown") == 0.12


def test_overlap_can_be_forced():
    cfg = ChunkConfig(overlap_ratio=0.3)
    assert cfg.overlap_for("md") == 0.3


def test_child_chunks_respect_size(md_file):
    doc = parse_document(md_file)
    cfg = ChunkConfig(child_tokens=40, min_child_tokens=1)
    chunks = split_document(doc, cfg)
    children = [c for c in chunks if c.kind == "child"]
    # 允许单个段落超出，但不能成倍超出
    assert all(c.tokens <= 40 * 3 for c in children)


def test_long_section_splits_into_multiple_parents(tmp_path):
    body = "\n\n".join(
        f"第 {i} 条：这是一段用于验证切分行为的长文本，包含若干说明性内容与举例。" for i in range(60)
    )
    path = tmp_path / "长文档.md"
    path.write_text(f"# 长文档\n\n## 一、大章节\n\n{body}", encoding="utf-8")

    doc = parse_document(path)
    cfg = ChunkConfig(parent_tokens=300, child_tokens=120, min_child_tokens=20)
    chunks = split_document(doc, cfg)

    parents = [c for c in chunks if c.kind == "parent"]
    children = [c for c in chunks if c.kind == "child"]
    assert len(parents) > 1, "超长章节必须拆成多个父块，否则会挤爆 prompt"
    assert len(children) > len(parents)
    assert sum(1 for c in children) > 0


def test_document_without_headings_falls_back(tmp_path):
    path = tmp_path / "无标题.txt"
    path.write_text("这是一段没有标题的正文内容。\n\n这是第二段内容。", encoding="utf-8")
    doc = parse_document(path)
    chunks = split_document(doc)
    assert any(c.kind == "parent" for c in chunks)
    assert any(c.kind == "child" for c in chunks)


def test_short_sections_merge_into_parent(tmp_path):
    """低于 min_child_tokens 的碎片应并入前一块，避免产生大量无效检索单元。"""
    path = tmp_path / "碎片.md"
    path.write_text(
        "# 制度\n\n## 一、总则\n\n第一条 内容。\n\n第二条 内容。\n\n第三条 内容。\n",
        encoding="utf-8",
    )
    doc = parse_document(path)
    cfg = ChunkConfig(child_tokens=200, min_child_tokens=60)
    children = [c for c in split_document(doc, cfg) if c.kind == "child"]
    assert len(children) == 1


def test_payload_carries_acl_fields(md_file):
    doc = parse_document(md_file, meta={"dept": "hr", "acl_tags": ["hr", "all"]})
    chunk = split_document(doc)[0]
    payload = chunk.to_payload({**doc.meta, "source_type": doc.source_type})

    assert payload["acl_tags"] == ["hr", "all"]
    assert payload["dept"] == "hr"
    assert payload["source_type"] == "md"
    assert payload["tokens"] > 0


def test_chunk_ids_unique(md_file):
    doc = parse_document(md_file)
    chunks = split_document(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
