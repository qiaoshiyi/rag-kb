"""父子块切分。

父块 = 一个标题层级下的完整内容（送进 LLM 的上下文）
子块 = 父块内的小片（进向量库检索的粒度）

这样解决了「块小则上下文断裂、块大则语义被稀释」的矛盾。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ragkb.models.document import Block, Document
from ragkb.tokens import estimate_tokens

ChunkKind = Literal["parent", "child"]

# 按文档类型的默认重叠比例：结构化程度越低，重叠越大
DEFAULT_OVERLAP: dict[str, float] = {"txt": 0.20, "pdf": 0.15, "md": 0.10, "docx": 0.10}


@dataclass(slots=True)
class ChunkConfig:
    parent_tokens: int = 2048
    child_tokens: int = 512
    overlap_ratio: float | None = None
    parent_level: int = 2
    min_child_tokens: int = 80

    def overlap_for(self, source_type: str) -> float:
        if self.overlap_ratio is not None:
            return self.overlap_ratio
        return DEFAULT_OVERLAP.get(source_type, 0.12)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    kind: ChunkKind
    text: str
    section_path: str
    order: int
    parent_id: str | None = None
    level: int | None = None
    page: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    def to_payload(self, doc_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Qdrant payload。ACL 字段现在就写进去，二期加权限不用重建索引。"""
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "kind": self.kind,
            "text": self.text,
            "section_path": self.section_path,
            "order": self.order,
            "parent_id": self.parent_id,
            "level": self.level,
            "page": self.page,
            "tokens": self.tokens,
        }
        if doc_meta:
            payload.update(
                {
                    "title": doc_meta.get("title", ""),
                    "source_type": doc_meta.get("source_type", ""),
                    "filename": doc_meta.get("filename", ""),
                    "dept": doc_meta.get("dept", ""),
                    "acl_tags": doc_meta.get("acl_tags", []),
                    "version": doc_meta.get("version", 1),
                }
            )
        return payload


@dataclass
class _Section:
    path: list[str]
    level: int
    blocks: list[Block] = field(default_factory=list)


def _join(units: list[str]) -> str:
    return "\n\n".join(u for u in units if u.strip())


def _block_units(blocks: list[Block]) -> list[str]:
    return [b.text for b in blocks if b.text.strip()]


def _build_sections(blocks: list[Block], parent_level: int) -> list[_Section]:
    titles = [b.level or 1 for b in blocks if b.is_title]
    if titles:
        # 找文档里实际存在的、最接近目标层级的标题层作为父块边界。
        # 有 H2 就用 H2；只有 H1 和 H3 时用 H3；只有 H1 时退回 H1。
        candidates = [t for t in titles if t >= parent_level]
        parent_level = min(candidates) if candidates else max(titles)

    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(path=[], level=0)

    for block in blocks:
        if block.is_title:
            level = block.level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block.text))

            if level <= parent_level:
                if current.blocks:
                    sections.append(current)
                current = _Section(path=[t for _, t in stack], level=level, blocks=[block])
                continue
        current.blocks.append(block)

    if current.blocks:
        sections.append(current)
    return sections


def _split_units(units: list[str], size: int, overlap: int) -> list[str]:
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    acc = 0

    for unit in units:
        cost = estimate_tokens(unit)
        if current and acc + cost > size:
            chunks.append(_join(current))
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                prev_cost = estimate_tokens(prev)
                if tail_tokens + prev_cost > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += prev_cost
            current = tail
            acc = tail_tokens

        current.append(unit)
        acc += cost

    if current:
        chunks.append(_join(current))
    return [c for c in chunks if c.strip()]


def _merge_short(chunks: list[str], min_tokens: int) -> list[str]:
    if len(chunks) <= 1:
        return chunks
    merged: list[str] = []
    for chunk in chunks:
        if merged and estimate_tokens(chunk) < min_tokens:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)
    return merged


def _group_for_parents(child_texts: list[str], parent_tokens: int) -> list[tuple[int, int]]:
    """父块超长时，按子块归组，返回每个父块覆盖的子块区间 [start, end)。"""
    groups: list[tuple[int, int]] = []
    start = 0
    acc = 0
    for idx, text in enumerate(child_texts):
        cost = estimate_tokens(text)
        if idx > start and acc + cost > parent_tokens:
            groups.append((start, idx))
            start = idx
            acc = 0
        acc += cost
    groups.append((start, len(child_texts)))
    return groups


def split_document(
    doc: Document,
    config: ChunkConfig | None = None,
    doc_id: str | None = None,
) -> list[Chunk]:
    cfg = config or ChunkConfig()
    doc_id = doc_id or doc.sha256[:16]
    overlap = int(cfg.child_tokens * cfg.overlap_for(doc.source_type))

    sections = _build_sections(doc.blocks, cfg.parent_level)
    chunks: list[Chunk] = []
    order = 0

    for section in sections:
        # 只有标题、没有正文的章节（典型是文档大标题）不产出块，
        # 否则会生成大量无检索价值的碎片；标题信息已经保留在子块的 section_path 里
        if len(section.blocks) == 1 and section.blocks[0].is_title:
            continue

        units = _block_units(section.blocks)
        full_text = _join(units)
        if not full_text.strip():
            continue

        path_str = " > ".join(section.path) or doc.title or doc.meta.get("filename", "")
        child_texts = _merge_short(
            _split_units(units, cfg.child_tokens, overlap), cfg.min_child_tokens
        )
        if not child_texts:
            continue

        page = next((b.page for b in section.blocks if b.page is not None), None)

        if estimate_tokens(full_text) <= cfg.parent_tokens:
            parent_id = f"{doc_id}:p{order:04d}"
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    doc_id=doc_id,
                    kind="parent",
                    text=full_text,
                    section_path=path_str,
                    order=order,
                    level=section.level,
                    page=page,
                )
            )
            order += 1
            ranges = [(0, len(child_texts))]
            parent_ids = [parent_id]
        else:
            groups = _group_for_parents(child_texts, cfg.parent_tokens)
            ranges = []
            parent_ids = []
            for start, end in groups:
                pid = f"{doc_id}:p{order:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=pid,
                        doc_id=doc_id,
                        kind="parent",
                        text=_join(child_texts[start:end]),
                        section_path=path_str,
                        order=order,
                        level=section.level,
                        page=page,
                    )
                )
                order += 1
                parent_ids.append(pid)
                ranges.append((start, end))

        for range_idx, (start, end) in enumerate(ranges):
            parent_id = parent_ids[range_idx]
            for idx in range(start, end):
                text = child_texts[idx]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}:c{order:04d}",
                        doc_id=doc_id,
                        kind="child",
                        text=text,
                        section_path=path_str,
                        order=order,
                        parent_id=parent_id,
                        level=section.level,
                        page=page,
                    )
                )
                order += 1

    return chunks


def split_parents_first(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    """(父块, 子块)。父块只入库用于取上下文，不参与向量检索。"""
    return (
        [c for c in chunks if c.kind == "parent"],
        [c for c in chunks if c.kind == "child"],
    )
