"""入库编排：解析 → 切分。

M1 只到切分为止，向量化与写入 Qdrant 在下一步接。

命令行调试：
    python -m ragkb.pipeline.ingest <文件或目录>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ragkb.chunking.hierarchy import Chunk, ChunkConfig, split_document
from ragkb.models.document import Document
from ragkb.parsing import ParseError, parse_document, supported_suffixes


@dataclass
class IngestResult:
    document: Document
    chunks: list[Chunk]

    @property
    def parents(self) -> list[Chunk]:
        return [c for c in self.chunks if c.kind == "parent"]

    @property
    def children(self) -> list[Chunk]:
        return [c for c in self.chunks if c.kind == "child"]


def ingest_file(
    path: str | Path,
    meta: dict | None = None,
    config: ChunkConfig | None = None,
) -> IngestResult:
    doc = parse_document(path, meta)
    chunks = split_document(doc, config)
    return IngestResult(document=doc, chunks=chunks)


def iter_documents(root: str | Path):
    root_path = Path(root)
    if root_path.is_file():
        yield root_path
        return
    for p in sorted(root_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in supported_suffixes():
            yield p


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python -m ragkb.pipeline.ingest <文件或目录>")
        return 1

    target = Path(sys.argv[1])
    ok = failed = 0

    for path in iter_documents(target):
        try:
            result = ingest_file(path)
        except ParseError as exc:
            print(f"[FAIL] {path.name}: {exc}")
            failed += 1
            continue

        doc = result.document
        print(f"\n=== {path.name} ===")
        print(f"  类型/块数 : {doc.stats()}")
        extra = {k: v for k, v in doc.meta.items() if k != "size_bytes"}
        print(f"  元数据     : {extra}")
        print(f"  父块 {len(result.parents)} / 子块 {len(result.children)}")
        for chunk in result.children[:3]:
            preview = chunk.text.replace("\n", " ")[:60]
            print(f"    · [{chunk.tokens:>4} tok] {chunk.section_path[:24] or '(无标题)'} | {preview}")
        ok += 1

    print(f"\n完成：{ok} 成功，{failed} 失败")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
