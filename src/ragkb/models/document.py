"""统一文档中间表示（IR）。

四个解析适配器（txt / md / pdf / docx）都产出这个结构，
下游切分、索引、检索只看 IR，不感知原始格式。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

BlockType = Literal["title", "para", "table", "list", "code", "image"]


@dataclass(slots=True)
class Block:
    """IR 的最小单元。"""

    type: BlockType
    text: str
    level: int | None = None
    """标题层级，1 = 一级标题。仅 type == "title" 时有意义。"""
    page: int | None = None
    """页码，从 1 开始。仅 pdf 能提供。"""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 全角空格会干扰检索匹配，统一半角；块级已切开，strip 不丢结构
        self.text = self.text.replace("\u3000", " ").strip()

    @property
    def is_title(self) -> bool:
        return self.type == "title"

    def __bool__(self) -> bool:
        return bool(self.text)


@dataclass(slots=True)
class Document:
    source_path: str
    source_type: str
    """txt / md / pdf / docx"""
    blocks: list[Block] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 丢弃空块，避免后面到处判空
        self.blocks = [b for b in self.blocks if b]

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or "")

    @property
    def sha256(self) -> str:
        """内容指纹，用于增量更新判定。不包含路径，移动文件不会触发重建。"""
        h = hashlib.sha256()
        for b in self.blocks:
            h.update(b.type.encode("utf-8"))
            h.update(b.text.encode("utf-8"))
        return h.hexdigest()

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    def iter_blocks(self, *types: BlockType) -> Iterator[Block]:
        wanted = set(types) if types else None
        for b in self.blocks:
            if wanted is None or b.type in wanted:
                yield b

    def to_markdown(self) -> str:
        """还原成 Markdown 供人工核验解析质量，也用于 badcase 归因对比。"""
        out: list[str] = []
        for b in self.blocks:
            if b.type == "title":
                out.append(f"{'#' * min(b.level or 1, 6)} {b.text}")
            elif b.type == "code":
                lang = b.meta.get("lang", "")
                out.append(f"```{lang}\n{b.text}\n```")
            elif b.type == "table":
                out.append(b.text)
            elif b.type == "list":
                out.append(f"- {b.text}")
            else:
                out.append(b.text)
        return "\n\n".join(out)

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for b in self.blocks:
            counts[b.type] = counts.get(b.type, 0) + 1
        return {
            "source_type": self.source_type,
            "blocks": len(self.blocks),
            "by_type": counts,
            "chars": self.char_count,
            "titles": counts.get("title", 0),
        }
