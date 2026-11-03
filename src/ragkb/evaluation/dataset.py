from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class GoldenItem:
    question: str
    expected_files: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    note: str = ""

    # ---- 多轮专用字段（单轮金标集不填）----
    history: list[dict[str, str]] = field(default_factory=list)
    standalone: str = ""
    """人工写好的标准独立问法，用来对照改写结果，不参与自动判分。"""
    must_contain: list[str] = field(default_factory=list)
    """改写后必须出现的上下文锚点。这些词只可能来自历史，测的就是「有没有补上上下文」。"""
    expect_reset: bool = False
    """话题跳转用例：期望系统识别出话题已换、清空历史。"""

    # ---- 回答质量评测专用字段 ----
    reference: str = ""
    """参考答案。判分时用要点覆盖度，不要求逐字一致。"""
    expect_refusal: bool = False
    """语料里没有答案的问题：期望系统拒答，答了就是幻觉。"""

    @property
    def is_multi_turn(self) -> bool:
        return bool(self.history)


def load_jsonl(path: str | Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 解析失败：{exc}") from exc
            items.append(
                GoldenItem(
                    question=raw["question"],
                    expected_files=list(raw.get("expected_files", [])),
                    keywords=list(raw.get("keywords", [])),
                    note=raw.get("note", ""),
                    history=list(raw.get("history", [])),
                    standalone=raw.get("standalone", ""),
                    must_contain=list(raw.get("must_contain", [])),
                    expect_reset=bool(raw.get("expect_reset", False)),
                    reference=raw.get("reference", ""),
                    expect_refusal=bool(raw.get("expect_refusal", False)),
                )
            )
    return items
