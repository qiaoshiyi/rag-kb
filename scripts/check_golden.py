"""金标集体检：把每条用例对语料做块级核验，避免用错的标注去评测。

为什么需要它：
关键词只在校验「整个文件里有没有」是不够的。实测踩过——「断电」在应急预案里
确实出现过（机械伤害那一条），但着火那一条写的是「切断着火区域电源」。
按文件校验能过，跑到块级才发现召回对了块、关键词却不在块里，
指标被算成 miss，看起来像检索的锅。

    python scripts/check_golden.py --docs corpus_full eval/full_golden.jsonl
    python scripts/check_golden.py --docs corpus_full eval/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragkb.chunking import split_document  # noqa: E402
from ragkb.parsing import parse_document  # noqa: E402


def build_index(docs: Path) -> dict[str, list[str]]:
    """文件名 → 该文件所有可检索块的文本（子块用父块文本兜底，与评测口径一致）。"""
    index: dict[str, list[str]] = {}
    for path in sorted(docs.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".docx", ".pdf"}:
            continue
        try:
            doc = parse_document(path)
            chunks = split_document(doc)
        except Exception as exc:  # 解析失败要让调用方看见，不静默
            print(f"  ⚠ 解析失败 {path.name}: {exc}")
            continue
        # 检索命中的是子块，但评测的 span 指标读的是 parent_text 兜底，
        # 所以核验时也把父块文本一起算进来，口径必须和评测一致
        parent_text = {c.chunk_id: c.text for c in chunks if c.kind == "parent"}
        index[path.name] = [
            c.text + "\n" + parent_text.get(c.parent_id or "", "") for c in chunks
        ]
    return index


def check(path: Path, index: dict[str, list[str]]) -> tuple[int, list[str]]:
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    problems: list[str] = []

    for i, item in enumerate(items, 1):
        question = str(item.get("question", ""))[:26]
        expected = item.get("expected_files") or []
        for name in expected:
            if name not in index:
                problems.append(f"#{i} 「{question}」期望文件不在语料中：{name}")

        if item.get("expect_refusal"):
            continue

        for kw in item.get("keywords") or []:
            hit_chunks = [
                name
                for name in expected
                if any(kw in text for text in index.get(name, []))
            ]
            if not hit_chunks:
                problems.append(
                    f"#{i} 「{question}」关键词「{kw}」不在期望文件的任何块里"
                    "（可能是标错了，也可能切分把它切断了）"
                )
    return len(items), problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=str(ROOT / "corpus_full"))
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    docs = Path(args.docs)
    index = build_index(docs)
    total_chunks = sum(len(v) for v in index.values())
    print(f"语料 {docs.name}：{len(index)} 份文档 / {total_chunks} 块\n")

    bad_total = 0
    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            print(f"{path}: 文件不存在")
            bad_total += 1
            continue
        count, problems = check(path, index)
        status = "通过" if not problems else f"{len(problems)} 处需要修正"
        print(f"{path.name:<28} {count:>3} 条  {status}")
        for p in problems:
            print(f"    - {p}")
        bad_total += len(problems)

    print(f"\n合计问题 {bad_total} 处")
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
