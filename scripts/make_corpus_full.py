"""生成一套真实企业知识库语料：txt / docx / pdf / md 各三份。

    python scripts/make_corpus_full.py                # 生成到 corpus_full/
    python scripts/make_corpus_full.py --out data/kb  # 指定输出目录
    python scripts/make_corpus_full.py --index        # 顺便解析+切分并打印统计

为什么不用同一份正文转四种格式：那样四份文件会长得一模一样，
既不像真实企业资料，也测不出解析器的差异。这里按各格式的"本土习惯"排版，
同一制度不同章节承载不同密级与细节，便于做检索评测。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from corpus_docs import ALL  # noqa: E402
from corpus_docs.render import render  # noqa: E402


def build(out: Path, clean: bool) -> list[dict]:
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for d in ALL:
        path = render(d, out)
        chars = sum(len(b[1]) for b in d["blocks"] if isinstance(b[1], str))
        chars += sum(len(c) for b in d["blocks"] if b[0] == "table" for r in b[2] for c in r)
        rows.append(
            {
                "format": d["fmt"],
                "file": str(path.relative_to(out)),
                "doc_id": d["meta"]["文档编号"],
                "title": d["title"],
                "classification": d["meta"]["密级"],
                "blocks": len(d["blocks"]),
                "tables": sum(1 for b in d["blocks"] if b[0] == "table"),
                "chars": chars,
                "bytes": path.stat().st_size,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "corpus_full"))
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--index", action="store_true", help="生成后跑一次解析与切分统计")
    args = parser.parse_args()

    out = Path(args.out)
    rows = build(out, args.clean)

    print(f"已生成 {len(rows)} 份文档 → {out}\n")
    header = f"  {'格式':<6}{'文档编号':<12}{'标题':<26}{'密级':<6}{'表':>3}{'字数':>7}{'大小':>9}"
    print(header)
    print("  " + "─" * 74)
    for r in rows:
        print(
            f"  {r['format']:<6}{r['doc_id']:<12}{r['title']:<26}{r['classification']:<6}"
            f"{r['tables']:>3}{r['chars']:>7}{r['bytes'] / 1024:>8.1f}K"
        )
    total = sum(r["chars"] for r in rows)
    print(f"\n  合计 {len(rows)} 份 · {total} 字 · {sum(r['bytes'] for r in rows) / 1024:.0f} KB")

    (out / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.index:
        from ragkb.pipeline.ingest import ingest_file, iter_documents

        print("\n  解析与切分结果：")
        print(f"  {'文件':<46}{'块':>5}{'父':>5}{'子':>5}{'标题':>5}{'表':>4}{'通道':>8}")
        print("  " + "─" * 78)
        totals = {"chunks": 0, "parents": 0, "children": 0}
        failures: list[str] = []
        for path in iter_documents(out):
            try:
                result = ingest_file(path)
            except Exception as exc:  # 解析失败要显式暴露，不静默跳过
                failures.append(f"{path.name}: {exc}")
                print(f"  {path.name[:44]:<46} 解析失败: {type(exc).__name__}")
                continue
            doc = result.document
            titles = sum(1 for b in doc.blocks if b.is_title)
            tables = sum(1 for b in doc.blocks if b.type == "table")
            totals["chunks"] += len(result.chunks)
            totals["parents"] += len(result.parents)
            totals["children"] += len(result.children)
            print(
                f"  {path.name[:44]:<46}{len(result.chunks):>5}{len(result.parents):>5}"
                f"{len(result.children):>5}{titles:>5}{tables:>4}"
                f"{doc.meta.get('channel', doc.source_type):>8}"
            )
        print(
            f"\n  合计 {totals['chunks']} 块（父 {totals['parents']} / 子 {totals['children']}）"
        )
        if failures:
            print(f"  ⚠ 解析失败 {len(failures)} 个：")
            for f in failures:
                print(f"    - {f}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
