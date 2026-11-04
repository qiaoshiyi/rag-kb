"""一条命令起服务：语料目录和标定阈值一起配好。

    export DASHSCOPE_API_KEY=sk-xxx
    python scripts/serve.py                      # 默认 corpus_full，阈值取 eval/thresholds_full.json
    python scripts/serve.py --docs corpus        # 换成 16 篇那套，阈值自动跟着换
    python scripts/serve.py --offline            # 无 key 跑通界面（hashing + lexical + 替身模型）

为什么要有这个脚本：`python -m ragkb.api.app` 的 KB_DOCS / KB_MIN_SCORE 是两个独立变量，
只改目录忘了改阈值，就会拿 A 语料标出来的 0.771 去卡 B 语料的检索分——
表现是"打开页面全是拒答"，但日志里一句报错都没有。这里把两者绑在一起。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 语料目录 → 对应的标定文件。阈值不能跨语料复用，所以这里必须一对一。
THRESHOLD_FILES = {
    "corpus_full": "thresholds_full.json",
    "corpus": "thresholds_corpus.json",
    "demo": "thresholds.json",
}
DEFAULT_MODEL = "deepseek-v4.1-flash"


def pick_docs(name: str | None) -> Path:
    if name:
        path = (ROOT / name).resolve()
        if not path.is_dir():
            choices = [p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith((".", "src"))]
            raise SystemExit(f"语料目录不存在：{path}\n可选：{', '.join(sorted(choices))}")
        return path
    for fallback in ("corpus_full", "corpus", "demo"):
        path = ROOT / fallback
        if path.is_dir():
            return path
    raise SystemExit(f"{ROOT} 下没有找到语料目录（corpus_full / corpus / demo）")


def load_threshold(docs: Path) -> tuple[float | None, str | None]:
    fname = THRESHOLD_FILES.get(docs.name)
    if not fname:
        return None, None
    path = ROOT / "eval" / fname
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("threshold")
    return (float(value) if value is not None else None), f"eval/{fname}"


def count_docs(docs: Path) -> int:
    from ragkb.pipeline.ingest import iter_documents

    return sum(1 for _ in iter_documents(docs))


def cloud_key() -> str | None:
    """返回第一个存在的云端 key 名。三家任一即可，不再绑死百炼。"""
    for name in ("DASHSCOPE_API_KEY", "SILICONFLOW_API_KEY", "KB_EMBED_API_KEY"):
        if os.environ.get(name, "").strip():
            return name
    return None


def load_threshold_file(path: str) -> tuple[float, str]:
    """显式指定标定文件。部署时阈值跟着索引走，不跟目录名走。"""
    target = Path(path)
    if not target.is_absolute():
        target = (ROOT / target).resolve()
    if not target.exists():
        raise SystemExit(f"阈值文件不存在：{target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    value = payload.get("threshold")
    if value is None:
        raise SystemExit(f"{target} 里没有 threshold 字段")
    return float(value), str(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 rag-kb 服务（自动匹配语料与阈值）")
    parser.add_argument("--docs", help="语料目录名，默认 corpus_full")
    parser.add_argument("--port", type=int, default=int(os.environ.get("KB_PORT", "8000")))
    parser.add_argument("--host", default=os.environ.get("KB_HOST", "127.0.0.1"))
    parser.add_argument("--threshold", type=float, help="手工指定拒答阈值，覆盖标定文件")
    parser.add_argument("--model", help=f"作答模型，默认 {DEFAULT_MODEL}（只有硅基流动 key 时为 Qwen/Qwen3-8B）")
    parser.add_argument("--offline", action="store_true", help="不调 API：hashing + lexical + 替身模型")
    parser.add_argument("--no-examples", action="store_true", help="启动时不探测示例问题（省几次 API 调用）")
    parser.add_argument("--seed", help="加载预建索引目录，跳过启动时的解析与向量化（部署用）")
    parser.add_argument("--threshold-file", help="显式指定标定文件（部署用，优先于按目录名匹配）")
    args = parser.parse_args()

    docs = pick_docs(args.docs)
    threshold, source = load_threshold(docs)
    if args.threshold_file:
        threshold, source = load_threshold_file(args.threshold_file)
    if args.threshold is not None:
        threshold, source = args.threshold, "命令行 --threshold"

    # 只有硅基流动的 key 时，默认走它的重排与作答，不用每次手写四个环境变量
    siliconflow_only = bool(os.environ.get("SILICONFLOW_API_KEY")) and not os.environ.get(
        "DASHSCOPE_API_KEY"
    )
    model_name = args.model or os.environ.get("KB_MODEL")
    if not model_name:
        model_name = "Qwen/Qwen3-8B" if siliconflow_only else DEFAULT_MODEL

    if not args.offline and not cloud_key():
        raise SystemExit(
            "缺少云端 API key（三家任一）：DASHSCOPE_API_KEY / SILICONFLOW_API_KEY / KB_EMBED_API_KEY。\n"
            "  Windows:  set DASHSCOPE_API_KEY=sk-xxx\n"
            "  bash:     export DASHSCOPE_API_KEY=sk-xxx\n"
            "或者加 --offline 先用替身模型把界面跑起来。"
        )

    if args.offline:
        os.environ.update(KB_EMBEDDER="hashing", KB_RERANKER="lexical", KB_LLM="mock", KB_REWRITER="heuristic")
        threshold = None  # 替身的分数分布没有意义，别拿真阈值去卡
        source = None
    else:
        os.environ.setdefault("KB_EMBEDDER", "api")
        os.environ.setdefault("KB_RERANKER", "siliconflow" if siliconflow_only else "dashscope")
        os.environ.setdefault("KB_LLM", "siliconflow" if siliconflow_only else "dashscope")
        os.environ.setdefault("KB_REWRITER", "llm")
        os.environ["KB_MODEL"] = model_name

    os.environ["KB_DOCS"] = str(docs)
    seed_dir: Path | None = None
    if args.seed:
        seed_dir = Path(args.seed)
        if not seed_dir.is_absolute():
            seed_dir = (ROOT / seed_dir).resolve()
        if not seed_dir.exists():
            raise SystemExit(
                f"预建索引目录不存在：{seed_dir}\n"
                "先生成：python scripts/build_seed_index.py"
            )
        # 维度必须对得上，否则检索会在运行时才炸，报错还很难懂
        meta_file = seed_dir / "seed_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            want = os.environ.get("KB_EMBEDDER", "auto")
            if meta.get("backend") and meta["backend"] != want:
                raise SystemExit(
                    f"预建索引是用 {meta['backend']}（dim={meta.get('dim')}）建的，"
                    f"当前 KB_EMBEDDER={want}，维度对不上。\n"
                    f"要么加 --offline（用自己的 hashing），要么重新建索引："
                    f"python scripts/build_seed_index.py --backend {want}"
                )
        os.environ["KB_QDRANT_PATH"] = str(seed_dir)
    if threshold is None:
        os.environ.pop("KB_MIN_SCORE", None)
    else:
        os.environ["KB_MIN_SCORE"] = f"{threshold:.4f}"
    if args.no_examples:
        os.environ["KB_EXAMPLES"] = "off"
    os.environ["KB_BANNER"] = "1"

    print("rag-kb 知识库问答", flush=True)
    print(f"  语料    {docs.name}（{count_docs(docs)} 篇）", flush=True)
    if seed_dir:
        try:
            shown = seed_dir.relative_to(ROOT)
        except ValueError:
            shown = seed_dir
        print(f"  索引    预建 {shown}（跳过解析与向量化）", flush=True)
    print(
        f"  阈值    {threshold if threshold is not None else '关闭'}"
        + (f"  ← {source}" if source else ""),
        flush=True,
    )
    if args.offline:
        print("  模式    offline（hashing 向量 + 词面重排 + 替身作答，只看链路通不通）", flush=True)
    else:
        print(
            f"  模型    embedding={os.environ.get('KB_EMBEDDER')}"
            f"（{os.environ.get('KB_EMBED_MODEL', '默认')}）"
            f"  重排={os.environ.get('KB_RERANKER')}"
            f"  作答={os.environ.get('KB_LLM')}/{os.environ.get('KB_MODEL')}",
            flush=True,
        )

    print(f"  地址    http://{args.host}:{args.port}", flush=True)
    print(f"  前端    http://{args.host}:{args.port}/  或直接双击 web/index.html", flush=True)

    import uvicorn

    uvicorn.run("ragkb.api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
