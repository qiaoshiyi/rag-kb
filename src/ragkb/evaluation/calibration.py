"""拒答阈值标定。

阈值不能凭经验拍。做法是拿验证集统计「有答案」和「无答案」两类问题的最高分分布，
再在两者之间取值。三个必须遵守的前提：

1. **阈值绑定具体 reranker**。cross-encoder 的 sigmoid 分与字面重叠分分布完全不同，
   换重排器必须重标。0.3 在 lexical 上是合理的，在 qwen-rerank 上会误杀。
2. **标定集要含硬负样本**。只拿「天气怎么样」这类域外问题测，阈值会偏得离谱；
   真正难的是「报销发票贴哪里」这种和语料同主题但确实没有答案的问题。
3. **误杀与漏放代价不对称**。误杀是直接丢答案，漏放还有生成层兜底，
   所以代价比默认 2:1 偏向避免误杀。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class Calibration:
    threshold: float
    cost_ratio: float
    positives: list[float] = field(default_factory=list)
    negatives: list[float] = field(default_factory=list)
    false_refusal: float = 0.0
    false_accept: float = 0.0
    margin: float = 0.0
    """阈值到最近样本的距离。margin 越大越鲁棒，同样代价下优先选它。"""
    overlap: float = 0.0
    """正样本最低分 − 负样本最高分。>0 表示两类完全可分，<0 是重叠区间宽度。"""
    sweep_threshold: float | None = None
    """未加安全边距时的扫描最优点，保留下来便于对照。"""

    @property
    def errors(self) -> float:
        return self.false_refusal + self.false_accept

    def summary(self) -> str:
        return (
            f"阈值={self.threshold:.3f} 误杀={self.false_refusal:.3f} "
            f"漏放={self.false_accept:.3f} margin={self.margin:.3f} "
            f"重叠={self.overlap:+.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "sweep_threshold": round(self.sweep_threshold, 4) if self.sweep_threshold else None,
            "cost_ratio": self.cost_ratio,
            "false_refusal": round(self.false_refusal, 4),
            "false_accept": round(self.false_accept, 4),
            "margin": round(self.margin, 4),
            "overlap": round(self.overlap, 4),
            "positives": [round(s, 4) for s in sorted(self.positives, reverse=True)],
            "negatives": [round(s, 4) for s in sorted(self.negatives, reverse=True)],
        }


def _rate(values: Sequence[float], predicate) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if predicate(v)) / len(values)


def sweep(
    positives: Sequence[float],
    negatives: Sequence[float],
    cost_ratio: float = 2.0,
) -> Calibration:
    """在候选阈值上扫描，取加权错误最少的那个点。

    候选点取相邻分数的中点：阈值落在两个样本之间才可能改变判定，
    区间内部任意取值等价，取中点是为了让 margin 有意义。
    """
    pos = [float(p) for p in positives]
    neg = [float(n) for n in negatives]

    if not pos and not neg:
        raise ValueError("正负样本都为空，无法标定")

    # 没有负样本时只能取正样本下限：保证不误杀，代价是漏放全靠生成层兜底
    if not neg:
        return Calibration(threshold=min(pos), cost_ratio=cost_ratio, positives=pos, margin=0.0)
    if not pos:
        return Calibration(threshold=max(neg), cost_ratio=cost_ratio, negatives=neg, margin=0.0)

    scores = sorted(set(pos) | set(neg))
    candidates = [scores[0] - 0.01]
    candidates.extend((a + b) / 2 for a, b in zip(scores, scores[1:]))
    candidates.append(scores[-1] + 0.01)

    best: Calibration | None = None
    best_key: tuple[float, float] = (float("inf"), 0.0)
    for t in candidates:
        fr = _rate(pos, lambda v: v < t)
        fa = _rate(neg, lambda v: v >= t)
        cost = cost_ratio * fr + fa
        margin = min(abs(t - s) for s in scores)
        # 同等代价时选 margin 更大（离样本更远）的，鲁棒性更好
        key = (cost, -margin)
        if key < best_key:
            best_key = key
            best = Calibration(
                threshold=t,
                cost_ratio=cost_ratio,
                positives=pos,
                negatives=neg,
                false_refusal=fr,
                false_accept=fa,
                margin=margin,
                overlap=min(pos) - max(neg),
            )

    assert best is not None
    return best


def evaluate_at(threshold: float, positives: Sequence[float], negatives: Sequence[float]) -> tuple[float, float]:
    """给定阈值返回 (误杀率, 漏放率)，用于对比某个候选值。"""
    return (
        _rate([float(p) for p in positives], lambda v: v < threshold),
        _rate([float(n) for n in negatives], lambda v: v >= threshold),
    )


def apply_safety_margin(cal: Calibration, margin: float) -> Calibration:
    """把阈值往「少误杀」的方向挪一点。

    扫描出的最优点常常贴在最低正样本旁边（两类可分但间隔很窄时尤其如此），
    线上换一批文档就会开始误杀。误杀的代价是不对称的：直接丢答案；
    漏放只是多调一次 LLM，prompt 还有一道拒答兜底。
    所以宁可贴住正样本下限再往下留 margin，也不站在间隔正中间。
    """
    if not cal.positives or margin <= 0:
        return cal

    floor = min(cal.positives) - margin
    if cal.threshold <= floor:
        return cal

    fr, fa = evaluate_at(floor, cal.positives, cal.negatives)
    return Calibration(
        threshold=round(floor, 4),
        cost_ratio=cal.cost_ratio,
        positives=cal.positives,
        negatives=cal.negatives,
        false_refusal=fr,
        false_accept=fa,
        margin=cal.margin,
        overlap=cal.overlap,
        sweep_threshold=cal.threshold,
    )


def save(calibration: Calibration, path: str | Path, meta: dict | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = calibration.to_dict()
    if meta:
        payload["meta"] = meta
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
