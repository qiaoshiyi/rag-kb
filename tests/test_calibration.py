"""拒答阈值标定的纯逻辑测试（不调模型）。"""

from __future__ import annotations

import json

import pytest

from ragkb.evaluation.calibration import (
    apply_safety_margin,
    evaluate_at,
    load,
    save,
    sweep,
)


def test_clean_separation_lands_between():
    pos = [0.95, 0.90, 0.72]
    neg = [0.69, 0.40, 0.05]
    cal = sweep(pos, neg)
    assert 0.69 < cal.threshold < 0.72
    assert cal.false_refusal == 0.0
    assert cal.false_accept == 0.0
    assert cal.overlap == pytest.approx(0.03, abs=1e-6)


def test_overlap_trades_errors_by_cost_ratio():
    pos = [0.9, 0.7, 0.5]
    neg = [0.8, 0.6, 0.3]

    strict = sweep(pos, neg, cost_ratio=1.0)
    lenient = sweep(pos, neg, cost_ratio=10.0)

    # 误杀代价越高，阈值越应该往下走（宁可漏放）
    assert lenient.threshold <= strict.threshold
    assert lenient.false_refusal <= strict.false_refusal


def test_no_negatives_uses_lowest_positive():
    cal = sweep([0.9, 0.6], [])
    assert cal.threshold == pytest.approx(0.6)
    assert cal.false_refusal == 0.0


def test_no_positives_uses_highest_negative():
    cal = sweep([], [0.3, 0.8])
    assert cal.threshold == pytest.approx(0.8)


def test_both_empty_raises():
    with pytest.raises(ValueError):
        sweep([], [])


def test_safety_margin_shifts_down_and_never_refuses():
    pos = [0.95, 0.712]
    neg = [0.69, 0.30]
    raw = sweep(pos, neg)
    shifted = apply_safety_margin(raw, 0.05)

    assert shifted.threshold < raw.threshold
    assert shifted.threshold == pytest.approx(0.712 - 0.05, abs=1e-3)
    assert shifted.false_refusal == 0.0
    # 一旦越过负样本上界，就必然多放进来一些负样本
    assert shifted.false_accept >= raw.false_accept
    # 扫描点要留下来做对照
    assert shifted.sweep_threshold == pytest.approx(raw.threshold)


def test_safety_margin_keeps_threshold_when_already_low():
    raw = sweep([0.95, 0.90], [0.3])
    # 扫描点（0.6）本来就低得多，不应被抬高
    shifted = apply_safety_margin(raw, 0.05)
    assert shifted.threshold == pytest.approx(raw.threshold)


def test_evaluate_at_reports_both_rates():
    fr, fa = evaluate_at(0.5, [0.9, 0.4], [0.6, 0.2])
    assert fr == pytest.approx(0.5)
    assert fa == pytest.approx(0.5)


def test_save_and_load_roundtrip(tmp_path):
    cal = sweep([0.9, 0.8], [0.3, 0.2])
    path = save(cal, tmp_path / "thresholds.json", meta={"reranker": "dashscope"})

    data = load(path)
    assert data["threshold"] == pytest.approx(cal.threshold, abs=1e-4)
    assert data["meta"]["reranker"] == "dashscope"
    assert data["positives"] == pytest.approx([0.9, 0.8])
    assert json.loads(path.read_text(encoding="utf-8"))["negatives"] == pytest.approx([0.3, 0.2])
