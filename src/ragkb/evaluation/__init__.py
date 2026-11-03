"""检索质量评测与拒答阈值标定。

指标只用不需要 LLM 判分的那几个：期望文档是否被召回、排在第几、答案关键词是否落在命中块里。
这样评测可以在离线反复跑，作为每次调参的对照组。
"""

from ragkb.evaluation.answer_eval import (
    AnswerItemResult,
    AnswerReport,
    LLMJudge,
    JudgeVerdict,
    evaluate_answers,
    parse_verdict,
)
from ragkb.evaluation.calibration import Calibration, evaluate_at, load, save, sweep
from ragkb.evaluation.dataset import GoldenItem, load_jsonl
from ragkb.evaluation.retrieval_eval import EvalReport, evaluate_retrieval

__all__ = [
    "GoldenItem",
    "load_jsonl",
    "EvalReport",
    "evaluate_retrieval",
    "Calibration",
    "sweep",
    "evaluate_at",
    "save",
    "load",
    "AnswerReport",
    "AnswerItemResult",
    "LLMJudge",
    "JudgeVerdict",
    "evaluate_answers",
    "parse_verdict",
]
