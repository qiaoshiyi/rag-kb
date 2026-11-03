"""关键词归一化匹配的测试：数字写法与空白差异不应影响判定。"""

from __future__ import annotations

import pytest

from ragkb.evaluation.text_match import contains, contains_all, cn_to_int, normalize


@pytest.mark.parametrize(
    "text,needle",
    [
        ("应在三十分钟内报告", "30 分钟"),
        ("15 分钟内首次响应", "十五分钟"),
        ("产假为一百五十八天", "158天"),
        ("每九十天更换一次", "90 天"),
        ("扣减当月全勤奖五十元", "50 元"),
        ("单次不超过 2000 元", "两千元"),
        ("次年 1 月 31 日前发放", "1月31日"),
    ],
)
def test_digit_spelling_does_not_matter(text, needle):
    assert contains(text, needle)


def test_wan_is_not_expanded_to_avoid_asymmetry():
    # 五十万 → 50万（而不是 500000），才能和「50 万元」对上
    assert normalize("五十万元") == "50万元"
    assert contains("单笔 50 万元（含）以上", "五十万元")


def test_common_words_with_digit_chars_are_left_alone():
    assert normalize("一致同意") == "一致同意"
    assert normalize("十分重要") == "十分重要"


def test_missing_keyword_really_reports_false():
    assert not contains("资料中未找到相关内容", "30 分钟")
    assert not contains_all("只有八小时", ["八小时", "30 分钟"])


def test_cn_to_int_basics():
    assert cn_to_int("十") == 10
    assert cn_to_int("十五") == 15
    assert cn_to_int("三十") == 30
    assert cn_to_int("一百五十八") == 158
    assert cn_to_int("两千") == 2000
    assert cn_to_int("一致") is None
