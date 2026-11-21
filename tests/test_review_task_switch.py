# -*- coding: utf-8 -*-
"""基准任务怎么开双复核（W5）：顶层 ``review: true`` 或 ``requirements`` 里任一条带 ``review: true``。

计划原文是「开关进 requirements，默认关」——所以交付级任务能在 requirements 里声明，
不必为每个任务另加顶层键；两处都没写就是不覆盖 ``engine.review_enabled``（默认关）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from runner import review_switch  # noqa: E402


def test_review_switch_defaults_to_none():
    assert review_switch({}) is None
    assert review_switch({"requirements": [{"id": "r1", "text": "投影到 4508"}]}) is None


def test_review_switch_top_level_wins():
    assert review_switch({"review": True}) is True
    assert review_switch({"review": False, "requirements": [{"review": True}]}) is False


def test_review_switch_from_requirements_entry():
    """交付级要求在 requirements 里声明：只对某条要求开复核。"""
    task = {
        "requirements": [
            {"id": "r1", "text": "出图"},
            {"id": "r2", "text": "交付级成果", "review": True},
        ]
    }
    assert review_switch(task) is True
    assert review_switch({"requirements": [{"id": "r1", "review": False}]}) is False
