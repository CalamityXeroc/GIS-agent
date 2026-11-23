# -*- coding: utf-8 -*-
"""分析正确性双复核（P1-5 / W5）。

两个独立可用的层：

- :mod:`gis_cli.review.assertions` —— 守恒/拓扑/量纲断言库。纯函数，吃
  ``claim``（主代理自报的数字）+ ``facts``（量测事实），不依赖配方、不依赖 LLM；
- :mod:`gis_cli.review.facts` —— 把数据路径量测成 ``facts``（ArcPy 内核 / 纯 Python）。

复核 subagent（``engine.subagent.SubAgentRunner.run_review``）只拿这三块：
**输入数据摘要 + 主代理结论 + 断言库**，看不到主代理的轨迹。开关默认关：
``config/llm_config.json -> engine.review_enabled``。
"""

from __future__ import annotations

from .assertions import ASSERTIONS, Assertion, AssertionSkipped, assertion_catalog, run_assertions
from .claims import (
    SEVERITIES,
    Finding,
    ReviewClaim,
    ReviewReport,
    ReviewVerdict,
    parse_review_payload,
)
from .facts import FactCollectionError, collect_facts, crs_facts_from_epsg, one_dataset

__all__ = [
    "ASSERTIONS",
    "Assertion",
    "AssertionSkipped",
    "FactCollectionError",
    "Finding",
    "ReviewClaim",
    "ReviewReport",
    "ReviewVerdict",
    "SEVERITIES",
    "assertion_catalog",
    "collect_facts",
    "crs_facts_from_epsg",
    "one_dataset",
    "parse_review_payload",
    "run_assertions",
]
