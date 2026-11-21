# -*- coding: utf-8 -*-
"""蒸馏的 JSON 解析失败必须重试一次 —— 不是只在被截断时重试。

背景：2026-10-22 端到端验证时，`gis-agent distill` 首跑返回
``finish_reason=stop`` 但内容不是合法 JSON，直接被判「模型没有产出可解析的
配方」而放弃；同一份输入再跑一次就成功了。说明重试条件漏了「能正常结束但
不是 JSON」这一档。
"""

from __future__ import annotations

import json

from gis_cli.engine.llm_client import LLMResponse
from gis_cli.recipes.distiller import RecipeDistiller

RECIPE = {
    "id": "csv_group_sum",
    "name": "CSV 分组汇总",
    "description": "按字段分组汇总",
    "domain": "data_management",
    "triggers": ["分组汇总"],
    "params": {"input_csv": {"type": "string", "required": True, "default": "input/a.csv"}},
    "steps": ["读 CSV"],
    "code_template": "import csv\nset_result({'output': {{input_csv}}})\n",
    "validation": [{"type": "file_exists", "path": "{{input_csv}}", "value": True}],
}


class _StubLLM:
    def __init__(self, *replies: LLMResponse):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs) -> LLMResponse:
        self.calls.append(messages)
        return self.replies.pop(0)


def _distill(llm: _StubLLM):
    return RecipeDistiller(llm, library=None).distill(
        goal="按地类汇总面积",
        methodology="csv",
        final_summary="完成",
        code_snippets=["print('hi')"],
        artifacts=["output/summary.md"],
    )


def test_retries_once_when_output_is_not_json():
    llm = _StubLLM(
        LLMResponse(content="我先分析一下这段代码……", finish_reason="stop"),
        LLMResponse(content=json.dumps(RECIPE, ensure_ascii=False), finish_reason="stop"),
    )
    recipe = _distill(llm)
    assert recipe is not None and recipe.id == "csv_group_sum"
    assert len(llm.calls) == 2
    # 重试要把「不是合法 JSON」的约束带回去
    assert "不是合法 JSON" in llm.calls[1][1]["content"]


def test_retries_once_when_output_is_truncated():
    llm = _StubLLM(
        LLMResponse(content='{"id": "csv_group_sum", "code_templ', finish_reason="length"),
        LLMResponse(content=json.dumps(RECIPE, ensure_ascii=False), finish_reason="stop"),
    )
    recipe = _distill(llm)
    assert recipe is not None
    assert len(llm.calls) == 2
    assert "token 上限" in llm.calls[1][1]["content"]


def test_gives_up_after_one_retry():
    llm = _StubLLM(
        LLMResponse(content="不是 JSON", finish_reason="stop"),
        LLMResponse(content="还是不是", finish_reason="stop"),
    )
    assert _distill(llm) is None
    assert len(llm.calls) == 2
