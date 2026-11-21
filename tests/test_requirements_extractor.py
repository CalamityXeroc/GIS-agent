# -*- coding: utf-8 -*-
"""Unit tests for the requirement extractor (mock LLM, no network)."""
from __future__ import annotations

import json

from gis_cli.engine.llm_client import LLMResponse
from gis_cli.requirements.extractor import RequirementExtractor


class MockLLM:
    def __init__(self, content):
        self.content = content
        self.last_task_type = None

    def chat(self, messages, **kwargs):
        self.last_task_type = kwargs.get("task_type")
        return LLMResponse(content=self.content, model="mock")


def test_extract_parses_requirements():
    payload = json.dumps(
        {
            "requirements": [
                {"id": "r1", "text": "投影到4508", "acceptance": "wkid=4508"},
                {"id": "r2", "text": "保留name/code", "acceptance": "字段存在"},
            ]
        },
        ensure_ascii=False,
    )
    llm = MockLLM(payload)
    ex = RequirementExtractor(llm)
    reqs = ex.extract("把county投影到4508并保留属性")
    assert [r.id for r in reqs] == ["r1", "r2"]
    assert reqs[0].acceptance == "wkid=4508"
    assert reqs[0].source == "extracted"
    # extraction uses the requirements routing for the strong model
    assert llm.last_task_type == "requirements"


def test_extract_returns_empty_on_garbage():
    llm = MockLLM("这不是JSON")
    ex = RequirementExtractor(llm)
    assert ex.extract("任意目标") == []


def test_extract_skips_empty_items():
    payload = json.dumps({"requirements": [{"id": "r1", "text": ""}, {"id": "r2", "text": "有效"}]}, ensure_ascii=False)
    reqs = RequirementExtractor(MockLLM(payload)).extract("g")
    assert [r.id for r in reqs] == ["r2"]
