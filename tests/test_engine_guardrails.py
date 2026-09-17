# -*- coding: utf-8 -*-
"""Tests for loop speed guardrails: exploration budget + failure escalation."""
from __future__ import annotations

import json
from pathlib import Path

from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.trace.recorder import RunRecorder


class ScriptedLLM:
    class _Cfg:
        model = "mock"

    config = _Cfg()

    def __init__(self, responses):
        self.responses = list(responses)
        self.task_type_history: list[str] = []

    def chat(self, messages, **kwargs):
        # surface the requested task_type so tests can assert escalation
        self.last_task_type = kwargs.get("task_type")
        self.task_type_history.append(self.last_task_type)
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(
            content=json.dumps({"tool": "finish", "args": {"summary": "done"}}), model="mock"
        )


class StubVerifier:
    def verify(self, state):
        return {"pass": True, "summary": "ok"}


def _tc(name, args=None):
    args = args or {}
    return LLMResponse(
        tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=args, raw_arguments=json.dumps(args))],
        model="mock",
    )


def _make(tmp_path: Path, responses):
    llm = ScriptedLLM(responses)
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=llm,
        registry=build_default_registry(),
        context=EngineContext(workspace=tmp_path),
        verifier=StubVerifier(),
        recorder=RunRecorder("run_guard", tmp_path / "traces"),
        config=LoopConfig(max_turns=12, max_diagnosis_turns=2, escalate_after_failures=2),
        auto_extract_requirements=False,
    )
    return loop, llm


def test_diagnosis_budget_triggers_nudge(tmp_path: Path):
    responses = [
        _tc("catalog_query", {"query": "a"}),
        _tc("read_document", {"path": "input/doc.txt"}),
        _tc("list_recipes", {}),
        _tc("update_tasklist", {"items": [{"id": "t1", "title": "干活", "status": "running"}]}),
        _tc("finish", {"summary": "完成"}),
    ]
    loop, llm = _make(tmp_path, responses)
    state = loop.run("测试")
    # 3 consecutive diagnosis turns > budget 2 -> nudge must have been injected
    nudges = [m for m in loop._messages if m["role"] == "user" and "侦查已连续" in str(m.get("content"))]
    assert nudges, "expected exploration-budget nudge"
    assert state.status == "completed"


def test_escalation_after_consecutive_failures(tmp_path: Path):
    responses = [
        # two consecutive failed turns -> escalate to agent_strong
        _tc("catalog_query", {"query": "x"}),
        _tc("catalog_query", {"query": "x"}),
        _tc("update_tasklist", {"items": [{"id": "t1", "title": "恢复", "status": "running"}]}),
        _tc("finish", {"summary": "完成"}),
    ]
    # make catalog_query fail: no catalog in context -> tool returns ok? build_default_registry
    # catalog_query returns ok=False when ctx.catalog is None.
    loop, llm = _make(tmp_path, responses)
    loop.run("测试")
    # turn1/turn2 failures (agent), turn3 must be escalated to agent_strong,
    # then success resets back to agent for the finish turn.
    assert llm.task_type_history[:3] == ["agent", "agent", "agent_strong"]
    assert llm.task_type_history[-1] == "agent"


# ---------------------------------------------------------------- recipe hint
class _FakeRecipe:
    def __init__(self, rid, description="", triggers=()):
        self.id = rid
        self.name = rid
        self.description = description
        self.triggers = list(triggers)

    def match_score(self, query: str) -> int:
        text = (query or "").strip().lower()
        if not text:
            return 1
        tokens = [t for t in __import__("re").split(r"[\s,，。;；/]+", text) if t]
        score = 0
        for token in tokens:
            if token in self.id.lower() or token in self.name.lower():
                score += 6
            if token in self.description.lower():
                score += 3
            for trigger in self.triggers:
                tl = trigger.lower()
                if token and (token in tl or tl in token):
                    score += 4
        return score


class _FakeRecipeStore:
    def __init__(self, recipes):
        self._recipes = {r.id: r for r in recipes}

    def get(self, rid):
        return self._recipes.get(rid)

    def all(self):
        return list(self._recipes.values())


def _hint_ctx(recipes):
    from types import SimpleNamespace

    return SimpleNamespace(recipes=recipes)


def test_recipe_hint_by_code_marker():
    from gis_cli.engine.tools import _recipe_hint

    ctx = _hint_ctx(_FakeRecipeStore([_FakeRecipe("graduated_colors_map")]))
    hint = _recipe_hint(ctx, "制作高亮地图", "p = arcpy.mp.ArcGISProject(...)\nlyt.exportToJPEG(out)")
    assert "graduated_colors_map" in hint and "run_recipe" in hint
    # marker absent -> no hint
    assert _recipe_hint(ctx, "随便算算", "x = 1 + 1") == ""


def test_recipe_hint_by_description_score():
    from gis_cli.engine.tools import _recipe_hint

    store = _FakeRecipeStore(
        [_FakeRecipe("空间连接", description="空间连接汇总统计", triggers=[])]
    )
    ctx = _hint_ctx(store)
    hint = _recipe_hint(ctx, "空间连接", "")
    assert "空间连接" in hint and "run_recipe" in hint
    # weak match below threshold -> no hint
    assert _recipe_hint(ctx, "检查一下", "") == ""


def test_recipe_hint_no_recipes():
    from gis_cli.engine.tools import _recipe_hint

    assert _recipe_hint(_hint_ctx(None), "制作高亮地图", "exportToJPEG") == ""
    assert _recipe_hint(_hint_ctx(None), "", "x=1") == ""
