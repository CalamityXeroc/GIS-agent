# -*- coding: utf-8 -*-
"""复核 subagent：盲上下文、工具收窄、JSON 判定解析、限额与工具接线。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from gis_cli.engine import subagent as subagent_mod
from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.state import TaskState
from gis_cli.engine.subagent import (
    REVIEW_PREAMBLE,
    REVIEW_TOOLS,
    SubAgentRunner,
    build_review_task,
)
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.review import Finding, ReviewClaim, ReviewVerdict
from gis_cli.trace.recorder import RunRecorder


class CaptureLLM:
    class _Cfg:
        model = "mock"

    config = _Cfg()

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": kwargs.get("tools")})
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content=json.dumps({"tool": "finish", "args": {"summary": "done"}}), model="mock")

    def usage_summary(self):
        return {"prompt_tokens": 100 * len(self.calls), "completion_tokens": 10 * len(self.calls)}


class StubVerifier:
    def verify(self, state):
        return {"pass": True, "summary": "验收通过"}


def _tc(name, args):
    return LLMResponse(
        tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=args, raw_arguments=json.dumps(args))],
        model="mock",
    )


def _parent(tmp_path: Path, responses, *, config: LoopConfig | None = None):
    llm = CaptureLLM(responses)
    context = EngineContext(workspace=tmp_path, llm=llm)
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=llm,
        registry=build_default_registry(),
        context=context,
        code_runner=None,
        verifier=StubVerifier(),
        recorder=RunRecorder("parent_run", tmp_path / ".gis_agent" / "traces"),
        config=config or LoopConfig(max_turns=10, report_enabled=False),
        auto_extract_requirements=False,
    )
    return loop, llm


def _claim():
    return ReviewClaim(
        goal="按城市汇总人口",
        summary="4 个城市合计 842 万人",
        data_paths=["output/zones.shp"],
        metrics={"total_pop": 8_420_000, "zone_pop": {"Z01": 2_100_000, "Z02": 6_320_000}},
        units={"area": "km2"},
        assertions=[
            {
                "type": "conservation.sum_equals",
                "total": "total_pop",
                "parts": "zone_pop",
            }
        ],
    )


def _facts():
    return {
        "datasets": {
            "output/zones.shp": {
                "exists": True,
                "kind": "vector",
                "collector": "arcpy",
                "count": 4,
                "area_total": 152_340_000.0,
                "area_unit": "m2",
                "group_sums": {"POP": {"Z01": 2_100_000.0, "Z02": 6_320_000.0}},
            }
        },
        "paths": ["output/zones.shp"],
        "collector": ["arcpy"],
    }


def _verdict_payload(**extra):
    payload = {
        "verdict": "pass",
        "findings": [],
        "checked": ["conservation.sum_equals"],
        "notes": "总量与分区合计一致",
    }
    payload.update(extra)
    return payload


def test_review_runs_blind_and_parses_verdict(tmp_path: Path):
    claim = _claim()
    facts = _facts()
    loop, llm = _parent(
        tmp_path,
        [
            _tc(
                "assertion_check",
                {
                    "assertions": claim.assertions,
                    "claim": claim.to_dict(),
                    "facts": facts,
                },
            ),
            _tc("finish", {"summary": json.dumps(_verdict_payload(), ensure_ascii=False)}),
        ],
        config=LoopConfig(max_turns=10, review_enabled=True, report_enabled=False),
    )
    # 主轨迹的“过程”标记：复核 prompt 里绝不能出现
    loop._messages = [{"role": "assistant", "content": "SECRET_STEP：先用 buffer 再相交"}]
    loop.state = TaskState(run_id="parent_run", goal="父任务", workspace=str(tmp_path))
    loop.state.methodology = "SECRET_METHOD：先投影再算"

    runner = SubAgentRunner(loop, review_max_turns=8)
    verdict = runner.run_review(claim=claim, facts=facts)

    assert verdict.status == "pass" and verdict.ok
    assert verdict.subtask_id == "rv1"
    assert verdict.checked == ["conservation.sum_equals"]
    assert verdict.trace.endswith("parent_run__rv1.jsonl")
    assert verdict.turns >= 1

    blob = json.dumps(llm.calls, ensure_ascii=False)
    assert "SECRET_STEP" not in blob and "SECRET_METHOD" not in blob
    # 只给“目标 + 结论 + 数据摘要 + 断言库”
    assert "按城市汇总人口" in blob
    assert "8,420,000" in blob or "8420000" in blob
    assert "conservation.sum_equals" in blob

    tool_names = {item["function"]["name"] for item in llm.calls[0]["tools"]}
    assert tool_names == set(REVIEW_TOOLS)
    assert "delegate" not in tool_names and "review_result" not in tool_names
    assert "run_recipe" not in tool_names and "create_map" not in tool_names


def test_review_prompt_is_composed_only_from_claim_and_facts(tmp_path: Path):
    """复核 prompt 只有目标/结论/事实/断言库，不带父的 requirement、任务清单或 trace。"""
    claim = _claim()
    prompt = build_review_task(claim, _facts())
    assert "SECRET_PARENT_REQUIREMENT" not in prompt
    assert "SECRET_TASKLIST" not in prompt
    assert "output/zones.shp" in prompt
    assert "断言库" in prompt and "REVIEW_PREAMBLE" not in prompt


def test_review_verdict_fail_and_unparsed(tmp_path: Path):
    claim = _claim()
    fail = {
        "verdict": "fail",
        "findings": [
            {
                "code": "conservation.partition_mismatch",
                "severity": "error",
                "detail": "Z02 声明 632 万，数据量测 230 万",
                "claimed": 6_320_000,
                "measured": 2_300_000,
            }
        ],
        "checked": ["conservation.sum_equals"],
        "notes": "",
    }
    loop, _ = _parent(
        tmp_path,
        [_tc("finish", {"summary": json.dumps(fail, ensure_ascii=False)})],
    )
    runner = SubAgentRunner(loop)
    verdict = runner.run_review(claim=claim, facts=_facts())
    assert verdict.status == "fail" and not verdict.ok
    assert verdict.findings[0].code == "conservation.partition_mismatch"
    assert verdict.findings[0].claimed == 6_320_000

    loop2, _ = _parent(tmp_path / "u", [_tc("finish", {"summary": "看起来没问题"})])
    verdict2 = SubAgentRunner(loop2).run_review(claim=claim, facts=_facts())
    assert verdict2.status == "unparsed" and not verdict2.ok
    assert "可解析" in verdict2.summary()


def test_review_limit_and_child_config(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class CapturingLoop(AgentLoop):
        def __init__(self, **kwargs):
            captured.setdefault("configs", []).append(kwargs["config"])
            super().__init__(**kwargs)

    monkeypatch.setattr(subagent_mod, "AgentLoop", CapturingLoop)
    loop, llm = _parent(
        tmp_path,
        [
            _tc("finish", {"summary": json.dumps(_verdict_payload())}),
            _tc("finish", {"summary": json.dumps(_verdict_payload())}),
        ],
    )
    runner = SubAgentRunner(loop, max_reviews_per_run=1, review_max_turns=7)
    first = runner.run_review(claim=_claim(), facts=_facts())
    second = runner.run_review(claim=_claim(), facts=_facts())

    assert first.status == "pass"
    assert second.status == "refused" and "最多复核" in second.notes
    cfg = captured["configs"][0]
    assert cfg.role_preamble == REVIEW_PREAMBLE
    assert cfg.review_enabled is False, "复核子代理的 finish 不能再触发复核（防套娃）"
    assert cfg.review_require_before_finish is False
    assert cfg.max_turns == 7
    assert cfg.report_enabled is False


def test_review_result_tool_records_signature_and_findings(tmp_path: Path):
    loop, _ = _parent(tmp_path, [])
    ctx = loop.context
    state = TaskState(run_id="r", goal="g", workspace=str(tmp_path))
    state.artifacts = ["output/zones.shp"]
    ctx.state = state

    seen: dict = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return ReviewVerdict(
            status="fail",
            findings=[
                Finding(
                    code="conservation.partition_mismatch",
                    detail="Z02 对不上",
                    claimed=6_320_000,
                    measured=2_300_000,
                )
            ],
            checked=["conservation.sum_equals"],
        )

    ctx.extra["review_factory"] = factory
    observation = loop.registry.execute(
        "review_result",
        {"claim": _claim().to_dict(), "facts": _facts()},
        ctx,
    )
    assert seen["claim"].goal == "按城市汇总人口"
    assert observation.ok is False
    assert "双复核未通过" in observation.summary
    assert observation.data["findings"][0]["code"] == "conservation.partition_mismatch"
    assert state.review["status"] == "fail"
    assert state.review["signature"] == ["output/zones.shp"]
    assert state.review["paths"] == ["output/zones.shp"]


def test_review_result_requires_claim_summary_and_factory(tmp_path: Path):
    loop, _ = _parent(tmp_path, [])
    ctx = loop.context
    ctx.state = TaskState(run_id="r", goal="g", workspace=str(tmp_path))
    observed = loop.registry.execute(
        "review_result",
        {"claim": _claim().to_dict(), "facts": _facts()},
        ctx,
    )
    assert observed.ok is False
    assert "engine.review_enabled 未开" in observed.summary

    ctx.extra["review_factory"] = lambda **kwargs: ReviewVerdict(status="pass")
    empty = loop.registry.execute("review_result", {"claim": {"metrics": {"x": 1}}}, ctx)
    assert empty.ok is False and "claim.summary 为空" in empty.summary


def test_assertion_check_tool_returns_structured_report(tmp_path: Path):
    loop, _ = _parent(tmp_path, [])
    ctx = loop.context
    ctx.state = TaskState(run_id="r", goal="g", workspace=str(tmp_path))
    claim = _claim()
    claim.metrics["zone_pop"] = {"Z01": 2_100_000, "Z02": 6_420_000}  # 多出 10 万

    observation = loop.registry.execute(
        "assertion_check",
        {"assertions": claim.assertions, "claim": claim.to_dict()},
        ctx,
    )
    assert observation.ok is True  # 工具跑成了；断言结论在 data 里
    report = observation.data["report"]
    assert report["ok"] is False
    assert report["findings"][0]["code"] == "conservation.sum_mismatch"
