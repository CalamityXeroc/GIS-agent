# -*- coding: utf-8 -*-
"""双复核的装配与 finish 闸门（默认关、开启后先复核再验收）。"""

from __future__ import annotations

import json
from pathlib import Path

from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.runner import build_engine
from gis_cli.engine.state import TaskState
from gis_cli.engine.tool_codec import Action
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.review import Finding, ReviewVerdict
from gis_cli.trace.recorder import RunRecorder


class StubVerifier:
    def verify(self, state):
        return {"pass": True, "summary": "验收通过"}


def _loop(tmp_path: Path, *, factory=None, **config_kwargs) -> AgentLoop:
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "output" / "x.shp").write_text("placeholder", encoding="utf-8")
    context = EngineContext(workspace=tmp_path)
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=object(),
        registry=build_default_registry(),
        context=context,
        code_runner=None,
        verifier=StubVerifier(),
        config=LoopConfig(report_enabled=False, **config_kwargs),
        recorder=RunRecorder("wiring", tmp_path / ".gis_agent" / "traces"),
        auto_extract_requirements=False,
    )
    loop.state = TaskState(run_id="wiring", goal="g", workspace=str(tmp_path))
    context.state = loop.state
    if factory is not None:
        context.extra["review_factory"] = factory
    return loop


def _finish(loop: AgentLoop, summary: str = "完成"):
    return loop._handle_finish(
        Action(kind="tool", tool="finish", args={"summary": summary, "artifacts": ["output/x.shp"]})
    )


def test_review_disabled_by_default_does_not_gate_finish(tmp_path: Path):
    loop = _loop(tmp_path)
    assert loop.config.review_enabled is False
    observation, stop = _finish(loop)
    assert observation.ok is True and stop is True
    assert loop.state.review == {}


def test_finish_requires_review_once_then_lets_through_with_missing_mark(tmp_path: Path):
    loop = _loop(tmp_path, review_enabled=True)
    observation, stop = _finish(loop)
    assert observation.ok is False and stop is False
    assert "双复核未完成" in observation.summary
    assert loop.state.review == {}

    # 第 2 次仍不复核：放行，但必须留痕（“没复核”不能看起来像“复核通过”）
    observation2, stop2 = _finish(loop)
    assert observation2.ok is True and stop2 is True
    assert loop.state.review["status"] == "missing"
    assert loop.state.review["signature"] == ["output/x.shp"]
    assert "没有执行 review_result" in observation2.summary


def test_finish_attaches_pass_verdict(tmp_path: Path):
    loop = _loop(tmp_path, review_enabled=True)
    loop.state.review = {
        "status": "pass",
        "summary": "双复核通过（检查 2 条断言）",
        "signature": ["output/x.shp"],
        "checked": ["conservation.sum_equals", "dimension.area_unit"],
    }
    observation, stop = _finish(loop)
    assert observation.ok is True and stop is True
    assert observation.data["review"]["status"] == "pass"
    assert "双复核：双复核通过" in observation.summary


def test_stale_review_after_artifact_change_requires_rerun(tmp_path: Path):
    loop = _loop(tmp_path, review_enabled=True)
    loop.state.review = {
        "status": "pass",
        "summary": "旧产物上的复核",
        "signature": ["output/old.shp"],
    }
    observation, stop = _finish(loop)
    assert observation.ok is False and stop is False
    assert "双复核未完成" in observation.summary


def test_review_block_on_fail_rejects_until_limit(tmp_path: Path):
    verdict = ReviewVerdict(
        status="fail",
        findings=[
            Finding(
                code="conservation.partition_mismatch",
                detail="Z02 多算 10 万",
                claimed=6_320_000,
                measured=2_300_000,
            )
        ],
        checked=["conservation.sum_equals"],
    )
    loop = _loop(
        tmp_path,
        factory=lambda **kwargs: verdict,
        review_enabled=True,
        review_block_on_fail=True,
        max_finish_rejections=2,
    )
    loop.state.review = {**verdict.to_dict(), "signature": ["output/x.shp"]}

    first, stop1 = _finish(loop)
    assert first.ok is False and stop1 is False
    assert "双复核未通过（第 1 次）" in first.summary

    second, stop2 = _finish(loop)
    assert second.ok is False and stop2 is False

    # 超过 max_finish_rejections 后放行，但必须在 hint 里写清“人工复核”
    third, stop3 = _finish(loop)
    assert third.ok is True and stop3 is True
    assert "人工复核" in third.summary


def test_review_fail_is_only_a_warning_by_default(tmp_path: Path):
    loop = _loop(tmp_path, review_enabled=True, review_block_on_fail=False)
    loop.state.review = {
        "status": "fail",
        "summary": "双复核未通过（1 个问题）",
        "signature": ["output/x.shp"],
        "findings": [{"code": "topology.overlap", "severity": "error", "detail": "重叠 6%"}],
    }
    observation, stop = _finish(loop)
    assert observation.ok is True and stop is True
    assert observation.data["review"]["status"] == "fail"


def test_build_engine_registers_review_tools_only_when_enabled(tmp_path: Path):
    def config(enabled: bool) -> Path:
        path = tmp_path / f"config_{enabled}.json"
        path.write_text(
            json.dumps(
                {
                    "model": "mock",
                    "api_key": "x",
                    "api_base": "http://127.0.0.1:1",
                    "engine": {
                        "review_enabled": enabled,
                        "subagent_enabled": False,
                        "report_enabled": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    off = build_engine(
        workspace=tmp_path / "ws_off",
        config_path=config(False),
        refresh_catalog=False,
    )
    try:
        names = off.loop.registry.names()
        assert "assertion_check" not in names and "review_result" not in names
        assert "review_factory" not in off.context.extra
        assert off.loop.config.review_enabled is False
    finally:
        off.close()

    on = build_engine(
        workspace=tmp_path / "ws_on",
        config_path=config(True),
        refresh_catalog=False,
    )
    try:
        names = on.loop.registry.names()
        assert "assertion_check" in names and "review_result" in names
        assert callable(on.context.extra.get("review_factory"))
        assert on.loop.config.review_enabled is True
    finally:
        on.close()
