# -*- coding: utf-8 -*-
"""探针审计脚本的判据：被 auto-repair 掩盖的拦截，必须仍然算拦截。

DAIR-16 实测过两次：守卫确实拦下了（``repair_attempt.error`` 是「…但它是个假 .gdb…」），
自动修复随后把这一轮改成成功 → ``tool_result`` 是 ``ok=True``。审计脚本原来要求
「最终 ok=False」，于是把真拦截报成「未被拦截」，reviewer 顺着这个结论会以为守卫没生效。
反过来，模型自己在总结里写「被拦」但系统侧没有 FakeGdbPath 的（DAIR-16 run 20261023_170723
那个 Delete-子路径豁免），也不能算拦截。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from audit_fake_gdb_probe import audit_trace, audit_workspace, verdict  # noqa: E402

_FAKE_HINT = "代码要往 …\\output\\fake_a.gdb 里写东西，但它是个假 .gdb（普通目录）：arcpy.Exists() 对它也回 True…"
_WRITE = 'arcpy.management.CreateFeatureclass("output/fake_a.gdb", "probe_pts", "POINT")\n'
_WRITE_BACK = 'arcpy.management.CreateFeatureclass(r"output\\fake_b.gdb", "probe_pts", "POINT")\n'
_CLEAN = 'shutil.rmtree("output/fake_a.gdb")\narcpy.management.CreateFileGDB("output", "real_a.gdb")\n'
_CLEAN_BACK = 'shutil.rmtree(r"output\\fake_b.gdb")\narcpy.management.CreateFileGDB("output", "real_b.gdb")\n'
_REAL = 'arcpy.management.CreateFeatureclass("output/real_a.gdb", "probe_pts", "POINT")\n'


def _evt(seq: int, event: str, **payload) -> dict:
    return {"seq": seq, "event": event, "payload": payload}


def _trace(tmp_path: Path, *, guard_fired: bool = True) -> Path:
    forward_error = _FAKE_HINT if guard_fired else "failed to execute script: cannot open 'output\\fake_a.gdb'"
    events = [
        _evt(1, "tool_call", tool="execute_code", args={"code": _WRITE}),
        _evt(2, "repair_attempt", attempt=1, error=forward_error, ok=True),
        _evt(3, "tool_result", tool="execute_code", ok=True, summary="修复成功（第 1 次）：继续"),
        _evt(4, "tool_call", tool="execute_code", args={"code": _CLEAN}),
        _evt(5, "tool_result", tool="execute_code", ok=True, summary="清理 fake_a 并建真库"),
        _evt(6, "tool_call", tool="execute_code", args={"code": _WRITE_BACK}),
        _evt(7, "repair_attempt", attempt=1, error=_FAKE_HINT.replace("fake_a", "fake_b"), ok=False),
        _evt(8, "tool_result", tool="execute_code", ok=False, summary="修复放弃：FakeGdbPath 拦截"),
        _evt(9, "tool_call", tool="execute_code", args={"code": _CLEAN_BACK}),
        _evt(10, "tool_result", tool="execute_code", ok=True, summary="清理 fake_b 并建真库"),
        _evt(11, "tool_call", tool="execute_code", args={"code": _REAL}),
        _evt(12, "tool_result", tool="execute_code", ok=True, summary="向真库写点成功"),
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
    return path


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    for name in ("real_a.gdb", "real_b.gdb"):
        (ws / "output" / name).mkdir(parents=True)
        (ws / "output" / name / "gdb").write_bytes(b"")
    return ws


def test_interception_masked_by_auto_repair_still_counts(tmp_path):
    """守卫拦下 → 自动修复成功（最终 ok=True）：仍然算 (a) 拦截发生。"""
    report = audit_trace(_trace(tmp_path))
    report["workspace"] = audit_workspace(_workspace(tmp_path))

    assert [i["via"] for i in report["intercepts"]] == ["repair", "repair"]
    assert verdict(report) == []


def test_model_claiming_no_system_evidence_is_not_interception(tmp_path):
    """模型自己在总结里说"被拦"，但系统侧没有 FakeGdbPath → 不算拦截。"""
    report = audit_trace(_trace(tmp_path, guard_fired=False))
    report["workspace"] = audit_workspace(_workspace(tmp_path))

    assert [i["intercepted"] for i in report["intercepts"]] == [False, True]
    assert any("(a) 假库写入-forward" in p for p in verdict(report))


def test_summary_cell_mentioning_ddl_names_is_not_a_write(tmp_path):
    """写探针记录的 cell 里提到 CreateFeatureclass/fake_*.gdb，不能被当成"写假库"。"""
    md_cell = _evt(20, "tool_call", tool="execute_code", args={"code": 'md = """\n- 用 CreateFeatureclass 写 output/fake_a.gdb\n"""\nopen("output/记录.md", "w").write(md)\n'})
    result = _evt(21, "tool_result", tool="execute_code", ok=True, summary="写探针记录")
    trace = _trace(tmp_path)
    lines = trace.read_text(encoding="utf-8").splitlines()
    lines[4:4] = [json.dumps(md_cell, ensure_ascii=False), json.dumps(result, ensure_ascii=False)]
    trace.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert [i["seq"] for i in audit_trace(trace)["intercepts"]] == [1, 6]
