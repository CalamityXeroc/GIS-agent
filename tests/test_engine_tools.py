# -*- coding: utf-8 -*-
"""引擎工具面单测（不需要 ArcGIS）：create_map 的失败契约与设计边车。

对应用户可见的两处实测缺陷（DAIR-32/DAIR-35）：
1. 版面体检不通过时 create_map 返回 ok=True，代理照样走到 finish，交付的 .aprx
   带着未取整的图例标签；
2. 同名 .spec.json 边车没落 renderer，验收（map_layout_ok）无法区分分类图例与
   分级图例，对同一张图得出与运行内体检相反的结论。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gis_cli.engine.tools import EngineContext, build_default_registry


class _Runner:
    """把 create_map 工具生成的代码在进程内执行（create_map 由用例打桩）。"""

    def __init__(self) -> None:
        self.codes: list[str] = []

    def run(self, code, timeout=None, workspace=None):
        self.codes.append(code)
        ns: dict = {"set_result": lambda value: ns.__setitem__("_result", value)}
        exec(compile(code, "<tools-test>", "exec"), ns)
        result = ns.get("_result")

        class R:
            ok = True
            error = None
            stdout = ""
            stderr = ""

        R.result = result
        return R


class _StaticRunner:
    """直接回一个固定 payload（不需要真的执行代码）。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def run(self, code, timeout=None, workspace=None):
        class R:
            ok = True
            error = None
            stdout = ""
            stderr = ""

        R.result = self.payload
        return R


def _run_tool(tmp_path: Path, payload_or_runner, args: dict | None = None):
    registry = build_default_registry()
    runner = payload_or_runner if hasattr(payload_or_runner, "run") else _StaticRunner(payload_or_runner)
    ctx = EngineContext(workspace=tmp_path, code_runner=runner)
    tool_args = {"layers": ["input/a.shp"], "output_path": str(tmp_path / "output" / "map.jpg")}
    tool_args.update(args or {})
    return registry.execute("create_map", tool_args, ctx)


def test_create_map_tool_blocks_delivery_when_layout_qc_fails(tmp_path):
    """版面体检不通过必须是一条失败观察，不能 ok=True 放行。

    DAIR-32 基准 4a 第 2 次：工具返回“（版面体检 未通过）”但 ok=True，代理直接
    走到 finish，交付的 .aprx 图例里是 193833.858800001 → map_layout_ok 断言失败。
    """
    failed = _run_tool(tmp_path, {
        "output": str(tmp_path / "output" / "map.jpg"),
        "aprx": str(tmp_path / "output" / "map.aprx"),
        "qc_ok": False,
        "qc_summary": "版面体检未通过 → legend_labels_unrounded: 分界值不整齐 [193833.858800001]",
    })
    assert failed.ok is False
    assert "版面体检未通过" in failed.summary
    assert "legend_labels_unrounded" in (failed.hint or "")
    assert "explicit_bounds" in (failed.hint or "")      # 必须给出可修的下一步
    assert (failed.error or {}).get("type") == "LayoutQCFailed"
    # 产物照常登记：已经落盘，登记着才不会重复出图/丢交付物
    assert any(str(path).endswith("map.aprx") for path in failed.artifacts)

    passed = _run_tool(tmp_path, {
        "output": str(tmp_path / "output" / "map.jpg"),
        "aprx": str(tmp_path / "output" / "map.aprx"),
        "qc_ok": True,
        "qc_summary": "版面体检全部通过",
    })
    assert passed.ok is True and "版面体检 通过" in passed.summary


def test_create_map_sidecar_carries_renderer_for_layout_assertions(tmp_path, monkeypatch):
    """验收（map_layout_ok）读的是同名 .spec.json，里面必须有 renderer。

    DAIR-35 实测：边车只落了 output/qc/notes，没有 renderer，验收那边无法区分
    “分类图例”与“分级图例”，会对同一张图得出与运行内体检相反的结论
    （运行内：“类名无原始浮点，通过”；验收：“分界值不整齐 [193, 834]”）。
    """
    import gis_cli.cartography as carto

    def _fake_create_map(layers, intent, *, output_path, aprx_path="", code_runner=None):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"jpg")
        return {
            "output": str(out),
            "aprx": str(out.with_suffix(".aprx")),
            "qc_ok": True,
            "qc_summary": "版面体检全部通过",
            "spec": {"renderer": {"mode": "unique", "field": "DIST_M",
                                  "legend_entries": [{"label": "193,834", "color": "#E8E8E8"}]}},
        }

    monkeypatch.setattr(carto, "create_map", _fake_create_map)
    observation = _run_tool(tmp_path, _Runner())
    assert observation.ok is True

    sidecar = json.loads((tmp_path / "output" / "map.spec.json").read_text(encoding="utf-8"))
    assert "spec" not in sidecar                      # 不落整份 spec（体积）
    assert sidecar["renderer"]["mode"] == "unique"    # 但渲染器要在
    assert sidecar["renderer"]["legend_entries"][0]["label"] == "193,834"
