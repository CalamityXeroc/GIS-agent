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

    def _fake_create_map(layers, intent, *, output_path, aprx_path="", code_runner=None, catalog_rows=None):
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


# --------------------------------------------------- 质量分（P1-3，只是信息）
def test_create_map_data_carries_quality_without_changing_ok(tmp_path):
    """``create_map`` 在 data 里带上 0–100 质量分，但 **ok 语义不变**。

    质量分是给人/模型看的信息：闸门仍是 qc_ok（版面体检），分数高低不能让一张
    体检通过的图变成失败，也不能让体检不通过的图侥幸通过。
    """
    payload = {
        "output": str(tmp_path / "output" / "map.jpg"),
        "aprx": str(tmp_path / "output" / "map.aprx"),
        "qc_ok": True,
        "qc_summary": "版面体检全部通过",
        "qc": {
            "ok": False,
            "checks": [
                {"name": "legend_covers_data", "ok": False, "detail": "图例框内非白像素占比 0.72"},
                {"name": "title_too_small", "ok": True, "detail": "图名字号 18pt"},
            ],
        },
    }
    observation = _run_tool(tmp_path, payload)
    assert observation.ok is True                      # 闸门只看 qc_ok
    quality = observation.data["quality"]
    assert quality["readability"] == 90.0              # 图例压数据扣 10
    assert quality["normativity"] == 100.0             # 图名正常，不跨维度扣分
    assert quality["total"] == 90.0
    assert quality["deductions"][0]["name"] == "legend_covers_data"

    failed = _run_tool(tmp_path, {**payload, "qc_ok": False, "qc_summary": "版面体检未通过"})
    assert failed.ok is False
    assert failed.data["quality"]["total"] == 90.0     # 失败路径也照样带分（便于解释）


def test_create_map_does_not_invent_quality_without_qc(tmp_path):
    """拿不到体检结果时不硬造分数（缺证据 ≠ 满分）。"""
    observation = _run_tool(tmp_path, {
        "output": str(tmp_path / "output" / "map.jpg"),
        "aprx": str(tmp_path / "output" / "map.aprx"),
        "qc_ok": True,
    })
    assert observation.ok is True
    assert "quality" not in observation.data


def test_check_map_project_returns_quality_without_changing_ok(tmp_path):
    """``check_map_project`` 的 data 带上质量分；ok 仍由断言结果决定。"""
    from gis_cli.recipes.store import ValidationResult

    class _Recipes:
        def __init__(self, result) -> None:
            self.result = result

        def run_assertions(self, raws):
            return [self.result]

    registry = build_default_registry()
    args = {"aprx_path": str(tmp_path / "map.aprx")}

    ok_ctx = EngineContext(
        workspace=tmp_path,
        recipes=_Recipes(ValidationResult("aprx_map_check", "map.aprx", True, "渲染器=['GraduatedColorsRenderer']")),
    )
    passed = registry.execute("check_map_project", args, ok_ctx)
    assert passed.ok is True
    assert passed.data["kind"] == "aprx_map_check"     # 原有 data 字段不动
    assert passed.data["quality"]["total"] == 100.0
    assert passed.data["quality"]["correctness"] == 100.0

    bad_ctx = EngineContext(
        workspace=tmp_path,
        recipes=_Recipes(ValidationResult("aprx_map_check", "map.aprx", False, "布局缺少「图例」")),
    )
    failed = registry.execute("check_map_project", args, bad_ctx)
    assert failed.ok is False                          # 语义不变：仍是断言结论
    quality = failed.data["quality"]
    assert quality["correctness"] == 0.0               # 一条断言失败 = 该条全扣
    assert quality["total"] == 0.0
    assert quality["deductions"][0]["name"] == "aprx_map_check"
    assert "布局缺少" in quality["deductions"][0]["reason"]


def test_check_map_project_env_failure_is_not_a_wrong_map(tmp_path):
    """B1 回归（工具层）：缺 arcpy 是环境失败，既不能算 0 分，也不能说“版面不合格”。

    实测（DAIR-57）：同一份合格 .aprx，CLI（无 arcpy）拿 0 分，arcgispro-py3 ok=true。
    """
    from gis_cli.recipes.store import ENV_NEEDS_ARCGIS, ValidationResult

    class _Recipes:
        def run_assertions(self, raws):
            return [
                ValidationResult(
                    "aprx_map_check",
                    "map.aprx",
                    False,
                    "未核验（需 ArcGIS）：当前进程没有 arcpy，也没有可用的代码执行器",
                    error_type=ENV_NEEDS_ARCGIS,
                )
            ]

    registry = build_default_registry()
    observation = registry.execute(
        "check_map_project",
        {"aprx_path": str(tmp_path / "map.aprx")},
        EngineContext(workspace=tmp_path, recipes=_Recipes()),
    )
    assert observation.ok is False                          # 没核验过不能报通过
    assert observation.error["type"] == ENV_NEEDS_ARCGIS     # 环境失败 ≠ 断言失败
    assert observation.data["error_type"] == ENV_NEEDS_ARCGIS
    assert "quality" not in observation.data                # 缺证据不硬造分数（既不 0 分也不 100 分）
    assert "未核验" in observation.summary
    assert "不是版面不合格" in observation.hint


def test_check_map_project_summary_does_not_repeat_prefix(tmp_path):
    """DAIR-63 复查 1：summary 只在 detail 没带前缀时补，别补成两层。

    旧行为：``summary="工程未核验（需 ArcGIS）: 未核验（需 ArcGIS）：当前进程没有 arcpy…"``。
    """
    from gis_cli.recipes.store import ENV_NEEDS_ARCGIS, ValidationResult

    class _Recipes:
        def run_assertions(self, raws):
            return [
                ValidationResult(
                    "aprx_map_check",
                    "map.aprx",
                    False,
                    "未核验（需 ArcGIS）：当前进程没有 arcpy，也没有可用的代码执行器",
                    error_type=ENV_NEEDS_ARCGIS,
                )
            ]

    registry = build_default_registry()
    observation = registry.execute(
        "check_map_project",
        {"aprx_path": str(tmp_path / "map.aprx")},
        EngineContext(workspace=tmp_path, recipes=_Recipes()),
    )
    assert observation.summary.count("未核验（需 ArcGIS）") == 1, observation.summary
    assert observation.summary.startswith("未核验（需 ArcGIS）"), observation.summary


def test_check_map_project_unresolved_path_is_not_a_wrong_map(tmp_path):
    """DAIR-63 复查 2（工具层）：路径没解析出来既不是版面不合格，也不说成「需 ArcGIS」。"""
    from gis_cli.recipes.store import ENV_PATH_UNRESOLVED, ValidationResult

    class _Recipes:
        def run_assertions(self, raws):
            return [
                ValidationResult(
                    "aprx_map_check",
                    "output/成果图/x.aprx",
                    False,
                    "工程路径无法解析：output/成果图/x.aprx（当前目录与代码执行器工作区都没有这个文件）",
                    error_type=ENV_PATH_UNRESOLVED,
                )
            ]

    registry = build_default_registry()
    observation = registry.execute(
        "check_map_project",
        {"aprx_path": "output/成果图/x.aprx"},
        EngineContext(workspace=tmp_path, recipes=_Recipes()),
    )
    assert observation.ok is False
    assert observation.error["type"] == ENV_PATH_UNRESOLVED
    assert "quality" not in observation.data, "没判定过就不给分"
    assert observation.summary.count("工程路径无法解析") == 1, observation.summary
    assert "ArcGIS" not in observation.summary, observation.summary
    assert "路径问题" in observation.hint


def test_check_map_project_plain_failure_has_no_error_type(tmp_path):
    """真断言失败（版面问题）不沾 error_type：与「没测 / 报错」在 to_dict() 上仍可分。"""
    from gis_cli.recipes.store import ValidationResult

    class _Recipes:
        def run_assertions(self, raws):
            return [ValidationResult("aprx_map_check", "map.aprx", False, "布局缺少「图例」")]

    registry = build_default_registry()
    observation = registry.execute(
        "check_map_project",
        {"aprx_path": str(tmp_path / "map.aprx")},
        EngineContext(workspace=tmp_path, recipes=_Recipes()),
    )
    assert observation.ok is False
    assert "error_type" not in observation.data
    assert observation.data["quality"]["total"] == 0.0


def test_check_map_project_open_failure_hint_is_not_self_contradictory(tmp_path, monkeypatch):
    """DAIR-73 复查 1（端到端）：工程打不开不能一边说「按提示改渲染器/配色/布局」一边说
    「工程损坏」。

    走真 ``RecipeLibrary`` + 假 arcpy（``ArcGISProject`` 抛异常），不 stub detail。
    """
    import sys
    import types

    from gis_cli.recipes import store as store_mod
    from gis_cli.recipes.store import ENV_CHECK_ERROR

    def _boom(_path):
        raise RuntimeError("aprx is corrupt")

    fake_arcpy = types.ModuleType("arcpy")
    fake_arcpy.mp = types.SimpleNamespace(ArcGISProject=_boom)
    monkeypatch.setitem(sys.modules, "arcpy", fake_arcpy)
    monkeypatch.setattr(store_mod, "_arcpy_importable", lambda: True)

    aprx = tmp_path / "map.aprx"
    aprx.write_text("x", encoding="utf-8")
    registry = build_default_registry()
    observation = registry.execute(
        "check_map_project",
        {"aprx_path": str(aprx)},
        EngineContext(workspace=tmp_path, recipes=store_mod.RecipeLibrary()),
    )
    assert observation.ok is False
    assert observation.error["type"] == ENV_CHECK_ERROR     # 核验没做成 ≠ 版面不合格
    assert observation.data["error_type"] == ENV_CHECK_ERROR
    assert "quality" not in observation.data, "没判定成就不给分（既不 0 分也不 100 分）"
    assert "无法打开工程" in observation.summary, observation.summary
    # 不再让两句提示互相矛盾：hint 说「工程损坏」，就别再叫人去改渲染器/配色/布局。
    assert "工程损坏" in observation.hint, observation.hint
    assert "按提示修正渲染器/配色/布局" not in observation.hint, observation.hint


def test_check_map_project_real_store_detail_is_not_double_prefixed(tmp_path, monkeypatch):
    """DAIR-73 复查 2（端到端）：去重不能靠中文字面量，措辞漂移要被抓住。

    旧用例 stub 了 ``ValidationResult`` 的 detail 字面量（``tests/test_engine_tools.py``
    的 ``_Recipes``），``store.py`` 那两句 detail 一改，工具层就会静默回到双前缀而测试
    仍绿。这里用真 ``RecipeLibrary`` 产出 detail，再原样喂给工具层。
    """
    from gis_cli.recipes import store as store_mod

    monkeypatch.setattr(store_mod, "_arcpy_importable", lambda: False)
    aprx = tmp_path / "map.aprx"
    aprx.write_text("x", encoding="utf-8")
    lib = store_mod.RecipeLibrary()          # 真库：无 arcpy、无 code_runner
    expected = lib.run_assertions([{"type": "aprx_map_check", "path": str(aprx)}])[0]
    assert expected.error_type == store_mod.ENV_NEEDS_ARCGIS, expected.detail

    registry = build_default_registry()
    observation = registry.execute(
        "check_map_project",
        {"aprx_path": str(aprx)},
        EngineContext(workspace=tmp_path, recipes=lib),
    )
    # detail 已自带前缀 → summary 就是 detail 本身：多补一层、或 store 换了措辞导致
    # 工具层又补一层，这里都会红。
    assert observation.summary == expected.detail, observation.summary
    assert observation.summary.count("未核验") == 1, observation.summary
