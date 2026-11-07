# -*- coding: utf-8 -*-
"""工程核验（aprx_map_check）的环境回退测试（不需要 ArcGIS）。

对应缺陷 B1（DAIR-57）：benchmark/CLI 跑在没有 arcpy 的 .venv 里，
``from arcpy import mp`` 必然失败，一个**完全合格**的 .aprx 19.9 ms 拿到
``ok=false / quality.total=0.0``——环境失败被算成断言失败。做法与
``map_layout_ok`` 一致：有 arcpy 就地算，没有就交给 code_runner 去 ArcGIS
解释器里算；两样都没有就明确报「未核验」并带上 ``error_type``。

本文件的用例把 ``_arcpy_importable`` 打桩成 False，不碰真 arcpy。
"""
from __future__ import annotations


def test_aprx_map_check_routes_through_code_runner(tmp_path, monkeypatch):
    """B1 回归：没有 arcpy 时工程核验交给 code_runner，不能判「版面不合格」。

    实测（DAIR-57）：benchmark/CLI 的 .venv 里 ``from arcpy import mp`` 必然失败，
    一个完全合格的 .aprx 19.9 ms 拿到 ``ok=false / quality.total=0.0``；同一个
    工程在 arcgispro-py3 里同一条断言 ``ok=true``（12.37 s）。
    """
    from gis_cli.recipes import store as store_mod

    class _Result:
        ok = True
        result = {"ok": True, "detail": "渲染器=['UniqueValueRenderer'] 配色={'Bison': '#B07FB2'}"}
        error = None

    class _Runner:
        def __init__(self) -> None:
            self.codes: list[str] = []

        def run(self, code, timeout=None, workspace=None):
            self.codes.append(code)
            return _Result()

    monkeypatch.setattr(store_mod, "_arcpy_importable", lambda: False)
    runner = _Runner()
    lib = store_mod.RecipeLibrary(code_runner=runner)
    aprx = tmp_path / "map.aprx"
    aprx.write_text("x", encoding="utf-8")

    result = lib.run_assertions(
        [{"type": "aprx_map_check", "path": str(aprx), "renderer": "UniqueValueRenderer"}]
    )[0]
    assert result.ok is True, result.detail
    assert result.error_type == "", "走回退拿到真实结果，不是环境失败"
    assert "#B07FB2" in result.detail, result.detail
    assert runner.codes, "必须让 code_runner 去 ArcGIS 解释器里跑"
    assert "_aprx_map_verdict" in runner.codes[0], "判据只此一份，别在子进程里另抄一套"


def test_aprx_map_check_without_arcpy_is_unverified_not_failed(tmp_path, monkeypatch):
    """B1 回归：既没 arcpy 也没 code_runner 时，明确报「需 ArcGIS」，不是断言失败。"""
    from gis_cli.recipes import store as store_mod

    monkeypatch.setattr(store_mod, "_arcpy_importable", lambda: False)
    lib = store_mod.RecipeLibrary()
    aprx = tmp_path / "map.aprx"
    aprx.write_text("x", encoding="utf-8")

    result = lib.run_assertions([{"type": "aprx_map_check", "path": str(aprx)}])[0]
    assert result.ok is False, "没核验过就不能报通过"
    assert result.error_type == store_mod.ENV_NEEDS_ARCGIS, "错误类型要能与断言失败分开"
    assert "需 ArcGIS" in result.detail, result.detail
    assert result.to_dict()["error_type"] == store_mod.ENV_NEEDS_ARCGIS


def test_aprx_map_check_runner_failure_is_unverified(tmp_path, monkeypatch):
    """内核没跑出结论（超时/崩）也不能算「版面不合格」——那是没测。"""
    from gis_cli.recipes import store as store_mod

    class _Result:
        ok = False
        result = None
        error = {"type": "KernelTimeout", "message": "execution exceeded 300s"}

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            return _Result()

    monkeypatch.setattr(store_mod, "_arcpy_importable", lambda: False)
    lib = store_mod.RecipeLibrary(code_runner=_Runner())
    aprx = tmp_path / "map.aprx"
    aprx.write_text("x", encoding="utf-8")

    result = lib.run_assertions([{"type": "aprx_map_check", "path": str(aprx)}])[0]
    assert result.ok is False
    assert result.error_type == store_mod.ENV_NEEDS_ARCGIS
    assert "未核验" in result.detail, result.detail


def test_resolve_assert_target_falls_back_to_runner_workspace(tmp_path, monkeypatch):
    """相对路径要先解析成绝对路径：内核的 CWD 是工作区，不是调用方目录。

    实测（DAIR-60）：CLI 传 ``output/成果图/x.aprx``，直接交给内核会按内核自己的
    CWD 解析而报「无法打开工程」——同一条断言只是换个进程就假失败。
    """
    from pathlib import Path

    from gis_cli.recipes import store as store_mod

    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "m.aprx").write_text("x", encoding="utf-8")

    class _Runner:
        workspace = str(ws)

    monkeypatch.chdir(tmp_path)  # CWD 下没有 output/m.aprx
    resolved = store_mod._resolve_assert_target(Path("output/m.aprx"), _Runner())
    assert resolved == ws / "output" / "m.aprx"
    assert resolved.is_absolute()

    absolute = (tmp_path / "abs.aprx").resolve()
    assert store_mod._resolve_assert_target(absolute, None) == absolute


def test_aprx_map_check_unresolved_path_is_not_a_missing_file(tmp_path, monkeypatch):
    """DAIR-63 复查 2：相对路径哪都找不到 = 路径问题，不是「工程文件不存在」的断言失败。

    旧行为：``_resolve_assert_target`` 解析失败仍返回原相对路径 → ``exists()`` 为假
    → ``ok=False`` 且 ``error_type=""``，看上去就是「版面不合格」。
    """
    from pathlib import Path

    from gis_cli.recipes import store as store_mod

    class _Runner:
        workspace = str(tmp_path / "ws")

    (tmp_path / "ws").mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # CWD 与 runner.workspace 下都没有 output/成果图/x.aprx

    lib = store_mod.RecipeLibrary(code_runner=_Runner())
    result = lib.run_assertions([{"type": "aprx_map_check", "path": "output/成果图/x.aprx"}])[0]

    assert result.ok is False
    assert result.error_type == store_mod.ENV_PATH_UNRESOLVED
    assert result.error_type != store_mod.ENV_NEEDS_ARCGIS, "路径问题不能并成「需 ArcGIS」"
    assert result.to_dict()["error_type"] == store_mod.ENV_PATH_UNRESOLVED
    assert "工程文件不存在" not in result.detail, "路径没解析出来 ≠ 工程文件不存在"
    assert "output/成果图/x.aprx" in result.detail, result.detail


def test_aprx_map_verdict_traversal_error_carries_error_type(tmp_path, monkeypatch):
    """DAIR-63 复查 3：打开工程后遍历抛异常是真报错，不能落进 NeedsArcGIS（没测）。

    旧行为：遍历阶段的异常冒出 ``_aprx_map_verdict``，外层吞成 ``ok=False`` 且没有
    ``error_type``（换进程跑则落进 ``NeedsArcGIS``）。
    """
    import sys
    import types

    from gis_cli.recipes import store as store_mod

    class _BrokenProject:
        def listMaps(self):
            raise RuntimeError("project is corrupt")

    fake_arcpy = types.ModuleType("arcpy")
    fake_arcpy.mp = types.SimpleNamespace(ArcGISProject=lambda _path: _BrokenProject())
    monkeypatch.setitem(sys.modules, "arcpy", fake_arcpy)

    verdict = store_mod._aprx_map_verdict("map.aprx", {"renderer": "UniqueValueRenderer"})
    assert verdict["ok"] is False
    assert verdict["error_type"] == store_mod.ENV_CHECK_ERROR
    assert "RuntimeError" in verdict["detail"], verdict["detail"]

    # 就地（有 arcpy）这条路：error_type 要一路带到 ValidationResult 上
    monkeypatch.setattr(store_mod, "_arcpy_importable", lambda: True)
    aprx = tmp_path / "map.aprx"
    aprx.write_text("x", encoding="utf-8")
    result = store_mod.RecipeLibrary().run_assertions(
        [{"type": "aprx_map_check", "path": str(aprx)}]
    )[0]
    assert result.ok is False
    assert result.error_type == store_mod.ENV_CHECK_ERROR
    assert result.error_type != store_mod.ENV_NEEDS_ARCGIS, "真报错不能算「没测」"
    assert result.to_dict()["error_type"] == store_mod.ENV_CHECK_ERROR


def test_aprx_map_via_runner_keeps_error_type_from_subprocess(tmp_path, monkeypatch):
    """子进程回传的「核验报错」要原样带回，不能一律改写成 NeedsArcGIS。"""

    from gis_cli.recipes import store as store_mod

    class _Result:
        ok = True
        result = {
            "ok": False,
            "detail": "核验过程报错: RuntimeError: project is corrupt",
            "error_type": store_mod.ENV_CHECK_ERROR,
        }
        error = None

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            return _Result()

    monkeypatch.setattr(store_mod, "_arcpy_importable", lambda: False)
    lib = store_mod.RecipeLibrary(code_runner=_Runner())
    aprx = tmp_path / "map.aprx"
    aprx.write_text("x", encoding="utf-8")

    result = lib.run_assertions([{"type": "aprx_map_check", "path": str(aprx)}])[0]
    assert result.ok is False
    assert result.error_type == store_mod.ENV_CHECK_ERROR
    assert result.to_dict()["error_type"] == store_mod.ENV_CHECK_ERROR


def test_aprx_map_verdict_open_failure_carries_error_type(tmp_path, monkeypatch):
    """DAIR-73 复查 1：工程打不开也是「核验没做成」，必须带 ``error_type``。

    旧行为：``mp.ArcGISProject()`` 抛异常只回 ``ok=False`` 且没有 ``error_type`` →
    工具层落进「工程核验未通过 + 按提示修正渲染器/配色/布局」并算出 0 分，而
    ``CheckError`` 的 hint 写的又是「工程损坏」——同一个「工程打不开」的事实落在
    两处、提示互相矛盾。
    """
    import sys
    import types

    from gis_cli.recipes import store as store_mod

    def _boom(_path):
        raise RuntimeError("aprx is corrupt")

    fake_arcpy = types.ModuleType("arcpy")
    fake_arcpy.mp = types.SimpleNamespace(ArcGISProject=_boom)
    monkeypatch.setitem(sys.modules, "arcpy", fake_arcpy)

    verdict = store_mod._aprx_map_verdict("map.aprx", {})
    assert verdict["ok"] is False
    assert verdict["error_type"] == store_mod.ENV_CHECK_ERROR
    assert "无法打开工程" in verdict["detail"], verdict["detail"]

    # 就地（有 arcpy）这条路：error_type 要一路带到 ValidationResult / to_dict()
    monkeypatch.setattr(store_mod, "_arcpy_importable", lambda: True)
    aprx = tmp_path / "map.aprx"
    aprx.write_text("x", encoding="utf-8")
    result = store_mod.RecipeLibrary().run_assertions(
        [{"type": "aprx_map_check", "path": str(aprx)}]
    )[0]
    assert result.ok is False
    assert result.error_type == store_mod.ENV_CHECK_ERROR
    assert result.to_dict()["error_type"] == store_mod.ENV_CHECK_ERROR
    assert "无法打开工程" in result.detail, result.detail
