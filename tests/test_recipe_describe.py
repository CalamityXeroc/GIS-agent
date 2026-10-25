# -*- coding: utf-8 -*-
"""Describe 否定结论的复核：同会话 arcpy 状态脏了不能直接判"产出缺失"。

实测：``real_{a,b}.gdb/probe_pts`` 在同一个内核会话里 ``arcpy.Exists``
恒 False（先删假库、再建真库留下的状态残留；hygiene 也判 ``real_*.gdb: 0MB/0项``），
换一个全新的 Pro 进程 exists=True、count=3。旧实现只有"会话窗口"这一路判定，
``_describe_in_process`` 在 venv 里又没有 arcpy（必然 None），于是 4 条断言假阴性、
整轮探针被判 7/11 —— 假阴性比假阳性更危险：它把"数据其实好了"的 run 标成 failed。

修法：会话内判"不存在"之后，再换一路真 arcpy（Pro python 子进程）复核，
只有复核也拿不到结论才维持原判。肯定分支不付这趟开销。
"""

from __future__ import annotations

from types import SimpleNamespace

from gis_cli.recipes.store import RecipeLibrary

_FC = r"E:\ws\output\real_a.gdb\probe_pts"


def _feature_class(tmp_path) -> str:
    """真库目录存在、要素类只是"不在会话视野里"的那种路径。"""
    gdb = tmp_path / "output" / "real_a.gdb"
    gdb.mkdir(parents=True)
    return str(gdb / "probe_pts")


class _SessionRunner:
    """单一会话窗口的内核桩：对外一律报同一个 ``exists``。"""

    def __init__(self, exists: bool):
        self.exists = exists
        self.codes: list[str] = []

    def run(self, code, timeout=None, workspace=None):
        self.codes.append(code)
        return SimpleNamespace(
            ok=True, stdout="", stderr="", error=None, display_images=[],
            result={
                "exists": self.exists,
                "feature_count": 3 if self.exists else None,
                "fields": [],
            },
        )


class _DeadRunner:
    """内核已经不可用（ok=False）的桩。"""

    def run(self, code, timeout=None, workspace=None):
        return SimpleNamespace(
            ok=False, stdout="", stderr="", error={"message": "kernel dead"},
            display_images=[], result=None,
        )


def _stub_pro(monkeypatch, data, status: str = "success") -> list[str]:
    calls: list[str] = []

    def fake_run_arcpy_code(code, **kwargs):
        calls.append(code)
        return SimpleNamespace(
            status=status,
            data=data if status == "success" else None,
            error=None if status == "success" else {"message": "no ArcGIS Pro"},
            stdout="",
            stderr="",
        )

    import gis_cli.arcpy_bridge as bridge

    monkeypatch.setattr(bridge, "run_arcpy_code", fake_run_arcpy_code)
    return calls


def test_describe_rechecks_with_real_arcpy_when_session_says_missing(monkeypatch, tmp_path):
    calls = _stub_pro(monkeypatch, {"exists": True, "feature_count": 3})
    lib = RecipeLibrary(code_runner=_SessionRunner(exists=False))

    record = lib._describe(_feature_class(tmp_path))

    assert record is not None and record["exists"] is True
    assert record["feature_count"] == 3
    assert calls, "会话说没有时必须换一路真 arcpy 复核，不能直接判不存在"


def test_describe_does_not_pay_subprocess_on_positive_path(monkeypatch, tmp_path):
    calls = _stub_pro(monkeypatch, {"exists": True})
    lib = RecipeLibrary(code_runner=_SessionRunner(exists=True))

    assert lib._describe(_feature_class(tmp_path))["exists"] is True
    assert calls == [], "会话能看到数据时不该为复核多花 15s"


def test_describe_stays_none_when_recheck_unavailable(monkeypatch, tmp_path):
    calls = _stub_pro(monkeypatch, None, status="error")
    lib = RecipeLibrary(code_runner=_DeadRunner())

    assert lib._describe(_feature_class(tmp_path)) is None
    assert len(calls) == 1, "内核不可用时也要复核一次再下结论"


def test_assertions_stop_false_negatives_from_a_dirty_session(monkeypatch, tmp_path):
    _stub_pro(monkeypatch, {"exists": True, "feature_count": 3})
    lib = RecipeLibrary(code_runner=_SessionRunner(exists=False))
    target = _feature_class(tmp_path)

    results = lib.run_assertions(
        [
            {"type": "vector_exists", "path": target},
            {"type": "feature_count_between", "path": target, "value": [1, 10]},
        ]
    )

    assert [r.ok for r in results] == [True, True], [r.detail for r in results]


def test_describe_skips_recheck_when_parent_directory_is_missing(monkeypatch):
    """父目录都没有就不必复核：数据不可能存在，脏会话也只能这么判。"""
    calls = _stub_pro(monkeypatch, {"exists": True, "feature_count": 3})
    lib = RecipeLibrary(code_runner=_SessionRunner(exists=False))

    record = lib._describe(r"E:\ws\output\nope.gdb\probe_pts")

    assert record is not None and record["exists"] is False
    assert calls == [], "真不存在的产物不该再花一趟子进程"
