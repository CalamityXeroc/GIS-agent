# -*- coding: utf-8 -*-
"""覆盖写入的备份（P3）：``overwriteOutput = True`` 下不能静默毁数据。

**实测的缺口**：内核 prelude 里 ``arcpy.env.overwriteOutput = True`` 是全局
开着的，而 ``looks_destructive()`` 只认 Delete/rmtree/remove 这类显式删除。
于是"正常写产出"的代码（FeatureClassToFeatureClass / Buffer / RasterTo…）
在目标已存在时会**静默覆盖原数据，且不触发任何备份**：

    looks_destructive('arcpy.analysis.Buffer(fc, out, "100 Meters")') == False

用户已有的结果库被模型一次"重跑"就没了，且没有任何回滚点。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gis_cli.runtime.code_runner import backup_targets, looks_destructive


# ------------------------------------------------------------------ 目标识别
def test_detects_existing_output_target_of_write_tool(tmp_path: Path):
    """写工具的输出参数指向已存在的数据 → 必须被识别为"会覆盖"。"""
    gdb = tmp_path / "output" / "result_data.gdb"
    gdb.mkdir(parents=True)
    existing = gdb / "study_area"
    existing.write_text("原有数据", encoding="utf-8")

    code = (
        "import arcpy\n"
        f'arcpy.management.FeatureClassToFeatureClass(src, r"{gdb}", "study_area")\n'
    )
    targets = backup_targets(code, tmp_path)
    assert str(existing) in targets, f"没识别出会被覆盖的产物: {targets}"


def test_absolute_output_path_is_detected(tmp_path: Path):
    """输出参数直接给绝对路径（最常见的写法）也要识别。"""
    target = tmp_path / "output" / "buf.shp"
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")
    code = f'arcpy.analysis.Buffer(fc, r"{target}", "100 Meters")\n'
    assert str(target) in backup_targets(code, tmp_path)


def test_nonexistent_target_is_not_a_backup_candidate(tmp_path: Path):
    """目标不存在 → 新建，没有覆盖风险，不该浪费磁盘去备份。"""
    code = f'arcpy.analysis.Buffer(fc, r"{tmp_path / "output" / "new.shp"}", "100")'
    assert backup_targets(code, tmp_path) == []


def test_readonly_input_dirs_are_skipped(tmp_path: Path):
    """input/ 是只读数据源，不该被当成"要备份的产出"。"""
    src = tmp_path / "input" / "roads.shp"
    src.parent.mkdir(parents=True)
    src.write_text("x", encoding="utf-8")
    code = f'arcpy.analysis.Buffer(r"{src}", out, "100")'
    assert backup_targets(code, tmp_path) == []


def test_unchanged_destructive_detection_still_works():
    """原有的显式删除识别不能退化。"""
    assert looks_destructive('arcpy.management.Delete(r"C:/x.gdb/fc")')
    assert looks_destructive("shutil.rmtree(d)")
    assert looks_destructive("os.remove(p)")
    assert not looks_destructive("arcpy.analysis.Buffer(a, b, '100')")


def test_delete_call_target_is_backup_candidate(tmp_path: Path):
    """Delete 的目标即使不是"输出参数位置"，也要能备份（原有行为）。"""
    target = tmp_path / "output" / "old.shp"
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")
    assert str(target) in backup_targets(f'arcpy.management.Delete(r"{target}")', tmp_path)


def test_distance_argument_is_not_mistaken_for_output_name(tmp_path: Path):
    """回归：``Buffer(in, out, "100 Meters")`` 里的距离不能当成输出名。

    实测 bug：盲目把输出参数后面那个字符串拼成 ``out/100 Meters``，于是
    真正的输出目标漏备份（Buffer 是最常用的分析工具之一）。
    """
    target = tmp_path / "output" / "study_area.shp"
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")
    code = f'arcpy.analysis.Buffer(src, r"{target}", "100 Meters")'
    found = backup_targets(code, tmp_path)
    assert str(target) in found
    assert not any("100 Meters" in item for item in found), found


def test_new_output_name_does_not_backup_whole_container(tmp_path: Path):
    """输出到新名字 → 无覆盖风险，不该把整个 GDB 备份一遍。

    实测 bug：只要容器库存在就备份整个库，一次误备份 30 MB（库里几十个
    要素类全被复制）。
    """
    gdb = tmp_path / "output" / "result_data.gdb"
    gdb.mkdir(parents=True)
    code = f'arcpy.management.FeatureClassToFeatureClass(src, r"{gdb}", "brand_new")'
    assert backup_targets(code, tmp_path) == []


def test_three_position_tool_clip(tmp_path: Path):
    """三参数写工具（Clip(in, clip, out)）的输出位置也要对。"""
    gdb = tmp_path / "output" / "x.gdb"
    gdb.mkdir(parents=True)
    existing = gdb / "clipped"
    existing.write_text("x", encoding="utf-8")
    code = f'arcpy.analysis.Clip(src, clip, r"{gdb}", "clipped")'
    assert str(existing) in backup_targets(code, tmp_path)


def test_looks_like_name_filters_non_names():
    """名字判据：排除距离/单位/占位符/路径/超长串。"""
    from gis_cli.runtime.code_runner import _looks_like_name

    assert _looks_like_name("study_area")
    assert _looks_like_name("耕地面积")
    for bad in ("100 Meters", "5 Kilometers", "#", "a/b", "c:\\x", ""):
        assert not _looks_like_name(bad), bad


def test_gdb_member_uses_arcpy_backup_path(tmp_path: Path):
    """GDB 内部要素类不能用 copytree，必须走 arcpy 导出（有内核时）。"""
    from gis_cli.runtime.code_runner import _is_inside_gdb

    assert _is_inside_gdb(Path("x/output/result_data.gdb/study_area"))
    assert _is_inside_gdb(Path("x/output/a.gdb/tbl"))
    assert not _is_inside_gdb(Path("x/output/result_data.gdb"))
    assert not _is_inside_gdb(Path("x/output/buf.shp"))


def test_backup_paths_returns_empty_without_arcpy_for_gdb_member(tmp_path: Path, monkeypatch):
    """没有 arcpy 时 GDB 成员备份应安全返回空（不抛异常、不静默丢数据）。

    真实运行环境里 GDB 成员走内核（有 arcpy）；这条只保证降级路径不炸。
    """
    import builtins

    from gis_cli.runtime.code_runner import backup_paths

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "arcpy":
            raise ImportError("no arcpy in venv")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    gdb = tmp_path / "output" / "r.gdb"
    gdb.mkdir(parents=True)
    created = backup_paths([str(gdb / "study_area")], tmp_path / "backups")
    assert created == []
