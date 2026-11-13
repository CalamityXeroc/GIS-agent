# -*- coding: utf-8 -*-
"""内核执行层的 arcpy.env 按次隔离（DAIR-289）。

背景：配方各自用 ``try/finally`` 还原自己改过的 env，但 11 个内置配方的**第 1 步**
（``CopyFeatures``）在各自 ``try`` 之前，跑前 env 里留着别的任务的 ``extent`` /
``outputCoordinateSystem`` 时，第 1 步就会被静默重投影、按范围裁空，最后报一个
与真因无关的错（DAIR-288：``zone_raster_sum`` 落到 ``np.bincount`` 的
``zero-size array``）。

这些用例**真跑** ``execute()`` 拼出来的那段代码（不是断言字符串在不在）：
``_execute_raw`` 换成"compile + exec 到同一个持久命名空间"，配一个假 arcpy，
里面的 ``CopyFeatures`` 按真 arcpy 的脾气办事 —— ``env.extent`` 非 None 时
把范围外的要素裁掉（返回 0 行）。于是：

- 不放快照/基线 → 第 1 步拿到 0 行（红）；
- 不放还原 → 跑完 env 停在 None（红）。

``test_removing_restore_goes_red`` 是负向对照：把还原挪掉后能观察到残留（= 上面那些
断言的依据）；实测在 ``kernel.py`` 里只删掉还原语句时，两个 ``test_env_restored_*`` 会红。
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import types

import pytest

from gis_cli.runtime import env_guard
from gis_cli.runtime.kernel import ArcpyKernel, ExecResult

#: 预置的"上一个任务留下的" env：另一套 CRS + 不相交的范围（就是 DAIR-288 的现场）
PRESET = {
    "extent": "PRESET_EXTENT",
    "snapRaster": "PRESET_SNAP",
    "outputCoordinateSystem": "PRESET_CS",
    "cellSize": 30,
    "mask": "PRESET_MASK",
    "overwriteOutput": False,
}


class _FakeEnv:
    """arcpy.env 的替身：一堆可读可写的属性（真 arcpy 就是这德行）。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in env_guard.GUARDED_ENV_KEYS}


def _fake_arcpy(env: _FakeEnv) -> types.ModuleType:
    mod = types.ModuleType("arcpy")
    mod.env = env
    mod.copied: list[int] = []

    def _copy_features(src, out):
        # 真 arcpy 的脾气：env.extent 非 None 就按那个范围裁，范围外全丢（0 行）。
        mod.copied.append(0 if env.extent is not None else 1)
        return out

    mod.management = types.SimpleNamespace(CopyFeatures=_copy_features)
    return mod


def _kernel(monkeypatch, env: _FakeEnv):
    """一个不连真内核的 ArcpyKernel：``_execute_raw`` 真跑拼出来的代码。"""
    mod = _fake_arcpy(env)
    monkeypatch.setitem(sys.modules, "arcpy", mod)

    kernel = ArcpyKernel.__new__(ArcpyKernel)
    kernel.workspace = None
    kernel._exec_lock = threading.RLock()
    kernel._started = True
    kernel._exec_count = 0
    kernel.watchdog_grace = 45.0
    kernel.is_alive = lambda: True
    kernel.start = lambda *a, **k: None
    kernel._fetch_result = lambda: None
    #: 持久命名空间：内核里 globals() 跨执行存活，快照变量就住在这儿。
    #: ``set_result`` 与结果变量由 prelude 定义（这里手抄一份，别让它缺席）
    namespace: dict = {
        "_cwd_launch": os.getcwd(),
        "__arcpy_engine_result__": None,
    }
    namespace["set_result"] = lambda value: namespace.update({"__arcpy_engine_result__": value})
    calls: list[str] = []

    def fake_raw(code, *, timeout):
        calls.append(code)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<combined>", "exec"), namespace)
            return ExecResult(ok=True, stdout=buf.getvalue())
        except Exception as exc:  # 用户代码抛异常 = cell 中断，尾巴不再执行
            return ExecResult(
                ok=False,
                stdout=buf.getvalue(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )

    kernel._execute_raw = fake_raw
    return kernel, calls, mod


CODE = (
    "arcpy.management.CopyFeatures('zone.shp', 'zz_zone_int_tmp')\n"
    "set_result({'copied': list(arcpy.copied),\n"
    "            'seen': {k: getattr(arcpy.env, k) for k in " + repr(env_guard.GUARDED_ENV_KEYS) + "}})\n"
)


def test_guard_scope_is_the_six_env_keys():
    """隔离范围锁死：本单点名的 6 项都要在，不多不少（第 7 项要单开）。"""
    assert set(env_guard.GUARDED_ENV_KEYS) == {
        "extent",
        "snapRaster",
        "outputCoordinateSystem",
        "cellSize",
        "mask",
        "overwriteOutput",
    }


def test_first_call_is_not_poisoned_by_preset_env(monkeypatch):
    """第 1 步 ``CopyFeatures`` 不许被跑前 env 带偏（DAIR-288 的实质）。

    预置 extent/CRS 后跑同一段代码：真 arcpy 下这一步会静默重投影 + 裁空
    （假 arcpy 用 ``copied == [0]`` 表达），修好后必须是 ``[1]``。
    """
    env = _FakeEnv(**PRESET)
    kernel, calls, mod = _kernel(monkeypatch, env)

    outcome = kernel.execute(CODE)

    assert outcome.ok, outcome.error
    assert mod.copied == [1], "第 1 步拿到 0 行 = env 还没被盖成干净基线"
    # 执行期间：5 项归零（不继承外部格网/范围/CRS/掩膜），overwriteOutput 仍为 True
    assert outcome.result["seen"] == {
        "extent": None,
        "snapRaster": None,
        "outputCoordinateSystem": None,
        "cellSize": None,
        "mask": None,
        "overwriteOutput": True,
    }
    assert len(calls) == 1, "成功路径不该多跑一趟（还原拼在同一趟里）"


def test_env_restored_after_success(monkeypatch):
    """跑完 env 逐字段回到跑前值（不是一律清空）；overwriteOutput 也回原值。"""
    env = _FakeEnv(**PRESET)
    kernel, _, _ = _kernel(monkeypatch, env)

    assert kernel.execute(CODE).ok
    assert env.as_dict() == PRESET, "跑完 env 没还原成预置值"


def test_env_restored_after_failure(monkeypatch):
    """用户代码抛异常 → cell 中断，还原尾巴没跑到，错误路径必须补一趟。"""
    env = _FakeEnv(**PRESET)
    kernel, calls, _ = _kernel(monkeypatch, env)

    outcome = kernel.execute("raise RuntimeError('boom')")

    assert outcome.ok is False
    assert outcome.error["type"] == "RuntimeError"
    assert len(calls) == 2, "失败路径要多一趟归位"
    assert env.as_dict() == PRESET, "失败后 env 没还原"


def test_removing_restore_goes_red(monkeypatch):
    """负向对照：把还原挪掉，上面的观察必须变红（证明用例不是空转）。

    没有还原时：跑完 env 停在被盖过的 None 上（``test_env_restored_after_success``
    会红）；而这恰恰是"下一个任务接着被带偏"的那条链。
    """
    monkeypatch.setattr(env_guard, "restore_code", lambda: "")
    env = _FakeEnv(**PRESET)
    kernel, _, mod = _kernel(monkeypatch, env)

    assert kernel.execute(CODE).ok
    assert mod.copied == [1], "基线仍在（这一步不是被移除的对象）"
    residue = env.as_dict()
    assert residue["extent"] is None and residue["outputCoordinateSystem"] is None
    assert residue != PRESET, "还原都挪掉了还等于预置值 —— 说明上面的断言空转"
