# -*- coding: utf-8 -*-
"""内核往返合并（P1-6）：取结果并入同一次执行。

以前 ``execute()`` = 跑用户代码（1 趟 ZMQ 往返）+ ``_fetch_result()``（第 2 趟）。
每趟都要“发消息 → 收 iopub → 等 idle → 抽 shell 回复”，固定开销明显。
现在把“把结果打成一行哨兵”拼在用户代码后面，从同一条 stdout 里剖出来。
"""

from __future__ import annotations

import threading

import pytest

from gis_cli.runtime.kernel import (
    _RESULT_MARKER,
    _RESULT_MISSING,
    ArcpyKernel,
    ExecResult,
    _split_result,
)


def _fake_kernel(scripted):
    """造一个不连真内核的 ArcpyKernel（沿用已有测试的 __new__ 手法）。"""
    kernel = ArcpyKernel.__new__(ArcpyKernel)
    kernel.workspace = None
    kernel._exec_lock = threading.RLock()
    kernel._started = True
    kernel._exec_count = 0
    calls: list[str] = []

    def fake_raw(code, *, timeout):
        calls.append(code)
        return scripted(code)

    kernel._execute_raw = fake_raw
    kernel.is_alive = lambda: True
    kernel.start = lambda *a, **k: None
    kernel._fetch_result = lambda: pytest.fail("不该走兜底的两趟路径")
    return kernel, calls


# ------------------------------------------------------------------ 纯函数
def test_split_result_extracts_and_strips():
    out = "普通输出\n" + _RESULT_MARKER + '{"n": 2}\n'
    result, rest = _split_result(out)
    assert result == {"n": 2}
    assert rest == "普通输出\n", "结果行必须从 stdout 里去掉（不能漏给模型看）"


def test_split_result_none_vs_missing():
    """``set_result(None)`` 与“压根没调用 set_result”要能区分。"""
    assert _split_result(_RESULT_MARKER + "null\n")[0] is None
    assert _split_result("只有普通输出")[0] is _RESULT_MISSING


def test_split_result_takes_last_marker():
    """用户自己 print 里恰好含同样字样时，取最后一次（那才是真结果）。"""
    out = _RESULT_MARKER + '"用户伪造"\n' + _RESULT_MARKER + '"真结果"\n'
    assert _split_result(out)[0] == "真结果"


def test_split_result_keeps_stdout_after_result_line():
    out = _RESULT_MARKER + "1\n尾巴输出\n"
    result, rest = _split_result(out)
    assert result == 1
    assert rest == "尾巴输出\n"


# ------------------------------------------------------------ execute 行为
def test_execute_uses_single_roundtrip():
    """核心断言：一次 execute 只发一条消息（不再有第二次取结果）。"""
    kernel, calls = _fake_kernel(
        lambda code: ExecResult(ok=True, stdout="结果前的输出\n" + _RESULT_MARKER + '{"k": 1}\n')
    )
    outcome = kernel.execute("set_result({'k': 1})")
    assert len(calls) == 1, f"应只执行一趟，实际 {len(calls)} 趟"
    assert outcome.result == {"k": 1}
    # 用户代码与取结果尾巴拼在同一条消息里
    assert "set_result({'k': 1})" in calls[0]
    assert _RESULT_MARKER in calls[0]


def test_execute_falls_back_when_marker_missing():
    """哨兵没出现（用户重定向了 stdout 等）→ 退回旧的两趟路径，行为不变。"""
    kernel, calls = _fake_kernel(lambda code: ExecResult(ok=True, stdout="没有哨兵"))
    kernel._fetch_result = lambda: {"fallback": True}
    outcome = kernel.execute("print('x')")
    assert outcome.result == {"fallback": True}
    assert len(calls) == 1  # 兜底那趟走的是 _fetch_result（已被替换）


def test_execute_does_not_fetch_result_on_error():
    """执行失败时不该再去取结果（旧行为也不取）。"""
    kernel, calls = _fake_kernel(
        lambda code: ExecResult(ok=False, stdout="", error={"type": "E", "message": "boom"})
    )
    outcome = kernel.execute("raise ValueError('boom')")
    assert outcome.ok is False
    assert outcome.result is None
    assert len(calls) == 1


def test_execute_preserves_workspace_prefix():
    """workspace 前缀仍要注入（arcpy.env.workspace / overwriteOutput）。"""
    kernel, calls = _fake_kernel(
        lambda code: ExecResult(ok=True, stdout=_RESULT_MARKER + "null\n")
    )
    kernel.execute("pass", workspace=r"C:\ws")
    assert "arcpy.env.workspace" in calls[0]
    assert "overwriteOutput = True" in calls[0]


def test_execute_returns_plain_stdout_without_result_line():
    """结果行不能混进 outcome.stdout（否则会当成普通输出喂给模型）。"""
    kernel, _ = _fake_kernel(
        lambda code: ExecResult(ok=True, stdout="真正有用的输出\n" + _RESULT_MARKER + '{"a": 1}\n')
    )
    outcome = kernel.execute("pass")
    assert outcome.stdout == "真正有用的输出\n"
    assert _RESULT_MARKER not in outcome.stdout


def test_execute_resets_result_before_running():
    """**关键**：本次没调 set_result 时必须返回 None，而不是上一步的旧值。

    实测踩到：``execute("x = 1")`` 返回了上一次 ``set_result([1,2,3])`` 的结果。
    结果变量是持久命名空间里的全局量，不清就会静默串值 —— 模型会拿着
    过期数据继续干活，而且看起来"一切正常"。
    """
    from gis_cli.runtime.kernel import _RESULT_HEAD

    kernel, calls = _fake_kernel(
        lambda code: ExecResult(ok=True, stdout=_RESULT_MARKER + "null\n")
    )
    outcome = kernel.execute("x = 1")
    assert outcome.result is None
    # 清结果的头必须排在用户代码之前
    head_at = calls[0].find(_RESULT_HEAD)
    code_at = calls[0].find("x = 1")
    assert head_at >= 0, "缺少清结果的头"
    assert head_at < code_at, "清结果必须在用户代码之前执行"


def test_result_tail_clears_after_reading():
    """读完结果后要把变量置 None（下一次执行的兜底）。"""
    from gis_cli.runtime.kernel import _RESULT_TAIL, _RESULT_VAR

    assert f"globals()['{_RESULT_VAR}'] = None" in _RESULT_TAIL
