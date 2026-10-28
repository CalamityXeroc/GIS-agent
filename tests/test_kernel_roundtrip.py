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
        lambda code: ExecResult(ok=True, stdout=_RESULT_MARKER + "null" + chr(10) + "n"),
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
        lambda code: ExecResult(ok=True, stdout=_RESULT_MARKER + "null" + chr(10) + "n"),
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


def test_execute_chdirs_to_workspace_and_restores(tmp_path, monkeypatch):
    """CWD 跟着工作区走，且**无论成败**都回到启动目录（DAIR-12/13）。

    真执行拼装出来的语句（不只断言字符串在不在）：

    - 正常跑完 → 用户代码期间 cwd = 工作区，归位语句后 = 启动目录；
    - 用户代码抛异常 → cell 中断，归位语句后仍回启动目录（execute 的错误路径
      会补一趟 _chdir_back_stmt()，这里用同一条语句验证）；
    - _cwd_launch 在 prelude 里只捕获一次，连续两次失败也不会把恢复目标挪走。
    """
    import os
    import sys
    import types

    from gis_cli.runtime.kernel import _chdir_back_stmt, _cwd_prefix_statements

    launch = os.getcwd()
    ws1 = tmp_path / "ws1"
    ws1.mkdir()
    ws2 = tmp_path / "ws2"
    ws2.mkdir()

    # 假 arcpy：组合代码里有 import arcpy / env.workspace
    fake = types.ModuleType("arcpy")
    fake.env = types.SimpleNamespace(workspace=None, overwriteOutput=None)
    monkeypatch.setitem(sys.modules, "arcpy", fake)

    def run_combined(combined: str):
        g: dict = {"_cwd_launch": launch}
        try:
            exec(compile(combined, "<combined>", "exec"), g)
            error = None
        except Exception as exc:  # 模拟 execute 的错误路径：cell 中断后补归位
            error = exc
            exec(compile("import os as _os" + chr(10) + _chdir_back_stmt(), "<restore>", "exec"), g)
        return error

    safe1 = str(ws1).replace(chr(92), chr(92) * 2)
    combined_ok = "import os as _os" + chr(10) + _cwd_prefix_statements(safe1) + "x = 1"
    assert run_combined(combined_ok) is None
    assert os.getcwd() == str(ws1), "执行期间 cwd = 本次工作区（成功路径不需要额外归位）"

    safe2 = str(ws2).replace(chr(92), chr(92) * 2)
    combined_boom = (
        "import os as _os" + chr(10) + _cwd_prefix_statements(safe2)
        + "raise RuntimeError('boom')"
    )
    assert run_combined(combined_boom) is not None, "用户代码的异常要如实抛出"
    assert os.getcwd() == launch, "异常路径也要回启动目录"

    # 无漂移：_cwd_launch 是常量，第二个工作区照样能进
    error2 = run_combined(combined_ok.replace(safe1, safe2))
    assert error2 is None and os.getcwd() == str(ws2), "第二个工作区照样进得去（无漂移）"

    os.chdir(launch)  # 别把 pytest 留在 tmp 里


def test_failed_execute_sends_cwd_repair(tmp_path):
    """用户代码抛异常会让 cell 中断，cwd 停在工作区 —— 错误路径必须补一趟归位（DAIR-13）。

    以前这个归位语句拼在用户代码后面，cell 一中断它根本不执行；
    而且每次 execute 都 getcwd() 当恢复目标，失败一次就永久挪一格。
    """
    kernel, calls = _fake_kernel(lambda code: ExecResult(ok=False, error={"type": "RuntimeError"}))
    ws = tmp_path / "ws"
    ws.mkdir()
    kernel.execute("raise RuntimeError('boom')", workspace=str(ws))

    assert len(calls) == 2, f"错误路径要多一趟归位，实际 {len(calls)} 趟"
    assert "_os.chdir(_cwd_launch)" in calls[1]
    # 漂移修复：拼装里不再有“每次 execute 都 getcwd 当恢复目标”
    assert "_cwd_restore = _os.getcwd()" not in calls[0]
    assert "_cwd_launch" in calls[0]
