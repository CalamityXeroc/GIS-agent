# -*- coding: utf-8 -*-
"""内核子进程 stdout/stderr 改道到日志（DAIR-312）。

背景：arcpy/GDAL 的原生库直接写 fd 1/2，绕过 Python 的 ``sys.stdout``，所以
``runtime/kernel.py`` 里 ``_strip_arcgis_noise`` 那条 IOPub 过滤拦不住。
实测（``运行示例.txt``）这些 warning 会直接打进聊天输入框，把框和状态行拆成
碎片：

    │  ›  WARNING: Unable to load code page translation table …  │
    WARNING: … ──────────────────────────────────────────────╯

所以内核启动时把子进程的 stdout/stderr 接到 ``.gis_agent/logs/kernel.log``：
噪声不进终端，事后还能查。
"""

from __future__ import annotations

from pathlib import Path

from gis_cli.runtime.kernel import ArcpyKernel


def _kernel(workspace: Path) -> ArcpyKernel:
    kernel = ArcpyKernel.__new__(ArcpyKernel)
    kernel.workspace = str(workspace)
    kernel.state_dir = workspace / "state"
    kernel._child_log = None
    return kernel


def test_child_stream_writes_into_workspace_log(tmp_path):
    kernel = _kernel(tmp_path)
    kwargs = kernel._child_stream_kwargs()
    assert set(kwargs) == {"stdout", "stderr"}
    assert kwargs["stdout"] is kwargs["stderr"], "一个句柄：两路输出不乱序"
    kwargs["stdout"].write(b"WARNING: Unable to load code page translation table\n")
    kernel._child_log.close()

    log = tmp_path / ".gis_agent" / "logs" / "kernel.log"
    assert log.is_file()
    assert "Unable to load code page" in log.read_text(encoding="utf-8", errors="replace")


def test_child_stream_is_reused_across_kernel_restarts(tmp_path):
    """restart_kernel 会重放 _launch_args，句柄必须还是原来那个。"""
    kernel = _kernel(tmp_path)
    first = kernel._child_stream_kwargs()["stdout"]
    assert kernel._child_stream_kwargs()["stdout"] is first
    kernel._child_log.close()


def test_child_stream_appends_instead_of_truncating_after_shutdown(tmp_path):
    """重开日志不能把上一次的内容截掉（DAIR-316）。

    ``shutdown()`` 会关句柄并置 None，启动失败重试 / ``restart()`` 回退都会再开
    一次——用 ``"wb"`` 恰好把上一次的 ``kernel.log``（最需要它的时候）清空。
    """
    kernel = _kernel(tmp_path)
    kernel._child_stream_kwargs()["stdout"].write(b"first attempt\n")
    # shutdown() 对日志做的全部事情：关句柄 + 置 None（其余字段本测试不涉及）
    kernel._child_log.close()
    kernel._child_log = None

    kernel._child_stream_kwargs()["stdout"].write(b"second attempt\n")
    kernel._child_log.close()

    body = (tmp_path / ".gis_agent" / "logs" / "kernel.log").read_text(encoding="utf-8", errors="replace")
    assert "first attempt" in body, "重开日志把上一次的内容截掉了"
    assert "second attempt" in body


def test_child_stream_degrades_when_log_cannot_be_opened(tmp_path, monkeypatch):
    """日志写不了不能连带内核起不动：退化为旧行为（继承控制台）。"""

    def _boom(*_args, **_kwargs):
        raise OSError("read-only dir")

    monkeypatch.setattr(Path, "mkdir", _boom)
    kernel = _kernel(tmp_path / "nowhere")
    assert kernel._child_stream_kwargs() == {}
    assert kernel._child_log is None
