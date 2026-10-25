# -*- coding: utf-8 -*-
"""看门狗硬杀内核（不需要 ArcGIS）。

实测 2026-10-22 组合任务：父会话一次 `KernelDensity` 卡进 600s 执行超时，
日志只留一行 `kernel hard kill failed: 'KernelManager' object has no attribute 'kill'`
——`jupyter_client` 的 `KernelManager` 根本没有 `kill()`，硬杀从来没成功过：
卡死的 ArcGIS 内核一直活着攥着 .tif 句柄，下一次执行直接吃 PermissionError。
"""

from __future__ import annotations

import threading

import pytest

from gis_cli.runtime.kernel import ArcpyKernel


class _Channels:
    def __init__(self) -> None:
        self.stopped = False

    def stop_channels(self) -> None:
        self.stopped = True


def _kernel(km) -> ArcpyKernel:
    kernel = ArcpyKernel.__new__(ArcpyKernel)
    kernel._km = km
    kernel._kc = _Channels()
    kernel._started = True
    kernel._exec_lock = threading.RLock()
    return kernel


def test_hard_kill_uses_shutdown_kernel_now():
    """正常路径走 ``shutdown_kernel(now=True)``——那是 KernelManager 真有的方法。"""

    class Km:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def shutdown_kernel(self, now=False, restart=False):
            self.calls.append(now)

        def kill(self):  # pragma: no cover - 存在就该被忽略（它不在 jupyter_client 里）
            pytest.fail("KernelManager 没有 kill()，不该调它")

    km = Km()
    _kernel(km)._hard_kill(km)
    assert km.calls == [True]


def test_hard_kill_falls_back_to_private_kill():
    """``shutdown_kernel`` 抛异常时退回私有的 ``_kill_kernel()``。"""

    class Km:
        def __init__(self) -> None:
            self.killed = 0

        def shutdown_kernel(self, now=False, restart=False):
            raise RuntimeError("provisioner gone")

        def _kill_kernel(self):
            self.killed += 1

    km = Km()
    _kernel(km)._hard_kill(km)
    assert km.killed == 1


def test_destroy_kernel_clears_handles_so_start_does_not_orphan():
    """杀掉后必须清 ``_km``/``_kc``：只置 _started=False，下一次 start() 会 new 一个
    KernelManager，旧进程和旧 ZMQ 通道就成了没人收的僵尸。"""

    class Km:
        def __init__(self) -> None:
            self.shutdown = 0

        def shutdown_kernel(self, now=False, restart=False):
            self.shutdown += 1

    km = Km()
    kernel = _kernel(km)
    channels = kernel._kc
    kernel._destroy_kernel()

    assert km.shutdown == 1
    assert channels.stopped is True
    assert kernel._km is None and kernel._kc is None
    assert kernel._started is False
