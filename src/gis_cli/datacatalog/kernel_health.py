# -*- coding: utf-8 -*-
"""ArcPy 内核熔断（DAIR-365）：连续失败后退避，别让 10s 轮询反复起整轮。

## 为什么需要

``web/catalog_scan.py`` 把内核失败当成**可恢复**的（不锁存失败指纹，下一拍
自己重试）。可内核**持续**失败（崩了 / 授权丢了 / arcpy 起不来）且还没成功
扫过一次时，前端每 10s 一次的 ``POST /api/catalog/scan`` 就真的每 10s 起一整
轮 ``refresh_catalog``（``discover`` + ``remove_missing`` + 一次
``kernel.execute``），``runs`` 一直涨、日志每 10s 一条。

这一层只做一件事：把「连续几次内核实打实失败」变成一段明确的退避窗口，
窗口内不再起扫，窗口过去自动恢复——既不假装成功，也不永久熔断。

## 边界

- 只统计**内核级**失败（``kernel.execute`` 不 ok / 抛异常）；单文件画像失败
  不算，那种失败下一轮本来就会重试，也不该把内核判成坏的。
- 会自愈：成功一次立刻清零；``force`` 放行一次立刻重试；退避有上限。
- 状态由调用方持有（每个 ``CatalogScanService`` 一份），不做进程级单例：
  测试与多 workspace 互不污染；同一时刻只有一轮扫描在跑，调用方串行使用。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

#: 连续失败到这个次数才进入退避（前两次仍按下一拍重试，保持"可恢复"语义）。
DEFAULT_THRESHOLD = 3

#: 退避间隔（秒）：第 3 次失败 10s，第 4 次 60s，第 5 次起 300s 封顶。
DEFAULT_BACKOFF: tuple[float, ...] = (10.0, 60.0, 300.0)


@dataclass
class KernelHealth:
    """内核健康：连续失败计数 + 一段有上限的退避窗口。"""

    threshold: int = DEFAULT_THRESHOLD
    backoff: tuple[float, ...] = DEFAULT_BACKOFF
    #: 连续失败次数；成功一次归零。
    strikes: int = 0
    #: 下次允许重试的 monotonic 时间；``None`` 表示不挡。
    retry_at: float | None = None
    last_error: str = ""
    #: 时间源（测试注入；默认 monotonic，不受系统时钟改动影响）。
    clock: Callable[[], float] = time.monotonic

    def record_failure(self, error: str, *, now: float | None = None) -> None:
        """记一次内核级失败；达到阈值后按档位开退避窗口。"""
        now = self._now(now)
        self.strikes += 1
        self.last_error = str(error)
        if self.strikes < self.threshold:
            self.retry_at = None
            return
        step = min(self.strikes - self.threshold, len(self.backoff) - 1)
        self.retry_at = now + max(0.0, float(self.backoff[step]))

    def record_success(self) -> None:
        """内核这一轮真的干活了：连续失败与退避一起清零（不留永久熔断）。"""
        self.strikes = 0
        self.retry_at = None
        self.last_error = ""

    def blocked_for(self, *, now: float | None = None) -> float:
        """还要等多少秒才允许再扫；``0`` 表示现在就可以。"""
        if self.retry_at is None:
            return 0.0
        return max(0.0, self.retry_at - self._now(now))

    def allow_retry(self) -> None:
        """``?force=1``：立刻放行一次，但保留 ``strikes``（再失败就按下一档）。"""
        self.retry_at = None

    def message(self, *, now: float | None = None) -> str:
        """给界面看的一句话（退避期间才会用到）。"""
        seconds = max(1, math.ceil(self.blocked_for(now=now)))
        return (
            f"内核连续失败 {self.strikes} 次，{seconds} 秒后重试"
            "（?force=1 立即重试）"
        )

    def _now(self, now: float | None) -> float:
        return self.clock() if now is None else float(now)


__all__ = ["DEFAULT_BACKOFF", "DEFAULT_THRESHOLD", "KernelHealth"]
