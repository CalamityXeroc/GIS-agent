# -*- coding: utf-8 -*-
"""计划卡闸门（web 版）：``threading.Event`` 等浏览器决策。

与 ``ui.plan_card.PlanGate`` 同接口（``interactive`` / ``ask(card)``），引擎不需要
知道对面是终端还是浏览器：

- 引擎发完 ``plan_card`` 事件后调 :meth:`ThreadingPlanGate.ask`，本闸门挂起工作线程；
- ``POST /api/runs/{id}/plan`` 调 :meth:`decide` 唤醒，返回 ``start`` / ``revise`` /
  ``cancel``（与 ``ui.plan_card.PlanDecision`` 字段一致，引擎只读 ``action``/``text``）；
- 超时（默认 300s，可配）自动 ``start``，浏览器关掉了也不会把队列永久占住。

``pauses = True`` 让引擎把 ``plan_pending`` 写进状态与 trace —— 前端刷新后能凭
``GET /api/runs/{id}`` 的 ``plan`` 字段恢复这张卡。

本模块不 import ``gis_cli.ui``（它会在包初始化时连带 import rich 渲染层）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

#: 闸门认的动作词（与 ``ui.plan_card.parse_decision`` 的返回值一致）。
ACTIONS = ("start", "revise", "cancel")

#: 默认等待时间（秒）：没人决策就自动开始。
DEFAULT_TIMEOUT = 300.0


@dataclass
class PlanDecision:
    """与 ``ui.plan_card.PlanDecision`` 同构；引擎只按属性名读取。"""

    action: str = "start"
    text: str = ""


class ThreadingPlanGate:
    """一个 run 一个闸门实例；可被 ``ask`` 多次调用（改一版再确认一次）。"""

    #: 引擎据此把计划卡渲染成"可按键"的底栏（显示 s/e/q 提示）。
    interactive = True
    #: 引擎据此停在 ``plan_pending`` 状态并写 trace。
    pauses = True

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        on_pending: Callable[[Any], None] | None = None,
        on_resolved: Callable[[PlanDecision, bool], None] | None = None,
    ) -> None:
        self.timeout = max(0.0, float(timeout))
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._decision: PlanDecision | None = None
        self._pending: Any = None
        self._pending_since: float | None = None
        self._on_pending = on_pending
        self._on_resolved = on_resolved

    # ------------------------------------------------------------- 引擎侧
    def ask(self, card: Any) -> PlanDecision:
        with self._lock:
            self._decision = None
            self._event.clear()
            self._pending = card
            self._pending_since = time.monotonic()
        self._notify(self._on_pending, card)

        got = self._event.wait(self.timeout)

        with self._lock:
            decision = self._decision
            self._pending = None
            self._pending_since = None
        timed_out = not got or decision is None
        if decision is None:
            decision = PlanDecision("start")
        self._notify(self._on_resolved, decision, timed_out)
        return decision

    # ------------------------------------------------------------- HTTP 侧
    def decide(self, action: str, text: str = "") -> bool:
        """投递一次决策；没有等待中的卡片（或已经决策过）返回 ``False``。"""
        normalized = str(action or "").strip().lower()
        if normalized not in ACTIONS:
            raise ValueError(f"未知的计划动作: {action}")
        with self._lock:
            if self._pending is None or self._event.is_set():
                return False
            self._decision = PlanDecision(normalized, str(text or "").strip())
        self._event.set()
        return True

    def pending_card(self) -> Any:
        with self._lock:
            return self._pending

    @property
    def waiting(self) -> bool:
        with self._lock:
            return self._pending is not None and not self._event.is_set()

    @property
    def deadline(self) -> float | None:
        """Unix 时间戳形式的超时时刻；没有等待时为 ``None``。"""
        with self._lock:
            since = self._pending_since
        if since is None:
            return None
        return time.time() + max(0.0, self.timeout - (time.monotonic() - since))

    @staticmethod
    def _notify(callback: Callable[..., None] | None, *args: Any) -> None:
        """回调异常不能影响引擎（闸门炸了会把任务拖死）。"""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # pragma: no cover - 防御性
            pass


__all__ = ["ACTIONS", "DEFAULT_TIMEOUT", "PlanDecision", "ThreadingPlanGate"]
