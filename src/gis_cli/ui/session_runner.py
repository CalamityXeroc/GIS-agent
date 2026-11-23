# -*- coding: utf-8 -*-
"""交互会话：把引擎放到工作线程里跑，主线程专心收发输入。

职责：
- ``start(text)`` 起一个任务（工作线程执行 ``loop.run``，事件直接渲染）；
- ``guide(text)`` 运行中注入用户消息（走 ``loop.inject_user_message``）——
  注入失败说明这轮已结束，交给调用方排队等下一轮；
- ``interrupt()`` 请求停止（``loop.request_stop``，引擎下一轮检查点退出）；
- ``status``/``busy`` 供输入框提示与快捷键判断。

线程约定：引擎是纯同步代码（无 asyncio/signal），放工作线程安全；
渲染在事件回调里直接 print，由 ``patch_stdout`` 保证不冲乱输入框。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from . import events as ui_events


class EngineSession:
    """驱动一个 EngineBundle 的交互式会话。"""

    def __init__(
        self,
        bundle: Any,
        *,
        on_event_extra: Callable[[str, dict], None] | None = None,
        renderer: Callable[[str, dict], None] | None = None,
        on_finish: Callable[[Any], None] | None = None,
    ) -> None:
        self.bundle = bundle
        self.tracker = ui_events.StatusTracker()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state: Any = None
        self._error: BaseException | None = None
        self._queued: list[str] = []
        self._renderer = renderer
        self._on_event_extra = on_event_extra
        self._on_finish = on_finish

        loop = getattr(bundle, "loop", None)
        config = getattr(getattr(bundle, "llm", None), "config", None)
        if config is not None:
            self.tracker.model = str(getattr(config, "model", "") or "")
            self.tracker.provider = str(getattr(config, "api_base", "") or "")
        workspace = getattr(bundle, "workspace", None)
        if workspace is not None:
            self.tracker.workspace = str(workspace)

    # ------------------------------------------------------------ 状态

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def state(self) -> Any:
        return self._state

    @property
    def queued(self) -> list[str]:
        return list(self._queued)

    def drain_events(self, name: str, payload: dict) -> None:
        """引擎事件入口（由 CLI 传给 build_engine 的回调里调用）。"""
        self.tracker.apply(name, payload)
        if self._on_event_extra is not None:
            try:
                self._on_event_extra(name, payload)
            except Exception:
                pass
        if self._renderer is not None:
            try:
                self._renderer(name, payload)
            except Exception:
                pass

    # ------------------------------------------------------------ 运行

    def start(self, task: str) -> bool:
        """起一个任务；已在运行时返回 False（调用方应改用 :meth:`guide`）。"""
        if self.busy:
            return False
        self._error = None
        self._state = None
        self.tracker.status = "running"

        def _worker() -> None:
            try:
                self._state = self.bundle.loop.run(task)
            except BaseException as exc:  # noqa: BLE001 - 交给主线程报告
                self._error = exc
            finally:
                self.tracker.current_action = ""
                if self._on_finish is not None:
                    try:
                        self._on_finish(self._state)
                    except Exception:
                        pass

        self._thread = threading.Thread(target=_worker, name="gis-agent-engine", daemon=True)
        self._thread.start()
        return True

    def wait(self, timeout: float | None = None) -> Any:
        """等当前任务结束（返回 state；异常会原样抛出）。"""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        return self._state

    def guide(self, text: str) -> bool:
        """运行中的追加输入：排队等本轮结束，作为**同一个任务**的补充。

        为什么是排队而不是注入当前任务：注入是静默的（消息看起来消失了，
        用户会以为没提交成功而重复输入 —— 实测就是这么发生的）。
        为什么合并成一个任务而不是各自成任务：用户的真实用法是为了完成
        **同一件事**多次补充说明；要换任务会直接开新对话。

        返回 ``False`` 表示已排队（调用方据此提示用户）。
        """
        content = (text or "").strip()
        if not content:
            return True
        with self._lock:
            self._queued.append(content)
        return False

    @property
    def queued_count(self) -> int:
        """已排队条数（界面据此显示提示）。"""
        with self._lock:
            return len(self._queued)

    def queue_hint(self) -> str:
        """排队提示文案（空串 = 没有排队）。"""
        count = self.queued_count
        if count <= 0:
            return ""
        return f"已排队 {count} 条，本轮结束后会合并成一个任务继续"

    def merged_task(self, base: str) -> str:
        """把原始目标与排队补给合并成一条任务（取走队列）。"""
        queued = self.take_queued()
        if not queued:
            return base
        extra = "\n".join(queued)
        return f"{base}\n\n{extra}" if base else extra

    def take_queued(self) -> list[str]:
        """取走排队消息（本轮结束后由调用方合并进下一条任务）。"""
        with self._lock:
            items, self._queued = self._queued, []
        return items

    def interrupt(self) -> bool:
        """请求停止当前任务（Esc）。"""
        if not self.busy:
            return False
        loop = getattr(self.bundle, "loop", None)
        if loop is not None and hasattr(loop, "request_stop"):
            try:
                return bool(loop.request_stop("用户中断（Esc）"))
            except Exception:
                return False
        return False

    def reset_after_run(self) -> str:
        """一轮结束后同步状态行（幂等）。"""
        state = self._state
        if state is not None:
            self.tracker.turn = int(getattr(state, "turn", self.tracker.turn) or self.tracker.turn)
            self.tracker.status = str(getattr(state, "status", self.tracker.status) or self.tracker.status)
        return self.tracker.status


__all__ = ["EngineSession"]
