# -*- coding: utf-8 -*-
"""每个 run 的事件序号 + 环形缓冲 + 订阅广播（SSE 的数据源）。

设计约束（``docs/p2-console-plan.md`` §6）：

- ``push()`` 只做三件事：**加 seq → 入环（2000 条）→ 广播**，不做任何渲染；
- ``llm_delta`` 是逐 token 的流式增量，按 100ms 窗口合并后再入环，否则一条
  SSE 连接会被几万条单字事件冲爆，前端也来不及渲染；
- 订阅者拿到的 seq 严格递增，断线后用 ``?since=<seq>`` / ``Last-Event-ID`` 补发。

本模块刻意不 import 引擎与 ui 层（``gis_cli.ui`` 的 ``__init__`` 会连带
import rich 渲染层，服务进程里不能碰）。
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

#: 每个 run 保留的事件条数（断线补发/页面刷新回放的上限）。
RING_SIZE = 2000

#: ``llm_delta`` 合并窗口（毫秒）。
DELTA_WINDOW_MS = 100

#: 单条合并事件的正文上限：超过就先把手上的发出去，避免一条事件几十 KB。
MAX_DELTA_CHARS = 8000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Event:
    """一条已编号的引擎事件。"""

    seq: int
    event: str
    ts: str
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "seq": self.seq,
            "event": self.event,
            "ts": self.ts,
            "payload": self.payload,
        }
        if self.duration_ms:
            data["duration_ms"] = self.duration_ms
        return data

    def to_sse(self) -> str:
        """SSE 帧：``id`` 让浏览器断线重连时自动带 ``Last-Event-ID``。"""
        body = json.dumps(self.to_dict(), ensure_ascii=False)
        return f"id: {self.seq}\nevent: {self.event}\ndata: {body}\n\n"


class Subscription:
    """一个 SSE 连接对应一个订阅：先拿补发，再收实时。"""

    def __init__(
        self,
        channel: "_Channel",
        backlog: Iterable[Event],
        *,
        from_seq: int,
        truncated: bool,
    ) -> None:
        self._channel = channel
        self._cond = threading.Condition()
        self._queue: deque[Event] = deque(backlog)
        self._closed = channel.closed
        self.from_seq = from_seq
        #: ``since`` 早于环形缓冲最旧一条 → 中间有缺口（前端据此提示"历史被截断"）
        self.truncated = truncated

    # ------------------------------------------------------------- 消费
    def wait(self, timeout: float) -> list[Event]:
        """最多等 ``timeout`` 秒，把手上已有的事件一次取走（可能是空）。"""
        with self._cond:
            if not self._queue and not self._closed:
                self._cond.wait(timeout)
            batch = list(self._queue)
            self._queue.clear()
        return batch

    def pending(self) -> bool:
        with self._cond:
            return bool(self._queue)

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # ------------------------------------------------------------- 生产
    def _deliver(self, event: Event) -> None:
        with self._cond:
            if self._closed:
                return
            self._queue.append(event)
            self._cond.notify_all()

    def _finish(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class _Channel:
    """一个 run 的序号/环/订阅集合。"""

    def __init__(self, run_id: str, *, ring_size: int, delta_window_s: float) -> None:
        self.run_id = run_id
        self.lock = threading.Lock()
        self.seq = 0
        self.ring: deque[Event] = deque(maxlen=ring_size)
        self.subscribers: set[Subscription] = set()
        self.closed = False
        self.delta_window_s = delta_window_s
        self.pending_delta: dict[str, Any] | None = None
        self.delta_timer: threading.Timer | None = None

    # 调用方必须持有 self.lock
    def append_locked(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        ts: str | None = None,
        duration_ms: int = 0,
    ) -> Event:
        self.seq += 1
        item = Event(
            seq=self.seq,
            event=event,
            ts=ts or _now_iso(),
            payload=payload,
            duration_ms=duration_ms,
        )
        self.ring.append(item)
        for sub in list(self.subscribers):
            sub._deliver(item)
        return item


class EventBus:
    """``run_id → {seq, ring, subscribers}``。

    同一个实例由 ``create_app`` 建一次，引擎线程写、HTTP 线程读，内部用锁串行化。
    """

    def __init__(
        self,
        *,
        ring_size: int = RING_SIZE,
        delta_window_ms: int = DELTA_WINDOW_MS,
    ) -> None:
        self._lock = threading.Lock()
        self._channels: dict[str, _Channel] = {}
        self.ring_size = max(1, int(ring_size))
        self.delta_window_s = max(0.0, float(delta_window_ms) / 1000.0)

    # ------------------------------------------------------------- 内部
    def _channel(self, run_id: str) -> _Channel:
        with self._lock:
            channel = self._channels.get(run_id)
            if channel is None:
                channel = _Channel(
                    run_id, ring_size=self.ring_size, delta_window_s=self.delta_window_s
                )
                self._channels[run_id] = channel
            return channel

    def _flush_delta_locked(self, channel: _Channel) -> Event | None:
        pending = channel.pending_delta
        if pending is None:
            return None
        channel.pending_delta = None
        if channel.delta_timer is not None:
            channel.delta_timer.cancel()
            channel.delta_timer = None
        return channel.append_locked(
            "llm_delta",
            {
                "kind": pending.get("kind", ""),
                "text": "".join(pending.get("parts") or []),
                "merged": int(pending.get("count") or 1),
            },
        )

    def _merge_delta_locked(self, channel: _Channel, payload: dict[str, Any]) -> None:
        text = str(payload.get("text") or "")
        pending = channel.pending_delta
        if pending is None:
            pending = {
                "kind": str(payload.get("kind") or ""),
                "parts": [],
                "count": 0,
            }
            channel.pending_delta = pending
            timer = threading.Timer(self.delta_window_s, self._flush_delta, args=(channel.run_id,))
            timer.daemon = True
            channel.delta_timer = timer
            timer.start()
        pending["parts"].append(text)
        pending["count"] += 1
        if sum(len(part) for part in pending["parts"]) >= MAX_DELTA_CHARS:
            self._flush_delta_locked(channel)

    def _flush_delta(self, run_id: str) -> None:
        channel = self._channel(run_id)
        with channel.lock:
            self._flush_delta_locked(channel)

    # ------------------------------------------------------------- 写入
    def push(self, run_id: str, event: str, payload: dict[str, Any] | None = None) -> Event | None:
        """引擎 ``on_event`` 的落点；``llm_delta`` 合并期间返回 ``None``。"""
        channel = self._channel(str(run_id))
        data = dict(payload or {})
        with channel.lock:
            if channel.closed:
                return None
            if event == "llm_delta" and channel.delta_window_s > 0:
                self._merge_delta_locked(channel, data)
                return None
            # 非 delta 事件前先把挂起的增量发出去，保持时间顺序。
            self._flush_delta_locked(channel)
            return channel.append_locked(event, data)

    def seed(self, run_id: str, events: Iterable[dict[str, Any]]) -> int:
        """把磁盘上的 trace 事件灌进环形缓冲（重启后仍能回放历史）。

        只在通道里还没有事件时调用；保留原 ``seq`` 会与实时 seq 冲突，
        所以重新编号，``ts`` 照抄（时间线显示的是真实发生时间）。
        """
        channel = self._channel(str(run_id))
        count = 0
        with channel.lock:
            if channel.ring:
                return 0
            for raw in events:
                if not isinstance(raw, dict):
                    continue
                name = str(raw.get("event") or "")
                if not name:
                    continue
                payload = raw.get("payload")
                channel.append_locked(
                    name,
                    payload if isinstance(payload, dict) else {},
                    ts=str(raw.get("ts") or "") or None,
                    duration_ms=int(raw.get("duration_ms") or 0),
                )
                count += 1
        return count

    def flush(self, run_id: str | None = None) -> None:
        """立刻结束当前的 delta 合并窗口（测试与 run 结束时用）。"""
        run_ids = [run_id] if run_id else list(self._channels)
        for rid in run_ids:
            channel = self._channels.get(rid)
            if channel is None:
                continue
            with channel.lock:
                self._flush_delta_locked(channel)

    def close(self, run_id: str) -> None:
        """run 结束：冲掉挂起的增量，通知所有订阅者“没有后续了”。"""
        channel = self._channel(str(run_id))
        with channel.lock:
            self._flush_delta_locked(channel)
            channel.closed = True
            subscribers = list(channel.subscribers)
            channel.subscribers.clear()
        for sub in subscribers:
            sub._finish()

    # ------------------------------------------------------------- 读取
    def subscribe(self, run_id: str, since: int | None = None) -> Subscription:
        """订阅一个 run：``since`` 之后的事件先补发，再收实时。"""
        channel = self._channel(str(run_id))
        start = int(since) if since is not None else 0
        if start < 0:
            start = 0
        with channel.lock:
            backlog = [item for item in channel.ring if item.seq > start]
            truncated = bool(channel.ring) and start + 1 < channel.ring[0].seq
            sub = Subscription(channel, backlog, from_seq=start, truncated=truncated)
            if not channel.closed:
                channel.subscribers.add(sub)
        return sub

    def unsubscribe(self, run_id: str, sub: Subscription) -> None:
        channel = self._channels.get(str(run_id))
        if channel is None:
            return
        with channel.lock:
            channel.subscribers.discard(sub)
        sub.close()

    def snapshot(self, run_id: str) -> list[Event]:
        """当前环形缓冲里的全部事件（调试/测试/补发核对用）。"""
        channel = self._channels.get(str(run_id))
        if channel is None:
            return []
        with channel.lock:
            return list(channel.ring)

    def has_events(self, run_id: str) -> bool:
        channel = self._channels.get(str(run_id))
        if channel is None:
            return False
        with channel.lock:
            return bool(channel.ring)

    def last_seq(self, run_id: str) -> int:
        channel = self._channels.get(str(run_id))
        if channel is None:
            return 0
        with channel.lock:
            return channel.seq

    def is_closed(self, run_id: str) -> bool:
        channel = self._channels.get(str(run_id))
        if channel is None:
            return False
        with channel.lock:
            return channel.closed


__all__ = [
    "DELTA_WINDOW_MS",
    "Event",
    "EventBus",
    "RING_SIZE",
    "Subscription",
]
