# -*- coding: utf-8 -*-
"""SSE（``text/event-stream``）端点：把 EventBus 里的 run 事件推到浏览器。

契约（``docs/p2-console-plan.md`` §6）：

- 断线重连：优先 ``?since=<seq>``，其次浏览器自动带的 ``Last-Event-ID``；
  只补发 ``seq > since`` 的段，不重发已收到的；
- 心跳：15s 一次 ``: ping`` 注释帧，防止代理/浏览器把空闲长连接掐掉；
- 结束：引擎的 ``run_end`` 发完即关连接（前端据此把状态条定住）。

服务重启后环形缓冲是空的：订阅前从 ``.gis_agent/traces/<engine_run_id>.jsonl``
把历史灌进去，这样刷新页面（甚至重启服务）时间线仍在。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from . import trace_api
from .event_bus import EventBus, Subscription

logger = logging.getLogger(__name__)

#: 心跳间隔（秒）。测试可调小。
HEARTBEAT_S = 15.0

#: 浏览器断线后建议的重连间隔（毫秒）。
RETRY_MS = 2000


def _seed_history(workspace: Path, bus: EventBus, manager: Any, run_id: str) -> None:
    """通道里没事件时，从磁盘 trace 补一份历史（重启后的回放）。"""
    if bus.has_events(run_id):
        return
    job = manager.get_or_none(run_id) if manager is not None else None
    trace_id = (job.engine_run_id if job is not None else None) or run_id
    try:
        events = trace_api.read_events(workspace, trace_id)
    except Exception:  # pragma: no cover - 坏目录不该让 SSE 500
        logger.warning("trace replay failed for %s", trace_id, exc_info=True)
        return
    if events:
        bus.seed(run_id, events)


def _hello(sub: Subscription, run_id: str) -> str:
    payload = {
        "run_id": run_id,
        "from_seq": sub.from_seq,
        "truncated": sub.truncated,
    }
    return f"event: hello\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _frames(sub: Subscription, run_id: str, *, heartbeat: float) -> Iterable[str]:
    """同步生成器：Starlette 会把它放进线程池迭代，``sub.wait`` 阻塞不会卡事件循环。"""
    yield f"retry: {RETRY_MS}\n\n"
    yield _hello(sub, run_id)
    last_send = time.monotonic()
    while True:
        batch = sub.wait(min(heartbeat, 1.0))
        for event in batch:
            yield event.to_sse()
        if batch:
            last_send = time.monotonic()
            if any(event.event == "run_end" for event in batch):
                return
        elif sub.closed and not sub.pending():
            return
        if time.monotonic() - last_send >= heartbeat:
            yield ": ping\n\n"
            last_send = time.monotonic()


def build_router(
    workspace: str | Path,
    bus: EventBus,
    manager: Any = None,
    *,
    heartbeat: float = HEARTBEAT_S,
) -> APIRouter:
    ws = Path(workspace).resolve()
    router = APIRouter()

    @router.get("/api/runs/{run_id}/events")
    def stream_events(run_id: str, request: Request, since: int | None = None):
        job = manager.get_or_none(run_id) if manager is not None else None
        if job is None:
            # 没有 job 记录（服务重启过/直接看 trace）时，磁盘 history 就是全部事实
            if not trace_api.read_events(ws, run_id):
                raise HTTPException(status_code=404, detail="run not found")
        _seed_history(ws, bus, manager, run_id)

        cursor = since
        if cursor is None:
            header = request.headers.get("last-event-id")
            if header:
                try:
                    cursor = int(header)
                except ValueError:
                    cursor = None

        subscription = bus.subscribe(run_id, since=cursor)
        return StreamingResponse(
            _frames(subscription, run_id, heartbeat=max(0.05, float(heartbeat))),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return router


__all__ = ["HEARTBEAT_S", "RETRY_MS", "build_router"]
