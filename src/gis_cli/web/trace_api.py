# -*- coding: utf-8 -*-
"""只读 trace API：解析 ``<workspace>/.gis_agent/traces/*.jsonl``。

trace 是引擎的追加写日志，读取端必须容忍两种现实情况：

- 坏行：进程被硬杀时可能留下半行 JSON，跳过这一行，不要整份报错；
- 半截文件：run 还在跑（或崩了）时没有 ``run_end``，按已有事件给一个
  ``incomplete`` 状态，照样返回。

事件按 ``seq`` 排序（子代理/续跑会让写盘顺序和 seq 偶尔错位）。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

#: trace 目录（相对 workspace）。
TRACE_DIR = Path(".gis_agent") / "traces"


def _parse_ts(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _read_events(path: Path) -> list[dict[str, Any]]:
    """逐行解析；坏行/半行跳过，绝不抛给上层。"""
    events: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return events
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _by_seq(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 seq 升序；没有合法 seq 的事件保持原顺序排在后面。"""

    def _key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, event = item
        try:
            return (0, int(event.get("seq")))
        except (TypeError, ValueError):
            return (1, index)

    return [event for _, event in sorted(enumerate(events), key=_key)]


def local_day(ts: Any) -> str:
    """把事件时间戳转成**本地**日期（看板分桶口径）。

    引擎写的是 UTC（``+00:00``），但看板是给人看的：本地 00:30 起跑的 run
    必须落在"今天"那一根柱子上，而不是按 UTC 落到昨天。所有分桶/边界比较
    都走这一个函数，避免"KPI 数得对、柱子却是空的"。
    """
    parsed = _parse_ts(ts)
    if parsed is None:
        return str(ts or "")[:10]
    return parsed.astimezone().date().isoformat()


def read_trace_file(path: str | Path) -> list[dict[str, Any]]:
    """解析单个 trace 文件并按 ``seq`` 排序（坏行跳过、不抛异常）。"""
    return _by_seq(_read_events(Path(path)))


def summarize(run_id: str, events: list[dict[str, Any]], mtime: float) -> dict[str, Any]:
    """把一份 run 的事件流压成列表页/看板要的一行。

    除列表展示字段外，这里同时算出看板聚合用的 ``pass`` / ``problems`` /
    ``models`` 等：一次解析两处共用，避免列表和看板各写一套口径。
    """
    started_at: str | None = None
    finished_at: str | None = None
    status: str | None = None
    goal: str | None = None
    turns = 0
    tokens = 0
    cached = 0
    duration_s: float | None = None
    verdict: str | None = None
    problems: list[str] = []
    repairs = 0
    repair_events = 0
    tool_failures = 0
    models: dict[str, int] = {}
    model_calls: dict[str, int] = {}
    model_cached: dict[str, int] = {}
    model_prompt: dict[str, int] = {}
    prompt_tokens = 0

    for event in events:
        name = event.get("event")
        payload = event.get("payload") or {}
        if name == "turn":
            turns += 1
        elif name == "llm_result":
            usage = payload.get("usage") or {}
            model = str(payload.get("model") or "unknown")
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            used = int(usage.get("total_tokens") or (prompt + completion) or 0)
            hit = int(usage.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0)
            tokens += used
            cached += hit
            prompt_tokens += prompt
            models[model] = models.get(model, 0) + used
            model_calls[model] = model_calls.get(model, 0) + 1
            model_cached[model] = model_cached.get(model, 0) + hit
            model_prompt[model] = model_prompt.get(model, 0) + prompt
        elif name == "tool_result":
            if payload.get("ok") is False:
                tool_failures += 1
        elif name == "repair_attempt":
            repair_events += 1
        elif name == "verification":
            verdict = "pass" if payload.get("pass") is True else "fail"
            # 只保留最后一条验收的缺口：一次 run 多轮验收时以最终结论为准。
            raw = payload.get("problems") or []
            problems = [str(item) for item in raw if str(item).strip()] if isinstance(raw, list) else []
        elif name in {"run_start", "run_end"}:
            if payload.get("goal"):
                goal = str(payload["goal"])
        if name == "run_end":
            status = str(payload.get("status") or status or "") or None
            started_at = str(payload.get("started_at") or started_at or "") or None
            finished_at = str(payload.get("finished_at") or finished_at or "") or None
            turns = int(payload.get("turns") or turns)
            tokens = int(payload.get("total_tokens") or tokens)
            if payload.get("repairs") is not None:
                repairs = int(payload["repairs"] or 0)
            if payload.get("duration_seconds") is not None:
                duration_s = round(float(payload["duration_seconds"]), 2)

    first_ts = next((event.get("ts") for event in events if event.get("ts")), None)
    last_ts = next((event.get("ts") for event in reversed(events) if event.get("ts")), None)
    started_at = started_at or first_ts or datetime.fromtimestamp(
        mtime, timezone.utc
    ).isoformat()

    if duration_s is None:
        begin, end = _parse_ts(first_ts), _parse_ts(last_ts)
        if begin and end:
            duration_s = round(max(0.0, (end - begin).total_seconds()), 2)

    if status is None:
        last_event = events[-1].get("event") if events else None
        if last_event == "stop_requested":
            status = "stopped"
        elif last_event in {"run_error", "llm_error", "deadline_exceeded"}:
            status = "error"
        else:
            status = "incomplete"

    return {
        "run_id": run_id,
        "started_at": started_at,
        #: 分桶用的日期：事件时间戳转到本地时区后的日期。
        "day": local_day(started_at),
        "finished_at": finished_at,
        "status": status,
        "turns": turns,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "duration_s": duration_s,
        "verdict": verdict,
        #: 验收通过与否：**只有 verification.pass 为 true 才算通过**，
        #: 没跑到验收（None）既不算过也不算失败，单独统计。
        "passed": True if verdict == "pass" else (False if verdict == "fail" else None),
        "goal": (goal or "")[:200],
        "problems": problems[:5],
        "repairs": repairs if repairs else repair_events,
        "tool_failures": tool_failures,
        #: 事件条数：0 说明文件完全不可解析（截断/空文件），聚合时不计入。
        "event_count": len(events),
        "models": models,
        "model_calls": model_calls,
        "model_cached": model_cached,
        "model_prompt": model_prompt,
    }


def parse_bound(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    """解析筛选边界：支持 ``YYYY-MM-DD`` 与 ISO 时间戳。

    ``until`` 传日期时按当天 23:59:59 处理，否则 ``until=2026-11-10`` 会把
    当天的 run 全部漏掉 —— 这是最常见的“筛选结果少一天”来源。
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = _parse_ts(text)
    if parsed is None:
        try:
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time(), timezone.utc)
        except ValueError:
            return None
    if len(text) == 10 and end_of_day:
        parsed = parsed + timedelta(days=1) - timedelta(seconds=1)
    return parsed


def _matches(
    item: dict[str, Any],
    *,
    since: datetime | None,
    until: datetime | None,
    statuses: set[str],
    passed: bool | None,
    unverified: bool,
    keyword: str,
) -> bool:
    started = _parse_ts(item.get("started_at"))
    if since is not None and (started is None or started < since):
        return False
    if until is not None and (started is None or started > until):
        return False
    if statuses and str(item.get("status") or "") not in statuses:
        return False
    verdict = item.get("passed")
    if unverified:
        if verdict is not None:
            return False
    elif passed is not None and verdict is not passed:
        return False
    if keyword:
        haystack = f"{item.get('goal') or ''} {item.get('run_id') or ''}".casefold()
        if keyword not in haystack:
            return False
    return True


def filter_traces(
    items: list[dict[str, Any]],
    *,
    since: str | None = None,
    until: str | None = None,
    status: str | None = None,
    passed: bool | None = None,
    unverified: bool = False,
    q: str | None = None,
) -> list[dict[str, Any]]:
    """按时间范围/状态/验收结果/关键词筛列表（纯函数，便于单测）。"""
    begin = parse_bound(since)
    end = parse_bound(until, end_of_day=True)
    statuses = {item.strip() for item in str(status or "").split(",") if item.strip()}
    keyword = str(q or "").strip().casefold()
    return [
        item
        for item in items
        if _matches(
            item,
            since=begin,
            until=end,
            statuses=statuses,
            passed=passed,
            unverified=unverified,
            keyword=keyword,
        )
    ]



def list_traces(workspace: str | Path) -> list[dict[str, Any]]:
    """列出所有 run，最新的在前。目录不存在时返回空列表。"""
    trace_dir = Path(workspace) / TRACE_DIR
    if not trace_dir.is_dir():
        return []

    summaries: list[dict[str, Any]] = []
    for path in trace_dir.glob("*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:  # pragma: no cover - 文件刚被删
            continue
        events = read_trace_file(path)
        summaries.append(summarize(path.stem, events, mtime))

    summaries.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    return summaries


def read_events(workspace: str | Path, run_id: str) -> list[dict[str, Any]]:
    """读一个 run 的原始事件（按 seq 排序）；不存在/越界返回空表。

    与 :func:`load_trace` 的区别：不做摘要、不读 ``.md``、不抛 404 ——
    SSE 重连时用它把磁盘历史灌进环形缓冲。
    """
    if not run_id or any(sep in run_id for sep in ("/", "\\")) or run_id in {".", ".."}:
        return []
    path = Path(workspace) / TRACE_DIR / f"{run_id}.jsonl"
    if not path.is_file():
        return []
    return _by_seq(_read_events(path))


def load_trace(workspace: str | Path, run_id: str) -> dict[str, Any]:
    """读取单个 run 的完整事件流 + Markdown 报告（没有就是 None）。"""
    trace_dir = Path(workspace) / TRACE_DIR
    # run_id 来自 URL：只允许普通文件名，挡掉 ``../`` 之类的拼接。
    if not run_id or any(sep in run_id for sep in ("/", "\\")) or run_id in {".", ".."}:
        raise HTTPException(status_code=404, detail="trace not found")
    path = trace_dir / f"{run_id}.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="trace not found")

    events = _by_seq(_read_events(path))
    result = summarize(run_id, events, path.stat().st_mtime)
    report_path = trace_dir / f"{run_id}.md"
    result["events"] = events
    result["report"] = (
        report_path.read_text(encoding="utf-8", errors="replace")
        if report_path.is_file()
        else None
    )
    return result


def _resolve_trace_path(workspace: str | Path, run_id: str, suffix: str) -> Path:
    """把 run_id + 后缀解析成 trace 目录内的真实文件；不存在就 404。"""
    if not run_id or any(sep in run_id for sep in ("/", "\\")) or run_id in {".", ".."}:
        raise HTTPException(status_code=404, detail="trace not found")
    path = Path(workspace) / TRACE_DIR / f"{run_id}{suffix}"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="trace not found")
    return path


def build_router(workspace: str | Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/traces")
    def get_traces(
        since: str | None = Query(None, description="起始时间（YYYY-MM-DD 或 ISO 时间戳）"),
        until: str | None = Query(None, description="结束时间（日期按当天 23:59:59 处理）"),
        status: str | None = Query(None, description="状态，逗号分隔：completed,failed,stopped…"),
        pass_: str | None = Query(None, alias="pass", description="pass / fail / none"),
        q: str | None = Query(None, description="关键词：目标摘要或 run_id"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        passed: bool | None = None
        unverified = False
        flag = str(pass_ or "").strip().lower()
        if flag in {"pass", "true", "1"}:
            passed = True
        elif flag in {"fail", "false", "0"}:
            passed = False
        elif flag in {"none", "unverified"}:
            unverified = True
        elif flag:
            raise HTTPException(status_code=422, detail="pass 只接受 pass / fail / none")

        items = filter_traces(
            list_traces(workspace),
            since=since,
            until=until,
            status=status,
            passed=passed,
            unverified=unverified,
            q=q,
        )
        page = items[offset : offset + limit]
        # count 是**本页**条数（S1 起的旧约定），total 才是筛选后的总数。
        return {
            "items": page,
            "count": len(page),
            "total": len(items),
            "limit": limit,
            "offset": offset,
        }

    @router.get("/api/traces/{run_id}")
    def get_trace(run_id: str) -> dict[str, Any]:
        return load_trace(workspace, run_id)

    @router.get("/api/traces/{run_id}/download")
    def download_trace(run_id: str, kind: str = Query("jsonl")) -> FileResponse:
        """原样下载 trace：``kind=jsonl`` 是原始事件流，``kind=md`` 是报告。"""
        suffix = {"jsonl": ".jsonl", "md": ".md"}.get(kind)
        if suffix is None:
            raise HTTPException(status_code=422, detail="kind 只接受 jsonl / md")
        path = _resolve_trace_path(workspace, run_id, suffix)
        media = "application/x-ndjson" if suffix == ".jsonl" else "text/markdown"
        return FileResponse(path, media_type=media, filename=f"{run_id}{suffix}")

    return router
