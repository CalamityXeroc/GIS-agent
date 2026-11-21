# -*- coding: utf-8 -*-
"""看板指标聚合：把 ``<workspace>/.gis_agent/traces/*.jsonl`` 压成总览页要的数。

口径（本单 S4 的硬规矩，逐条对应验收）：

- **成功率只认 ``verification.pass == true``**，不看 ``run_end.status``。
  引擎历史上出现过 ``status=completed`` 但验收没过（``verification.pass=false``）
  的 run，按 run_end 算会把失败刷成成功，这是必须避免的误报。
- **耗时**取 ``run_end.duration_seconds``（没有 run_end 的半截 run 不参与分位数）。
- **token** 逐条 ``llm_result.usage`` 求和，按 ``model`` 分组；``cached_tokens``
  （命中前缀缓存的部分）单独累计 —— 它直接对应钱，仓库里没有价格表，
  所以这里只报 token 与缓存命中率，不编造金额。
- **返工量** = ``run_end.repairs`` 与 ``tool_result.ok == false`` 两个计数。
- **缺口 TOP** 对 ``verification.problems`` 做关键词归并（纯规则，**不做 LLM 归类**）。

性能：逐文件按 ``(mtime, size)`` 缓存解析结果，文件没变就只做聚合；
聚合本身是 O(run 数 × 模型数)，18 个真 trace 约 1ms。
"""

from __future__ import annotations

import json
import math
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, Query, Request

from .trace_api import TRACE_DIR, read_trace_file, summarize

#: 一次最多看多少天：再多也不会有分辨率的柱子，只会更慢。
MAX_DAYS = 90

#: 缺口归并的规则表：**先匹配先赢**。顺序按“越具体的越靠前”排，
#: 否则 ``文件不存在`` 会被 ``不存在`` 这类宽规则吃掉。
GAP_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"坐标系|投影|CRS|EPSG|WGS ?84|CGCS ?2000|4508|4326|3857", re.I), "坐标系/投影不一致"),
    (re.compile(r"不存在|未找到|找不到|缺失|missing|not found|no such", re.I), "交付物缺失"),
    (re.compile(r"为空|空文件|0 ?字节|无内容|empty|zero", re.I), "输出为空"),
    (re.compile(r"几何|要素|geometry|geometr|feature", re.I), "几何/要素异常"),
    (re.compile(r"字段|属性|field|attribute|column", re.I), "字段/属性缺失"),
    (re.compile(r"格式|类型|format|extension|后缀", re.I), "格式不符"),
    (re.compile(r"范围|extent|越界|out of", re.I), "范围/越界"),
    (re.compile(r"路径|目录|path|folder", re.I), "路径问题"),
    (re.compile(r"大小|体量|size|MB|KB", re.I), "体量不符"),
    (re.compile(r"编码|乱码|encoding|decode", re.I), "编码问题"),
)

#: 每文件解析结果缓存：``path -> (mtime, size, 摘要)``。
_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


def _percentile(values: Iterable[float], q: float) -> float | None:
    """线性插值分位数（与 numpy 的默认实现同口径），空集合返回 None。"""
    data = sorted(float(item) for item in values)
    if not data:
        return None
    if len(data) == 1:
        return round(data[0], 2)
    position = (len(data) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return round(data[low], 2)
    weight = position - low
    return round(data[low] + (data[high] - data[low]) * weight, 2)


def _load_trace(path: Path) -> dict[str, Any] | None:
    """读一个 trace 的摘要；``(mtime, size)`` 没变就直接用缓存。"""
    try:
        stat = path.stat()
    except OSError:  # 文件刚被删/权限变化
        return None
    key = str(path)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == stat.st_mtime and hit[1] == stat.st_size:
            return hit[2]
    events = read_trace_file(path)
    record = summarize(path.stem, events, stat.st_mtime)
    record["mtime"] = stat.st_mtime
    record["size"] = stat.st_size
    with _CACHE_LOCK:
        _CACHE[key] = (stat.st_mtime, stat.st_size, record)
    return record


def _prune_cache(alive: set[str]) -> None:
    """删掉已经不存在的 trace 缓存项，避免长期运行后越攒越多。"""
    with _CACHE_LOCK:
        for key in [item for item in _CACHE if item not in alive]:
            _CACHE.pop(key, None)


def load_records(workspace: str | Path) -> tuple[list[dict[str, Any]], str | None]:
    """读出全部 run 的摘要；目录缺失/全部损坏时返回 ``([], warning)``。"""
    trace_dir = Path(workspace) / TRACE_DIR
    if not trace_dir.is_dir():
        return [], f"trace 目录不存在：{TRACE_DIR.as_posix()}（还没有运行记录）"

    paths = sorted(trace_dir.glob("*.jsonl"))
    records: list[dict[str, Any]] = []
    corrupt = 0
    for path in paths:
        record = _load_trace(path)
        if record is None:
            continue
        if not record.get("event_count"):
            # 空文件/全坏行：不是错误（run 可能正在写），只是没有可聚合的事实。
            corrupt += 1
            continue
        records.append(record)
    _prune_cache({str(path) for path in paths})

    if not paths:
        return [], f"trace 目录为空：{TRACE_DIR.as_posix()}"
    if not records:
        return [], f"{corrupt or len(paths)} 个 trace 全部无法解析（文件可能被截断）"
    return records, None


def _window(days: int, today: date | None = None) -> list[str]:
    """最近 ``days`` 个自然日的日期串（旧 → 新，含今天；本地时区）。"""
    end = today or date.today()
    return [(end - timedelta(days=offset)).isoformat() for offset in range(days - 1, -1, -1)]


def _in_window(record: dict[str, Any], bucket_days: list[str]) -> bool:
    day = str(record.get("day") or "")
    return bool(day) and bucket_days[0] <= day <= bucket_days[-1]


def _daily(records: list[dict[str, Any]], bucket_days: list[str]) -> list[dict[str, Any]]:
    rows = {
        day: {
            "date": day,
            "runs": 0,
            "passed": 0,
            "failed": 0,
            "tokens": 0,
            "tool_failures": 0,
            "by_model": {},
        }
        for day in bucket_days
    }
    for record in records:
        row = rows.get(str(record.get("day") or ""))
        if row is None:
            continue
        row["runs"] += 1
        if record.get("passed") is True:
            row["passed"] += 1
        elif record.get("passed") is False:
            row["failed"] += 1
        row["tokens"] += int(record.get("tokens") or 0)
        row["tool_failures"] += int(record.get("tool_failures") or 0)
        for model, tokens in (record.get("models") or {}).items():
            row["by_model"][model] = row["by_model"].get(model, 0) + int(tokens)
    return list(rows.values())


def _models(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for record in records:
        for model, tokens in (record.get("models") or {}).items():
            slot = stats.setdefault(
                model,
                {
                    "model": model,
                    "calls": 0,
                    "runs": 0,
                    "tokens": 0,
                    "cached_tokens": 0,
                    "prompt_tokens": 0,
                },
            )
            slot["runs"] += 1
            slot["tokens"] += int(tokens)
            slot["calls"] += int((record.get("model_calls") or {}).get(model) or 0)
            slot["cached_tokens"] += int((record.get("model_cached") or {}).get(model) or 0)
            slot["prompt_tokens"] += int((record.get("model_prompt") or {}).get(model) or 0)
    rows = []
    for slot in stats.values():
        prompt_like = slot["prompt_tokens"]
        slot["cache_ratio"] = (
            round(slot["cached_tokens"] / prompt_like, 4) if prompt_like else 0.0
        )
        rows.append(slot)
    rows.sort(key=lambda item: (-item["tokens"], item["model"]))
    return rows


def gap_keyword(text: str) -> str:
    """把一条 ``verification.problems`` 归到一个固定标签（纯规则）。"""
    raw = str(text or "").strip()
    if not raw:
        return "未分类"
    for pattern, label in GAP_RULES:
        if pattern.search(raw):
            return label
    # 兜底：去掉路径与数字后取前 16 字，保证相同缺口还能合并到一起。
    cleaned = re.sub(r"[A-Za-z]:[\\/][^\s，。;；、]+", "<路径>", raw)
    cleaned = re.sub(r"\d+(?:\.\d+)?", "N", cleaned)
    return cleaned[:16] or "未分类"


def _gaps(records: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for record in records:
        seen: set[str] = set()
        for problem in record.get("problems") or []:
            label = gap_keyword(problem)
            slot = stats.setdefault(label, {"label": label, "count": 0, "runs": 0})
            slot["count"] += 1
            if label not in seen:
                slot["runs"] += 1
                seen.add(label)
    rows = sorted(stats.values(), key=lambda item: (-item["count"], item["label"]))
    return rows[:limit]


def _kpis(records: list[dict[str, Any]]) -> dict[str, Any]:
    runs = len(records)
    passed = sum(1 for record in records if record.get("passed") is True)
    failed = sum(1 for record in records if record.get("passed") is False)
    unverified = runs - passed - failed
    durations = [
        float(record["duration_s"])
        for record in records
        if record.get("duration_s") is not None
    ]
    tokens_total = sum(int(record.get("tokens") or 0) for record in records)
    prompt_total = sum(int(record.get("prompt_tokens") or 0) for record in records)
    cached_total = sum(int(record.get("cached_tokens") or 0) for record in records)
    return {
        "runs": runs,
        # 分母是全窗口 run 数：没跑验收的 run 不算"通过"（宁严不宽）。
        "pass_rate": round(passed / runs, 4) if runs else 0.0,
        "passed": passed,
        "failed": failed,
        "unverified": unverified,
        "duration_p50_s": _percentile(durations, 0.5),
        "duration_p90_s": _percentile(durations, 0.9),
        "duration_n": len(durations),
        "tokens_total": tokens_total,
        "prompt_tokens": prompt_total,
        "tokens_cached": cached_total,
        # 命中率 = 命中缓存的 prompt token / 全部 prompt token（与 latency_report 同口径）
        "cache_ratio": round(cached_total / prompt_total, 4) if prompt_total else 0.0,
        "tool_failures": sum(int(record.get("tool_failures") or 0) for record in records),
        "repairs": sum(int(record.get("repairs") or 0) for record in records),
        "turns": sum(int(record.get("turns") or 0) for record in records),
    }


def _kernel_health(workspace: Path) -> dict[str, Any]:
    """内核就绪 = kernelspec 已注册（首次跑任务时 ``ArcpyKernel`` 会自动建）。"""
    candidates = (
        workspace / "kernel" / "kernels" / "arcpy-engine" / "kernel.json",
        Path.home() / ".gis_agent" / "kernel" / "kernels" / "arcpy-engine" / "kernel.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return {"label": "内核", "status": "ok", "detail": "arcpy-engine 已注册 · 串行 1 路"}
    return {"label": "内核", "status": "warn", "detail": "未注册（首次运行会自动创建）"}


def _arcgis_health() -> dict[str, Any]:
    """ArcGIS 就绪只看解释器能不能发现；**绝不 import arcpy**（那要几秒且要许可）。"""
    try:
        from ..arcpy_bridge import discover_arcgis_pro_python

        info = discover_arcgis_pro_python()
    except Exception as exc:  # 发现失败是常态（没装 ArcGIS），不是服务错误
        detail = str(exc).splitlines()[0] if str(exc) else "未找到 ArcGIS Pro Python"
        return {"label": "ArcGIS", "status": "warn", "detail": detail[:120]}
    source = getattr(info, "source", "") or "unknown"
    return {
        "label": "ArcGIS",
        "status": "ok",
        "detail": f"python 就绪 · {source}",
    }


def _model_health(workspace: Path, config_path: str | Path | None) -> dict[str, Any]:
    from .config_api import resolve_config_path

    try:
        path = resolve_config_path(workspace, config_path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"label": "模型", "status": "warn", "detail": f"配置不可读：{exc}"[:120]}
    if not isinstance(data, dict):
        return {"label": "模型", "status": "warn", "detail": "配置不是 JSON 对象"}
    model = str(data.get("model") or "").strip()
    if not model:
        return {"label": "模型", "status": "warn", "detail": "未配置 model"}
    fallbacks = [str(item) for item in (data.get("fallback_models") or []) if str(item).strip()]
    detail = model + (f" → {' / '.join(fallbacks[:3])}" if fallbacks else "")
    return {"label": "模型", "status": "ok", "detail": detail}


def _queue_health(manager: Any) -> dict[str, Any]:
    if manager is None:
        return {"label": "队列", "status": "warn", "detail": "服务未接线", "length": 0, "running": 0}
    length = int(manager.queue_length)
    try:
        running = len(manager.list_jobs(status="running", limit=50))
    except Exception:  # pragma: no cover - 管理器异常不该拖垮看板
        running = 0
    detail = "空闲" if length == 0 and running == 0 else f"运行 {running} · 排队 {length}"
    return {
        "label": "队列",
        "status": "ok",
        "detail": detail,
        "length": length,
        "running": running,
    }


def health(
    workspace: str | Path, config_path: str | Path | None = None, manager: Any = None
) -> dict[str, Any]:
    """总览页健康条：内核 / ArcGIS / 模型 / 队列，四项都只做低成本的本地探测。"""
    ws = Path(workspace)
    return {
        "kernel": _kernel_health(ws),
        "arcgis": _arcgis_health(),
        "model": _model_health(ws, config_path),
        "queue": _queue_health(manager),
    }


def summary(
    workspace: str | Path,
    days: int = 7,
    *,
    config_path: str | Path | None = None,
    manager: Any = None,
    today: date | None = None,
) -> dict[str, Any]:
    """总览页的一次完整聚合。"""
    days = max(1, min(int(days), MAX_DAYS))
    records, warning = load_records(workspace)
    bucket_days = _window(days, today)
    windowed = [record for record in records if _in_window(record, bucket_days)]

    if warning is None and len(records) == 1:
        warning = "仅 1 个 run：中位/分位耗时与趋势不具代表性"

    return {
        "window": {
            "days": days,
            "since": bucket_days[0],
            "until": bucket_days[-1],
            "timezone": f"本地时区 UTC{datetime.now().astimezone().strftime('%z')}"
            "（事件时间戳先转本地日期再分桶）",
        },
        "kpis": _kpis(windowed),
        "daily": _daily(windowed, bucket_days),
        "models": _models(windowed),
        "gaps": _gaps(windowed),
        "health": health(workspace, config_path, manager),
        "total_runs": len(records),
        "runs_in_window": len(windowed),
        "warning": warning,
    }


def build_router(workspace: str | Path, config_path: str | Path | None = None) -> APIRouter:
    ws = Path(workspace)
    router = APIRouter()

    @router.get("/api/metrics/summary")
    def get_summary(request: Request, days: int = Query(7, ge=1, le=MAX_DAYS)) -> dict[str, Any]:
        # 队列长度只有 JobManager 知道；从 app.state 取，避免把 router 和
        # 管理器耦合进构造签名（测试里 create_app 也未必传 manager）。
        manager = getattr(request.app.state, "jobs", None)
        return summary(ws, days, config_path=config_path, manager=manager)

    @router.get("/api/metrics/health")
    def get_health(request: Request) -> dict[str, Any]:
        manager = getattr(request.app.state, "jobs", None)
        return health(ws, config_path, manager)

    return router
