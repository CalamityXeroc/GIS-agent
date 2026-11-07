# -*- coding: utf-8 -*-
"""合规自检的落盘与汇总：``create_map`` 写边车，``verify_outputs`` / ``finish`` 读回来。

为什么落盘而不是只留在返回值里：

- 出图在 ArcGIS 内核里（另一个进程），验收在主进程 —— 结果必须能跨进程传；
- 边车里存的是**上下文**（图层/文字/声明），验收时按**当前**规则库重算：
  规则改了、某条规则被关了，交付闸门立刻按新口径判，不用重新出图；
- 上下文坏了还能退回存档结论（``report``），不至于让验收直接瞎掉。

边车命名：``<图名>.compliance.json``，与图同目录（``output/地图.jpg`` →
``output/地图.compliance.json``）。**不是**新产物类型，不进产物清单。
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import NOTICE, ComplianceReport, check_map
from .facts import MapContext

logger = logging.getLogger(__name__)

#: 边车后缀（与 ``.spec.json`` 同一思路：图旁边的那份机器可读证据）。
SIDECAR_SUFFIX = ".compliance.json"
#: 边车格式版本。
SIDECAR_VERSION = 1


def sidecar_path(image_path: str | Path) -> Path:
    """图 → 边车路径（``a/b/地图.jpg`` → ``a/b/地图.compliance.json``）。"""
    target = Path(str(image_path))
    return target.with_name(target.stem + SIDECAR_SUFFIX)


def write_sidecar(
    image_path: str | Path,
    report: ComplianceReport,
    *,
    context: MapContext | None = None,
) -> Path:
    """把本次自检写成边车（覆盖同名旧文件）；返回路径。写失败抛 OSError 由调用方兜。"""
    target = sidecar_path(image_path)
    payload: dict[str, Any] = {
        "version": SIDECAR_VERSION,
        "image": str(image_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "notice": report.notice or NOTICE,
        "report": report.to_dict(),
    }
    if context is not None:
        payload["context"] = context.to_dict()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def read_sidecar(image_path: str | Path) -> dict[str, Any] | None:
    """读边车；不存在/不是 JSON 对象/读不动都返回 ``None``（不抛）。"""
    target = sidecar_path(image_path)
    try:
        if not target.is_file():
            return None
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:  # 半写/权限/编码：都不该让验收崩
        logger.debug("读合规边车失败 %s: %s", target.name, exc)
        return None
    return data if isinstance(data, dict) else None


def report_from_sidecar(
    image_path: str | Path,
    *,
    rules: Any = None,
    enabled: Mapping[str, bool] | None = None,
) -> ComplianceReport | None:
    """边车 → 报告：优先用存的上下文按当前规则重算，退路是存档结论。"""
    data = read_sidecar(image_path)
    if not data:
        return None
    archived = data.get("report")
    context = MapContext.from_dict(data.get("context"))
    if context is not None:
        if not context.image:
            context = replace(context, image=str(image_path))
        report = check_map(context, rules=rules, enabled=enabled)
        # 规则库坏了（error 非空）就用存档结论顶着，别让闸门自己把验收判成"没记录"
        if not report.error:
            return report
        logger.warning("合规边车重算失败（%s），退回存档结论", report.error)
    if isinstance(archived, Mapping):
        return ComplianceReport.from_dict(archived)
    return None


def collect_reports(
    paths: Iterable[str | Path],
    *,
    rules: Any = None,
    enabled: Mapping[str, bool] | None = None,
) -> list[tuple[str, ComplianceReport]]:
    """从一批产物里挑出带边车的图，返回 ``[(图路径, 报告)]``（顺序保持入参顺序）。

    去重键是**边车路径**而不是入参路径：``create_map`` 会把同一次出图的
    ``地图.jpg`` 与 ``地图.aprx`` 都登记成产物，两者 ``sidecar_path()`` 相同
    （``地图.compliance.json``），按入参去重会把一张图算成两张、阻断数翻倍。
    """
    out: list[tuple[str, ComplianceReport]] = []
    seen: set[str] = set()
    seen_sidecars: set[str] = set()
    for raw in paths:
        path = str(raw or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        target = sidecar_path(path)
        if not target.is_file():  # 没边车就不是候选产物（.aprx/.spec.json 也就不用去读）
            continue
        try:
            key = str(target.resolve()).lower()
        except OSError:  # 路径解析失败（超长/权限）也按原样当键，别把图漏掉
            key = str(target).lower()
        if key in seen_sidecars:
            continue
        seen_sidecars.add(key)
        report = report_from_sidecar(path, rules=rules, enabled=enabled)
        if report is not None:
            out.append((path, report))
    return out


def aggregate(entries: Iterable[tuple[str, ComplianceReport]]) -> dict[str, Any]:
    """多张图的自检结论汇总成 ``verdict["compliance"]``（验收/卡片/UI 共用这一份）。"""
    items = list(entries)
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    problems: list[str] = []
    for image, report in items:
        name = Path(str(image)).name
        if report.error:
            problems.append(f"{name}: {report.error}")
        for hit in report.hits:
            key = (name, hit.rule_id)
            if key in seen:
                continue
            seen.add(key)
            findings.append({**hit.to_dict(), "image": name})

    level = "none"
    for item in findings:
        if _ORDER.get(str(item.get("level")), 0) > _ORDER.get(level, -1):
            level = str(item.get("level"))
    counts = {name: sum(1 for item in findings if item.get("level") == name) for name in ("block", "warn", "info")}
    checked = len(items)
    if not checked:
        summary = "合规闸门：没有可核验的自检记录（图不是 create_map 出的，或没写边车）"
    elif not findings:
        summary = f"合规闸门：未发现问题（{checked} 张图）"
    else:
        bits = [f"{_LABELS[name]} {counts[name]} 项" for name in ("block", "warn", "info") if counts[name]]
        summary = f"合规闸门：{'、'.join(bits)}（{checked} 张图）"
        first = next((item for item in findings if item.get("level") == level), findings[0])
        if first.get("message"):
            summary += "。" + str(first["message"])
    if problems:
        # 规则库读不到/边车坏了：不当成阻断，但绝不能静默 —— 否则“没查”看着像“查过了”
        summary += f"；自检异常 {len(problems)} 张图：{problems[0][:120]}"
    return {
        "level": level,
        "blocked": level == "block",
        "checked": checked,
        "images": [Path(str(image)).name for image, _report in items],
        "counts": counts,
        "findings": findings,
        "problems": problems,
        "summary": summary,
        "notice": NOTICE,
    }


_ORDER: dict[str, int] = {"none": 0, "pass": 0, "info": 1, "warn": 2, "block": 3}
_LABELS: dict[str, str] = {"block": "阻断", "warn": "警告", "info": "提示"}


__all__ = [
    "SIDECAR_SUFFIX",
    "SIDECAR_VERSION",
    "aggregate",
    "collect_reports",
    "read_sidecar",
    "report_from_sidecar",
    "sidecar_path",
    "write_sidecar",
]
