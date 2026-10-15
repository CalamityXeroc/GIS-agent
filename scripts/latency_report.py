# -*- coding: utf-8 -*-
"""从 trace 里算每段耗时占比（回答"到底慢在哪"）。

用法::

    python scripts/latency_report.py                     # 最近一次 run
    python scripts/latency_report.py --all               # 所有 run 汇总
    python scripts/latency_report.py path/to/run.jsonl   # 指定文件

输出三段划分（就是我们能优化的三类）：

- **模型等待**：每次 LLM 往返的墙钟时间（trace 里相邻事件的间隔）
- **工具执行**：execute_code / 配方等实际跑活的时间
- **上下文压缩**：_compact_if_needed 触发时的 LLM 摘要调用

注意：trace 里没有逐条 LLM 的耗时字段，这里用"事件时间差"估算——
``llm_result`` 的时间戳 - 前一个事件的时间戳 就是这次模型调用的等待。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

#: 默认 trace 目录（相对仓库根）
DEFAULT_TRACE_DIR = Path("workspace") / ".gis_agent" / "traces"


def _parse_ts(value: str) -> float:
    """ISO 时间戳 → epoch 秒（解析失败返回 0）。"""
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return 0.0


@dataclass
class Segment:
    """一段耗时。"""

    kind: str
    seconds: float
    label: str = ""


@dataclass
class RunReport:
    """单个 run 的耗时分解。"""

    path: Path
    run_id: str = ""
    status: str = ""
    turns: int = 0
    tool_calls: int = 0
    total_seconds: float = 0.0
    total_tokens: int = 0
    segments: list[Segment] = field(default_factory=list)
    compactions: int = 0
    llm_calls: int = 0

    def add(self, kind: str, seconds: float, label: str = "") -> None:
        if seconds > 0:
            self.segments.append(Segment(kind, seconds, label))

    def by_kind(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for seg in self.segments:
            out[seg.kind] = out.get(seg.kind, 0.0) + seg.seconds
        return out

    def worst(self, limit: int = 8) -> list[Segment]:
        return sorted(self.segments, key=lambda s: -s.seconds)[:limit]


def read_run(path: Path) -> RunReport:
    """读一个 jsonl trace，算出耗时分解。"""
    report = RunReport(path=path)
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    except OSError:
        return report

    if not events:
        return report

    report.run_id = path.stem
    last_ts = _parse_ts(str(events[0].get("ts", "")))
    started = last_ts
    pending_tool: tuple[str, float] | None = None
    llm_started = last_ts
    #: 本轮压缩耗时（压缩发生在 turn 内部、模型调用之前，
    #: 而“模型等待”= turn→llm_result 的整段，所以必须把它扣掉，否则重复计数）
    turn_compaction = 0.0

    for event in events:
        name = str(event.get("event", ""))
        ts = _parse_ts(str(event.get("ts", "")))
        gap = max(0.0, ts - last_ts) if ts else 0.0
        payload = event.get("payload") or {}

        if name == "run_start":
            started = ts or started
        elif name == "turn":
            # 轮次开始 → 接下来是（可选的压缩 +）模型调用
            llm_started = ts
            turn_compaction = 0.0
        elif name == "llm_result":
            report.llm_calls += 1
            net = max(0.0, ts - llm_started - turn_compaction)
            report.add("模型等待", net, "turn")
            turn_compaction = 0.0
        elif name == "context_compacted":
            report.compactions += 1
            report.add("上下文压缩", gap, f"{payload.get('summarized_messages')} 条")
            turn_compaction += gap
        elif name == "tool_call":
            pending_tool = (str(payload.get("tool", "")), ts)
        elif name == "tool_result":
            tool = str(payload.get("tool", ""))
            if pending_tool and pending_tool[1]:
                report.add("工具执行", max(0.0, ts - pending_tool[1]), tool)
            else:
                report.add("工具执行", gap, tool)
            pending_tool = None
        elif name == "run_end":
            report.status = str(payload.get("status", ""))
            report.turns = int(payload.get("turns") or 0)
            report.tool_calls = int(payload.get("tool_calls") or 0)
            report.total_tokens = int(payload.get("total_tokens") or 0)
            report.total_seconds = max(0.0, ts - started) if ts and started else 0.0

        last_ts = ts or last_ts

    if not report.total_seconds and last_ts and started:
        report.total_seconds = max(0.0, last_ts - started)
    # 未归因的水位：启动开销、需求抽取（独立 LLM 调用）、验收里的语义检查、
    # 写 trace/落盘等。不把它报出来，报告会留下一个莫名的大缺口。
    accounted = sum(report.by_kind().values())
    leftover = report.total_seconds - accounted
    if leftover > 1.0:
        report.add("其他", leftover, "启动/需求抽取/验收等")
    return report


def _bar(seconds: float, total: float, width: int = 28) -> str:
    if total <= 0:
        return ""
    filled = int(round(width * seconds / total))
    return "█" * max(0, min(width, filled))


def print_report(report: RunReport) -> None:
    totals = report.by_kind()
    accounted = sum(totals.values())
    total = report.total_seconds or accounted
    print(f"\n=== {report.run_id} ===")
    print(f"状态 {report.status or '?'} · {report.turns} 轮 · {report.tool_calls} 次工具 · "
          f"{report.llm_calls} 次模型调用 · {report.total_tokens:,} tokens")
    print(f"总耗时 {total:.1f}s（已归因 {accounted:.1f}s，占 {accounted / total * 100 if total else 0:.0f}%）")
    print()
    for kind in ("模型等待", "工具执行", "上下文压缩", "其他"):
        seconds = totals.get(kind, 0.0)
        share = (seconds / total * 100) if total else 0.0
        print(f"  {kind:6} {seconds:8.1f}s  {share:5.1f}%  {_bar(seconds, total)}")
    if report.compactions:
        print(f"\n  压缩触发 {report.compactions} 次（每次都是一次完整 LLM 调用）")
    print("\n最慢的几段:")
    for seg in report.worst():
        print(f"  {seg.seconds:8.1f}s  {seg.kind:6} {seg.label[:40]}")


def latest_trace(trace_dir: Path) -> Path | None:
    if not trace_dir.is_dir():
        return None
    files = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 trace 里算耗时占比")
    parser.add_argument("path", nargs="?", help="trace 文件（默认取最近一次）")
    parser.add_argument("--all", action="store_true", help="汇总全部 run")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR), help="trace 目录")
    parser.add_argument("--limit", type=int, default=8, help="--all 时最多看几个")
    args = parser.parse_args(list(argv) if argv is not None else None)

    trace_dir = Path(args.trace_dir)
    targets: list[Path] = []
    if args.path:
        targets = [Path(args.path)]
    elif args.all:
        targets = sorted(
            trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
        )[: args.limit]
    else:
        one = latest_trace(trace_dir)
        targets = [one] if one else []

    if not targets:
        print(f"没有找到 trace 文件（看的是 {trace_dir}）")
        return 1

    reports = [read_run(path) for path in targets]
    for report in reports:
        print_report(report)

    if len(reports) > 1:
        grand = sum(r.by_kind().get(k, 0.0) for r in reports for k in ("模型等待", "工具执行", "上下文压缩"))
        print("\n=== 汇总 ===")
        agg: dict[str, float] = {}
        for report in reports:
            for kind, seconds in report.by_kind().items():
                agg[kind] = agg.get(kind, 0.0) + seconds
        for kind, seconds in sorted(agg.items(), key=lambda kv: -kv[1]):
            share = (seconds / grand * 100) if grand else 0.0
            print(f"  {kind:6} {seconds:9.1f}s  {share:5.1f}%  {_bar(seconds, grand)}")
        total_compactions = sum(r.compactions for r in reports)
        print(f"  压缩共 {total_compactions} 次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
