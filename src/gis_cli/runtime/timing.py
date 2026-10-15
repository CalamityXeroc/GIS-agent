# -*- coding: utf-8 -*-
"""轻量分段计时：只为"哪里慢"提供证据，不改变任何行为。

为什么要它：agent 的等待来自很多段（导入、建引擎、起内核、刷数据目录、
每次 LLM 往返、上下文压缩…），没有分解就只能凭感觉优化。这里的约定是
**只记数据**——测量失败、抛异常一律吞掉，绝不影响主流程。

用法::

    with stage("kernel.start"):
        kernel.start()

    span = StageTimer("catalog.refresh")
    ...
    span.done(scanned=133)

日志行统一前缀 ``[timing]``，便于 grep::

    [timing] kernel.start            28.41s
    [timing] catalog.refresh         61.20s  scanned=133
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

LOG_PREFIX = "[timing]"


def _fmt_fields(extra: dict[str, Any] | None) -> str:
    if not extra:
        return ""
    bits = []
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, float):
            bits.append(f"{key}={value:.2f}")
        else:
            bits.append(f"{key}={value}")
    return ("  " + " ".join(bits)) if bits else ""


@dataclass
class StageTimer:
    """一个分段的秒表（``done()`` 后打印一行日志）。"""

    name: str
    started: float = field(default_factory=time.perf_counter)
    extra: dict[str, Any] = field(default_factory=dict)
    _finished: bool = False

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def done(self, **extra: Any) -> float:
        """结束计时并写日志；返回耗时（秒）。重复调用只记一次。"""
        if self._finished:
            return self.elapsed
        self._finished = True
        seconds = self.elapsed
        self.extra.update(extra)
        try:
            logger.info("%s %-28s %7.2fs%s", LOG_PREFIX, self.name, seconds, _fmt_fields(self.extra))
        except Exception:  # pragma: no cover - 日志永远不能影响主流程
            pass
        return seconds


@contextmanager
def stage(name: str, **extra: Any) -> Iterator[StageTimer]:
    """``with stage("name") as t: ...`` —— 退出时（含异常）都记一笔。"""
    timer = StageTimer(name, extra=dict(extra))
    try:
        yield timer
    finally:
        timer.done()


@dataclass
class Counter:
    """累计计时（同一段被反复调用时用，如"LLM 调用总耗时"）。"""

    name: str = ""
    count: int = 0
    total: float = 0.0
    worst: float = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total += seconds
        self.worst = max(self.worst, seconds)

    def report(self, **extra: Any) -> None:
        try:
            logger.info(
                "%s %-28s total=%7.2fs n=%d avg=%6.2fs max=%6.2fs%s",
                LOG_PREFIX,
                self.name,
                self.total,
                self.count,
                (self.total / self.count if self.count else 0.0),
                self.worst,
                _fmt_fields(extra),
            )
        except Exception:  # pragma: no cover
            pass


__all__ = ["LOG_PREFIX", "StageTimer", "Counter", "stage"]
