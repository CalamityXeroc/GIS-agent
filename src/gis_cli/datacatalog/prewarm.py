# -*- coding: utf-8 -*-
"""数据目录后台预热：把"扫描工作区"从启动阻塞挪到首轮之后台并行。

## 为什么要重做

原来 ``build_engine`` 在**启动时同步**扫完所有数据才出界面。这有两个问题：

1. 用户要盯着"启动引擎（加载数据目录/内核）…"干等（实测修复 GDB 缓存后
   仍要全量扫一次，慢则一两分钟）；
2. 更关键的是**顺序错了**：模型第一轮只发 LLM 请求（不需要数据目录），
   而扫描完全可以和这次请求重叠。

## 现在的做法

- 启动时**不扫**（``refresh_catalog=False``），界面立刻可用；
- 真正开始做事时（``start_prewarm``）在后台线程里扫，与首轮模型调用并行；
- ``catalog_query`` 等工具首次用到目录时，先 ``wait_ready()`` 等它完成——
  对模型完全透明：拿到的永远是真实数据，不需要学新工具。

任何一步失败都不致命：扫描失败就退回"空目录 + 提示"，与旧行为一致。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PrewarmState:
    """预热进度（供界面显示"正在扫描数据目录…"）。"""

    started: bool = False
    finished: bool = False
    failed: bool = False
    scanned: int = 0
    skipped: int = 0
    discovered: int = 0
    error: str = ""
    stages: list[str] = field(default_factory=list)

    @property
    def busy(self) -> bool:
        return self.started and not self.finished and not self.failed

    def summary(self) -> str:
        if self.failed:
            return f"数据目录扫描失败：{self.error[:80]}"
        if not self.started:
            return "未扫描"
        if not self.finished:
            return f"正在扫描数据目录（已发现 {self.discovered} 项）…"
        if self.scanned == 0:
            return f"数据目录已就绪（{self.skipped} 项无变化）"
        return f"数据目录已就绪（新扫 {self.scanned} 项，跳过 {self.skipped} 项）"


class CatalogPrewarmer:
    """后台扫描数据目录（线程安全、可重复 start）。"""

    def __init__(
        self,
        *,
        catalog: Any,
        code_runner: Any,
        workspace: str | Path,
        roots: list[str] | None = None,
        on_progress: Callable[[PrewarmState], None] | None = None,
    ) -> None:
        self.catalog = catalog
        self.code_runner = code_runner
        self.workspace = Path(workspace)
        self._roots = roots
        self.on_progress = on_progress
        self.state = PrewarmState()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 控制
    def start(self) -> bool:
        """开始后台扫描；已在跑或已完成/已失败则返回 False。

        “失败后不再重试”是故意的：失败通常来自环境（arcpy 不可用、目录权限），
        每个任务开始时重试一次只会反复烧同样的时间。
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if self.state.finished or self.state.failed:
                return False
            self.state = PrewarmState(started=True)
            self._thread = threading.Thread(
                target=self._run, name="gis-agent-catalog-prewarm", daemon=True
            )
            self._thread.start()
            return True

    def wait_ready(self, timeout: float | None = None) -> PrewarmState:
        """等扫描完成（``catalog_query`` 首次用到目录时调）。

        没有在跑（未 start / 已结束）时立即返回——不能把工具调用卡住。
        """
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        return self.state

    @property
    def busy(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # ------------------------------------------------------------------ 执行
    def _run(self) -> None:
        try:
            from ..datacatalog.scanner import refresh_catalog
            from ..runtime.timing import StageTimer

            roots = self._roots or self._candidate_roots()
            if not roots:
                self.state.finished = True
                self.state.stages.append("无数据根目录")
                self._progress()
                return
            timer = StageTimer("catalog.prewarm")
            stats = refresh_catalog(self.catalog, roots, self.code_runner, force=False)
            timer.done(discovered=stats.discovered, scanned=stats.scanned, skipped=stats.skipped)
            self.state.discovered = stats.discovered
            self.state.scanned = stats.scanned
            self.state.skipped = stats.skipped
            self.state.finished = True
            self.state.stages.append("扫描完成")
            logger.info("catalog prewarm done: %s", stats)
        except Exception as exc:  # 扫描失败不影响任务（与旧行为一致：空目录 + 提示）
            self.state.failed = True
            self.state.error = str(exc)
            logger.warning("catalog prewarm failed (continuing): %s", exc)
        finally:
            self._progress()

    def _progress(self) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(self.state)
        except Exception:  # pragma: no cover - 界面回调不能影响扫描
            pass

    def _candidate_roots(self) -> list[str]:
        """默认只扫输入目录（``output`` 是产物，不是数据源）。

        以前把 ``output`` 也当数据源扫，既慢又会让"自己刚生成的成果"混进
        数据目录摘要。需要时（``--refresh-catalog``）调用方可以显式覆盖。
        """
        candidates = [
            self.workspace / "input",
            self.workspace / "data",
        ]
        roots = [str(p) for p in candidates if p.exists()]
        return roots or [str(self.workspace)]


__all__ = ["CatalogPrewarmer", "PrewarmState"]
