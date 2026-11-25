# -*- coding: utf-8 -*-
"""Web 端数据目录扫描服务（DAIR-346）：打开控制台就能看见 ``input/`` 里的数据。

## 为什么要单独一层

以前只有两处会扫数据目录：引擎跑任务时（``jobs.make_engine_runner`` 的
``refresh_catalog=True``）和 CLI 的预热（``datacatalog/prewarm.py``）。web 侧
的 ``/api/catalog/*`` 只读库，所以「把数据放进 ``input/`` → 打开控制台 →
目录是空的」必须先提交一个任务才刷新。

这一层只负责四件事，且**不进入引擎启动路径**（``prewarm.py`` 头注释里写的
启动阻塞是前车之鉴）：

1. 起服务时在后台线程先扫一遍（首屏不等它）；
2. 暴露一个可重复调用的入口给前端「刷新」按钮与轮询（同一时刻只扫一遍）；
3. 扫描失败只落 ``error`` 文本，**不动库里已有记录**，GET 目录照常返回旧数据；
4. 扫完顺手给大 ``.gdb`` 要素类建**视口包络索引**（DAIR-393）：第一次打开
   13 万行图层不必再整层扫，直接按 OID 读视口。有门槛与预算，失败只记日志。

## 「自动识别」怎么做的

前端每 ``10s`` 调一次 ``POST /api/catalog/scan``（见 ``use-catalog.ts``）。
服务端收到请求先算一次**文件系统指纹**（``discover``：只 stat、不读内容，
实测 82 项 / 37ms），指纹与上次成功扫描的一致就直接回 ``unchanged``——空闲
时这一轮只有一次目录遍历的开销，不会反复重扫。指纹变了才真正开扫，所以
「放入新文件 → 出现」的延迟是「下一次轮询 + 增量扫描」。

内核**持续**失败时不再每 10s 起一整轮：失败计数与退避规则在
``datacatalog/kernel_health.py``（DAIR-365），这一层只在 ``trigger`` 里读一眼
「现在能不能扫」，退避期间直接回 ``backoff`` + 可读的 ``error``。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..datacatalog.kernel_health import KernelHealth
from ..datacatalog.prewarm import candidate_roots
from ..datacatalog.store import sanitize_text

logger = logging.getLogger(__name__)

#: catalog 库位置（相对 workspace），与 ``catalog_api.CATALOG_DB`` 一致。
CATALOG_DB = Path(".gis_agent") / "catalog.db"

#: 指纹里保留的 mtime 精度：文件系统给的是浮点数，比较到微秒足够。
_MTIME_PRECISION = 6

#: 视口包络索引的预热门槛（DAIR-393）：小图层整层扫也只要几十毫秒，不值得提前建；
#: 大 `.gdb` 要素类第一次打开才需要它（64k 面整层扫 ~4 s）。
INDEX_WARM_MIN_FEATURES = 20_000
#: 每轮扫描最多预热几个图层（别让「扫数据目录」变成「建库」）。
INDEX_WARM_MAX_LAYERS = 3
#: 每轮预热的墙钟预算（秒）：超了就下一轮再做，不拖住扫描状态。
INDEX_WARM_BUDGET_S = 120.0
#: 最近这么长时间内有过预览请求就让路（预览比预热索引更等不起）。
INDEX_WARM_IDLE_S = 8.0
#: 单层建索引的内核超时（秒）。预热和用户预览共用内核串行锁（``_exec_lock``），
#: 而预览等锁的预算是 ``read_timeout + 45 s``（``runtime/kernel.py``）：单层放到
#: 180 s 会把「慢」变成 PreviewTimeout → 422（DAIR-394 非阻塞项 2）。30 s 足够
#: 建几十万行的包络索引；超时就放弃这一层、下次扫描再试，退回整层扫不影响正确性。
INDEX_WARM_LAYER_TIMEOUT_S = 30.0

#: ``scanner.refresh_catalog`` 在 ArcPy 整批失败时写进 ``stats.errors`` 的前缀。
#: 这种失败不是「某几个文件有问题」，而是内核崩了 / 超时 900s / 脚本级失败：
#: 一条记录都没落库，必须让界面看见，也不能记成「扫过了」（DAIR-352 打回项 1）。
KERNEL_FAILURE_PREFIX = "kernel scan failed"


def _kernel_failed(errors: Iterable[str]) -> bool:
    return any(str(item).startswith(KERNEL_FAILURE_PREFIX) for item in errors)


def _warm_failure_reason(outcome: Any, payload: Any) -> str:
    """把「这一层没建成索引」的原因收成一行（页面上看不到，至少日志里要有）。"""
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    error = getattr(outcome, "error", None)
    if isinstance(error, dict):
        reason = error.get("message") or error.get("type")
        if reason:
            return str(reason)
    elif error:
        return str(error)
    return "内核未返回结果" if getattr(outcome, "ok", False) else "未返回错误信息"


def _is_total_failure(stats: Any) -> bool:
    """有 stats，但这一轮实质什么都没扫成（内核崩 / 内核缺席 / 发现的文件全部失败）。"""
    if getattr(stats, "kernel_unavailable", ""):
        # 内核没起来：本地能扫的扫了，但 .shp/.gdb/.tif 一批压根没扫（DAIR-374）。
        return True
    errors = [str(item) for item in (getattr(stats, "errors", None) or ())]
    if _kernel_failed(errors):
        return True
    return int(getattr(stats, "scanned", 0) or 0) == 0 and bool(errors)


def _iso(ts: float | None) -> str | None:
    """UTC 时间戳（带毫秒）：前端拿它当"这一轮扫描结束了"的变化标记。"""
    if not ts:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int(ts % 1 * 1000):03d}Z"


@dataclass
class ScanState:
    """一次扫描的状态（对外就是 ``snapshot()`` 那点字段）。"""

    running: bool = False
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_ms: int | None = None
    #: 谁触发的：``startup``（起服务）/ ``request``（前端轮询或刷新按钮）
    triggered_by: str = ""
    discovered: int = 0
    scanned: int = 0
    skipped: int = 0
    removed: int = 0
    merged: int = 0
    #: 致命错误（扫描整轮失败）；``errors`` 是单文件级失败，不影响其余记录。
    error: str = ""
    errors: list[str] = field(default_factory=list)
    #: 进程内第几次扫描（前端可以据此判断"有没有真的扫"）
    runs: int = 0

    def snapshot(self) -> dict[str, Any]:
        # 错误文本里会嵌原始路径（坏名 = 孤立代理项）：JSONResponse 遇到它们
        # 直接抛 UnicodeEncodeError，扫描接口自己先 500（DAIR-369）。这里是
        # 扫描状态的序列化边界，统一按展示文本洗干净。
        return {
            "running": self.running,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "elapsed_ms": self.elapsed_ms,
            "triggered_by": self.triggered_by,
            "discovered": self.discovered,
            "scanned": self.scanned,
            "skipped": self.skipped,
            "removed": self.removed,
            "merged": self.merged,
            "error": sanitize_text(self.error),
            "errors": [sanitize_text(item) for item in self.errors],
            "runs": self.runs,
        }


class CatalogScanService:
    """后台扫描工作区数据目录（线程安全；同一 workspace 不并发扫两遍）。"""

    def __init__(
        self,
        workspace: str | Path,
        *,
        kernel: Callable[[], Any] | None = None,
        roots: Iterable[str | Path] | None = None,
        health: KernelHealth | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self._kernel = kernel
        self._roots = [str(item) for item in roots] if roots else None
        #: 内核熔断（DAIR-365）：连续失败进入退避，退避期间不再起整轮。
        #: 计数与退避规则在 ``datacatalog/kernel_health.py``，这一层只读它。
        self._health = health if health is not None else KernelHealth()
        self._state = ScanState()
        #: 上一轮预热出来的视口索引层数（诊断用；只写日志，不进 scan 状态字段）。
        self._index_warmed = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        #: 上次**成功**扫描时的文件系统指纹；``None`` 表示还没成功过。
        self._scanned_signature: tuple | None = None
        #: 上次**失败**的指纹：同一份数据不反复重试（重试只在文件变化后发生）。
        self._failed_signature: tuple | None = None

    # ------------------------------------------------------------------ 查询
    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state.snapshot()

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        """等这一轮扫描结束（测试与关服务时用；没有在跑就立刻返回）。"""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        return self.state()

    # ------------------------------------------------------------------ 控制
    def trigger(self, *, reason: str = "request", force: bool = False) -> dict[str, Any]:
        """开一轮扫描（异步），返回 ``{triggered, reason, scan}``。

        ``reason``：``started``（真的开了）/ ``running``（已经有一轮在跑，
        直接复用）/ ``unchanged``（文件与上次成功扫描时一致，省掉一轮）/
        ``backoff``（内核连续失败，还没到重试时间，这一拍不起扫）。
        ``force``：跳过上次**失败**的指纹与内核退避窗口——用户点「刷新」时得
        真的再试一次，否则看到「扫描失败：…」之后按钮永远回 ``unchanged``
        （DAIR-352 打回项 2）或一直被退避挡着（DAIR-365）。
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"triggered": False, "reason": "running", "scan": self._state.snapshot()}
            if force:
                # 立刻放行一次；strikes 保留，再失败就按下一档退避。
                self._health.allow_retry()
            else:
                retry_in = self._health.blocked_for()
                if retry_in > 0:
                    scan = self._state.snapshot()
                    scan["error"] = self._health.message()
                    return {
                        "triggered": False,
                        "reason": "backoff",
                        "retry_in": round(retry_in, 1),
                        "error": scan["error"],
                        "scan": scan,
                    }
            roots = self._roots or candidate_roots(self.workspace)
            signature = self._fs_signature(roots)
            if self._scanned_signature is not None and signature == self._scanned_signature:
                return {"triggered": False, "reason": "unchanged", "scan": self._state.snapshot()}
            if (
                not force
                and self._failed_signature is not None
                and signature == self._failed_signature
            ):
                return {"triggered": False, "reason": "unchanged", "scan": self._state.snapshot()}
            self._state = ScanState(
                running=True,
                started_at=time.time(),
                triggered_by=reason,
                # 指纹就是这次要扫的条目：先告诉界面「已发现 N 项」，
                # 而不是等整轮结束才从 0 跳到 N。
                discovered=len(signature),
                runs=self._state.runs + 1,
            )
            thread = threading.Thread(
                target=self._run,
                args=(signature, roots),
                name="gis-agent-catalog-scan",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return {"triggered": True, "reason": "started", "scan": self._state.snapshot()}

    def close(self) -> None:
        """关服务：只等一小会儿，不等整轮扫描（daemon 线程随进程退出）。"""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(1.0)

    # ------------------------------------------------------------------ 执行
    def _run(self, signature: tuple, roots: list[str]) -> None:
        stats: Any = None
        error = ""
        #: 这一轮要不要预热视口索引，以及预热用的内核。预热最长几十秒，只能出锁之后做。
        warm = False
        warm_kernel: Any = None
        try:
            from ..datacatalog.scanner import refresh_catalog
            from ..datacatalog.store import Catalog

            kernel = self._kernel() if self._kernel is not None else None
            catalog = Catalog(self.workspace / CATALOG_DB)
            try:
                stats = refresh_catalog(
                    catalog,
                    roots,
                    kernel,
                    workspace=self.workspace,
                    force=False,
                    health=self._health,
                )
            finally:
                catalog.close()
        except Exception as exc:  # 扫描失败不影响读接口：库里旧记录照常用
            error = f"{type(exc).__name__}: {exc}".strip()[:300]
            logger.warning("catalog scan failed (GET 仍回旧数据): %s", exc)

        with self._lock:
            state = self._state
            state.running = False
            state.finished_at = time.time()
            if state.started_at is not None:
                state.elapsed_ms = int((state.finished_at - state.started_at) * 1000)
            if stats is not None:
                state.discovered = int(getattr(stats, "discovered", 0))
                state.scanned = int(getattr(stats, "scanned", 0))
                state.skipped = int(getattr(stats, "skipped", 0))
                state.removed = int(getattr(stats, "removed", 0))
                state.merged = int(getattr(stats, "merged", 0))
                state.errors = list(getattr(stats, "errors", []))[:5]

            if stats is None or _is_total_failure(stats):
                # 整轮失败：界面要看见原因，且**不写** ``_scanned_signature``——
                # 写了的话之后每 10s 轮询都是 unchanged，文件不变就再也不重试。
                # 内核缺席时直接把可读原因（“ArcGIS 内核不可用：…”）当 error，
                # 不要退化成“内核扫描失败”这种没头没脑的一句话（DAIR-374）。
                unavailable = str(getattr(stats, "kernel_unavailable", "") or "")
                state.error = error or unavailable or (
                    "内核扫描失败" if _kernel_failed(state.errors) else f"{len(state.errors)} 项数据扫描失败"
                )
                # 内核崩溃 / 超时 / 缺席是可恢复的：不锁存失败指纹，让下一轮轮询自己重试；
                # 其它整轮失败（库被锁、权限、脚本异常）锁存，只由 force 重试。
                if unavailable or _kernel_failed(state.errors):
                    # 清掉上一轮（异常失败）锁下的同一份指纹，否则「异常失败 → 点刷新
                    # → 这一轮内核失败」之后，轮询会命中旧指纹直接回 unchanged，
                    # 内核失败的自动重试就失效了（DAIR-354 遗留）。
                    self._failed_signature = None
                else:
                    self._failed_signature = signature
                logger.warning("catalog scan failed: %s", state.error)
            else:
                self._scanned_signature = signature
                self._failed_signature = None
                # 数据目录这一轮扫完了：顺手给大 `.gdb` 层建视口包络索引（DAIR-393）。
                # 建在后台扫描线程里，用户打开数据目录的这几十秒正好把它做完，
                # 点开图层就直接走索引，不用再等第一次整层扫。
                #
                # 预热**必须出锁**再跑（DAIR-394 B1）：``state()``（GET
                # /api/catalog/scan 的轮询）与 ``trigger()``（刷新按钮）拿的是同一把
                # 锁，锁里跑最长 120 s 的预热会把轮询和刷新一起卡住，``finished_at``
                # 也迟迟送不出去。锁内只取内核，状态字段在下面这段里已经全部落定。
                try:
                    warm_kernel = self._kernel() if self._kernel is not None else None
                except Exception:  # pragma: no cover - 内核构造失败不影响扫描结果
                    warm_kernel = None
                warm = True
                logger.info(
                    "catalog scan done: discovered=%s scanned=%s skipped=%s removed=%s",
                    state.discovered,
                    state.scanned,
                    state.skipped,
                    state.removed,
                )

        if warm:
            self._index_warmed = self._warm_indexes(warm_kernel)

    def _warm_indexes(self, kernel: Any) -> int:
        """给大 `.gdb` 要素类预建视口包络索引，返回这次建了几个（DAIR-393）。

        只做「要素数 ≥ :data:`INDEX_WARM_MIN_FEATURES`、索引缺失或过期」的
        ``feature_class``；每轮最多 :data:`INDEX_WARM_MAX_LAYERS` 个、总预算
        :data:`INDEX_WARM_BUDGET_S` 秒。任何一步失败都只记日志——索引是加速项，
        不能把「数据目录扫描」本身弄失败。
        """
        if kernel is None or not hasattr(kernel, "run"):
            return 0
        if self._health.blocked_for() > 0:
            return 0
        from ..datacatalog.store import Catalog
        from ..report import filegdb_view, html_pack

        try:
            catalog = Catalog(self.workspace / CATALOG_DB)
            try:
                records = catalog.list_datasets(kind="feature_class", limit=500)
            finally:
                catalog.close()
        except Exception as exc:  # pragma: no cover - 库被锁/损坏
            logger.debug("视口索引预热跳过（读目录失败）：%s", exc)
            return 0

        deadline = time.time() + INDEX_WARM_BUDGET_S
        built = 0
        for record in records:
            if built >= INDEX_WARM_MAX_LAYERS or time.time() > deadline:
                break
            # 用户正在看图层就别抢内核：预览和建索引都在同一个内核里串行执行，
            # 抢过去会让「首图」白等一整轮建索引（DAIR-393 实测 15.8 s）。
            from ..web import preview_api as _preview_api

            if time.time() - _preview_api.last_preview_at() < INDEX_WARM_IDLE_S:
                logger.debug("视口索引预热让路：最近有预览请求")
                break
            try:
                count = int(record.feature_count or 0)
                if count < INDEX_WARM_MIN_FEATURES:
                    continue
                if ".gdb" not in str(record.path).lower():
                    continue
                layer = Path(record.path)
                if not layer.is_absolute():
                    layer = self.workspace / layer
                index_file = filegdb_view.index_path(self.workspace, layer)
                key = filegdb_view.index_key(layer, count)
                if not key or filegdb_view.load_index(index_file, key) is not None:
                    continue
                code = html_pack.layer_index_build_code(str(layer), str(index_file), key)
                outcome = kernel.run(
                    code, timeout=min(INDEX_WARM_LAYER_TIMEOUT_S, max(5.0, deadline - time.time()))
                )
                payload = getattr(outcome, "result", None)
                if isinstance(payload, dict) and payload.get("skipped"):
                    # 等内核锁的这段时间里预览已经把同一层的索引扫出来了，内核侧
                    # 拿到锁后按指纹跳过（DAIR-397 第 3 条，实测白建 3.7 s）。
                    logger.info("视口索引预热跳过：%s（已有索引，未重建）", record.path)
                    continue
                if getattr(outcome, "ok", False) and isinstance(payload, dict) and not payload.get("error"):
                    built += 1
                    logger.info("视口索引已建：%s（%s 条）", record.path, payload.get("count"))
                else:
                    # WARNING：30 s 预算不够时这是唯一现场证据，默认 INFO 下必须看得见。
                    logger.warning(
                        "视口索引未建成：%s %s",
                        record.path,
                        _warm_failure_reason(outcome, payload),
                    )
            except Exception as exc:  # pragma: no cover - 单个图层失败不影响其它
                logger.debug("视口索引预热失败 %s：%s", record.path, exc)
        return built

    def _fs_signature(self, roots: list[str]) -> tuple:
        """文件系统指纹：``(path, kind, mtime, size)`` 排序后的一串。

        只统计不读内容（``discover`` 就是 stat 遍历），所以每 10s 一次也不贵；
        目录读不了时返回 ``()``（当成"变了"，让扫描去给出可读的错误）。
        """
        from ..datacatalog.scanner import discover

        try:
            items = discover(roots, max_items=5000)
        except Exception as exc:  # pragma: no cover - 目录权限/被占用
            logger.warning("catalog signature failed: %s", exc)
            return ()
        return tuple(
            sorted(
                (d.path, d.kind, round(d.mtime, _MTIME_PRECISION), d.size) for d in items
            )
        )


__all__ = ["CatalogScanService", "ScanState"]
