# -*- coding: utf-8 -*-
"""数据目录自动扫描（DAIR-346）：触发一次扫描、并发只扫一遍、失败不影响 GET。

扫描本身用真实 ``refresh_catalog``（CSV 走 agent 侧本地画像，不需要 ArcPy），
只有需要控制时序/失败的两个用例才 monkeypatch 掉它。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from gis_cli.datacatalog.kernel_health import KernelHealth
from gis_cli.datacatalog.scanner import RefreshStats
from gis_cli.datacatalog.scanner import refresh_catalog as real_refresh_catalog
from gis_cli.datacatalog.store import Catalog, DatasetRecord
from gis_cli.report import filegdb_view
from gis_cli.web.app import create_app
from gis_cli.web.catalog_scan import CatalogScanService
from gis_cli.web.security import load_or_create_token


def _wait(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _client(
    workspace: Path,
    token: str,
    kernel: Callable[[], Any] | None = None,
    health: KernelHealth | None = None,
) -> TestClient:
    """不跑 lifespan（不起服务就不自动扫），扫描全由用例显式触发。"""
    scanner = CatalogScanService(
        workspace,
        kernel=kernel if kernel is not None else lambda: None,
        health=health,
    )
    return TestClient(create_app(workspace, token=token, scanner=scanner))


def _scan_state(client: TestClient, auth: dict[str, str]) -> dict[str, Any]:
    return client.get("/api/catalog/scan", headers=auth).json()


def _wait_scan_done(client: TestClient, auth: dict[str, str], timeout: float = 10.0) -> dict[str, Any]:
    assert _wait(lambda: not _scan_state(client, auth)["running"], timeout=timeout)
    return _scan_state(client, auth)


def _csv(workspace: Path, name: str = "demo.csv") -> Path:
    path = workspace / "input" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("region,pop\nA,100\nB,200\n", encoding="utf-8")
    return path


def _big_gdb_layer(workspace: Path, count: int = 90_000) -> Path:
    """造一个「够大」的 `.gdb` 要素类记录（预热门槛是 2 万要素）。"""
    gdb = workspace / "input" / "数据.gdb"
    gdb.mkdir(parents=True, exist_ok=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"fake")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(
        DatasetRecord(path=str(gdb / "big"), kind="feature_class", name="big", feature_count=count)
    )
    catalog.close()
    return gdb / "big"


def _shp(workspace: Path, name: str = "roads.shp") -> Path:
    """矢量文件：走 ArcPy 内核那一支（.shp 不在本地画像的白名单里）。"""
    path = workspace / "input" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake shp bytes")
    return path


class _BrokenKernel:
    """ArcPy 内核整批失败（``execute`` 返回 ``ok=False``）。"""

    calls = 0

    def execute(self, code: str, timeout: float | None = None) -> Any:
        _BrokenKernel.calls += 1
        return SimpleNamespace(ok=False, error="arcpy boom", result={})


class _FlakyKernel:
    """先坏后好的内核：``ok`` 翻成 True 后正常返回一条矢量记录。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.ok = False
        self.calls = 0

    def execute(self, code: str, timeout: float | None = None) -> Any:
        self.calls += 1
        if not self.ok:
            return SimpleNamespace(ok=False, error="arcpy boom", result={})
        return SimpleNamespace(
            ok=True,
            error="",
            result={
                "records": [
                    {
                        "path": self.path,
                        "kind": "vector",
                        "name": "roads",
                        "geom_type": "Polyline",
                        "wkid": 4547,
                    }
                ]
            },
        )


@pytest.fixture()
def token(workspace: Path) -> str:
    return load_or_create_token(workspace)


@pytest.fixture()
def auth(token: str) -> dict[str, str]:
    return {"X-Web-Token": token}


def test_scan_endpoint_populates_catalog(
    workspace: Path, token: str, auth: dict[str, str]
):
    """不跑任务、不点刷新，POST 一次就能把 input/ 里的数据落进目录。"""
    _csv(workspace)
    client = _client(workspace, token)

    assert _scan_state(client, auth)["running"] is False

    response = client.post("/api/catalog/scan", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["triggered"] is True
    assert body["reason"] == "started"
    assert body["scan"]["triggered_by"] == "request"
    assert body["scan"]["running"] is True

    state = _wait_scan_done(client, auth)
    assert state["error"] == ""
    assert state["scanned"] == 1
    assert state["finished_at"] and state["elapsed_ms"] is not None

    listing = client.get("/api/catalog/datasets", headers=auth).json()
    assert listing["count"] == 1
    item = listing["items"][0]
    assert item["kind"] == "table"
    # 落库口径与引擎扫描一致（DAIR-335/336）：相对 workspace、无 `_:` 绝对写法。
    assert not item["path"].startswith("_:")
    assert item["path"].replace("\\", "/") == "input/demo.csv"


def test_second_trigger_while_running_reuses_the_scan(
    workspace: Path, token: str, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    """并发两次只扫一遍：第二次直接返回 running，不再开一轮。"""
    _csv(workspace)
    calls: list[float] = []
    release = threading.Event()

    def slow_refresh(catalog: Any, roots: Any, kernel: Any, **kwargs: Any) -> RefreshStats:
        calls.append(time.monotonic())
        release.wait(10)  # 卡住第一轮，让第二个请求一定撞在"进行中"
        return RefreshStats(discovered=1, scanned=1)

    monkeypatch.setattr("gis_cli.datacatalog.scanner.refresh_catalog", slow_refresh)
    client = _client(workspace, token)

    first = client.post("/api/catalog/scan", headers=auth).json()
    assert _wait(lambda: len(calls) == 1, timeout=5)  # 等第一轮真的进了 refresh
    second = client.post("/api/catalog/scan", headers=auth).json()
    assert first["triggered"] is True
    assert second["triggered"] is False
    assert second["reason"] == "running"
    assert second["scan"]["runs"] == 1
    assert len(calls) == 1

    release.set()
    assert _wait_scan_done(client, auth)["running"] is False
    assert len(calls) == 1


def test_scan_failure_keeps_old_data_and_get_still_works(
    workspace: Path, token: str, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    """扫描整轮失败：目录树退回旧数据 + 一行可读提示，GET 不 500。"""
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(DatasetRecord(path="input\\old.shp", kind="vector", name="old", size=10))
    catalog.close()

    def boom(catalog: Any, roots: Any, kernel: Any, **kwargs: Any) -> RefreshStats:
        raise RuntimeError("arcpy 内核起不来")

    monkeypatch.setattr("gis_cli.datacatalog.scanner.refresh_catalog", boom)
    client = _client(workspace, token)

    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    state = _wait_scan_done(client, auth)
    assert "arcpy 内核起不来" in state["error"]
    assert state["running"] is False

    listing = client.get("/api/catalog/datasets", headers=auth)
    assert listing.status_code == 200
    assert [item["name"] for item in listing.json()["items"]] == ["old"]

    # 同一份数据不反复重试：客户端每 10s 轮询也不会把失败刷成循环。
    again = client.post("/api/catalog/scan", headers=auth).json()
    assert again["triggered"] is False
    assert again["reason"] == "unchanged"


def test_unchanged_files_skip_a_second_scan(
    workspace: Path, token: str, auth: dict[str, str]
):
    _csv(workspace)
    client = _client(workspace, token)

    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    _wait_scan_done(client, auth)
    assert _scan_state(client, auth)["runs"] == 1

    again = client.post("/api/catalog/scan", headers=auth).json()
    assert again["triggered"] is False
    assert again["reason"] == "unchanged"
    assert again["scan"]["runs"] == 1

    # 新文件进来 → 指纹变了 → 再扫一轮。
    _csv(workspace, "second.csv")
    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    state = _wait_scan_done(client, auth)
    assert state["runs"] == 2
    assert client.get("/api/catalog/datasets", headers=auth).json()["count"] == 2


def test_kernel_scan_failure_is_visible(
    workspace: Path, token: str, auth: dict[str, str]
):
    """内核整批失败（``result.ok=False``）不能当成「扫成功」：error 要让它跑出来。

    62dcbc03 时 ``stats`` 非空就直接算成功 → ``state.error`` 仍是 ``''``，
    UI 只读 ``state.error``（``workbench.tsx``），界面上没有任何提示。
    """
    _shp(workspace)
    _BrokenKernel.calls = 0
    client = _client(workspace, token, kernel=lambda: _BrokenKernel())

    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    state = _wait_scan_done(client, auth)
    assert state["error"] == "内核扫描失败"
    assert state["scanned"] == 0
    assert "kernel scan failed: arcpy boom" in state["errors"][0]


def test_kernel_failure_does_not_lock_the_fingerprint(
    workspace: Path, token: str, auth: dict[str, str]
):
    """内核失败后不能把指纹记成「扫过了」：同一份文件再 trigger 要真的重试。"""
    _shp(workspace)
    _BrokenKernel.calls = 0
    client = _client(workspace, token, kernel=lambda: _BrokenKernel())

    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    assert _wait_scan_done(client, auth)["error"]

    # 内核崩是可恢复的：下一轮轮询不该被 unchanged 挡住。
    retry = client.post("/api/catalog/scan", headers=auth).json()
    assert retry["triggered"] is True
    assert retry["reason"] == "started"
    assert _wait_scan_done(client, auth)["error"]
    assert _BrokenKernel.calls == 2


def test_kernel_failure_clears_a_stale_failed_fingerprint(
    workspace: Path, token: str, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    """「异常失败 → 点刷新 → 内核失败」后，轮询不能还被上一轮的失败指纹挡着。"""
    _shp(workspace)
    _BrokenKernel.calls = 0
    calls: list[int] = []

    def flaky_refresh(catalog: Any, roots: Any, kernel: Any, **kwargs: Any) -> RefreshStats:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("catalog 被占用")
        return real_refresh_catalog(catalog, roots, kernel, **kwargs)

    monkeypatch.setattr("gis_cli.datacatalog.scanner.refresh_catalog", flaky_refresh)
    client = _client(workspace, token, kernel=lambda: _BrokenKernel())

    # 1) 异常失败 → 失败指纹锁存，轮询 unchanged
    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    assert "catalog 被占用" in _wait_scan_done(client, auth)["error"]
    assert client.post("/api/catalog/scan", headers=auth).json()["reason"] == "unchanged"

    # 2) 用户点刷新 → 这一轮碰上内核失败
    assert client.post("/api/catalog/scan?force=1", headers=auth).json()["triggered"] is True
    assert _wait_scan_done(client, auth)["error"] == "内核扫描失败"

    # 3) 内核失败不锁存，且要清掉上一轮锁下的旧指纹——否则轮询直接 unchanged，
    #    「内核失败自动重试」在「异常失败 → 刷新」这个组合里失效（DAIR-354 遗留）。
    retry = client.post("/api/catalog/scan", headers=auth).json()
    assert retry["triggered"] is True
    assert retry["reason"] == "started"
    assert _wait_scan_done(client, auth)["error"] == "内核扫描失败"
    assert _BrokenKernel.calls == 2


def test_kernel_backoff_stops_the_polling_loop(
    workspace: Path, token: str, auth: dict[str, str]
):
    """连续 3 次内核失败后退避：第 4 拍不再起整轮，``?force=1`` 立刻重试（DAIR-365）。"""
    _shp(workspace)
    _BrokenKernel.calls = 0
    client = _client(workspace, token, kernel=lambda: _BrokenKernel())

    for _ in range(3):
        assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
        assert _wait_scan_done(client, auth)["error"] == "内核扫描失败"
    assert _BrokenKernel.calls == 3

    # 第 4 拍：文件没变、失败指纹也没锁存，唯一挡住它的就是退避窗口。
    blocked = client.post("/api/catalog/scan", headers=auth).json()
    assert blocked["triggered"] is False
    assert blocked["reason"] == "backoff"
    assert "内核连续失败 3 次" in blocked["error"]
    assert blocked["retry_in"] > 0
    assert _BrokenKernel.calls == 3  # 没起整轮，内核没被再调一次
    assert blocked["scan"]["runs"] == 3
    # 前端读的是 ``scan.error``：退避原因要能被看见。
    assert blocked["scan"]["error"] == blocked["error"]

    # 用户点刷新：跳过退避立刻重试一次。
    forced = client.post("/api/catalog/scan?force=1", headers=auth).json()
    assert forced["triggered"] is True
    assert forced["reason"] == "started"
    assert _wait_scan_done(client, auth)["error"] == "内核扫描失败"
    assert _BrokenKernel.calls == 4


def test_kernel_recovery_resets_the_backoff(
    workspace: Path, token: str, auth: dict[str, str]
):
    """内核恢复后下一拍正常扫并落库，退避计数归零（不留下永久熔断）。"""
    shp = _shp(workspace)
    kernel = _FlakyKernel(str(shp))
    # 退避窗口压到 0.2s，免得用例真等 10s；档位规则本身由
    # ``tests/test_catalog_kernel_health.py`` 覆盖。
    scanner = CatalogScanService(
        workspace,
        kernel=lambda: kernel,
        health=KernelHealth(backoff=(0.2, 0.2, 0.2)),
    )
    client = TestClient(create_app(workspace, token=token, scanner=scanner))

    for _ in range(3):
        assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
        assert _wait_scan_done(client, auth)["error"] == "内核扫描失败"
    assert client.post("/api/catalog/scan", headers=auth).json()["reason"] == "backoff"

    kernel.ok = True
    # 只等退避窗口过去（上限 5s，正常 <0.3s），不等扫描。
    deadline = time.monotonic() + 5.0
    while scanner._health.blocked_for() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    recovered = client.post("/api/catalog/scan", headers=auth).json()
    assert recovered["triggered"] is True
    assert recovered["reason"] == "started"
    assert _wait_scan_done(client, auth)["error"] == ""
    assert client.get("/api/catalog/datasets", headers=auth).json()["count"] == 1
    assert scanner._health.strikes == 0
    assert kernel.calls == 4


def test_force_bypasses_the_failed_fingerprint(
    workspace: Path, token: str, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
):
    """失败后刷新按钮要能真的重试：``?force=1`` 跳过失败指纹（DAIR-352 打回项 2）。"""
    _csv(workspace)
    calls: list[int] = []

    def flaky_refresh(catalog: Any, roots: Any, kernel: Any, **kwargs: Any) -> RefreshStats:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("catalog 被占用")
        return real_refresh_catalog(catalog, roots, kernel, **kwargs)

    monkeypatch.setattr("gis_cli.datacatalog.scanner.refresh_catalog", flaky_refresh)
    client = _client(workspace, token)

    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    state = _wait_scan_done(client, auth)
    assert "catalog 被占用" in state["error"]

    # 轮询不把失败刷成循环：文件没变 → unchanged（失败指纹锁存）。
    polling = client.post("/api/catalog/scan", headers=auth).json()
    assert polling["triggered"] is False
    assert polling["reason"] == "unchanged"

    forced = client.post("/api/catalog/scan?force=1", headers=auth).json()
    assert forced["triggered"] is True
    assert forced["reason"] == "started"

    # 重试成功后恢复指纹语义：文件没变就不再扫。
    assert _wait_scan_done(client, auth)["error"] == ""
    assert len(calls) == 2
    assert client.post("/api/catalog/scan", headers=auth).json()["reason"] == "unchanged"
    assert client.get("/api/catalog/datasets", headers=auth).json()["count"] == 1


def test_startup_scans_without_any_request(workspace: Path, token: str, auth: dict[str, str]):
    """起服务（lifespan startup）就后台扫一遍：不提交任务、不点刷新。"""
    _csv(workspace)
    scanner = CatalogScanService(workspace, kernel=lambda: None)
    app = create_app(workspace, token=token, scanner=scanner)
    with TestClient(app) as client:
        assert _wait(
            lambda: client.get("/api/catalog/datasets", headers=auth).json()["count"] == 1,
            timeout=10,
        )
        state = _scan_state(client, auth)
    assert state["triggered_by"] == "startup"
    assert state["error"] == ""


def test_scan_requires_the_web_token(workspace: Path, token: str):
    client = TestClient(create_app(workspace, token=token))
    assert client.post("/api/catalog/scan").status_code == 401
    assert client.get("/api/catalog/scan").status_code == 401


def test_scan_warms_viewport_index_for_large_gdb_layer(tmp_path: Path):
    """扫描之后给大 `.gdb` 要素类预建视口包络索引（DAIR-393）。

    只做「要素数 ≥ 门槛」的层；小图层整层扫也只要几十毫秒，不值得提前建。
    """
    workspace = tmp_path / "ws"
    gdb = workspace / "input" / "数据.gdb"
    gdb.mkdir(parents=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"fake")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(
        DatasetRecord(path=str(gdb / "big"), kind="feature_class", name="big", feature_count=90_000)
    )
    catalog.upsert(
        DatasetRecord(path=str(gdb / "small"), kind="feature_class", name="small", feature_count=100)
    )
    catalog.close()

    calls: list[str] = []

    class _Kernel:
        def run(self, code, timeout=None):  # noqa: ANN001
            calls.append(code)
            return SimpleNamespace(ok=True, result={"error": "", "count": 90_000})

    scanner = CatalogScanService(workspace, kernel=lambda: _Kernel())

    assert scanner._warm_indexes(_Kernel()) == 1
    assert len(calls) == 1, "只该给大图层建索引"
    assert "build_index" in calls[0] and "big" in calls[0] and "small" not in calls[0]


def test_index_warm_failure_is_visible_at_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """建不成索引不能再静默（DAIR-397 第 2 条）：30 s 预算超时是唯一的现场证据。"""
    workspace = tmp_path / "ws"
    layer = _big_gdb_layer(workspace)

    class _TimeoutKernel:
        def run(self, code, timeout=None):  # noqa: ANN001
            return SimpleNamespace(
                ok=False,
                result=None,
                error={"type": "TimeoutExpired", "message": "kernel timeout after 30.0s"},
            )

    scanner = CatalogScanService(workspace, kernel=lambda: _TimeoutKernel())
    with caplog.at_level(logging.WARNING, logger="gis_cli.web.catalog_scan"):
        assert scanner._warm_indexes(_TimeoutKernel()) == 0

    lines = [r for r in caplog.records if "视口索引未建成" in r.getMessage()]
    assert lines, "超时必须在日志里留下一条「未建成」"
    assert lines[0].levelno == logging.WARNING, "默认 INFO 级别下 DEBUG 等于没有"
    text = lines[0].getMessage()
    assert str(layer) in text, "要能看出是哪一层"
    assert "kernel timeout after 30.0s" in text, "要能看出原因"


def test_index_warm_counts_kernel_side_skip_as_not_built(tmp_path: Path):
    """内核拿到锁后按指纹跳过时不能记成「已建」（DAIR-397 第 3 条）。"""
    workspace = tmp_path / "ws"
    _big_gdb_layer(workspace)

    class _SkippingKernel:
        def run(self, code, timeout=None):  # noqa: ANN001
            # 等锁期间预览已经扫出索引，内核里的 build_index 直接返回 skipped。
            return SimpleNamespace(ok=True, result={"error": "", "count": 90_000, "skipped": True})

    scanner = CatalogScanService(workspace, kernel=lambda: _SkippingKernel())
    assert scanner._warm_indexes(_SkippingKernel()) == 0


def test_warm_kernel_code_skips_index_written_by_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """预热拿到内核锁时预览已把索引建好：不重扫整层（DAIR-397 第 3 条）。

    假内核按真实内核的做法：先执行「预览那次整层扫」（写出索引），再 exec 预热下发的
    建索引代码。生成的代码里 ``build_index`` 持锁后按指纹查一次 → 直接 skipped，
    不会再打开数据源扫整层。
    """
    workspace = tmp_path / "ws"
    layer = _big_gdb_layer(workspace)
    index_file = filegdb_view.index_path(workspace, layer)
    key = filegdb_view.index_key(layer, 90_000)
    opened: list[str] = []

    def _no_open(path):  # noqa: ANN001
        opened.append(str(path))
        raise AssertionError("预热不该再打开数据源重扫整层")

    monkeypatch.setattr(filegdb_view, "_open_layer", _no_open)

    class _Kernel:
        def run(self, code, timeout=None):  # noqa: ANN001
            # 预览（同一内核、先拿到锁）这次整层扫把索引写出来了。
            filegdb_view.write_index(index_file, key, [1, 2, 3], [(0.0, 0.0, 1.0, 1.0)] * 3)
            box: dict[str, Any] = {}
            exec(code, {"set_result": lambda value: box.setdefault("result", value)})
            return SimpleNamespace(ok=True, result=box.get("result"))

    scanner = CatalogScanService(workspace, kernel=lambda: _Kernel())
    with caplog.at_level(logging.INFO, logger="gis_cli.web.catalog_scan"):
        assert scanner._warm_indexes(_Kernel()) == 0

    assert opened == [], "内核里应该直接按指纹跳过"
    assert any("视口索引预热跳过" in r.getMessage() for r in caplog.records)


def test_scan_state_does_not_block_on_index_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """预热在建索引时 ``state()`` 必须立刻返回（DAIR-394 B1）。

    预热要占内核几十秒，**不能**同时占扫描状态锁：``GET /api/catalog/scan``
    （``state()``）和刷新按钮（``trigger()``）拿的是同一把锁，锁里跑预热会让
    前端轮询卡住、``finished_at`` 迟迟拿不到，目录树重拉被推迟到预热结束。
    """
    workspace = tmp_path / "ws"
    gdb = workspace / "input" / "数据.gdb"
    gdb.mkdir(parents=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"fake")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(
        DatasetRecord(path=str(gdb / "big"), kind="feature_class", name="big", feature_count=90_000)
    )
    catalog.close()

    warming = threading.Event()

    class _SlowKernel:
        def run(self, code, timeout=None):  # noqa: ANN001
            warming.set()
            time.sleep(2.0)
            return SimpleNamespace(ok=True, result={"error": "", "count": 90_000})

    monkeypatch.setattr(
        "gis_cli.datacatalog.scanner.refresh_catalog",
        lambda *args, **kwargs: RefreshStats(discovered=1, scanned=1),
    )
    scanner = CatalogScanService(workspace, kernel=lambda: _SlowKernel())
    thread = threading.Thread(target=scanner._run, args=((("x",),), [str(gdb)]), daemon=True)
    thread.start()
    try:
        assert warming.wait(10.0), "预热没开始"
        started = time.monotonic()
        state = scanner.state()
        elapsed = time.monotonic() - started
    finally:
        thread.join(10.0)

    assert elapsed < 0.5, f"预热期间 state() 阻塞了 {elapsed:.2f}s"
    assert state["running"] is False and state["finished_at"], "扫描结果要在预热结束前就可见"
    assert scanner._index_warmed == 1, "预热本身仍要在后台做完"
