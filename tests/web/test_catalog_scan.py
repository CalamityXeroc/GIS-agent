# -*- coding: utf-8 -*-
"""数据目录自动扫描（DAIR-346）：触发一次扫描、并发只扫一遍、失败不影响 GET。

扫描本身用真实 ``refresh_catalog``（CSV 走 agent 侧本地画像，不需要 ArcPy），
只有需要控制时序/失败的两个用例才 monkeypatch 掉它。
"""

from __future__ import annotations

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
