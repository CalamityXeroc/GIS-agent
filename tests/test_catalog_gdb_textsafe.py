# -*- coding: utf-8 -*-
"""DAIR-369/374：.gdb 当文件夹记账 + 坏文件名（孤立代理项）不再带崩整轮扫描。

锁住五件事：

1. 每个 ``.gdb`` 都落一条 ``kind='gdb'`` 容器记录——内核不可用 / 某个库读不出来
   时它仍然以「文件夹」形态出现在目录里，并带着可读原因，而不是凭空消失或
   变成一个展开不了的叶子节点；
2. 名字含孤立代理项（WTF-8 破损名）的记录**只降级自己**：同一轮里其它记录照常
   落库，``/api/catalog/datasets`` 回 200，不再出现「扫描失败：UnicodeEncodeError」；
3. 容器的新鲜度口径：内核失败那一轮不能把容器伪装成「已扫过」，内核恢复后必须
   重扫出成员；
4. 扫描状态里的错误文本（可能嵌着坏名路径）也必须能 JSON 序列化；
5. **磁盘上的真坏名**（DAIR-374）：坏名文件真实存在于目录里时，新鲜度比较 /
   对账也不能拿原始路径去绑 sqlite——否则第一条 `is_fresh` 就把整轮扫描带掉。
   内核缺席（``kernel=None``）时给可读原因，不是 ``AttributeError``。

DAIR-374 之前这里以为「Windows 上造不出真实坏名文件」，实际可以
（``Path.write_text`` 写 NTFS 未配对代理项名）：bench 用真文件一注入就复现了
`is_fresh` 的崩。所以``test_real_broken_filename_*`` 走真文件；只有文件系统
真拒绝这种名字时（非 Windows）才退回 ``discover`` 桩。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from fastapi.testclient import TestClient

from gis_cli.datacatalog import scanner as scanner_module
from gis_cli.datacatalog.scanner import (
    BAD_NAME_NOTE,
    KERNEL_UNAVAILABLE_NOTE,
    RefreshStats,
    refresh_catalog,
)
from gis_cli.datacatalog.store import Catalog
from gis_cli.web.app import create_app
from gis_cli.web.catalog_scan import CatalogScanService
from gis_cli.web.security import load_or_create_token

BROKEN = "bad\uddd7name"


# ------------------------------------------------------------------ 桩内核
class _GdbKernel:
    """假内核：按 arcpy.da.Walk 的真实口径返回 gdb 成员（path=…gdb/成员）。"""

    def __init__(self, gdb: Path, members: list[tuple[str, str, int]], *, ok: bool = True) -> None:
        self.gdb = gdb
        self.members = members
        self.ok = ok
        self.calls = 0

    def execute(self, code: str, timeout: float = 0) -> Any:
        self.calls += 1
        if not self.ok:
            return SimpleNamespace(ok=False, error="arcpy not available", result=None)
        records = [
            {
                "path": str(self.gdb / name),
                "kind": kind,
                "container": str(self.gdb),
                "name": name,
                "crs_name": "CGCS2000",
                "wkid": 4490,
                "geom_type": "Polygon" if kind == "feature_class" else "",
                "feature_count": count,
                "extent": [110.0, 30.0, 112.0, 32.0] if kind == "feature_class" else None,
                "fields": [],
            }
            for name, kind, count in self.members
        ]
        return SimpleNamespace(ok=True, error=None, result={"records": records, "count": len(records)})


class _WalkFailedKernel:
    """假内核：gdb 整库枚举失败时 ``_scan_target`` 返回的那条占位记录。"""

    def __init__(self, gdb: Path, reason: str = "arcpy.da.Walk failed: lock not released") -> None:
        self.gdb = gdb
        self.reason = reason
        self.calls = 0

    def execute(self, code: str, timeout: float = 0) -> Any:
        self.calls += 1
        return SimpleNamespace(
            ok=True,
            error=None,
            result={
                "records": [
                    {
                        "path": str(self.gdb),
                        "kind": "gdb",
                        "container": "",
                        "name": self.gdb.name,
                        "fields": [],
                        "extra": {"error": self.reason},
                    }
                ]
            },
        )


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    root = workspace / "input"
    gdb = root / "result_data.gdb"
    gdb.mkdir(parents=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"fake")
    return workspace, gdb


def _kinds(catalog: Catalog) -> dict[str, int]:
    return catalog.counts()


# ------------------------------------------------------------------ 容器记账
def test_gdb_container_record_kept_when_kernel_unavailable(tmp_path: Path) -> None:
    """内核起不来也要有 gdb 容器记录：目录里不能凭空少一整个数据库。"""
    workspace, gdb = _workspace(tmp_path)
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")

    stats = refresh_catalog(catalog, [workspace / "input"], _GdbKernel(gdb, [], ok=False), workspace=workspace)

    assert stats.errors and stats.errors[0].startswith("kernel scan failed")
    container = catalog.get("input\\result_data.gdb")
    assert container is not None, f"内核不可用时 .gdb 也要有容器记录：{[r.path for r in catalog.list_datasets()]}"
    assert container.kind == "gdb"
    assert container.name == "result_data.gdb"
    assert container.extra.get("members") is None, "没成功枚举过就不能留成员标记"
    catalog.close()


def test_gdb_container_members_and_freshness(tmp_path: Path) -> None:
    """.gdb 落容器 + 成员；第二轮不重扫（成员数由成功枚举写入）。"""
    workspace, gdb = _workspace(tmp_path)
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    root = workspace / "input"

    kernel = _GdbKernel(gdb, [("city", "feature_class", 49), ("county", "feature_class", 385)])
    stats = refresh_catalog(catalog, [root], kernel, workspace=workspace)
    assert kernel.calls == 1
    assert stats.scanned == 2, stats
    assert _kinds(catalog) == {"gdb": 1, "feature_class": 2}

    container = catalog.get("input\\result_data.gdb")
    assert container is not None and container.extra.get("members") == 2
    city = catalog.get("input\\result_data.gdb\\city")
    assert city is not None and city.container == "input\\result_data.gdb"

    again = _GdbKernel(gdb, [("city", "feature_class", 49), ("county", "feature_class", 385)])
    stats = refresh_catalog(catalog, [root], again, workspace=workspace)
    assert again.calls == 0, "成员没变就不该重扫整个 .gdb"
    assert stats.skipped == 1
    catalog.close()


def test_gdb_member_count_only_counts_stored_members(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """成员数只算真正落库的成员（DAIR-373）。

    内核报了 2 个成员、其中 1 个写库失败时，容器不能记成「已枚举完整」：
    成员数必须在 `_store` 循环之后、只统计成功落库的那几条。
    """
    workspace, gdb = _workspace(tmp_path)
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    kernel = _GdbKernel(gdb, [("city", "feature_class", 49), ("county", "feature_class", 385)])
    real = scanner_module.record_from_scan

    def flaky(payload: dict[str, Any]) -> Any:
        if str(payload.get("name")) == "county":
            raise ValueError("模拟单条写库失败")
        return real(payload)

    monkeypatch.setattr(scanner_module, "record_from_scan", flaky)
    stats = refresh_catalog(catalog, [workspace / "input"], kernel, workspace=workspace)

    assert len(stats.errors) == 1, f"只有失败那条记错：{stats.errors}"
    assert stats.scanned == 1
    container = catalog.get("input\\result_data.gdb")
    assert container is not None
    assert container.extra.get("members") == 1, "写库失败的成员不能算进成员数"
    assert catalog.get("input\\result_data.gdb\\city") is not None
    assert catalog.get("input\\result_data.gdb\\county") is None
    catalog.close()


def test_gdb_walk_failure_is_readable_and_retried(tmp_path: Path) -> None:
    """某个库枚举失败：容器记录带原因（不是空白），下一轮必须重试出来。"""
    workspace, gdb = _workspace(tmp_path)
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    root = workspace / "input"

    failed = _WalkFailedKernel(gdb)
    refresh_catalog(catalog, [root], failed, workspace=workspace)
    container = catalog.get("input\\result_data.gdb")
    assert container is not None and container.extra.get("error") == failed.reason
    assert container.extra.get("members") is None

    # 读接口要把原因带给前端（目录树上标「不可读」，而不是一个空文件夹）。
    from gis_cli.web.catalog_api import list_datasets

    listed = list_datasets(workspace, kind="gdb")
    assert listed["items"][0]["error"] == failed.reason
    assert "不可读" not in json.dumps(listed, ensure_ascii=False)  # 文案在前端，接口只给原因

    good = _GdbKernel(gdb, [("city", "feature_class", 49)])
    stats = refresh_catalog(catalog, [root], good, workspace=workspace)
    assert good.calls == 1, "上一轮枚举失败 → 这一轮不能被新鲜度跳过"
    assert stats.scanned == 1
    container = catalog.get("input\\result_data.gdb")
    assert container is not None and container.extra.get("members") == 1
    assert not container.extra.get("error"), "成功重扫后旧的瞬时错误要清掉"
    catalog.close()


# ------------------------------------------------------------------ 坏文件名
def test_broken_name_only_degrades_its_own_record(tmp_path: Path) -> None:
    """一条坏名不能带走整轮扫描：同一 gdb 的其它成员必须照常落库。"""
    workspace, gdb = _workspace(tmp_path)
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    kernel = _GdbKernel(
        gdb,
        [("city", "feature_class", 49), (BROKEN, "feature_class", 1), ("county", "feature_class", 385)],
    )

    stats = refresh_catalog(catalog, [workspace / "input"], kernel, workspace=workspace)

    assert stats.errors == [], f"坏名只降级单条，不该记成整轮失败：{stats.errors}"
    assert stats.scanned == 3
    paths = {r.path: r for r in catalog.list_datasets()}
    assert "input\\result_data.gdb\\city" in paths
    assert "input\\result_data.gdb\\county" in paths
    broken_key = next(path for path in paths if path.startswith("input\\result_data.gdb\\bad"))
    assert "\uddd7" not in broken_key, "落库的路径必须是合法 UTF-8（sqlite 绑定就要求）"
    assert paths[broken_key].extra.get("error") == BAD_NAME_NOTE, "降级条目要带说明"

    # 落库内容能按 Starlette JSONResponse 的口径编码（就是用户看到的那个报错点）。
    from gis_cli.web.catalog_api import list_datasets

    listed = list_datasets(workspace, limit=50)
    json.dumps(listed, ensure_ascii=False).encode("utf-8")
    catalog.close()


def _client(workspace: Path, kernel: Callable[[], Any]) -> tuple[TestClient, dict[str, str]]:
    token = load_or_create_token(workspace)
    scanner = CatalogScanService(workspace, kernel=kernel)
    client = TestClient(create_app(workspace, token=token, scanner=scanner))
    return client, {"X-Web-Token": token}


def _wait_scan_done(client: TestClient, auth: dict[str, str], timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/api/catalog/scan", headers=auth).json()
        if not state["running"]:
            return state
        time.sleep(0.02)
    raise AssertionError("扫描没在预算内结束")


def test_scan_and_datasets_api_stay_200_with_broken_name(tmp_path: Path) -> None:
    """注入 ``\\uddd7`` 名称：扫描成功、接口 200、目录里既有 .gdb 又有成员。"""
    workspace, gdb = _workspace(tmp_path)
    kernel = _GdbKernel(gdb, [("city", "feature_class", 49), (BROKEN, "feature_class", 1)])
    client, auth = _client(workspace, lambda: kernel)

    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    state = _wait_scan_done(client, auth)
    assert state["error"] == "", f"坏名不该让整轮扫描失败：{state['error']}"

    response = client.get("/api/catalog/datasets", params={"limit": 50}, headers=auth)
    assert response.status_code == 200
    body = response.json()
    kinds = [item["kind"] for item in body["items"]]
    assert kinds.count("gdb") == 1 and kinds.count("feature_class") == 2
    gdb_item = next(item for item in body["items"] if item["kind"] == "gdb")
    assert gdb_item["path"] == "input\\result_data.gdb"
    broken = next(item for item in body["items"] if item["path"].startswith("input\\result_data.gdb\\bad"))
    assert broken["error"] == BAD_NAME_NOTE
    # 响应体本身没有孤立代理项（FastAPI 的 JSONResponse 就是在这里炸的）
    assert "\uddd7" not in response.text


def test_scan_state_error_text_is_json_safe(tmp_path: Path, monkeypatch: Any) -> None:
    """扫描整轮失败、错误里嵌了坏名路径时，状态接口仍要 200 且原因可读。"""
    workspace, _ = _workspace(tmp_path)

    def _boom(*args: Any, **kwargs: Any) -> RefreshStats:
        raise UnicodeEncodeError("utf-8", f"input\\{BROKEN}", 1, 2, "surrogates not allowed")

    monkeypatch.setattr("gis_cli.datacatalog.scanner.refresh_catalog", _boom)
    client, auth = _client(workspace, lambda: None)

    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    state = _wait_scan_done(client, auth)
    assert "UnicodeEncodeError" in state["error"]
    assert "\uddd7" not in json.dumps(state, ensure_ascii=False)


# ------------------------------------------------------------------ DAIR-374
#: 磁盘上真实的坏名（WTF-8 破损）：先打头，后面接有意义的后缀给 discover 认类型。
REAL_BROKEN_NAME = chr(0xDDD7) + "bad.csv"


class _OneRecordKernel:
    """假内核：为单个走内核的条目回一条可用记录（无 arcpy 时替身用）。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.calls = 0

    def execute(self, code: str, timeout: float = 0) -> Any:
        self.calls += 1
        return SimpleNamespace(
            ok=True,
            error=None,
            result={
                "records": [
                    {
                        "path": self.path,
                        "kind": "vector",
                        "name": Path(self.path).stem,
                        "geom_type": "Point",
                        "feature_count": 1,
                        "fields": [],
                    }
                ],
                "count": 1,
            },
        )


def _plain_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    root = workspace / "input"
    root.mkdir(parents=True)
    (root / "ok.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
    )
    return workspace, root


def _plant_real_broken_name(root: Path, monkeypatch: Any) -> bool:
    """在目录里放一个名字含孤立代理项的真文件；建不出来时退回 discover 桩。

    Windows/NTFS 允许未配对代理项作文件名，``Path.write_text`` 真能建出来——
    这正是 DAIR-371 打进真实工作区的坏名，也正是 DAIR-374 复现失败的那条。
    """
    target = root / REAL_BROKEN_NAME
    try:
        target.write_text("a,b\n1,2\n", encoding="utf-8")
        return True
    except (OSError, UnicodeEncodeError):  # pragma: no cover - 非 Windows
        real_discover = scanner_module.discover
        stamp = time.time()
        fake = scanner_module.Discovery(
            path=str(target), kind="table", mtime=stamp, size=7
        )
        monkeypatch.setattr(
            scanner_module,
            "discover",
            lambda roots, **kwargs: [*real_discover(roots, **kwargs), fake],
        )
        return False


def test_real_broken_filename_only_degrades_its_own_record(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """磁盘上有真坏名时整轮扫描要成功（DAIR-374 打回点）。

    崩点在 ``is_fresh``/``remove_missing`` 拿磁盘原始路径绑 sqlite:
    ``UnicodeEncodeError: surrogates not allowed``，一条坏名就把同轮的正常
    ``ok.geojson`` 一起带掉。
    """
    workspace, root = _plain_workspace(tmp_path)
    real = _plant_real_broken_name(root, monkeypatch)
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    kernel = _OneRecordKernel(str(root / "ok.geojson"))

    stats = refresh_catalog(catalog, [root], kernel, workspace=workspace)

    assert stats.errors == [], f"坏名只降级自己，不能记成整轮失败：{stats.errors}"
    assert stats.scanned >= 2, f"坏名 + 正常 geojson 都该落库：{stats}"
    paths = {r.path: r for r in catalog.list_datasets()}
    assert "input\\ok.geojson" in paths, "同轮的正常文件必须照常落库"
    broken_key = next(path for path in paths if path.startswith("input\\" + chr(0xFFFD)))
    assert chr(0xDDD7) not in broken_key, "落库的键必须是合法 UTF-8"
    assert paths[broken_key].extra.get("error") == BAD_NAME_NOTE

    if real:
        # 第二轮：坏名记录不能被当成「文件已消失」每轮删一遍，且应判新鲜。
        again = refresh_catalog(catalog, [root], _OneRecordKernel(str(root / "ok.geojson")), workspace=workspace)
        assert again.errors == [], again.errors
        assert again.removed == 0, f"坏名记录还在磁盘上，不该被对账删掉：{again}"
        assert "input\\ok.geojson" in {r.path for r in catalog.list_datasets()}

    # 接口响应必须能按 JSONResponse 的口径编码（用户看到的就是这里）。
    from gis_cli.web.catalog_api import list_datasets

    listed = list_datasets(workspace, limit=50)
    json.dumps(listed, ensure_ascii=False).encode("utf-8")
    assert chr(0xDDD7) not in json.dumps(listed, ensure_ascii=False)
    catalog.close()


def test_scan_api_stays_200_with_real_broken_filename(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """bench 的复现路径：``POST /api/catalog/scan?force=1`` → 200，目录照常列出。"""
    workspace, root = _plain_workspace(tmp_path)
    _plant_real_broken_name(root, monkeypatch)
    kernel = _OneRecordKernel(str(root / "ok.geojson"))
    client, auth = _client(workspace, lambda: kernel)

    assert client.post("/api/catalog/scan?force=1", headers=auth).json()["triggered"] is True
    state = _wait_scan_done(client, auth)
    assert state["error"] == "", f"坏名只降级自己，整轮扫描不能失败：{state['error']}"
    assert "UnicodeEncodeError" not in state["error"]

    body = client.get("/api/catalog/datasets", params={"limit": 50}, headers=auth)
    assert body.status_code == 200
    paths = [item["path"] for item in body.json()["items"]]
    assert "input\\ok.geojson" in paths, f"同轮正常文件必须落库：{paths}"
    assert any(path.startswith("input\\" + chr(0xFFFD)) for path in paths), paths
    assert chr(0xDDD7) not in body.text, "接口响应里不能有孤立代理项"


def test_kernel_none_reports_readable_reason(tmp_path: Path) -> None:
    """``kernel=None``（arcpy 缺失的真实返回）：不抛 AttributeError，gdb 也带原因。"""
    workspace, gdb = _workspace(tmp_path)
    (workspace / "input" / "roads.shp").write_bytes(b"fake shp bytes")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")

    stats = refresh_catalog(catalog, [workspace / "input"], None, workspace=workspace)

    assert stats.kernel_unavailable == KERNEL_UNAVAILABLE_NOTE
    assert stats.errors == [KERNEL_UNAVAILABLE_NOTE]
    container = catalog.get("input\\result_data.gdb")
    assert container is not None, "内核缺席时 .gdb 仍要以文件夹形态出现"
    assert container.extra.get("error") == KERNEL_UNAVAILABLE_NOTE, "不能是空白：要给打不开的原因"
    catalog.close()


def test_scan_state_error_when_kernel_missing_is_readable(tmp_path: Path) -> None:
    """内核缺席时 ``state.error`` 要写「内核不可用」，而不是 ``AttributeError``。"""
    workspace, _ = _workspace(tmp_path)
    (workspace / "input" / "roads.shp").write_bytes(b"fake shp bytes")
    client, auth = _client(workspace, lambda: None)

    assert client.post("/api/catalog/scan", headers=auth).json()["triggered"] is True
    state = _wait_scan_done(client, auth)
    assert "内核不可用" in state["error"], state
    assert "AttributeError" not in state["error"], state
    assert "\uddd7" not in json.dumps(state, ensure_ascii=False)

    # gdb 容器在目录里带着同一句原因（不是 0 成员的空白条目）。
    gdb_item = client.get("/api/catalog/datasets", params={"kind": "gdb"}, headers=auth).json()
    assert gdb_item["items"][0]["error"] == KERNEL_UNAVAILABLE_NOTE

    # 内核缺席不算「扫过了」：同一份文件再轮询要真的重试，不能被 unchanged 挡住。
    again = client.post("/api/catalog/scan", headers=auth).json()
    assert again["triggered"] is True and again["reason"] == "started"
