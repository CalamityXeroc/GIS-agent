# -*- coding: utf-8 -*-
"""S6 预览 API 契约：抽稀上限、越界拦截、错误不 500、CRS 告警、成果包沙箱。

用真 TestClient + 真磁盘文件；只有「内核取几何」这一步用假 runner，因为
CI/开发机上不一定有 ArcPy（``html_pack.load_layer`` 的 GDB/shp 分支）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gis_cli.datacatalog.store import Catalog, DatasetRecord
from gis_cli.web.app import create_app
from gis_cli.web import preview_api
from gis_cli.web.preview_api import MAX_PREVIEW_BYTES, PREVIEW_POINT_BUDGET


class FailingRunner:
    """模拟「内核不可用」：用户看到的是 4xx + 原因，不是 500。"""

    def __init__(self, error: str = "arcpy 不可用") -> None:
        self.error = error
        self.calls = 0

    def run(self, code: str, **kwargs):
        self.calls += 1
        return SimpleNamespace(ok=False, result=None, error=self.error)


class GeometryRunner:
    """假内核：模拟 ``SearchCursor`` 成功返回 Esri JSON 几何（.gdb 要素类路径）。"""

    def __init__(self, geoms: list | None = None, *, error: str = "", sleep: float = 0.0) -> None:
        self.geoms = list(geoms or [])
        self.error = error
        self.sleep = sleep
        self.calls: list[dict] = []

    def run(self, code: str, **kwargs):
        self.calls.append({"code": code, **kwargs})
        if self.sleep:
            time.sleep(self.sleep)
        return SimpleNamespace(
            ok=True, result={"error": self.error, "geoms": self.geoms}, error=None
        )


class SlowFirstRunner:
    """第一次调用故意慢到超时，之后立刻返回：验证「超时先降级再返回」。"""

    def __init__(self, geoms: list | None = None, *, first_sleep: float = 1.0) -> None:
        self.geoms = list(geoms or [])
        self.first_sleep = first_sleep
        self.calls: list[dict] = []

    def run(self, code: str, **kwargs):
        self.calls.append({"code": code, **kwargs})
        if len(self.calls) == 1:
            time.sleep(self.first_sleep)
        return SimpleNamespace(
            ok=True, result={"error": "", "geoms": self.geoms, "simplify": {}}, error=None
        )


def _make_gdb(workspace: Path, member: str = "city", name: str = "数据.gdb") -> Path:
    """造一个 .gdb 目录 + 库内要素类路径（要素类在文件系统上不是文件）。"""
    gdb = workspace / "input" / name
    gdb.mkdir(parents=True, exist_ok=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"fake")
    return gdb / member


@pytest.fixture()
def preview_client(workspace: Path, token: str) -> TestClient:
    return TestClient(create_app(workspace, token=token, preview_runner=FailingRunner()))


@pytest.fixture()
def auth(token: str) -> dict[str, str]:
    return {"X-Web-Token": token}


def _write_geojson(path: Path, points: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = [
        {
            "type": "Feature",
            "properties": {"id": index},
            "geometry": {"type": "Point", "coordinates": [100 + index * 0.01, 30 + index * 0.01]},
        }
        for index in range(points)
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _seed(workspace: Path, path: Path, **overrides) -> None:
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    payload = {
        "path": str(path),
        "kind": "vector",
        "name": path.stem,
        "feature_count": 3,
        "crs_name": "WGS 1984",
        "wkid": 4326,
        "extent": [100.0, 30.0, 101.0, 31.0],
    }
    payload.update(overrides)
    catalog.upsert(DatasetRecord(**payload))
    catalog.close()


# ---------------------------------------------------------------- 抽稀 / 缓存
def test_preview_thins_100k_points_under_5mb_and_2s(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    layer = _write_geojson(workspace / "input" / "big.geojson", points=100_000)
    _seed(workspace, layer, feature_count=100_000, extent=[100.0, 30.0, 101.0, 31.0])

    started = time.perf_counter()
    response = preview_client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "max_features": 400},
        headers=auth,
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200, response.text
    assert len(response.content) <= MAX_PREVIEW_BYTES
    assert elapsed < 2.0, f"首次预览耗时 {elapsed:.2f}s"

    body = response.json()
    assert body["feature_count"] == 100_000
    assert body["returned_features"] == 400
    assert body["truncated"] is True
    # 顶点预算（preview 侧比报告的 20k 宽一些，5MB 才是硬闸）。
    assert body["point_count"] <= PREVIEW_POINT_BUDGET
    assert body["point_budget"] == PREVIEW_POINT_BUDGET
    assert body["geojson"]["type"] == "FeatureCollection"
    assert body["crs_warning"] is None


def test_unseeded_geojson_reports_true_count_and_truncation(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    """不在 catalog 里的 .geojson：feature_count 取文件全量，truncated 不能谎报 false。

    读盘时就截要素的话，feature_count 只剩 400，前端会以为「这层本来只有 400 个」，
    于是既不显示「部分」也不提示还有 600 个没画（DAIR-368 回归）。
    """
    layer = _write_geojson(workspace / "input" / "unseeded.geojson", points=1_000)

    response = preview_client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "max_features": 400},
        headers=auth,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["returned_features"] == 400
    assert body["feature_count"] == 1_000, "全量要素数不能是截断后的数字"
    assert body["truncated"] is True


def test_seeded_geojson_without_count_still_reports_truncation(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    """catalog 有记录但 ``feature_count`` 为空：直读仍知道全量，不能谎报完整。

    scanner 对 .geojson 只做轻量画像，记录的 feature_count 是 NULL；
    ``_fit_payload`` 的 truncated 依赖它，不补全就会把「只给了前 400 个」
    报成 truncated=false（DAIR-375 缺陷 1）。
    """
    layer = _write_geojson(workspace / "input" / "county.geojson", points=1_000)
    _seed(workspace, layer, feature_count=None, extent=[100.0, 30.0, 101.0, 31.0])

    response = preview_client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "max_features": 400},
        headers=auth,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["returned_features"] == 400
    assert body["feature_count"] == 1_000, "catalog 没画像时要用直读的全量条数"
    assert body["truncated"] is True


def test_same_path_second_call_hits_cache(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    layer = _write_geojson(workspace / "input" / "cache_me.geojson", points=2_000)
    _seed(workspace, layer, feature_count=2_000, extent=[100.0, 30.0, 120.0, 50.0])

    first = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    second = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()

    assert first["cache"]["hit"] is False
    assert first["cache"]["read_ms"] >= 0
    assert second["cache"]["hit"] is True
    # 命中缓存不再读几何：响应里不会出现 read_ms。
    assert "read_ms" not in second["cache"]
    assert second["geojson"] == first["geojson"]
    assert list((workspace / ".gis_agent" / "preview_cache").glob("*.json"))


# ---------------------------------------------------------------- 越界 / 错误
def test_path_outside_workspace_is_403(
    workspace: Path, preview_client: TestClient, auth: dict[str, str], tmp_path_factory
):
    outside = tmp_path_factory.mktemp("outside") / "secret.geojson"
    _write_geojson(outside)

    response = preview_client.get(
        "/api/catalog/preview", params={"path": str(outside)}, headers=auth
    )
    assert response.status_code == 403
    assert "outside workspace" in response.json()["detail"]

    traversal = preview_client.get(
        "/api/catalog/preview", params={"path": "../../secret.geojson"}, headers=auth
    )
    assert traversal.status_code == 403


def test_gdb_directory_empty_layer_and_permission_are_4xx(
    workspace: Path, preview_client: TestClient, auth: dict[str, str], token: str
):
    gdb = workspace / "input" / "fake.gdb"
    gdb.mkdir(parents=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"not really")

    response = preview_client.get(
        "/api/catalog/preview", params={"path": str(gdb)}, headers=auth
    )
    assert response.status_code == 422
    assert "要素类" in response.json()["detail"]

    empty = workspace / "input" / "empty.geojson"
    empty.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    response = preview_client.get(
        "/api/catalog/preview", params={"path": str(empty)}, headers=auth
    )
    assert response.status_code == 422
    assert "没有可预览的要素" in response.json()["detail"]

    shp = workspace / "input" / "locked.shp"
    shp.write_bytes(b"fake")

    class DenyRunner:
        def run(self, code, **kwargs):
            raise PermissionError("拒绝访问")

    denied = TestClient(create_app(workspace, token=token, preview_runner=DenyRunner()))
    response = denied.get(
        "/api/catalog/preview", params={"path": str(shp)}, headers=auth
    )
    assert response.status_code == 403
    assert "无权限" in response.json()["detail"]

    missing = preview_client.get(
        "/api/catalog/preview", params={"path": str(workspace / "input" / "nope.shp")},
        headers=auth,
    )
    assert missing.status_code == 404


def test_gdb_feature_class_with_missing_container_is_404(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    gone = workspace / "input" / "nope.gdb" / "city"
    response = preview_client.get(
        "/api/catalog/preview", params={"path": str(gone)}, headers=auth
    )
    assert response.status_code == 404


# ---------------------------------------------------------------- .gdb 要素类
def test_table_preview_is_422_no_geometry_without_kernel(workspace: Path, token: str):
    """非空间表（catalog kind=table）不打内核：直接 422 + code=no_geometry（DAIR-343）。"""
    layer = _make_gdb(workspace, member="统计")
    runner = GeometryRunner(
        [{"rings": [[[100, 30], [101, 30], [101, 31], [100, 31], [100, 30]]]}]
    )
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))
    _seed(workspace, layer, kind="table", name="统计", feature_count=49)
    auth = {"X-Web-Token": token}

    response = client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "no_geometry"
    assert detail["reason"] == "该数据集是非空间表（没有几何），只能看属性表"
    assert detail["suggestion"] == "双击打开属性表"
    assert detail["elapsed_ms"] >= 0
    # 关键：内核一次都没起（不然就是白等一次必然失败的 SearchCursor）。
    assert runner.calls == []


def test_kernel_table_has_no_geometry_maps_to_no_geometry(workspace: Path, token: str):
    """旧 catalog（kind 还是 feature_class）的兜底：内核那句报错映射成同一个 code。"""
    layer = _make_gdb(workspace)
    runner = GeometryRunner([], error="ERROR 999999: table has no geometry")
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))
    _seed(workspace, layer, kind="feature_class", name="city")
    auth = {"X-Web-Token": token}

    response = client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "no_geometry"
    assert detail["suggestion"] == "双击打开属性表"
    assert len(runner.calls) == 1, "旧记录还是要走一次内核才能知道它没有几何"


def test_gdb_feature_class_preview_reads_via_kernel(workspace: Path, token: str):
    """catalog 里大部分条目是 .gdb 内的要素类：不得再报「数据文件不存在」。"""
    layer = _make_gdb(workspace)
    runner = GeometryRunner([{"rings": [[[100, 30], [101, 30], [101, 31], [100, 31], [100, 30]]]}])
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))
    _seed(
        workspace,
        layer,
        kind="feature_class",
        name="city",
        feature_count=49,
        wkid=4508,
        extent=[100.0, 30.0, 101.0, 31.0],
    )
    auth = {"X-Web-Token": token}

    response = client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "feature_class"
    assert body["feature_count"] == 49
    assert body["returned_features"] == 1
    # 4508 不是 4326：DAIR-299 后实际重投影到 4326（100,30 那圈落在赤道非洲），无告警。
    assert body["crs"] == "EPSG:4326"
    assert body["crs_warning"] is None
    ring = body["geojson"]["features"][0]["geometry"]["coordinates"][0]
    assert all(0.0 < y < 45.0 for _, y in ring)  # 30,31 度纬度重投影后不变

    # 真走了内核，且传给 SearchCursor 的是要素类全路径（不是「文件」判定）。
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert ".gdb" in call["code"] and "SearchCursor" in call["code"]
    assert call["timeout"] > 0
    # 顶点预算下沉到内核（DAIR-368）：全精度几何不进程间回传，内核里先减点。
    assert "simplify_esri_geometries" in call["code"]
    assert "_max_points = %d" % PREVIEW_POINT_BUDGET in call["code"]
    # 预算按 catalog 的真实要素数放大，不再是死的 10s。
    assert call["timeout"] == pytest.approx(10 + 0.06 * 49)

    # 同路径第二次命中 (path, mtime) 缓存，不再读内核。
    again = client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert again["cache"]["hit"] is True
    assert len(runner.calls) == 1


class RowRunner:
    """假内核：模拟带属性列的 ``SearchCursor``（``fields=`` 路径，DAIR-330）。"""

    def __init__(self, rows: list | None = None, *, error: str = "") -> None:
        self.rows = list(rows or [])
        self.error = error
        self.calls: list[str] = []

    def run(self, code: str, **kwargs):
        self.calls.append(code)
        return SimpleNamespace(
            ok=True, result={"error": self.error, "rows": self.rows}, error=None
        )


def test_preview_fields_attach_whitelisted_attributes(workspace: Path, token: str):
    """``fields=`` 把真实字段带回 features.properties（唯一值符号化的前提）。"""
    layer = _make_gdb(workspace)
    runner = RowRunner(
        [
            [
                {"rings": [[[100, 30], [101, 30], [101, 31], [100, 31], [100, 30]]]},
                ["林地", 12.5],
            ]
        ]
    )
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))
    _seed(
        workspace,
        layer,
        kind="feature_class",
        name="city",
        fields=[
            {"name": "TDLYMC", "alias": "", "type": "String", "length": 40},
            {"name": "MJ_HA", "alias": "", "type": "Double", "length": 8},
        ],
    )
    auth = {"X-Web-Token": token}

    response = client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "fields": "TDLYMC,MJ_HA"},
        headers=auth,
    )
    assert response.status_code == 200, response.text
    props = response.json()["geojson"]["features"][0]["properties"]
    assert props == {"TDLYMC": "林地", "MJ_HA": 12.5}
    # 字段名以列表字面量注入内核代码，不是拼进语法。
    assert '["SHAPE@JSON"] + _names' in runner.calls[0]
    assert "'TDLYMC'" in runner.calls[0]


def test_preview_fields_ignore_unknown_columns(workspace: Path, token: str):
    """不在 catalog 白名单里的列名直接丢掉：不猜、不把未知文本拼进内核。"""
    layer = _make_gdb(workspace)
    runner = RowRunner([[{"x": 100.0, "y": 30.0}, ["林地"]]])
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))
    _seed(
        workspace,
        layer,
        kind="feature_class",
        name="city",
        fields=[{"name": "TDLYMC", "alias": "", "type": "String", "length": 40}],
    )

    response = client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "fields": "TDLYMC,; DROP TABLE x"},
        headers={"X-Web-Token": token},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["geojson"]["features"][0]["properties"] == {"TDLYMC": "林地"}
    literal = runner.calls[0].split("_names = ")[1].split("\n")[0]
    assert "DROP" not in literal and ";" not in literal


def test_preview_fields_change_cache_key(workspace: Path, token: str):
    """同一图层带不同 fields 不能命中同一份缓存（否则属性会串）。"""
    layer = _make_gdb(workspace)
    runner = RowRunner([[{"x": 100.0, "y": 30.0}, ["林地"]]])
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))
    _seed(
        workspace,
        layer,
        kind="feature_class",
        name="city",
        fields=[
            {"name": "TDLYMC", "alias": "", "type": "String", "length": 40},
            {"name": "MJ_HA", "alias": "", "type": "Double", "length": 8},
        ],
    )
    auth = {"X-Web-Token": token}
    params = {"path": str(layer), "fields": "TDLYMC"}

    first = client.get("/api/catalog/preview", params=params, headers=auth).json()
    again = client.get("/api/catalog/preview", params=params, headers=auth).json()
    other = client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "fields": "MJ_HA"},
        headers=auth,
    ).json()
    assert first["cache"]["hit"] is False
    assert again["cache"]["hit"] is True  # 同 fields 命中同一份缓存
    assert other["cache"]["hit"] is False  # 换字段就是另一份
    assert len(runner.calls) == 2


def test_gdb_feature_class_relative_catalog_path_resolves(workspace: Path, token: str):
    """catalog 里有 `workspace/input/....gdb/fc` 这种相对写法，不能因此 404。"""
    layer = _make_gdb(workspace)
    runner = GeometryRunner([{"x": 100.0, "y": 30.0}])
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))
    raw = f"{workspace.name}/input/数据.gdb/city"

    response = client.get(
        "/api/catalog/preview",
        params={"path": raw},
        headers={"X-Web-Token": token},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["path"].endswith("数据.gdb\\city") or body["path"].endswith("数据.gdb/city")
    assert body["returned_features"] == 1


def test_gdb_feature_class_without_kernel_is_503(workspace: Path, token: str):
    layer = _make_gdb(workspace)
    client = TestClient(
        create_app(workspace, token=token, preview_runner=FailingRunner("arcpy 导入挂死"))
    )
    response = client.get(
        "/api/catalog/preview",
        params={"path": str(layer)},
        headers={"X-Web-Token": token},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "kernel_unavailable"
    assert "arcpy 导入挂死" in detail["reason"]
    # 关键回归：路径判定已经对了，不会再谎报文件不存在。
    assert "数据文件不存在" not in response.text


def test_gdb_feature_class_timeout_is_422_with_suggestion(
    workspace: Path, token: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GIS_PREVIEW_READ_TIMEOUT", "0.4")
    layer = _make_gdb(workspace)
    runner = GeometryRunner(sleep=3.0)
    client = TestClient(
        create_app(workspace, token=token, preview_runner=runner)
    )

    started = time.perf_counter()
    response = client.get(
        "/api/catalog/preview",
        params={"path": str(layer)},
        headers={"X-Web-Token": token},
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "preview_timeout"
    # 报的仍是第一档预算（用户看到的 10s 语义），同时说明降级也试过了。
    assert detail["timeout_s"] == 0.4
    assert detail["elapsed_ms"] >= 300
    assert detail["degraded"]["attempt"]["feature_limit"] == 100
    assert "要素类太大或内核不可用" in detail["suggestion"]
    assert len(runner.calls) == 2, "超时后必须先降级重试，而不是直接 422"
    assert elapsed < 3.0, f"超时没有收敛在预算内：{elapsed:.2f}s"


def test_timeout_degrades_to_partial_features_instead_of_422(
    workspace: Path, token: str, monkeypatch: pytest.MonkeyPatch
):
    """第一次读不完就降级（少要素 + 更粗顶点预算）返回部分结果，不是 422。"""
    monkeypatch.setenv("GIS_PREVIEW_READ_TIMEOUT", "0.4")
    monkeypatch.setenv("GIS_PREVIEW_MAX_READ_TIMEOUT", "0.4")
    layer = _make_gdb(workspace)
    geoms = [{"rings": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}]
    runner = SlowFirstRunner(geoms, first_sleep=1.2)
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))

    started = time.perf_counter()
    response = client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers={"X-Web-Token": token}
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(runner.calls) == 2, "先超时再降级：两次内核调用"
    assert "_limit = 100" in runner.calls[1]["code"], "降级要把要素数压到 1/4"
    assert body["degraded"]["code"] == "preview_degraded"
    assert body["degraded"]["feature_limit"] == 100
    # 预算按服务端实际值下发，前端不用自己猜 10s。
    assert body["budget_s"] == 0.4
    assert body["returned_features"] == 1
    assert elapsed < 2.5


def test_degraded_result_is_not_cached(
    workspace: Path, token: str, monkeypatch: pytest.MonkeyPatch
):
    """降级（只读了部分要素）的结果不得落缓存（DAIR-370 复审阻塞项）。

    一次瞬时慢（内核冷启动/排队）不能把「前 100 个要素」写成这个图层的永久
    答案：降级只对本次请求有效，下一次预览必须重新尝试读全量。
    """
    monkeypatch.setenv("GIS_PREVIEW_READ_TIMEOUT", "0.4")
    monkeypatch.setenv("GIS_PREVIEW_MAX_READ_TIMEOUT", "0.4")
    layer = _make_gdb(workspace)
    geoms = [{"rings": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}]
    runner = SlowFirstRunner(geoms, first_sleep=1.2)  # 只有第一次慢
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))

    first = client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers={"X-Web-Token": token}
    )
    assert first.status_code == 200, first.text
    assert first.json()["degraded"]["code"] == "preview_degraded"
    assert len(runner.calls) == 2, "第一次：超时 + 降级重试"
    assert not (workspace / preview_api.CACHE_DIR).exists() or not list(
        (workspace / preview_api.CACHE_DIR).glob("*.json")
    ), "降级结果不能落盘"

    second = client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers={"X-Web-Token": token}
    )
    assert second.status_code == 200, second.text
    assert second.json()["cache"]["hit"] is False, "降级结果不能当缓存答案"
    assert second.json()["degraded"] is None, "第二次已经正常读完，不该再降级"
    assert len(runner.calls) == 3, "第二次必须重新读一次（然后命中新的正常结果）"


def test_read_timeout_scales_with_feature_count(monkeypatch: pytest.MonkeyPatch):
    """10s 不再是硬上限：要素多就给更多时间，两个环境变量可调。"""
    from gis_cli.web import preview_api

    monkeypatch.delenv("GIS_PREVIEW_READ_TIMEOUT", raising=False)
    monkeypatch.delenv("GIS_PREVIEW_MAX_READ_TIMEOUT", raising=False)
    assert preview_api._read_timeout() == 10.0
    assert preview_api._read_timeout(42) == pytest.approx(10 + 0.06 * 42)
    assert preview_api._read_timeout(5000) == preview_api.MAX_READ_TIMEOUT_S

    # 只设保底：上限跟着保底走，运维只改一个数时行为必须是确定的。
    monkeypatch.setenv("GIS_PREVIEW_READ_TIMEOUT", "3")
    assert preview_api._read_timeout(5000) == 3.0

    # 两个都设就是「保底 → 上限」的区间。
    monkeypatch.setenv("GIS_PREVIEW_MAX_READ_TIMEOUT", "30")
    assert preview_api._read_timeout(400) == pytest.approx(3 + 0.06 * 400)
    assert preview_api._read_timeout(5000) == 30.0


def test_preview_keeps_adjacent_polygons_seamless(
    workspace: Path, preview_client: TestClient, auth: dict[str, str], monkeypatch
):
    """相邻面共享边界两侧必须逐点相同（旧 stride 抽稀会各抽一半 → 白缝）。"""
    # 预算压到 40 点，保证这条 600 点的小层一定被简化（预算够时是全精度，见下条测试）。
    monkeypatch.setattr(preview_api, "PREVIEW_POINT_BUDGET", 40)
    detail = 300
    left = [[0.0, 0.0], [1.0, 0.0]] + [[1.0, i / detail] for i in range(1, detail)] + [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    right = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]] + [[1.0, i / detail] for i in range(detail - 1, 0, -1)] + [[1.0, 0.0]]
    layer = workspace / "input" / "邻里.geojson"
    layer.parent.mkdir(parents=True, exist_ok=True)
    layer.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"n": "左"}, "geometry": {"type": "Polygon", "coordinates": [left]}},
                    {"type": "Feature", "properties": {"n": "右"}, "geometry": {"type": "Polygon", "coordinates": [right]}},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _seed(workspace, layer, feature_count=2, extent=[0.0, 0.0, 2.0, 1.0])

    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()

    left_out, right_out = body["geojson"]["features"]
    left_ring = left_out["geometry"]["coordinates"][0]
    right_ring = right_out["geometry"]["coordinates"][0]
    shared_left = [tuple(p) for p in left_ring if p[0] == 1.0]
    shared_right = [tuple(p) for p in right_ring if p[0] == 1.0]
    assert len(shared_left) >= 2 and len(shared_right) >= 2
    assert shared_left == shared_right[::-1], "共享边两侧坐标必须完全一致"
    assert body["point_count"] < 2 * detail, "确实做了简化"
    assert body["simplified"] is True


def test_runner_timeout_error_maps_to_422_timeout(workspace: Path, token: str):
    """内核自己报 TimeoutExpired（软中断失败）时，同样是 422 + 建议。"""

    class TimeoutRunner:
        def run(self, code, **kwargs):
            return SimpleNamespace(
                ok=False,
                result=None,
                error={"type": "TimeoutExpired", "message": "Execution exceeded 10s"},
            )

    layer = _make_gdb(workspace)
    client = TestClient(create_app(workspace, token=token, preview_runner=TimeoutRunner()))
    response = client.get(
        "/api/catalog/preview",
        params={"path": str(layer)},
        headers={"X-Web-Token": token},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "preview_timeout"
    assert detail["suggestion"]


def test_kernel_failure_is_503_with_reason(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    shp = workspace / "input" / "no_kernel.shp"
    shp.parent.mkdir(parents=True, exist_ok=True)
    shp.write_bytes(b"fake")
    response = preview_client.get(
        "/api/catalog/preview", params={"path": str(shp)}, headers=auth
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "kernel_unavailable"
    assert "arcpy 不可用" in detail["reason"]


# ---------------------------------------------------------------- CRS
@pytest.mark.parametrize("wkid", [4490, 32650])
def test_known_non_4326_wkid_is_reprojected(
    workspace: Path, preview_client: TestClient, auth: dict[str, str], wkid
):
    """已知 wkid（DAIR-299）：实际重投影到 4326，不再报「未做投影变换」。"""
    layer = _write_geojson(workspace / "input" / f"crs_{wkid}.geojson")
    _seed(workspace, layer, wkid=wkid, crs_name="CGCS2000")
    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert body["crs"] == "EPSG:4326"
    assert body["crs_warning"] is None


def test_3857_has_no_warning(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    layer = _write_geojson(workspace / "input" / "mercator.geojson")
    _seed(workspace, layer, wkid=3857, crs_name="WGS 1984 Web Mercator")
    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert body["crs_warning"] is None


# ---------------------------------------------------------------- 字段画像
def test_profile_adds_numeric_range_from_samples(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    layer = _write_geojson(workspace / "input" / "profile.geojson")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(
        DatasetRecord(
            path=str(layer),
            kind="vector",
            name="profile",
            feature_count=3,
            wkid=4326,
            fields=[
                {"name": "POP", "type": "Double", "samples": [1.5, 10, None], "null_ratio": 0.33},
                {"name": "NAME", "type": "String", "samples": ["甲", "乙"], "null_ratio": 0.0},
            ],
        )
    )
    catalog.close()

    response = preview_client.get(
        f"/api/catalog/datasets/{layer.as_posix()}/profile", headers=auth
    )
    assert response.status_code == 200, response.text
    body = response.json()
    fields = {field["name"]: field for field in body["fields"]}
    assert body["numeric_fields"] == ["POP"]
    assert fields["POP"]["min"] == 1.5
    assert fields["POP"]["max"] == 10.0
    assert fields["POP"]["numeric"] is True
    assert fields["POP"]["sample_count"] == 3
    assert fields["NAME"]["numeric"] is False
    assert "min" not in fields["NAME"]


# ---------------------------------------------------------------- 成果包
def _seed_reports(workspace: Path) -> Path:
    out = workspace / "output" / "任务A"
    out.mkdir(parents=True)
    (out / "成果包.html").write_text("<!doctype html><title>包</title>", encoding="utf-8")
    (out / "溯源卡.html").write_text("<!doctype html><title>卡</title>", encoding="utf-8")
    (out / "notes.html").write_text("<p>不是成果包</p>", encoding="utf-8")
    return out


def test_reports_list_only_package_and_card(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    _seed_reports(workspace)
    body = preview_client.get("/api/reports", headers=auth).json()
    names = {item["name"] for item in body["items"]}
    assert names == {"任务A/成果包.html", "任务A/溯源卡.html"}
    kinds = {item["kind"] for item in body["items"]}
    assert kinds == {"package", "provenance"}


def test_report_preview_is_sandboxed_html(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    _seed_reports(workspace)

    response = preview_client.get("/api/reports/任务A/成果包.html", headers=auth)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "sandbox" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "包" in response.text

    download = preview_client.get(
        "/api/reports/任务A/成果包.html", params={"download": "true"}, headers=auth
    )
    assert "attachment" in download.headers["content-disposition"]


def test_report_rejects_non_html_and_escape(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    _seed_reports(workspace)
    (workspace / "output" / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "secret.html").write_text("<p>越界</p>", encoding="utf-8")

    assert (
        preview_client.get("/api/reports/任务A/no.txt", headers=auth).status_code
        in (403, 404, 415)
    )
    # 编码过的 ..%2F 才会保留到 path 参数里；解析后必须被 output/ 白名单挡下。
    escaped = preview_client.get("/api/reports/..%2F..%2Fsecret.html", headers=auth)
    assert escaped.status_code in (403, 404)
    assert "越界" not in escaped.text


# ---------------------------------------------------------------- CRS 归一（DAIR-299）
def test_4548_preview_is_reprojected_to_4326(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    """EPSG:4548（CGCS2000 3度带 CM=117E）坐标字面值 → 必须落到北京并去掉误报。"""
    layer = workspace / "input" / "bj4548.geojson"
    layer.parent.mkdir(parents=True, exist_ok=True)
    features = [
        {
            "type": "Feature",
            "properties": {"id": index},
            "geometry": {
                "type": "Point",
                # 4548 真值：445000,4405000 ≈ 116.358, 39.777（北京丰台一带）
                "coordinates": [445000.0 + index * 1000.0, 4405000.0 + index * 1000.0],
            },
        }
        for index in range(3)
    ]
    layer.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    _seed(
        workspace,
        layer,
        wkid=4548,
        crs_name="CGCS2000 / 3-degree Gauss-Kruger CM 117E",
        extent=[444000.0, 4404000.0, 447000.0, 4407000.0],
    )

    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert body["crs"] == "EPSG:4326"
    assert body["crs_warning"] is None
    coords = [f["geometry"]["coordinates"] for f in body["geojson"]["features"]]
    # 重投影后必须落在 116°E/39°N 一带，不再是 44 万/440 万的米制值
    assert all(115.0 < x < 118.0 and 38.0 < y < 41.0 for x, y in coords)
    assert body["bbox"][0] > 100.0  # bbox 也必须是度


def test_4548_without_pyproj_degrades_with_warning(
    workspace: Path, preview_client: TestClient, auth: dict[str, str], monkeypatch
):
    """pyproj 不可用时降级：坐标原样返回 + crs_warning。"""
    import gis_cli.web.preview_api as preview_api

    layer = workspace / "input" / "noproj.geojson"
    _write_geojson(layer)  # 4326 字面坐标，但 catalog 标成 4548
    _seed(workspace, layer, wkid=4548, crs_name="CGCS2000")

    monkeypatch.setattr(preview_api, "_HAS_PYPROJ", False)
    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert body["crs"] is None
    assert "未做投影变换" in (body["crs_warning"] or "")
    # 坐标没被动过（100,30 起点）
    assert body["geojson"]["features"][0]["geometry"]["coordinates"][0] == 100.0


def test_cache_key_includes_pyproj_availability(
    workspace: Path, preview_client: TestClient, auth: dict[str, str], monkeypatch
):
    """切换 pyproj 可用性后不能命中降级 payload（DAIR-303 D4）。"""
    import gis_cli.web.preview_api as preview_api

    layer = _write_geojson(workspace / "input" / "cache_crs.geojson")
    _seed(workspace, layer, wkid=4548, crs_name="CGCS2000")
    params = {"path": str(layer)}

    monkeypatch.setattr(preview_api, "_HAS_PYPROJ", False)
    degraded = preview_client.get("/api/catalog/preview", params=params, headers=auth).json()
    assert degraded["cache"]["hit"] is False
    assert degraded["crs"] is None
    assert "未做投影变换" in (degraded["crs_warning"] or "")

    # 同一个工作区、同一图层：补上 pyproj 后必须重新算，不能复用降级结果。
    monkeypatch.setattr(preview_api, "_HAS_PYPROJ", True)
    reprojected = preview_client.get("/api/catalog/preview", params=params, headers=auth).json()
    assert reprojected["cache"]["hit"] is False
    assert reprojected["crs"] == "EPSG:4326"
    assert reprojected["crs_warning"] is None
    assert reprojected["geojson"]["features"][0]["geometry"]["coordinates"][0] != 100.0


def test_4326_preview_reports_crs_4326(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    layer = _write_geojson(workspace / "input" / "wgs84.geojson")
    _seed(workspace, layer)
    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert body["crs"] == "EPSG:4326"
    assert body["crs_warning"] is None


def test_unknown_wkid_keeps_unknown_warning(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    """wkid 缺失：坐标系未知告警保留，crs 为 null。"""
    layer = _write_geojson(workspace / "input" / "unknown.geojson")
    _seed(workspace, layer, wkid=None, crs_name="")
    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert body["crs"] is None
    assert "坐标系未知" in (body["crs_warning"] or "")


def test_reproject_collection_bad_wkid_returns_false():
    """pyproj 不认识的 wkid：返回 False 而不是抛异常。"""
    from gis_cli.web.preview_api import _reproject_collection

    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [1.0, 2.0]}}
        ],
    }
    assert _reproject_collection(fc, 9999999) is False
    # 坐标未被破坏
    assert fc["features"][0]["geometry"]["coordinates"] == [1.0, 2.0]


def test_3857_without_pyproj_does_not_claim_4326(
    workspace: Path, preview_client: TestClient, auth: dict[str, str], monkeypatch
):
    """3857 + 无 pyproj：坐标是米制，不能说自己是 4326。

    3857 在 KNOWN_CRS 里，所以 ``_crs_warning`` 返回 None；旧代码就是靠这个
    把米制数据报成 ``crs: EPSG:4326``，前端按度画 → 落错位置且无提示。
    """
    import gis_cli.web.preview_api as preview_api

    layer = workspace / "input" / "mercator.geojson"
    features = [
        {
            "type": "Feature",
            "properties": {"id": index},
            "geometry": {"type": "Point", "coordinates": [12952000.0, 4823000.0]},
        }
        for index in range(2)
    ]
    layer.parent.mkdir(parents=True, exist_ok=True)
    layer.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    _seed(
        workspace,
        layer,
        wkid=3857,
        crs_name="WGS 84 / Pseudo-Mercator",
        extent=[12951000.0, 4822000.0, 12953000.0, 4824000.0],
    )

    monkeypatch.setattr(preview_api, "_HAS_PYPROJ", False)
    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert body["crs"] is None
    assert "未做投影变换" in (body["crs_warning"] or "")
    # 米制坐标原样返回（没被当成度改写），bbox 也是米制
    assert body["geojson"]["features"][0]["geometry"]["coordinates"] == [12952000.0, 4823000.0]
    assert body["bbox"][0] > 1_000_000


def test_3857_preview_is_reprojected_to_4326(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    """有 pyproj 时 3857 也要真的转成经纬度（不能让前端把米当度画）。"""
    layer = workspace / "input" / "mercator_ok.geojson"
    features = [
        {
            "type": "Feature",
            "properties": {"id": index},
            "geometry": {"type": "Point", "coordinates": [12952000.0, 4823000.0]},
        }
        for index in range(2)
    ]
    layer.parent.mkdir(parents=True, exist_ok=True)
    layer.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    _seed(
        workspace,
        layer,
        wkid=3857,
        crs_name="WGS 84 / Pseudo-Mercator",
        extent=[12951000.0, 4822000.0, 12953000.0, 4824000.0],
    )

    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert body["crs"] == "EPSG:4326"
    assert body["crs_warning"] is None
    [x, y] = body["geojson"]["features"][0]["geometry"]["coordinates"]
    assert 115.0 < x < 118.0 and 38.0 < y < 41.0
    assert body["bbox"][0] > 100.0


def test_reproject_collection_is_all_or_nothing():
    """重投影层逆失败时不能留半米半度的混合几何（DAIR-301）。"""
    from gis_cli.web.preview_api import _reproject_collection

    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 0},
                "geometry": {"type": "Point", "coordinates": [445000.0, 4405000.0]},
            },
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {"type": "Point", "coordinates": [None, None]},
            },
        ],
    }
    assert _reproject_collection(fc, 4548) is True
    first = fc["features"][0]["geometry"]["coordinates"]
    # 第一个要素确实转过（度），空坐标原样保留 —— 全集仍是 4326 语义
    assert 115.0 < first[0] < 118.0
    assert fc["features"][1]["geometry"]["coordinates"] == [None, None]


# ---------------------------------------------------------------- 逐波段元数据
class BandRunner:
    """假内核：只响应逐波段元数据代码，其余（预览取几何）交给 geometry 列表。"""

    def __init__(self, bands: list | None = None, *, error: str = "", sleep: float = 0.0) -> None:
        self.bands = bands
        self.error = error
        self.sleep = sleep
        self.calls: list[str] = []

    def run(self, code: str, **kwargs):
        self.calls.append(code)
        if "_arcpy.Raster" not in code:
            return SimpleNamespace(ok=True, result={"error": "", "geoms": []}, error=None)
        if self.sleep:
            time.sleep(self.sleep)
        return SimpleNamespace(
            ok=True,
            result={
                "bands": self.bands if self.bands is not None else [],
                "band_count": len(self.bands or []),
                "error": self.error,
            },
            error=None,
        )


def _three_bands() -> list[dict]:
    return [
        {
            "band": index,
            "name": f"Band_{index}",
            "dtype": "U16",
            "min": 100.0,
            "max": 2099.0,
            "mean": 1099.38,
            "nodata": 0.0,
            "width": 4096,
            "height": 3072,
            "cell_size": 10.0,
            "color_interpretation": None,
        }
        for index in (1, 2, 3)
    ]


def _seed_raster(workspace: Path, *, band_count: int = 3) -> Path:
    raster = workspace / "input" / "dem_10m.tif"
    raster.parent.mkdir(parents=True, exist_ok=True)
    raster.write_bytes(b"fake-tiff")
    _seed(
        workspace,
        raster,
        kind="raster",
        name="dem_10m",
        feature_count=None,
        geom_type="Raster",
        extent=None,
        wkid=4547,
        raster={
            "band_count": band_count,
            "cell_size": 10.0,
            "width": 4096,
            "height": 3072,
            "min": 100.0,
            "max": 2099.0,
            "mean": 1099.38,
            "nodata": 0.0,
        },
    )
    return raster


def test_profile_reports_per_band_metadata(workspace: Path, token: str, auth: dict[str, str]):
    raster = _seed_raster(workspace)
    runner = BandRunner(_three_bands())
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))

    response = client.get(f"/api/catalog/datasets/{raster.as_posix()}/profile", headers=auth)
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["bands"]) == 3
    first = body["bands"][0]
    assert first["band"] == 1
    assert first["dtype"] == "U16"
    assert first["min"] == 100.0
    assert first["max"] == 2099.0
    assert first["mean"] == 1099.38
    assert first["nodata"] == 0.0
    assert first["unavailable"] is None
    assert body["bands_info"]["cached"] is False
    # 色彩解释 arcpy 没暴露 → 如实记一条 note，不是编一个值
    assert any("色彩解释" in note for note in body["bands_info"]["notes"])

    calls = len(runner.calls)
    again = client.get(
        f"/api/catalog/datasets/{raster.as_posix()}/profile", headers=auth
    ).json()
    assert again["bands_info"]["cached"] is True
    assert len(again["bands"]) == 3
    assert len(runner.calls) == calls, "命中缓存后不应再调内核"


def test_bands_endpoint_returns_three_bands(workspace: Path, token: str, auth: dict[str, str]):
    raster = _seed_raster(workspace)
    client = TestClient(
        create_app(workspace, token=token, preview_runner=BandRunner(_three_bands()))
    )
    body = client.get(
        "/api/catalog/raster/bands", params={"path": str(raster)}, headers=auth
    ).json()
    assert body["band_count"] == 3
    assert [band["band"] for band in body["bands"]] == [1, 2, 3]
    assert body["raster"]["band_count"] == 3
    assert body["cached"] is False


def test_bands_without_stats_are_unavailable_not_zero(
    workspace: Path, token: str, auth: dict[str, str]
):
    """栅格没算统计值时 min/max/mean 是 None：必须给原因，不能写 0 冒充。"""
    raster = _seed_raster(workspace, band_count=1)
    bands = [
        {
            "band": 1,
            "name": "Band_1",
            "dtype": "U16",
            "min": None,
            "max": None,
            "mean": None,
            "nodata": None,
            "width": 4096,
            "height": 3072,
            "cell_size": 10.0,
            "color_interpretation": None,
        }
    ]
    client = TestClient(create_app(workspace, token=token, preview_runner=BandRunner(bands)))
    body = client.get(
        "/api/catalog/raster/bands", params={"path": str(raster)}, headers=auth
    ).json()
    band = body["bands"][0]
    assert band["min"] is None and band["max"] is None and band["mean"] is None
    assert "未计算统计值" in band["unavailable"]
    assert band["dtype"] == "U16"  # 便宜的元数据照常给


def test_bands_kernel_failure_is_200_with_reason(
    workspace: Path, token: str, auth: dict[str, str]
):
    """无 arcpy / 内核挂了：profile 照常 200，bands 为空 + 原因（不是 500）。"""
    raster = _seed_raster(workspace)
    client = TestClient(create_app(workspace, token=token, preview_runner=FailingRunner()))
    response = client.get(
        f"/api/catalog/datasets/{raster.as_posix()}/profile", headers=auth
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["bands"] == []
    assert "arcpy 不可用" in body["bands_info"]["unavailable"]
    # catalog 侧的整栅格汇总仍在（扫描时算过），没被一起丢掉
    assert body["raster"]["band_count"] == 3


def test_bands_budget_is_enforced(workspace: Path, token: str, auth: dict[str, str], monkeypatch):
    """逐波段读取超过预算就返回原因，不能把请求挂在大栅格上。"""
    monkeypatch.setenv("GIS_BAND_BUDGET", "0.1")
    raster = _seed_raster(workspace)
    client = TestClient(
        create_app(workspace, token=token, preview_runner=BandRunner(_three_bands(), sleep=1.5))
    )
    started = time.perf_counter()
    body = client.get(
        "/api/catalog/raster/bands", params={"path": str(raster)}, headers=auth
    ).json()
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"预算没生效：{elapsed:.2f}s"
    assert body["bands"] == []
    assert "预算" in body["unavailable"]


def test_bands_on_vector_is_422(workspace: Path, preview_client: TestClient, auth: dict[str, str]):
    layer = _write_geojson(workspace / "input" / "not_raster.geojson")
    _seed(workspace, layer)
    response = preview_client.get(
        "/api/catalog/raster/bands", params={"path": str(layer)}, headers=auth
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "not_raster"
