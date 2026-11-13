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
from gis_cli.web.preview_api import MAX_PREVIEW_BYTES


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
    # 点也被抽稀：20k 是 html_pack 的坐标预算。
    assert body["point_count"] <= 20_000
    assert body["geojson"]["type"] == "FeatureCollection"
    assert body["crs_warning"] is None


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
    assert body["bbox"] == [100.0, 30.0, 101.0, 31.0]
    assert body["feature_count"] == 49
    assert body["returned_features"] == 1
    # 4508 不是 4326/3857：照原样画，但必须带告警。
    assert "未做投影变换" in (body["crs_warning"] or "")

    # 真走了内核，且传给 SearchCursor 的是要素类全路径（不是「文件」判定）。
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert ".gdb" in call["code"] and "SearchCursor" in call["code"]
    assert call["timeout"] > 0

    # 同路径第二次命中 (path, mtime) 缓存，不再读内核。
    again = client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert again["cache"]["hit"] is True
    assert len(runner.calls) == 1


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
    client = TestClient(
        create_app(workspace, token=token, preview_runner=GeometryRunner(sleep=3.0))
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
    assert detail["timeout_s"] == 0.4
    assert detail["elapsed_ms"] >= 300
    assert "要素类太大或内核不可用" in detail["suggestion"]
    assert elapsed < 2.0, f"超时没有收敛在预算内：{elapsed:.2f}s"


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
@pytest.mark.parametrize("wkid", [4490, 32650, None])
def test_non_4326_preview_carries_crs_warning(
    workspace: Path, preview_client: TestClient, auth: dict[str, str], wkid
):
    layer = _write_geojson(workspace / "input" / f"crs_{wkid}.geojson")
    _seed(workspace, layer, wkid=wkid, crs_name="CGCS2000")
    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer)}, headers=auth
    ).json()
    assert body["crs_warning"]
    assert "未做投影变换" in body["crs_warning"]


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
