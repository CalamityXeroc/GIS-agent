# -*- coding: utf-8 -*-
"""视口裁剪的预览契约（DAIR-386）。

用户看到的是「线/面只显示一部分」：前端把 `max_features` 写死 400、后端也只读
前 400 条，13 万条道路因此只画出 0.3%。修法有两半，这里测的是服务端那一半：

- `bbox`/`zoom` 入参：只读视口内的要素；
- 返回体里的 `total_in_view` / `coverage` / `truncated` 必须是**真数字**——
  视口内给全了就 `truncated=false`，超上限才 `true` 并给覆盖率。

`.shp` 走 `.shx` 直读（不进内核），`.gdb` 走内核并回 `view_total`，`.geojson`
在主进程筛。三条口径都要验：数值错了，前端文案就跟着错。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from _shapefiles import SHP_POINT, SHP_POLYLINE, write_shapefile
from gis_cli.datacatalog.store import Catalog, DatasetRecord
from gis_cli.report import filegdb_view
from gis_cli.web.app import create_app
from gis_cli.web.preview_api import (
    MAX_MAX_FEATURES,
    PREVIEW_POINT_BUDGET,
    VIEW_FEATURE_LIMITS,
    _fit_payload,
)

VIEW_BYTES_GATE = 5 * 1024 * 1024


def encoded_size(payload: dict) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


class FailingRunner:
    """内核不可用：能证明 `.shp`/`.geojson` 的视口读根本没碰内核。"""

    def __init__(self, error: str = "arcpy 不可用") -> None:
        self.error = error
        self.calls = 0

    def run(self, code: str, **kwargs):
        self.calls += 1
        return SimpleNamespace(ok=False, result=None, error=self.error)


class ViewRunner:
    """假内核：`.gdb` 要素类按视口读（模板会回 `view_total`）。"""

    def __init__(self, geoms: list | None = None, *, view_total: int = 0, exact: bool = True) -> None:
        self.geoms = list(geoms or [])
        self.view_total = view_total
        self.exact = exact
        self.calls: list[dict] = []

    def run(self, code: str, **kwargs):
        self.calls.append({"code": code, **kwargs})
        return SimpleNamespace(
            ok=True,
            result={
                "error": "",
                "geoms": self.geoms,
                "rows": [],
                "simplify": {},
                "view_total": self.view_total,
                "view_count_exact": self.exact,
            },
            error=None,
        )


@pytest.fixture()
def preview_client(workspace: Path, token: str) -> TestClient:
    """内核不可用的普通客户端：`.geojson` 不该因此报错。"""
    return TestClient(create_app(workspace, token=token, preview_runner=FailingRunner()))


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


def _make_gdb(workspace: Path, member: str = "city", name: str = "数据.gdb") -> Path:
    gdb = workspace / "input" / name
    gdb.mkdir(parents=True, exist_ok=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"fake")
    return gdb / member


def _write_geojson(path: Path, points: int = 3) -> Path:
    import json

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


def _shp(tmp_path: Path, name: str, shape_type, shapes, **kwargs) -> Path:
    return write_shapefile(
        tmp_path / "input" / name, shape_type=shape_type, shapes=shapes, **kwargs
    )


# ---------------------------------------------------------------- .shp（纯 Python 直读）
def test_preview_bbox_reads_only_features_in_view_without_kernel(workspace: Path, token: str):
    """`.shp` + 视口：`.shx` 直接筛，**不碰内核**；视口内多少就是多少。"""
    layer = _shp(
        workspace,
        "线.shp",
        SHP_POLYLINE,
        [
            [[[100.0, 30.0], [100.2, 30.2]]],
            [[[100.1, 30.1], [100.3, 30.0]]],
            [[[120.0, 40.0], [120.2, 40.2]]],  # 视口外
        ],
    )
    _seed(
        workspace,
        layer,
        geom_type="Polyline",
        feature_count=3,
        extent=[100.0, 30.0, 120.2, 40.2],
    )
    runner = FailingRunner()
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))

    response = client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99.5,29.5,101.0,31.0", "zoom": 13},
        headers={"X-Web-Token": token},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["returned_features"] == 2, "视口外的要素不该被读进来"
    assert body["total_in_view"] == 2
    assert body["coverage"] == 1.0
    assert body["truncated"] is False, "视口内都给全了，就不能谎报「部分」"
    assert body["view_bbox"] == [99.5, 29.5, 101.0, 31.0]
    assert body["view_zoom"] == 13
    assert body["view_count_exact"] is True
    assert body["feature_count"] == 3, "全量要素数仍要说清"
    assert runner.calls == 0, ".shp 视口读不该起内核"


def test_preview_bbox_does_not_truncate_before_simplifying(workspace: Path, token: str):
    """``max_features`` 只是兜底、不是读盘上限（DAIR-387 D1）。

    30 条小线在 5MB / 60k 顶点内绰绰有余：传 ``max_features=10`` 也必须给全 30 条。
    修复前是先 ``features[:10]`` 再简化——z11 的 28,921 条只画出 12,000（41%）就是
    这一个顺序造成的，而当时 5MB 只用了 36%、顶点只用了 46%。
    """
    shapes = [
        [[[100.0 + index * 0.01, 30.0], [100.0 + index * 0.01, 30.01]]] for index in range(30)
    ]
    layer = _shp(workspace, "多线.shp", SHP_POLYLINE, shapes)
    _seed(
        workspace,
        layer,
        geom_type="Polyline",
        feature_count=30,
        extent=[100.0, 30.0, 100.3, 30.01],
    )
    client = TestClient(create_app(workspace, token=token, preview_runner=FailingRunner()))

    body = client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31", "max_features": 10},
        headers={"X-Web-Token": token},
    ).json()

    assert body["returned_features"] == 30, "max_features 只该在 5MB 压不下去时兜底"
    assert body["total_in_view"] == 30
    assert body["coverage"] == 1.0
    assert body["truncated"] is False


def test_preview_bbox_keeps_attributes_for_symbolization(workspace: Path, token: str):
    """符号化要的字段跟着视口读一起回来（DBF 按记录号定位，不扫全表）。"""
    layer = _shp(
        workspace,
        "带属性.shp",
        SHP_POINT,
        [[100.0, 30.0], [100.1, 30.1], [120.0, 40.0]],
        fields=[("NAME", "C", 20, 0), ("LEN", "N", 10, 2)],
        rows=[("辛房路", 1.5), ("永定路", 2.5), ("远处", 3.5)],
        encoding="gbk",
        ldid=0x4D,
    )
    _seed(
        workspace,
        layer,
        geom_type="Point",
        feature_count=3,
        extent=[100.0, 30.0, 120.0, 40.0],
        fields=[{"name": "NAME", "type": "String"}, {"name": "LEN", "type": "Double"}],
    )
    client = TestClient(create_app(workspace, token=token, preview_runner=FailingRunner()))

    body = client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31", "fields": "NAME,LEN"},
        headers={"X-Web-Token": token},
    ).json()

    names = [feature["properties"]["NAME"] for feature in body["geojson"]["features"]]
    assert sorted(names) == ["永定路", "辛房路"]
    assert body["total_in_view"] == 2


# ---------------------------------------------------------------- .geojson（主进程筛）
def test_preview_geojson_is_filtered_by_view_bbox(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    layer = _write_geojson(workspace / "input" / "视口.geojson", points=5)
    _seed(workspace, layer, feature_count=5, extent=[100.0, 30.0, 101.0, 31.0])

    body = preview_client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99.99,29.99,100.015,30.015"},
        headers=auth,
    ).json()

    assert body["returned_features"] == 2
    assert body["total_in_view"] == 2
    assert body["coverage"] == 1.0
    assert body["truncated"] is False


# ---------------------------------------------------------------- .gdb（内核 extent 判交）
def test_preview_bbox_is_transformed_into_layer_crs(workspace: Path, token: str):
    """视口是 4326，图层是米制时要先换单位——否则筛出来是空的。"""
    geoms = [
        {
            "rings": [
                [
                    [12946456.0, 4844168.0],
                    [12947456.0, 4844168.0],
                    [12947456.0, 4845168.0],
                    [12946456.0, 4844168.0],
                ]
            ]
        }
    ]
    runner = ViewRunner(geoms, view_total=7)
    layer = _make_gdb(workspace)
    _seed(
        workspace,
        layer,
        kind="feature_class",
        name="city",
        feature_count=49,
        wkid=3857,
        extent=[12946456.0, 4844168.0, 12947456.0, 4845168.0],
    )
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))

    body = client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "116.30,39.85,116.55,39.95"},
        headers={"X-Web-Token": token},
    ).json()

    code = runner.calls[0]["code"]
    assert "_bbox = (" in code
    numbers = [float(value) for value in code.split("_bbox = (", 1)[1].split(")", 1)[0].split(",")]
    assert len(numbers) == 4
    assert numbers[0] > 1e6, f"没换算到米制：{numbers}"
    assert body["total_in_view"] == 7, "内核报的视口内要素数要透出来"
    assert body["coverage"] == pytest.approx(1 / 7, abs=1e-4)


def test_kernel_collects_more_features_without_widening_the_count_window(
    workspace: Path, token: str
):
    """视口读盘放宽（收集上限 ×3），但 .gdb 的计数扫描窗口不变（DAIR-391）。

    面档 3,000 → 收集 9,000（先简化再兜底截），扫描窗口仍是 3,000×20 = 60,000 行：
    计数策略（``view_count_exact`` / 「≥N」）不该因为要收更多几何而变慢。
    """
    runner = ViewRunner([{"x": 100.0, "y": 30.0}], view_total=7)
    layer = _make_gdb(workspace)
    _seed(
        workspace,
        layer,
        kind="feature_class",
        name="city",
        feature_count=900000,
        wkid=4326,
        geom_type="Polygon",
    )

    TestClient(create_app(workspace, token=token, preview_runner=runner)).get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31"},
        headers={"X-Web-Token": token},
    )

    code = runner.calls[0]["code"]
    assert "_limit = 9000" in code
    assert "_scan_cap = 60000" in code


def test_gdb_view_reads_by_index_oids_when_index_is_fresh(workspace: Path, token: str):
    """`.gdb` 有包络索引时按 OID 读（DAIR-393）：不再整层扫，计数就是索引命中数。"""
    layer = _make_gdb(workspace)
    _seed(workspace, layer, kind="feature_class", name="city", feature_count=3)
    target = filegdb_view.index_path(workspace, layer)
    filegdb_view.write_index(
        target,
        filegdb_view.index_key(layer, 3),
        [5, 6, 7],
        [
            (100.0, 30.0, 100.1, 30.1),
            (100.05, 30.05, 100.15, 30.15),
            (120.0, 40.0, 120.1, 40.1),  # 视口外
        ],
    )
    runner = ViewRunner([{"x": 100.0, "y": 30.0}], view_total=2)

    body = TestClient(create_app(workspace, token=token, preview_runner=runner)).get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31"},
        headers={"X-Web-Token": token},
    ).json()

    code = runner.calls[0]["code"]
    assert '_view_ns["read_by_oids"](' in code, "有索引就不该再整层扫"
    assert "_oids = [5, 6]" in code, "只读视口里命中的 OID"
    assert body["total_in_view"] == 2
    assert body["view_count_exact"] is True
    assert body["coverage"] == pytest.approx(0.5, abs=1e-4)


def test_gdb_view_dumps_index_when_missing(workspace: Path, token: str):
    """没有索引：这次整层扫顺便把索引建出来，下一次视口才快得起来。"""
    layer = _make_gdb(workspace)
    _seed(workspace, layer, kind="feature_class", name="city", feature_count=3)
    runner = ViewRunner([{"x": 100.0, "y": 30.0}], view_total=1)

    TestClient(create_app(workspace, token=token, preview_runner=runner)).get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31"},
        headers={"X-Web-Token": token},
    )

    code = runner.calls[0]["code"]
    assert '_view_ns["read_view"](' in code
    assert "_index_path = " in code and ".idx" in code, "扫描要顺手写索引"


def test_gdb_view_index_stale_falls_back_to_scan(workspace: Path, token: str):
    """索引与数据指纹对不上（有人改过库）→ 重扫，不能拿旧索引画错图。"""
    layer = _make_gdb(workspace)
    _seed(workspace, layer, kind="feature_class", name="city", feature_count=3)
    target = filegdb_view.index_path(workspace, layer)
    filegdb_view.write_index(target, "旧指纹", [5], [(100.0, 30.0, 100.1, 30.1)])
    runner = ViewRunner([{"x": 100.0, "y": 30.0}], view_total=1)

    TestClient(create_app(workspace, token=token, preview_runner=runner)).get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31"},
        headers={"X-Web-Token": token},
    )

    assert '_view_ns["read_by_oids"](' not in runner.calls[0]["code"]


def test_preview_bbox_marks_inexact_when_kernel_stops_early(workspace: Path, token: str):
    """内核扫到上限就停：只能给下限，前端要能看出「≥N」。"""
    runner = ViewRunner([{"x": 100.0, "y": 30.0}], view_total=20000, exact=False)
    layer = _make_gdb(workspace)
    _seed(workspace, layer, kind="feature_class", name="city", feature_count=900000, wkid=4326)

    body = TestClient(create_app(workspace, token=token, preview_runner=runner)).get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31"},
        headers={"X-Web-Token": token},
    ).json()

    assert body["total_in_view"] == 20000
    assert body["view_count_exact"] is False


def test_inexact_view_must_not_claim_full_coverage(workspace: Path, token: str):
    """扫描停下时 ``total_in_view`` 只是下限，不能拿来当分母（DAIR-388 阻塞项）。

    内核模板扫到 ``_scan_cap`` 行就停（面档 limit=3000 ⇒ 上限 60,000 行），这时
    返回数 == 数到的视口内数。把它当精确数会得到 ``coverage=1.0`` +
    ``truncated=false``：地图明显缺要素，图例却写「视口内 3 / 全量 900,000」而不带
    任何「没给全」提示——正是本单要消灭的那种谎报。
    """
    runner = ViewRunner([{"x": 100.0, "y": 30.0}] * 3, view_total=3, exact=False)
    layer = _make_gdb(workspace)
    _seed(
        workspace,
        layer,
        kind="feature_class",
        name="city",
        feature_count=900000,
        wkid=4326,
        geom_type="Polygon",
    )

    body = TestClient(create_app(workspace, token=token, preview_runner=runner)).get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31"},
        headers={"X-Web-Token": token},
    ).json()

    assert body["view_count_exact"] is False
    # 下限原样透出（前端写「≥3」），但百分比与「给全了」的判断一律不给。
    assert body["total_in_view"] == 3
    assert body["coverage"] is None, "下限不能当分母：算出来的 100% 是假的"
    assert body["truncated"] is True, "900,000 的图层只读到 3 条，必须标「没给全」"


def test_inexact_view_without_feature_count_still_marks_partial(
    workspace: Path, token: str
):
    """计数不精确 + catalog 没有全量数：没有分母，但「只给了一部分」必须说出去。

    ``expected_total=None`` 会让 ``truncated`` 落到 false，「部分」角标随之消失，
    只剩图例的「（未给全）」（DAIR-390）。也不能退一步拿 ``total_in_view`` 当分母
    ——那是扫描窗口的下限，拿它算出来的 100% 正是 DAIR-388 打回的谎报。
    """
    runner = ViewRunner([{"x": 100.0, "y": 30.0}] * 3, view_total=20000, exact=False)
    layer = _make_gdb(workspace)
    _seed(
        workspace,
        layer,
        kind="feature_class",
        name="city",
        feature_count=None,
        wkid=4326,
        geom_type="Polygon",
    )

    body = TestClient(create_app(workspace, token=token, preview_runner=runner)).get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31"},
        headers={"X-Web-Token": token},
    ).json()

    assert body["view_count_exact"] is False
    assert body["feature_count"] is None, "catalog 没画像：前端只能写「视口内 ≥N」"
    assert body["total_in_view"] == 20000
    assert body["coverage"] is None, "下限不能当分母"
    assert body["truncated"] is True, "扫到上限就停的视口，没有分母也不能说「给全了」"


def test_preview_bbox_is_part_of_cache_key(workspace: Path, token: str):
    """平移换了视口就必须重读：缓存键不带 bbox 会拿上一个视口的图形冒充这个视口。"""
    runner = ViewRunner([{"x": 100.0, "y": 30.0}], view_total=1)
    layer = _make_gdb(workspace)
    _seed(workspace, layer, kind="feature_class", name="city", feature_count=10, wkid=4326)
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))
    params = {"path": str(layer), "bbox": "99,29,101,31"}

    first = client.get("/api/catalog/preview", params=params, headers={"X-Web-Token": token})
    again = client.get("/api/catalog/preview", params=params, headers={"X-Web-Token": token})
    moved = client.get(
        "/api/catalog/preview",
        params={**params, "bbox": "101,29,103,31"},
        headers={"X-Web-Token": token},
    )

    assert first.json()["cache"]["hit"] is False
    assert again.json()["cache"]["hit"] is True
    assert moved.json()["cache"]["hit"] is False
    assert len(runner.calls) == 2


# ---------------------------------------------------------------- 先容差后截断（DAIR-391）
def test_z11_viewport_returns_every_feature_under_the_gates(workspace: Path, token: str):
    """参考图同尺度（z11，28,921 条线）：视口内要素全给，且不越 5MB / 60k 顶点。

    DAIR-387 基准里同一 bbox 只拿到 12,000 条（41.49%）、12,000 条时响应才 1.79 MB。
    本用例把那条口径钉住：返回数 == 视口真实数，响应与顶点仍守闸门。
    """
    count = 28921
    minx, miny = 116.323242, 39.902344
    shapes = [
        [
            [
                [minx + index * 9.1e-6, miny + (index % 100) * 1e-4],
                [minx + index * 9.1e-6, miny + (index % 100) * 1e-4 + 4e-5],
            ]
        ]
        for index in range(count)
    ]
    layer = _shp(workspace, "路网.shp", SHP_POLYLINE, shapes)
    _seed(
        workspace,
        layer,
        geom_type="Polyline",
        feature_count=count,
        extent=[minx, miny, minx + 0.27, miny + 0.01],
    )
    client = TestClient(create_app(workspace, token=token, preview_runner=FailingRunner()))

    response = client.get(
        "/api/catalog/preview",
        params={
            "path": str(layer),
            "bbox": "116.323242,39.902344,116.586914,40.034180",
            "zoom": 11,
        },
        headers={"X-Web-Token": token},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_in_view"] == count
    assert body["returned_features"] == count, "视口内的要素要全给，不是先砍到 12,000"
    assert body["coverage"] == 1.0
    assert body["truncated"] is False, "给全了就不能标「部分」"
    assert body["point_count"] <= PREVIEW_POINT_BUDGET
    assert len(response.content) <= VIEW_BYTES_GATE
    assert body["max_features"] > body["returned_features"], "读盘上限要比兜底上限宽"


def test_viewport_read_cap_never_breaks_the_vertex_budget(workspace: Path, token: str):
    """视口要素多到画不下时（40,000 条线）：读盘上限卡在「60k 顶点 ÷ 2 点」，并标「部分」。"""
    count = 40000
    shapes = [
        [[[100.0 + index * 1e-5, 30.0], [100.0 + index * 1e-5, 30.0005]]]
        for index in range(count)
    ]
    layer = _shp(workspace, "超量线.shp", SHP_POLYLINE, shapes)
    _seed(
        workspace,
        layer,
        geom_type="Polyline",
        feature_count=count,
        extent=[100.0, 30.0, 100.4, 30.0005],
    )
    client = TestClient(create_app(workspace, token=token, preview_runner=FailingRunner()))

    response = client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "99,29,101,31"},
        headers={"X-Web-Token": token},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_in_view"] == count
    assert body["returned_features"] == PREVIEW_POINT_BUDGET // 2, "线最少 2 点/条"
    assert body["point_count"] <= PREVIEW_POINT_BUDGET
    assert body["truncated"] is True
    assert body["coverage"] == pytest.approx(0.75, abs=1e-4)
    assert len(response.content) <= VIEW_BYTES_GATE


def test_fit_payload_truncates_features_only_after_simplify():
    """顶点压不动、响应仍 >5MB 时才截要素，且 ``truncated`` 必须为 true（DAIR-391）。

    点要素不占顶点预算（没路径可简化），60,000 个的外壳就超 5MB：这是唯一必须
    按要素数兜底的情形。
    """
    features = [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Point", "coordinates": [100.0 + index * 1e-5, 30.0]},
        }
        for index in range(60_000)
    ]
    payload, final, _ = _fit_payload(
        {"type": "FeatureCollection", "features": features},
        base={"path": "x", "name": "x"},
        point_budget=PREVIEW_POINT_BUDGET,
        max_features=VIEW_FEATURE_LIMITS["point"],
        expected_total=len(features),
    )

    assert payload["returned_features"] == VIEW_FEATURE_LIMITS["point"]
    assert payload["returned_features"] == len(final["features"])
    assert payload["truncated"] is True, "截了要素还说给全了正是要消灭的谎报"
    assert encoded_size(payload) <= VIEW_BYTES_GATE


def test_fit_payload_falls_back_to_bbox_for_oversized_single_feature():
    """单个要素自己就超 5MB 时只回 bbox，绝不回一个打不开的响应（旧行为不回退）。"""
    coords = [[100.0 + index * 1e-6, 30.0] for index in range(400_000)]
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "MultiPoint", "coordinates": coords},
            }
        ],
    }

    payload, final, _ = _fit_payload(
        collection,
        base={"path": "x"},
        point_budget=PREVIEW_POINT_BUDGET,
        max_features=VIEW_FEATURE_LIMITS["point"],
        expected_total=1,
    )

    assert payload["oversized"] is True
    assert payload["returned_features"] == 0
    assert payload["truncated"] is True
    assert final["features"] == []
    assert encoded_size(payload) <= VIEW_BYTES_GATE


def test_viewport_empty_is_200_and_info(workspace: Path, token: str):
    """带 bbox 的空视口不是故障（DAIR-387 D2）：200 + 空 features + total_in_view=0。"""
    layer = _shp(workspace, "单线.shp", SHP_POLYLINE, [[[[100.0, 30.0], [100.1, 30.1]]]])
    _seed(
        workspace,
        layer,
        geom_type="Polyline",
        feature_count=1,
        extent=[100.0, 30.0, 100.1, 30.1],
    )
    client = TestClient(create_app(workspace, token=token, preview_runner=FailingRunner()))

    response = client.get(
        "/api/catalog/preview",
        params={"path": str(layer), "bbox": "110,40,111,41", "zoom": 11},
        headers={"X-Web-Token": token},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["geojson"]["features"] == []
    assert body["returned_features"] == 0
    assert body["total_in_view"] == 0
    assert body["coverage"] is None, "0 要素没有覆盖率可算"
    assert body["truncated"] is False
    assert body["feature_count"] == 1, "全量要素数仍要说清"
    assert body["bbox"] == [110.0, 40.0, 111.0, 41.0], "没几何时用视口说明看的是哪一片"


def test_max_features_ceiling_covers_every_tier():
    """显式上限不能比默认档位还小（DAIR-387 D3：5000 < 线默认 12,000）。"""
    assert MAX_MAX_FEATURES >= max(VIEW_FEATURE_LIMITS.values())


# ---------------------------------------------------------------- 边界
def test_preview_without_bbox_keeps_legacy_payload(
    workspace: Path, preview_client: TestClient, auth: dict[str, str]
):
    """不带视口（旧客户端/报告链路）口径不变：不回视口字段，上限仍由调用方给。"""
    layer = _write_geojson(workspace / "input" / "无视口.geojson", points=4)
    _seed(workspace, layer, feature_count=4, extent=[100.0, 30.0, 101.0, 31.0])

    body = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer), "max_features": 2}, headers=auth
    ).json()

    assert body["view_bbox"] is None
    assert body["total_in_view"] is None
    assert body["coverage"] is None
    assert body["returned_features"] == 2
    assert body["truncated"] is True


@pytest.mark.parametrize("bad", ["1,2,3", "a,b,c,d", "101,29,99,31", "1,2,3,nan"])
def test_preview_bad_bbox_is_422(
    workspace: Path, preview_client: TestClient, auth: dict[str, str], bad: str
):
    layer = _write_geojson(workspace / "input" / "坏框.geojson")
    _seed(workspace, layer)

    response = preview_client.get(
        "/api/catalog/preview", params={"path": str(layer), "bbox": bad}, headers=auth
    )

    assert response.status_code == 422, bad
    assert response.json()["detail"]["code"] == "bad_bbox"


# ---------------------------------------------------------------- 内核预热判断
def test_kernel_warm_only_for_layers_that_need_the_kernel(workspace: Path):
    """起服务时要不要热内核：只有 `.gdb` 要素类这类「非内核不可」的数据才热（DAIR-393）。"""
    from gis_cli.web.preview_api import kernel_warm_suggested

    assert kernel_warm_suggested(workspace) is False, "空工作区不该起内核"

    shp = _shp(workspace, "道路.shp", SHP_POLYLINE, [[[[100.0, 30.0], [100.1, 30.1]]]])
    _seed(workspace, shp, geom_type="Polyline", feature_count=1)
    assert kernel_warm_suggested(workspace) is False, ".shp 有 .shx 直读，不需要内核"

    layer = _make_gdb(workspace)
    _seed(workspace, layer, kind="feature_class", name="city", feature_count=3)
    assert kernel_warm_suggested(workspace) is True
