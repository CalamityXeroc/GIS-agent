# -*- coding: utf-8 -*-
"""catalog 只读 API：分页、kind/q 过滤、字段画像，以及 DAIR-328 的唯一值/
属性表分页/参数化 where。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gis_cli.datacatalog.store import Catalog, DatasetRecord
from gis_cli.web.app import create_app


def _seed(workspace: Path) -> None:
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    rows = [
        ("input/pop.shp", "vector", "pop", [{"name": "POP", "type": "Double", "samples": [1, 2]}]),
        ("input/dem.tif", "raster", "dem", []),
        ("output/stats.csv", "table", "stats", []),
    ]
    for relative, kind, name, fields in rows:
        catalog.upsert(
            DatasetRecord(
                path=str(workspace / relative),
                kind=kind,
                name=name,
                fields=fields,
                size=10,
                feature_count=1 if kind == "vector" else None,
            )
        )
    catalog.close()


def test_list_paginates_and_counts(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    _seed(workspace)

    response = client.get("/api/catalog/datasets", headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert body["counts"] == {"raster": 1, "table": 1, "vector": 1}
    assert "fields" not in body["items"][0]

    first = client.get(
        "/api/catalog/datasets", params={"limit": 1, "offset": 0}, headers=auth
    ).json()
    assert first["count"] == 1
    assert first["has_more"] is True

    last = client.get(
        "/api/catalog/datasets", params={"limit": 1, "offset": 2}, headers=auth
    ).json()
    assert last["count"] == 1
    assert last["has_more"] is False


def test_filters_by_kind_and_query(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    _seed(workspace)

    vectors = client.get(
        "/api/catalog/datasets", params={"kind": "vector"}, headers=auth
    ).json()
    assert vectors["count"] == 1
    assert vectors["items"][0]["name"] == "pop"

    found = client.get(
        "/api/catalog/datasets", params={"q": "po"}, headers=auth
    ).json()
    assert found["count"] == 1
    assert found["items"][0]["name"] == "pop"


def test_profile_returns_fields(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    _seed(workspace)
    response = client.get(
        "/api/catalog/datasets/profile",
        params={"path": "input/pop.shp"},
        headers=auth,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "vector"
    assert body["fields"][0]["name"] == "POP"
    assert body["fields"][0]["samples"] == [1, 2]


def test_profile_unknown_path_is_404(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    _seed(workspace)
    response = client.get(
        "/api/catalog/datasets/profile",
        params={"path": "input/missing.shp"},
        headers=auth,
    )
    assert response.status_code == 404


def test_missing_catalog_returns_empty_page(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    response = client.get("/api/catalog/datasets", headers=auth)
    assert response.status_code == 200
    assert response.json()["items"] == []


# ============================================================ DAIR-328 新接口
#: 12 行测试数据：3 类用地 + 1 个空值 + 数值列，覆盖唯一值/空值/排序/筛选。
_CSV_ROWS = [
    ("1", "林地", "12.5", "a"),
    ("2", "林地", "8.0", ""),
    ("3", "林地", "30.25", "c"),
    ("4", "草地", "1.5", "d"),
    ("5", "草地", "2.0", ""),
    ("6", "耕地", "100", "f"),
    ("7", "耕地", "", "g"),
    ("8", "林地", "7.75", ""),
    ("9", "草地", "5", "i"),
    ("10", "耕地", "50", "j"),
    ("11", "", "3", "k"),
    ("12", "林地", "15", "l"),
]


def _seed_table(workspace: Path, *, kind: str = "table", name: str = "tdly.csv") -> Path:
    """写一个真 CSV + 一条 catalog 记录（不需要 arcpy 的路径）。"""
    import csv

    path = workspace / "input" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["OBJECTID", "TDLYMC", "AREA", "NOTE"])
        writer.writerows(_CSV_ROWS)
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(
        DatasetRecord(
            path=str(path),
            kind=kind,
            name=name,
            size=path.stat().st_size,
            mtime=path.stat().st_mtime,
            feature_count=len(_CSV_ROWS),
        )
    )
    catalog.close()
    return path


def test_rows_paginates_with_honest_total(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    path = _seed_table(workspace)
    first = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "offset": 0, "limit": 5},
        headers=auth,
    ).json()
    assert first["total"] == len(_CSV_ROWS)
    assert first["returned"] == 5
    assert first["truncated"] is True
    assert first["rows"][0]["TDLYMC"] == "林地"
    assert first["fields"][0]["name"] == "OBJECTID"
    assert first["cache"]["hit"] is False

    last = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "offset": 10, "limit": 5},
        headers=auth,
    ).json()
    assert last["returned"] == 2
    assert last["truncated"] is False
    # 第二次命中行缓存，不再重读文件。
    assert last["cache"]["hit"] is True


def test_rows_orders_by_real_field_with_nulls_last(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    path = _seed_table(workspace)
    desc = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "order": "AREA", "dir": "desc", "limit": 50},
        headers=auth,
    ).json()
    assert desc["dir"] == "desc"
    areas = [row["AREA"] for row in desc["rows"]]
    assert areas[0] == 100.0
    assert areas[-1] is None, "空值必须排在最后，不能占首屏"

    asc = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "order": "AREA", "dir": "asc", "limit": 50},
        headers=auth,
    ).json()
    assert asc["rows"][0]["AREA"] == 1.5


def test_rows_where_supports_equality_like_and_numeric(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    path = _seed_table(workspace)
    forest = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "where": "TDLYMC=林地", "limit": 50},
        headers=auth,
    ).json()
    assert forest["total"] == 5
    assert {row["TDLYMC"] for row in forest["rows"]} == {"林地"}

    quoted = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "where": "TDLYMC='林地'", "limit": 50},
        headers=auth,
    ).json()
    assert quoted["total"] == 5

    like = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "where": "TDLYMC LIKE 耕%", "limit": 50},
        headers=auth,
    ).json()
    assert like["total"] == 3

    big = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "where": "AREA>=30", "limit": 50},
        headers=auth,
    ).json()
    assert sorted(row["AREA"] for row in big["rows"]) == [30.25, 50.0, 100.0]

    not_equal = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "where": "TDLYMC≠林地", "limit": 50},
        headers=auth,
    ).json()
    # SQL 三值逻辑：空值（NULL）不满足 <>，所以再少 1 行 —— 与 ArcGIS 一致。
    assert not_equal["total"] == len(_CSV_ROWS) - 5 - 1


@pytest.mark.parametrize(
    "where",
    [
        "1=1 OR TDLYMC=林地",  # 不是「字段 op 值」
        "TDLYMC='林地' OR 1=1",  # 引号未闭合 → 想混片段
        "TDLYMC=林地;DROP TABLE x",  # 分号
        "NOPE=1",  # 字段不在白名单
        "AREA=abc",  # 数值字段给了非数字
        "TDLYMC",  # 没有运算符
        "TDLYMC=",  # 没有值
    ],
)
def test_rows_rejects_bad_where_without_touching_cache(
    workspace: Path, client: TestClient, auth: dict[str, str], where: str
):
    path = _seed_table(workspace)
    ok = client.get(
        "/api/catalog/rows", params={"path": str(path), "limit": 1}, headers=auth
    )
    assert ok.status_code == 200

    bad = client.get(
        "/api/catalog/rows", params={"path": str(path), "where": where}, headers=auth
    )
    assert bad.status_code == 422, bad.text
    detail = bad.json()["detail"]
    assert detail["code"] == "bad_where"
    assert detail["reason"]

    # 拒绝之后缓存仍然可用（不是靠删表/清缓存来"防注入"）。
    again = client.get(
        "/api/catalog/rows", params={"path": str(path), "limit": 1}, headers=auth
    ).json()
    assert again["total"] == len(_CSV_ROWS)


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 501},
        {"limit": 0},
        {"offset": -1},
        {"order": "NOPE"},
        {"dir": "sideways"},
    ],
)
def test_rows_validates_limits_and_order(
    workspace: Path, client: TestClient, auth: dict[str, str], params: dict
):
    path = _seed_table(workspace)
    response = client.get(
        "/api/catalog/rows", params={"path": str(path), **params}, headers=auth
    )
    assert response.status_code == 422, response.text


def test_distinct_lists_values_counts_and_nulls(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    path = _seed_table(workspace)
    body = client.get(
        "/api/catalog/distinct",
        params={"path": str(path), "field": "TDLYMC"},
        headers=auth,
    ).json()
    assert body["field"] == "TDLYMC"
    assert body["total_distinct"] == 3  # 林地/草地/耕地（空单元格按 NULL 计）
    assert body["null_count"] == 1  # 第 11 行是空单元格
    assert body["truncated"] is False
    values = {item["value"]: item["count"] for item in body["values"]}
    assert values["林地"] == 5
    assert values["草地"] == 3
    assert values["耕地"] == 3
    assert body["values"][0]["value"] == "林地", "按出现次数降序"

    limited = client.get(
        "/api/catalog/distinct",
        params={"path": str(path), "field": "TDLYMC", "limit": 2},
        headers=auth,
    ).json()
    assert len(limited["values"]) == 2
    assert limited["truncated"] is True
    assert limited["total_distinct"] == 3


def test_distinct_quantile_breaks_are_real(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    path = _seed_table(workspace)
    body = client.get(
        "/api/catalog/distinct",
        params={
            "path": str(path),
            "field": "AREA",
            "breaks": "quantile",
            "classes": 3,
        },
        headers=auth,
    ).json()
    assert body["classes"] == 3
    assert len(body["breaks"]) == 4
    assert body["breaks"][0] == 1.5
    assert body["breaks"][-1] == 100.0
    assert body["breaks"] == sorted(body["breaks"])
    assert sum(body["counts"]) == len(_CSV_ROWS) - 1  # 1 个空值不参与分级
    assert body["break_basis"]["method"] == "quantile"


def test_distinct_on_text_field_has_no_breaks(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    path = _seed_table(workspace)
    body = client.get(
        "/api/catalog/distinct",
        params={"path": str(path), "field": "TDLYMC", "breaks": "quantile"},
        headers=auth,
    ).json()
    assert body["breaks"] == []
    assert body["counts"] == []
    assert body["break_basis"]["reason"]


def test_distinct_unknown_field_is_422(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    path = _seed_table(workspace)
    response = client.get(
        "/api/catalog/distinct",
        params={"path": str(path), "field": "NOPE"},
        headers=auth,
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "bad_field"
    assert "TDLYMC" in detail["allowed_fields"]


def test_rows_on_raster_is_422_not_500(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    raster = workspace / "input" / "dem.tif"
    raster.parent.mkdir(parents=True, exist_ok=True)
    raster.write_bytes(b"fake")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(DatasetRecord(path=str(raster), kind="raster", name="dem"))
    catalog.close()

    response = client.get(
        "/api/catalog/rows", params={"path": str(raster)}, headers=auth
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "not_tabular"


def test_rows_without_catalog_is_404(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    response = client.get(
        "/api/catalog/rows", params={"path": "input/none.csv"}, headers=auth
    )
    assert response.status_code == 404


def test_rows_geojson_without_arcpy(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    path = workspace / "input" / "points.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"NAME": "甲", "POP": 10}, "geometry": None},
                    {"type": "Feature", "properties": {"NAME": "乙", "POP": 20}, "geometry": None},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(DatasetRecord(path=str(path), kind="vector", name="points"))
    catalog.close()

    body = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "order": "POP", "dir": "desc"},
        headers=auth,
    ).json()
    assert body["total"] == 2
    assert body["rows"][0]["NAME"] == "乙"
    assert body["cache"]["source"] == "geojson"

    distinct = client.get(
        "/api/catalog/distinct", params={"path": str(path), "field": "NAME"}, headers=auth
    ).json()
    assert distinct["total_distinct"] == 2


def test_rows_gpkg_uses_parameterized_sql(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    """GeoPackage 走 SQLite 只读连接；where 的值必须是绑定参数而不是拼进 SQL。"""
    import sqlite3

    gpkg = workspace / "input" / "zones.gpkg"
    gpkg.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(gpkg))
    conn.executescript(
        """
        CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT);
        CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT);
        CREATE TABLE zones (fid INTEGER PRIMARY KEY, name TEXT, area REAL, geom BLOB);
        INSERT INTO gpkg_contents VALUES ('zones', 'features');
        INSERT INTO gpkg_geometry_columns VALUES ('zones', 'geom');
        INSERT INTO zones (fid, name, area, geom) VALUES
            (1, '甲', 1.5, x'00'), (2, '乙', 2.5, x'00'), (3, '乙', 3.5, x'00');
        """
    )
    conn.commit()
    conn.close()

    layer = f"{gpkg}\\zones"
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(DatasetRecord(path=layer, kind="feature_class", name="zones"))
    catalog.close()

    body = client.get(
        "/api/catalog/rows",
        params={"path": layer, "where": "name=乙", "order": "area", "dir": "desc"},
        headers=auth,
    ).json()
    assert body["total"] == 2
    assert [row["area"] for row in body["rows"]] == [3.5, 2.5]
    assert body["fields"][0]["name"] == "fid"
    assert "geom" not in {field["name"] for field in body["fields"]}


def test_cache_truncation_is_reported_not_hidden(
    workspace: Path, client: TestClient, auth: dict[str, str], monkeypatch
):
    from gis_cli.datacatalog import tables

    monkeypatch.setattr(tables, "MAX_CACHE_ROWS", 5)
    path = _seed_table(workspace)
    body = client.get(
        "/api/catalog/rows", params={"path": str(path), "limit": 500}, headers=auth
    ).json()
    assert body["cache"]["truncated"] is True
    assert body["total"] == 5
    assert body["cache"]["rows"] == 5


def test_rows_handles_quote_in_field_name(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    """字段名来自真实数据（可能带引号）；路径作为绑定参数，不拼 SQL 文本。"""
    path = workspace / "input" / "odd.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"A\"B": 7, "正常": "x"}, "geometry": None}
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(DatasetRecord(path=str(path), kind="vector", name="odd"))
    catalog.close()

    body = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "where": 'A"B=7'},
        headers=auth,
    ).json()
    assert body["total"] == 1
    assert body["rows"][0]['A"B'] == 7


def _seed_quoted_field_table(workspace: Path) -> Path:
    """4 行数据，字段名带双引号，其中一行空值。"""
    path = workspace / "input" / "odd_multi.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"A\"B": 3}, "geometry": None},
                    {"type": "Feature", "properties": {"A\"B": 1}, "geometry": None},
                    {"type": "Feature", "properties": {"A\"B": None}, "geometry": None},
                    {"type": "Feature", "properties": {"A\"B": 2}, "geometry": None},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(DatasetRecord(path=str(path), kind="vector", name="odd_multi"))
    catalog.close()
    return path


def test_rows_orders_by_quoted_field(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    """带引号的字段名也能排序（旧实现拼 JSON 路径，3.10 自带 SQLite 解析不了）。"""
    path = _seed_quoted_field_table(workspace)
    body = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "order": 'A"B', "dir": "asc"},
        headers=auth,
    ).json()
    assert [row['A"B'] for row in body["rows"]] == [1, 2, 3, None], "升序 + 空值排最后"

    desc = client.get(
        "/api/catalog/rows",
        params={"path": str(path), "order": 'A"B', "dir": "desc"},
        headers=auth,
    ).json()
    assert [row['A"B'] for row in desc["rows"]] == [3, 2, 1, None]


def test_distinct_handles_quote_in_field_name(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    """带引号字段的唯一值 / 基数 / 空值 / 分位断点都要取到真值。"""
    path = _seed_quoted_field_table(workspace)
    body = client.get(
        "/api/catalog/distinct",
        params={"path": str(path), "field": 'A"B', "breaks": "quantile", "classes": 2},
        headers=auth,
    ).json()
    assert body["total_distinct"] == 3
    assert body["null_count"] == 1
    assert {item["value"]: item["count"] for item in body["values"] if item["value"] is not None} == {
        1: 1,
        2: 1,
        3: 1,
    }
    assert body["breaks"][0] == 1.0
    assert body["breaks"][-1] == 3.0
    assert sum(body["counts"]) == 3


def _fake_gdb_layer(workspace: Path, member: str = "bldg") -> Path:
    gdb = workspace / "input" / "data.gdb"
    gdb.mkdir(parents=True, exist_ok=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"fake")
    layer = gdb / member
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(
        DatasetRecord(path=str(layer), kind="feature_class", name=member, feature_count=3)
    )
    catalog.close()
    return layer


class _TableRunner:
    """假内核：只响应取行代码（.gdb 要素类路径）。"""

    def __init__(self, rows: list | None = None, error: str = "") -> None:
        self.rows = rows or []
        self.error = error
        self.codes: list[str] = []

    def run(self, code: str, **kwargs):
        self.codes.append(code)
        return SimpleNamespace(
            ok=True,
            result={
                "fields": [
                    {"name": "OID", "alias": "", "type": "OID", "length": 4},
                    {"name": "TDLYMC", "alias": "用地", "type": "String", "length": 40},
                ],
                "rows": self.rows,
                "total": 12847 if not self.error else None,
                "error": self.error,
            },
            error=None,
        )


def test_gdb_rows_go_through_kernel_without_where_clause(
    workspace: Path, token: str, auth: dict[str, str]
):
    """要素类取行只用 ``SearchCursor(path, fields)``：没有 where_clause 可供注入。"""
    layer = _fake_gdb_layer(workspace)
    runner = _TableRunner([["1", "林地"], ["2", "草地"], ["3", "林地"]])
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))

    body = client.get(
        "/api/catalog/rows",
        params={"path": str(layer), "where": "TDLYMC=林地", "limit": 10},
        headers=auth,
    ).json()
    assert body["total"] == 2
    assert body["cache"]["source"] == "arcpy"
    assert "SearchCursor" in runner.codes[0]
    # 用户输入（where 的字段/值）根本不在内核代码里：筛选全在本地行缓存上做。
    assert "TDLYMC=林地" not in runner.codes[0]
    assert "where_clause" not in runner.codes[0]


def test_gdb_rows_accept_relative_catalog_path(
    workspace: Path, token: str, auth: dict[str, str]
):
    """catalog 里「扫描时 cwd = 仓库根」的相对写法也要能读。

    ``.gdb`` 内的要素类在文件系统上不存在，拿 ``exists()`` 当判据会把路径拼成
    ``workspace/workspace/...``（DAIR-330 在真实 catalog 上踩到）。
    """
    layer = _fake_gdb_layer(workspace)
    runner = _TableRunner([["1", "林地"]])
    client = TestClient(create_app(workspace, token=token, preview_runner=runner))

    relative = f"{workspace.name}/input/data.gdb/{layer.name}"
    body = client.get(
        "/api/catalog/rows", params={"path": relative, "limit": 10}, headers=auth
    ).json()
    assert body["total"] == 1
    # 送给内核的路径只拼一次工作区，不能是 workspace/workspace/...（代码里是 JSON 转义）
    escaped = str(workspace).replace("\\", "\\\\")
    assert escaped in runner.codes[0]
    assert escaped + "\\workspace" not in runner.codes[0]


def test_gdb_rows_without_arcpy_is_503_with_reason(
    workspace: Path, token: str, auth: dict[str, str]
):
    """无 arcpy 环境：明确降级（503 + 原因 + 建议），不是 500，也不是空表。"""

    class NoKernel:
        def run(self, code, **kwargs):
            return SimpleNamespace(ok=False, result=None, error="arcpy 不可用")

    layer = _fake_gdb_layer(workspace)
    client = TestClient(create_app(workspace, token=token, preview_runner=NoKernel()))
    response = client.get(
        "/api/catalog/rows", params={"path": str(layer)}, headers=auth
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "kernel_unavailable"
    assert "arcpy" in detail["reason"]
    assert detail["suggestion"]


def test_gdb_rows_kernel_timeout_is_504(
    workspace: Path, token: str, auth: dict[str, str]
):
    class SlowKernel:
        def run(self, code, **kwargs):
            return SimpleNamespace(
                ok=False,
                result=None,
                error={"type": "TimeoutExpired", "message": "Execution exceeded 60s"},
            )

    layer = _fake_gdb_layer(workspace)
    client = TestClient(create_app(workspace, token=token, preview_runner=SlowKernel()))
    response = client.get(
        "/api/catalog/rows", params={"path": str(layer)}, headers=auth
    )
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "table_read_timeout"


def _seed_relative_geojson(workspace: Path) -> Path:
    """按 DAIR-335 的新口径落库：catalog 里存工作区相对写法，文件真实存在。"""
    path = workspace / "input" / "points.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"NAME": "甲", "POP": 10}, "geometry": None},
                    {"type": "Feature", "properties": {"NAME": "乙", "POP": 20}, "geometry": None},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(
        DatasetRecord(path=str(Path("input") / "points.geojson"), kind="vector", name="points")
    )
    catalog.close()
    return path


def test_lookup_accepts_every_path_spelling(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    """库里是工作区相对写法时，三种入参都得命中（DAIR-336 审查要求补的回归）。

    catalog 的 path 从 DAIR-335 起统一成「相对 workspace」，但请求可能来自
    旧链接 / 抓包重放 / 浏览器缓存的旧树（绝对路径、``workspace/input/x`` 老写法）。
    ``_find_record`` 的兼容分支没有测试就会静默退化成 404。
    """
    path = _seed_relative_geojson(workspace)
    relative = str(Path("input") / "points.geojson")
    legacy = str(Path(workspace.name) / "input" / "points.geojson")

    for spelling in (relative, str(path), legacy):
        profile = client.get(
            "/api/catalog/datasets/profile", params={"path": spelling}, headers=auth
        )
        assert profile.status_code == 200, (spelling, profile.text)
        assert profile.json()["path"] == relative, "返回的应是库里的口径"

        rows = client.get(
            "/api/catalog/rows", params={"path": spelling, "order": "POP", "dir": "desc"},
            headers=auth,
        )
        assert rows.status_code == 200, (spelling, rows.text)
        assert rows.json()["total"] == 2
        assert rows.json()["rows"][0]["NAME"] == "乙"

    # 兼容不能突破 workspace 边界
    outside = client.get(
        "/api/catalog/datasets/profile", params={"path": "../outside.geojson"}, headers=auth
    )
    assert outside.status_code == 403


def test_lookup_absolute_record_via_relative_request(
    workspace: Path, client: TestClient, auth: dict[str, str]
):
    """反向兼容：旧库里存的是绝对写法，相对入参也得找得到（升级期混用）。"""
    path = workspace / "input" / "legacy.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"NAME": "丙"}, "geometry": None},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(DatasetRecord(path=str(path), kind="vector", name="legacy"))
    catalog.close()

    body = client.get(
        "/api/catalog/datasets/profile",
        params={"path": str(Path("input") / "legacy.geojson")},
        headers=auth,
    )
    assert body.status_code == 200, body.text
    assert body.json()["path"] == str(path)
