# -*- coding: utf-8 -*-
"""catalog 只读 API：分页、kind/q 过滤、字段画像。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from gis_cli.datacatalog.store import Catalog, DatasetRecord


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
