# -*- coding: utf-8 -*-
"""只读数据目录 API：给控制台的「数据资产」表提供分页查询。

数据来自引擎已经维护的 ``<workspace>/.gis_agent/catalog.db``，这里不触发
扫描、不写库（库不存在时返回空页，而不是新建一个空库）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..datacatalog.store import Catalog, DatasetRecord
from .security import safe_path

#: catalog 库位置（相对 workspace），与 ``engine/runner.py`` 保持一致。
CATALOG_DB = Path(".gis_agent") / "catalog.db"

#: 单页上限：本地控制台不需要一次拉几千行。
MAX_LIMIT = 200


def _open_catalog(workspace: str | Path) -> Catalog | None:
    db_path = Path(workspace) / CATALOG_DB
    if not db_path.is_file():
        return None
    return Catalog(db_path)


def _brief(record: DatasetRecord) -> dict[str, Any]:
    """列表页字段：不带 fields/samples（那可能几百个字段）。"""
    return {
        "path": record.path,
        "kind": record.kind,
        "container": record.container,
        "name": record.name,
        "size": record.size,
        "mtime": record.mtime,
        "crs_name": record.crs_name,
        "wkid": record.wkid,
        "geom_type": record.geom_type,
        "feature_count": record.feature_count,
        "extent": record.extent,
        "field_count": len(record.fields),
        "source": record.source,
        "license": record.license,
        "acquired_at": record.acquired_at,
    }


def _full(record: DatasetRecord) -> dict[str, Any]:
    data = _brief(record)
    data["fields"] = record.fields
    data["raster"] = record.raster
    data["extra"] = record.extra
    return data


def list_datasets(
    workspace: str | Path,
    *,
    kind: str = "",
    q: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """分页 + kind 精确过滤 + q 模糊匹配（名称或路径）。"""
    limit = max(1, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))

    # 多取一行用来判断还有没有下一页，避免额外的 COUNT 查询。
    fetch = offset + limit + 1
    catalog = _open_catalog(workspace)
    if catalog is None:
        return {
            "items": [],
            "count": 0,
            "limit": limit,
            "offset": offset,
            "has_more": False,
            "counts": {},
        }

    try:
        records = catalog.list_datasets(
            kind=kind or None,
            name_like=q or None,
            limit=fetch,
        )
        counts = catalog.counts()
    finally:
        catalog.close()

    page = records[offset : offset + limit]
    return {
        "items": [_brief(record) for record in page],
        "count": len(page),
        "limit": limit,
        "offset": offset,
        "has_more": len(records) > offset + limit,
        "counts": counts,
    }


def dataset_profile(workspace: str | Path, path: str) -> dict[str, Any]:
    """单个数据集的字段画像（只读 catalog，不打开数据文件）。"""
    resolved = safe_path(workspace, path)
    catalog = _open_catalog(workspace)
    if catalog is None:
        raise HTTPException(status_code=404, detail="catalog not found")

    try:
        record = catalog.get(str(resolved)) or catalog.get(str(path))
        if record is None:
            # 库里的 path 可能是扫描时的原始写法（大小写/相对路径不同），
            # 按文件名回退匹配一次，再比 resolve 后的绝对路径。
            for candidate in catalog.list_datasets(name_like=Path(path).name, limit=50):
                if Path(candidate.path).resolve() == resolved:
                    record = candidate
                    break
    finally:
        catalog.close()

    if record is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return _full(record)


def build_router(workspace: str | Path) -> APIRouter:
    router = APIRouter()

    @router.get("/api/catalog/datasets")
    def get_datasets(
        kind: str = Query("", description="数据集类型，精确匹配"),
        q: str = Query("", description="按名称或路径模糊搜索"),
        limit: int = Query(50, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        return list_datasets(workspace, kind=kind, q=q, limit=limit, offset=offset)

    @router.get("/api/catalog/datasets/profile")
    def get_profile(path: str = Query(..., description="数据集路径（必须在 workspace 内）")):
        return dataset_profile(workspace, path)

    return router
