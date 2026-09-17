# -*- coding: utf-8 -*-
"""SQLite-backed incremental catalog of GIS datasets.

The catalog is the single source of truth for "what data exists and what does
it look like". It is refreshed incrementally by mtime/size so planning no
longer triggers a full ArcPy scan on every call.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    path TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'other',
    container TEXT,
    name TEXT,
    mtime REAL,
    size INTEGER,
    crs_name TEXT,
    wkid INTEGER,
    geom_type TEXT,
    feature_count INTEGER,
    extent TEXT,
    extra TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS fields (
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    alias TEXT,
    ftype TEXT,
    length INTEGER,
    nullable INTEGER,
    samples TEXT,
    null_ratio REAL,
    PRIMARY KEY (path, name)
);
CREATE TABLE IF NOT EXISTS rasters (
    path TEXT PRIMARY KEY,
    band_count INTEGER,
    cell_size REAL,
    min REAL,
    max REAL,
    mean REAL,
    nodata REAL,
    width INTEGER,
    height INTEGER,
    crs_name TEXT,
    wkid INTEGER,
    extra TEXT
);
CREATE INDEX IF NOT EXISTS idx_datasets_kind ON datasets(kind);
CREATE INDEX IF NOT EXISTS idx_datasets_container ON datasets(container);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DatasetRecord:
    """A single dataset entry."""

    path: str
    kind: str = "other"
    container: str = ""
    name: str = ""
    mtime: float = 0.0
    size: int = 0
    crs_name: str = ""
    wkid: int | None = None
    geom_type: str = ""
    feature_count: int | None = None
    extent: list[float] | None = None
    fields: list[dict[str, Any]] = field(default_factory=list)
    raster: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> tuple:
        return (
            self.path,
            self.kind,
            self.container,
            self.name,
            self.mtime,
            self.size,
            self.crs_name,
            self.wkid,
            self.geom_type,
            self.feature_count,
            json.dumps(self.extent, ensure_ascii=False) if self.extent else None,
            json.dumps(self.extra, ensure_ascii=False) if self.extra else None,
            _now(),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DatasetRecord":
        return cls(
            path=row["path"],
            kind=row["kind"] or "other",
            container=row["container"] or "",
            name=row["name"] or "",
            mtime=row["mtime"] or 0.0,
            size=row["size"] or 0,
            crs_name=row["crs_name"] or "",
            wkid=row["wkid"],
            geom_type=row["geom_type"] or "",
            feature_count=row["feature_count"],
            extent=json.loads(row["extent"]) if row["extent"] else None,
            extra=json.loads(row["extra"]) if row["extra"] else {},
        )


class Catalog:
    """Incremental catalog store."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ write
    def upsert(self, record: DatasetRecord) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO datasets
                    (path, kind, container, name, mtime, size, crs_name, wkid,
                     geom_type, feature_count, extent, extra, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    kind=excluded.kind, container=excluded.container, name=excluded.name,
                    mtime=excluded.mtime, size=excluded.size, crs_name=excluded.crs_name,
                    wkid=excluded.wkid, geom_type=excluded.geom_type,
                    feature_count=excluded.feature_count, extent=excluded.extent,
                    extra=excluded.extra, updated_at=excluded.updated_at
                """,
                record.to_row(),
            )
            self._conn.execute("DELETE FROM fields WHERE path = ?", (record.path,))
            if record.fields:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO fields "
                    "(path, name, alias, ftype, length, nullable, samples, null_ratio) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            record.path,
                            str(f.get("name", "")),
                            str(f.get("alias", "") or ""),
                            str(f.get("type", "") or ""),
                            int(f.get("length", 0) or 0),
                            1 if f.get("nullable") else 0,
                            json.dumps(f.get("samples", []), ensure_ascii=False),
                            f.get("null_ratio"),
                        )
                        for f in record.fields
                        if str(f.get("name", "")).strip()
                    ],
                )
            if record.raster:
                r = record.raster
                self._conn.execute(
                    """
                    INSERT INTO rasters
                        (path, band_count, cell_size, min, max, mean, nodata,
                         width, height, crs_name, wkid, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        band_count=excluded.band_count, cell_size=excluded.cell_size,
                        min=excluded.min, max=excluded.max, mean=excluded.mean,
                        nodata=excluded.nodata, width=excluded.width,
                        height=excluded.height, crs_name=excluded.crs_name,
                        wkid=excluded.wkid, extra=excluded.extra
                    """,
                    (
                        record.path,
                        r.get("band_count"),
                        r.get("cell_size"),
                        r.get("min"),
                        r.get("max"),
                        r.get("mean"),
                        r.get("nodata"),
                        r.get("width"),
                        r.get("height"),
                        record.crs_name,
                        record.wkid,
                        json.dumps(r.get("extra", {}), ensure_ascii=False),
                    ),
                )

    def remove_missing(self, existing_paths: set[str], existing_containers: set[str] | None = None) -> int:
        """Delete entries whose files no longer exist.

        Records inside a container (GDB) are kept when the container itself is
        still present, even though their own path is not in ``existing_paths``.
        """
        containers = existing_containers or set()
        removed = 0
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT path, container FROM datasets").fetchall()
            for row in rows:
                if row["path"] in existing_paths:
                    continue
                if row["container"] and row["container"] in containers:
                    continue
                self._conn.execute("DELETE FROM datasets WHERE path = ?", (row["path"],))
                self._conn.execute("DELETE FROM fields WHERE path = ?", (row["path"],))
                self._conn.execute("DELETE FROM rasters WHERE path = ?", (row["path"],))
                removed += 1
        return removed

    # ------------------------------------------------------------------- read
    def is_fresh(self, path: str, mtime: float, size: int) -> bool:
        """Whether a dataset (or container) is unchanged since last scan."""
        with self._lock:
            row = self._conn.execute(
                "SELECT mtime, size FROM datasets WHERE path = ?", (path,)
            ).fetchone()
            if row is not None:
                return abs(float(row["mtime"] or 0) - mtime) < 1e-6 and int(row["size"] or 0) == size
            # Container (e.g. .gdb): fresh when all children carry the same stamp.
            rows = self._conn.execute(
                "SELECT mtime, size FROM datasets WHERE container = ?", (path,)
            ).fetchall()
        if not rows:
            return False
        return all(
            abs(float(r["mtime"] or 0) - mtime) < 1e-6 and int(r["size"] or 0) == size
            for r in rows
        )

    def get(self, path: str) -> DatasetRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM datasets WHERE path = ?", (path,)
            ).fetchone()
            if row is None:
                return None
            rec = DatasetRecord.from_row(row)
            rec.fields = self._fields(path)
        return rec

    def _fields(self, path: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM fields WHERE path = ? ORDER BY rowid", (path,)
        ).fetchall()
        return [
            {
                "name": r["name"],
                "alias": r["alias"],
                "type": r["ftype"],
                "length": r["length"],
                "nullable": bool(r["nullable"]),
                "samples": json.loads(r["samples"]) if r["samples"] else [],
                "null_ratio": r["null_ratio"],
            }
            for r in rows
        ]

    def list_datasets(
        self,
        *,
        kind: str | None = None,
        name_like: str | None = None,
        path_like: str | None = None,
        limit: int = 200,
    ) -> list[DatasetRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if name_like:
            clauses.append("(name LIKE ? OR path LIKE ?)")
            params.extend([f"%{name_like}%", f"%{name_like}%"])
        if path_like:
            clauses.append("path LIKE ?")
            params.append(f"%{path_like}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM datasets {where} ORDER BY kind, name LIMIT ?",
                (*params, limit),
            ).fetchall()
            records = []
            for row in rows:
                rec = DatasetRecord.from_row(row)
                rec.fields = self._fields(rec.path)
                records.append(rec)
        return records

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, COUNT(*) AS n FROM datasets GROUP BY kind"
            ).fetchall()
        return {row["kind"]: row["n"] for row in rows}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def record_from_scan(payload: dict[str, Any]) -> DatasetRecord:
    """Build a DatasetRecord from the kernel scanner payload."""
    return DatasetRecord(
        path=str(payload.get("path", "")),
        kind=str(payload.get("kind", "other")),
        container=str(payload.get("container", "") or ""),
        name=str(payload.get("name", "") or ""),
        mtime=float(payload.get("mtime", 0.0) or 0.0),
        size=int(payload.get("size", 0) or 0),
        crs_name=str(payload.get("crs_name", "") or ""),
        wkid=payload.get("wkid"),
        geom_type=str(payload.get("geom_type", "") or ""),
        feature_count=payload.get("feature_count"),
        extent=payload.get("extent"),
        fields=list(payload.get("fields", []) or []),
        raster=dict(payload.get("raster", {}) or {}),
        extra=dict(payload.get("extra", {}) or {}),
    )
