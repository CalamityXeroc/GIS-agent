# -*- coding: utf-8 -*-
"""SQLite-backed incremental catalog of GIS datasets.

The catalog is the single source of truth for "what data exists and what does
it look like". It is refreshed incrementally by mtime/size so planning no
longer triggers a full ArcPy scan on every call.
"""

from __future__ import annotations

import json
import os
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
    source TEXT,
    license TEXT,
    acquired_at TEXT,
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
-- 属性表行缓存（DAIR-328）：一次性把真实表读进来，之后的分页/排序/筛选
-- 全走参数化 SQL，不再每次请求重读数据文件。
CREATE TABLE IF NOT EXISTS table_cache_meta (
    path TEXT PRIMARY KEY,
    mtime REAL,
    size INTEGER,
    row_count INTEGER,
    field_count INTEGER,
    fields TEXT,
    truncated INTEGER DEFAULT 0,
    source TEXT,
    notes TEXT,
    built_at TEXT
);
CREATE TABLE IF NOT EXISTS table_cache_rows (
    path TEXT NOT NULL,
    row_no INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (path, row_no)
);
-- 栅格逐波段元数据（DAIR-328）：读不到的行保留 unavailable 原因，不写假数值。
CREATE TABLE IF NOT EXISTS raster_bands (
    path TEXT NOT NULL,
    band INTEGER NOT NULL,
    mtime REAL,
    size INTEGER,
    name TEXT,
    dtype TEXT,
    min REAL,
    max REAL,
    mean REAL,
    nodata REAL,
    width INTEGER,
    height INTEGER,
    cell_size REAL,
    color_interpretation TEXT,
    unavailable TEXT,
    updated_at TEXT,
    PRIMARY KEY (path, band)
);
"""

#: 行缓存单表上限：超过就只缓存前 N 行，并在响应里如实标 ``cache_truncated``。
MAX_CACHE_ROWS = 50_000
#: 单元格最长字符数：超长文本（备注/XML 片段）截断，避免一条记录撑爆缓存。
MAX_CELL_CHARS = 1_000


#: 数据来源（P0-2）：接入时就登记，扫描器不覆盖。
SOURCES = ("local", "csv_upload", "boundary", "download")
#: 许可（P0-2）：未知就写 unknown，绝不猜。
LICENSES = ("unknown", "ODbL", "天地图条款", "用户自有")

#: 老库补列用的迁移表（SQLite 只能 ALTER TABLE ADD COLUMN）。
_ADDED_COLUMNS = {
    "source": "TEXT",
    "license": "TEXT",
    "acquired_at": "TEXT",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_catalog_path(path: str | Path, workspace: str | Path) -> str:
    """把任意路径写法归一成「相对 workspace」的一条（catalog 的路径主键口径）。

    为什么必须归一（DAIR-335 实测）：同一份数据被两次扫描分别以绝对根
    （``E:\\...\\workspace\\input\\x.shp``）和「扫描时 cwd = 仓库根」的相对写法
    （``workspace\\input\\x.shp``）写进 catalog，379 条记录里 167 对是同一个
    文件；目录树于是在根下长出 ``E:`` 与 ``workspace`` 两个分支，用户看到
    同一数据集两份。

    规则（只做减法，不猜）：

    - 绝对路径且落在 workspace 内 → ``relpath``；落不进去（外部数据源）原样返回；
    - 相对路径且首段就是工作区目录名 → 去掉首段（老的仓库根写法）；
    - 其余原样返回。

    落库前调用（``refresh_catalog(workspace=...)`` / ``Catalog.renormalize_paths``），
    读的时候仍接受老写法（``catalog_api._resolve``），不强制迁移到某一刻才可用。
    """
    if not str(path):
        return ""
    raw = os.path.normpath(str(path))
    root = os.path.normpath(str(workspace))
    if os.path.isabs(raw):
        try:
            rel = os.path.relpath(raw, root)
        except ValueError:  # 跳盘符时 relpath 抛 ValueError：保持绝对写法
            return raw
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            return raw
        return rel
    head, sep, rest = raw.partition(os.sep)
    if sep and rest and head.lower() == os.path.basename(root).lower():
        return rest
    return raw


#: extra 里属于“上一次扫描的瞬时结论”的键：重扫没带就该清掉，不该继承。
_TRANSIENT_EXTRA_KEYS = {"error"}

#: 「上次扫描把类型归错了」的记录：有几何的标成表、没几何的标成要素类。
#: 这类记录光靠 mtime 判新鲜会永远自愈不了（DAIR-343：.gdb 里的非空间表）。
_KIND_MISMATCH_SQL = (
    "(kind = 'feature_class' AND COALESCE(geom_type, '') = '')"
    " OR (kind = 'table' AND COALESCE(geom_type, '') <> '')"
)


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
    #: 接入来源 / 许可 / 采集时间。空串表示"本次没带信息"：
    #: 写入时保留库里的旧值（扫描器重扫不能把 csv_upload 打回 local）。
    source: str = ""
    license: str = ""
    acquired_at: str = ""

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
            self.source,
            self.license,
            self.acquired_at,
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
            source=row["source"] or "local",
            license=row["license"] or "unknown",
            acquired_at=row["acquired_at"] or "",
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
            self._migrate()

    def _migrate(self) -> None:
        """给 P0-2 之前建的库补 source/license/acquired_at 三列。

        ``CREATE TABLE IF NOT EXISTS`` 对已存在的表不起作用，老库（用户机器上
        已经跑出来的 catalog.db）不会自动长出新列 —— 只能显式 ALTER。
        """
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(datasets)")}
        for name, decl in _ADDED_COLUMNS.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE datasets ADD COLUMN {name} {decl}")

    # ------------------------------------------------------------------ write
    def upsert(self, record: DatasetRecord) -> None:
        with self._lock, self._conn:
            self._merge_provenance(record)
            self._conn.execute(
                """
                INSERT INTO datasets
                    (path, kind, container, name, mtime, size, crs_name, wkid,
                     geom_type, feature_count, extent, extra,
                     source, license, acquired_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    kind=excluded.kind, container=excluded.container, name=excluded.name,
                    mtime=excluded.mtime, size=excluded.size, crs_name=excluded.crs_name,
                    wkid=excluded.wkid, geom_type=excluded.geom_type,
                    feature_count=excluded.feature_count, extent=excluded.extent,
                    extra=excluded.extra, source=excluded.source,
                    license=excluded.license, acquired_at=excluded.acquired_at,
                    updated_at=excluded.updated_at
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

    def _merge_provenance(self, record: DatasetRecord) -> None:
        """空串字段沿用库里已有值，缺省值落到 local/unknown。

        为什么不能在 SQL 里用 COALESCE：扫描器每次都 upsert 全量记录，
        它不知道"这份 CSV 是用户上传的"。若把它带来的空串写成 local，
        一次重扫就会把 `source=csv_upload, license=用户自有` 冲掉。

        注意：会就地补全传入的 ``record``（调用方拿到的对象与库里一致）。
        """
        row = self._conn.execute(
            "SELECT source, license, acquired_at, extra FROM datasets WHERE path = ?",
            (record.path,),
        ).fetchone()
        old = dict(row) if row is not None else {}
        record.source = record.source or old.get("source") or "local"
        record.license = record.license or old.get("license") or "unknown"
        record.acquired_at = record.acquired_at or old.get("acquired_at") or ""
        # extra：登记时写的元信息（来源格式、原始列名、跳过行数…）不能被一次重扫抹掉。
        # 扫描器不认识这些字段，它带来的 extra 往往是空的；而 error 是上一次扫描的
        # 瞬时结论，新扫描没带就该丢——所以“旧的先留、新的覆盖、瞬时键清掉”。
        old_extra = json.loads(old.get("extra") or "{}") if old.get("extra") else {}
        merged_extra = {k: v for k, v in old_extra.items() if k not in _TRANSIENT_EXTRA_KEYS}
        merged_extra.update(record.extra or {})
        record.extra = merged_extra

    def renormalize_paths(self, workspace: str | Path) -> int:
        """把历史记录的路径归一成 workspace 相对写法，并把重复记录合并掉。

        幂等：库已经是唯一写法时返回 0。合并规则：同一个文件的两种写法只留
        ``updated_at`` 更新的那条（它来自最近一次真实扫描），另一条的
        ``fields`` / ``rasters`` 一并删掉，不留孤儿行。

        不碰 ``table_cache_*``：那几个表按「resolve 后的绝对路径」记，
        与 datasets 的主键口径无关。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, container FROM datasets"
            ).fetchall()
        plan: list[tuple[str, str, str]] = []
        for row in rows:
            new_path = normalize_catalog_path(row["path"], workspace)
            new_container = (
                normalize_catalog_path(row["container"], workspace)
                if row["container"]
                else ""
            )
            if new_path != row["path"] or new_container != (row["container"] or ""):
                plan.append((row["path"], new_path, new_container))
        if not plan:
            return 0

        changed = 0
        with self._lock, self._conn:
            for old, new_path, new_container in plan:
                if self._conn.execute(
                    "SELECT 1 FROM datasets WHERE path = ?", (old,)
                ).fetchone() is None:
                    continue  # 前一轮已作为重复项删掉了
                if new_path == old:  # 只有 container 要归一
                    self._conn.execute(
                        "UPDATE datasets SET container = ? WHERE path = ?",
                        (new_container, old),
                    )
                    changed += 1
                    continue
                existing = self._conn.execute(
                    "SELECT updated_at FROM datasets WHERE path = ?", (new_path,)
                ).fetchone()
                if existing is not None:
                    old_row = self._conn.execute(
                        "SELECT updated_at FROM datasets WHERE path = ?", (old,)
                    ).fetchone()
                    if str(old_row["updated_at"] or "") > str(
                        existing["updated_at"] or ""
                    ):
                        self._drop_records({new_path})
                        self._rename_record(old, new_path, new_container)
                    else:
                        self._drop_records({old})
                        changed += 1
                        continue
                else:
                    self._rename_record(old, new_path, new_container)
                changed += 1
        return changed

    def _drop_records(self, paths: Iterable[str]) -> None:
        """删记录及其 fields/rasters（调用方已持锁与事务）。"""
        for path in paths:
            for table in ("datasets", "fields", "rasters"):
                self._conn.execute(f"DELETE FROM {table} WHERE path = ?", (path,))

    def _rename_record(self, old: str, new_path: str, new_container: str) -> None:
        """换主键（调用方已持锁与事务，且已删掉占着新键的那条）。"""
        self._conn.execute(
            "UPDATE datasets SET path = ?, container = ? WHERE path = ?",
            (new_path, new_container, old),
        )
        for table in ("fields", "rasters"):
            self._conn.execute(
                f"UPDATE {table} SET path = ? WHERE path = ?", (new_path, old)
            )

    def remove_missing(
        self,
        existing_paths: set[str],
        existing_containers: set[str] | None = None,
        within: list[str] | None = None,
        workspace: str | Path | None = None,
    ) -> int:
        """Delete entries whose files no longer exist.

        Records inside a container (GDB) are kept when the container itself is
        still present, even though their own path is not in ``existing_paths``.

        ``within``：限定只清理这些路径前缀（绝对路径，小写）下的记录。
        为什么需要：扫描可以只指定部分根目录（如只扫 ``input/``），
        此时不能把未参与扫描的目录（``output/``）里的记录当成“已消失”——
        实测旧逻辑会把自己刚生成的成果从目录里删掉（removed=19）。
        传 ``None`` 时保持旧行为（全库对账）。

        ``workspace``：把「全库对账」限定成「工作区内全库对账」。扫描根就是
        workspace 本身时（没有 input/data/output 的兜底根）走 ``within=None``，
        此时工作区**之外**的绝对路径记录（外部数据源，归一函数按设计原样保留）
        既不归这次扫描管、也没法证明已消失，不能顺手删掉。
        判据：归一后仍是绝对写法 = 落在工作区外（与 ``normalize_catalog_path``
        同一口径，不另造一套边界判断）。
        """
        prefixes = [str(p).replace("/", "\\").lower() for p in (within or []) if p]

        def _in_scope(path: str) -> bool:
            if workspace is not None:
                key = normalize_catalog_path(path, workspace)
                if os.path.isabs(key):
                    return False
                # 相对写法里以 ``..`` 开头的也落在工作区外（扫描根给的是相对路径时）
                return key != os.pardir and not key.startswith(os.pardir + os.sep)
            if not prefixes:
                return True
            normalized = str(path).replace("/", "\\").lower()
            return any(normalized.startswith(prefix) for prefix in prefixes)

        containers = existing_containers or set()
        removed = 0
        with self._lock, self._conn:
            rows = self._conn.execute("SELECT path, container FROM datasets").fetchall()
            for row in rows:
                if row["path"] in existing_paths:
                    continue
                if row["container"] and row["container"] in containers:
                    continue
                if not _in_scope(row["path"]):
                    continue
                self._conn.execute("DELETE FROM datasets WHERE path = ?", (row["path"],))
                self._conn.execute("DELETE FROM fields WHERE path = ?", (row["path"],))
                self._conn.execute("DELETE FROM rasters WHERE path = ?", (row["path"],))
                removed += 1
        return removed

    # ------------------------------------------------------------------- read
    def kind_mismatch_keys(self) -> set[str]:
        """kind 与 geom_type 矛盾的记录所在的 path / container。

        刷新时这些 key 不能按 mtime 跳过：必须重扫一遍才能把错分类改对。
        返回 container 是为了让 .gdb 容器整体重扫（成员记录单独扫不了）。
        """
        with self._lock:
            rows = self._conn.execute(
                f"SELECT path, container FROM datasets WHERE {_KIND_MISMATCH_SQL}"
            ).fetchall()
        keys: set[str] = set()
        for row in rows:
            keys.add(row["path"])
            if row["container"]:
                keys.add(row["container"])
        return keys

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
            rec.raster = self._raster(path)
        return rec

    def _raster(self, path: str) -> dict[str, Any]:
        """栅格汇总行（band_count/尺寸/min/max/mean/nodata）。

        以前 ``get()`` 只回填 ``fields``，栅格记录拿到的 ``raster`` 恒为空 ——
        画像接口于是永远不告诉前端「几段/多大/NoData 是多少」（DAIR-328 修）。
        """
        row = self._conn.execute(
            "SELECT * FROM rasters WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return {}
        data = {
            key: row[key]
            for key in row.keys()
            if key not in {"path", "crs_name", "wkid", "extra"}
        }
        data["extra"] = json.loads(row["extra"]) if row["extra"] else {}
        return data

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

    # ------------------------------------------------------- 属性表行缓存
    def table_meta(self, path: str) -> dict[str, Any] | None:
        """行缓存的元信息（含字段清单）；没有缓存返回 None。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM table_cache_meta WHERE path = ?", (path,)
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["fields"] = json.loads(data.get("fields") or "[]")
        data["notes"] = json.loads(data.get("notes") or "[]")
        data["truncated"] = bool(data.get("truncated"))
        return data

    def replace_table_cache(
        self,
        path: str,
        *,
        mtime: float,
        size: int,
        fields: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        truncated: bool,
        source: str,
        notes: list[str] | None = None,
    ) -> None:
        """整体替换某个数据集的行缓存（先删后插，同一事务）。"""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM table_cache_rows WHERE path = ?", (path,))
            self._conn.executemany(
                "INSERT OR REPLACE INTO table_cache_rows (path, row_no, data) VALUES (?, ?, ?)",
                [
                    (path, index, json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    for index, row in enumerate(rows)
                ],
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO table_cache_meta
                    (path, mtime, size, row_count, field_count, fields,
                     truncated, source, notes, built_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path,
                    float(mtime),
                    int(size),
                    len(rows),
                    len(fields),
                    json.dumps(fields, ensure_ascii=False),
                    1 if truncated else 0,
                    source,
                    json.dumps(notes or [], ensure_ascii=False),
                    _now(),
                ),
            )

    def drop_table_cache(self, path: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM table_cache_rows WHERE path = ?", (path,))
            self._conn.execute("DELETE FROM table_cache_meta WHERE path = ?", (path,))

    @staticmethod
    def _field_value(alias: str = "data") -> str:
        """按**键名**取一个字段值（键名作绑定参数），返回 SQL 片段。

        不用 ``json_extract(data, '$.路径')``：JSON 路径对含双引号的字段名没有
        跨版本都成立的写法（裸键 ``$.A"B`` 只在较新的 SQLite 上能解析，
        Python 3.10 自带的旧版直接返回 NULL）。改成 json_each 按键匹配，
        任何字段名都安全；排序时每行多一次子查询（12,847 行翻页实测 12.6 ms）。
        """
        return f"(SELECT value FROM json_each({alias}) WHERE key = ?)"

    def _where_sql(
        self, where: tuple[str, str, Any] | None
    ) -> tuple[str, list[Any]]:
        """把已校验的单条件转成 SQL 片段；字段/运算符/值全部来自白名单或绑定参数。"""
        if where is None:
            return "", []
        field, op, value = where
        # op 已在调用方按白名单映射成 SQL 运算符；这里再兜一层，防止以后有人传生文本。
        if op not in {"=", "<>", ">", "<", ">=", "<=", "LIKE"}:
            raise ValueError(f"unsupported operator: {op}")
        return (
            f" AND {self._field_value()} {op} ?",
            [field, value],
        )

    def query_rows(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 100,
        order: str = "",
        direction: str = "asc",
        where: tuple[str, str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """分页取行。返回 ``(rows, total)``；``total`` 是**筛选后**的总行数。

        排序/筛选的字段名只作为绑定参数按键取值，值一律参数化；
        方向只接受 asc/desc（映射成字面量 ASC/DESC），用户文本进不了 SQL 语法。
        """
        where_sql, params = self._where_sql(where)
        count_sql = f"SELECT COUNT(*) FROM table_cache_rows WHERE path = ?{where_sql}"
        if order:
            direction_sql = "DESC" if str(direction).lower() == "desc" else "ASC"
            field_value = self._field_value()
            # NULL 恒排在最后（与 ArcGIS/QGIS 属性表一致），避免空值把首屏占满。
            order_sql = (
                f"ORDER BY ({field_value} IS NULL), "
                f"{field_value} COLLATE NOCASE {direction_sql}"
            )
            order_params: list[Any] = [order, order]
        else:
            order_sql = "ORDER BY row_no ASC"
            order_params = []
        page_sql = (
            f"SELECT data FROM table_cache_rows WHERE path = ?{where_sql} "
            f"{order_sql} LIMIT ? OFFSET ?"
        )
        with self._lock:
            total = int(self._conn.execute(count_sql, (path, *params)).fetchone()[0])
            rows = self._conn.execute(
                page_sql, (path, *params, *order_params, int(limit), int(offset))
            ).fetchall()
        return [json.loads(row["data"]) for row in rows], total

    def distinct_values(
        self, path: str, field: str, *, limit: int = 200
    ) -> dict[str, Any]:
        """唯一值 + 计数（按出现次数降序），并给出真实基数与空值数。"""
        field_value = self._field_value()
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT {field_value} AS v, COUNT(*) AS n
                FROM table_cache_rows WHERE path = ?
                GROUP BY v ORDER BY n DESC, v ASC LIMIT ?
                """,
                (field, path, int(limit) + 1),
            ).fetchall()
            total_distinct = int(
                self._conn.execute(
                    f"SELECT COUNT(DISTINCT {field_value}) FROM table_cache_rows "
                    "WHERE path = ?",
                    (field, path),
                ).fetchone()[0]
            )
            null_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM table_cache_rows "
                    f"WHERE path = ? AND {field_value} IS NULL",
                    (path, field),
                ).fetchone()[0]
            )
        values = [{"value": row["v"], "count": int(row["n"])} for row in rows[:limit]]
        return {
            "values": values,
            "total_distinct": total_distinct,
            "null_count": null_count,
            "truncated": len(rows) > limit,
        }

    def numeric_values(
        self, path: str, field: str, *, max_values: int = 100_000
    ) -> list[float]:
        """取某数值列的**全部**取值（用于分位断点）；超上限时按行号等距抽样。"""
        field_value = self._field_value()
        with self._lock:
            row_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM table_cache_rows WHERE path = ?", (path,)
                ).fetchone()[0]
            )
            stride = max(1, (row_count + max_values - 1) // max_values) if row_count else 1
            rows = self._conn.execute(
                f"""
                SELECT {field_value} AS v FROM table_cache_rows
                WHERE path = ? AND row_no % ? = 0
                  AND typeof({field_value}) IN ('integer', 'real')
                ORDER BY v
                """,
                (field, path, stride, field),
            ).fetchall()
        return [float(row["v"]) for row in rows]

    # ------------------------------------------------------- 栅格逐波段元数据
    def raster_band_rows(self, path: str, mtime: float, size: int) -> list[dict[str, Any]]:
        """读逐波段缓存；mtime/size 对不上（数据被改过）就当没有缓存。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM raster_bands WHERE path = ? ORDER BY band", (path,)
            ).fetchall()
        if not rows:
            return []
        if any(
            abs(float(row["mtime"] or 0) - float(mtime)) > 1e-6
            or int(row["size"] or 0) != int(size)
            for row in rows
        ):
            return []
        return [dict(row) for row in rows]

    def save_raster_bands(
        self, path: str, *, mtime: float, size: int, bands: list[dict[str, Any]]
    ) -> None:
        """整体替换逐波段缓存（含 unavailable 原因；不写假数值）。"""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM raster_bands WHERE path = ?", (path,))
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO raster_bands
                    (path, band, mtime, size, name, dtype, min, max, mean, nodata,
                     width, height, cell_size, color_interpretation, unavailable, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        path,
                        int(band.get("band") or 0),
                        float(mtime),
                        int(size),
                        band.get("name"),
                        band.get("dtype"),
                        band.get("min"),
                        band.get("max"),
                        band.get("mean"),
                        band.get("nodata"),
                        band.get("width"),
                        band.get("height"),
                        band.get("cell_size"),
                        band.get("color_interpretation"),
                        band.get("unavailable"),
                        _now(),
                    )
                    for band in bands
                ],
            )

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
        source=str(payload.get("source", "") or ""),
        license=str(payload.get("license", "") or ""),
        acquired_at=str(payload.get("acquired_at", "") or ""),
    )
