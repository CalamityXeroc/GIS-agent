# -*- coding: utf-8 -*-
"""catalog 路径口径（DAIR-335）：一律存「相对 workspace」，重复记录要合并。

回归的症状：目录树根下出现 ``E: 193`` 与 ``workspace 186`` 两个分支，同一个
文件（同一个 feature_count）出现两遍。根因是两次扫描用了不同的 root 拼写
（绝对根 / 「扫描时 cwd = 仓库根」的相对根），落库时没有归一，path 主键因此不同。

这里锁住四件事：

1. 归一函数只做减法：工作区内的绝对路径 → 相对写法；老写法去掉首段；
   工作区外的绝对路径原样保留（外部数据源不能被我改写成相对路径）。
2. ``refresh_catalog(workspace=...)`` 落库的是相对写法，含 ``container``。
3. 归一后新鲜度判断仍然成立（不会因为换了写法就把整库重扫一遍）。
4. 历史重复能合并：留 ``updated_at`` 更新的那条，另一条的 fields/rasters 一起删，
   不留孤儿行；重复调用返回 0（幂等）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gis_cli.datacatalog.scanner import refresh_catalog
from gis_cli.datacatalog.store import Catalog, DatasetRecord, normalize_catalog_path


class _KernelMustNotRun:
    """KML 走本地路径，绝不能起内核（起了就说明分支判断错了）。"""

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, code: str, timeout: float = 0) -> object:  # pragma: no cover
        self.calls += 1
        raise AssertionError("不该起 ArcPy 内核")


class _FakeGdbKernel:
    """假内核：按给定 kind 返回 .gdb 内一个成员的扫描结果。"""

    def __init__(self, gdb: Path, member: str, kind: str, geom_type: str = "") -> None:
        self.gdb = gdb
        self.member = member
        self.kind = kind
        self.geom_type = geom_type
        self.calls = 0

    def execute(self, code: str, timeout: float = 0) -> object:
        self.calls += 1
        return SimpleNamespace(
            ok=True,
            error=None,
            result={
                "records": [
                    {
                        "path": str(self.gdb / self.member),
                        "kind": self.kind,
                        "container": str(self.gdb),
                        "name": self.member,
                        "crs_name": "",
                        "wkid": None,
                        # 这个成员是非空间表：扫描器给的 geom_type 必须是空，
                        # 差别只在 kind（旧版一律 feature_class）。
                        "geom_type": self.geom_type,
                        "feature_count": 12,
                        "extent": None,
                        "fields": [
                            {
                                "name": "code",
                                "alias": "",
                                "type": "String",
                                "length": 8,
                                "nullable": True,
                            }
                        ],
                    }
                ]
            },
        )


def _rel(*parts: str) -> str:
    """工作区相对写法的期望值（与 normalize_catalog_path 的返回口径一致）。"""
    return str(Path(*parts))


def _kml(path: Path) -> Path:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>点位</name>
<Placemark><name>甲</name><Point><coordinates>116.4,39.9,0</coordinates></Point></Placemark>
</Document></kml>
""",
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------------------ 归一函数
def test_normalize_absolute_path_inside_workspace() -> None:
    workspace = Path("E:/data/ws")
    assert (
        normalize_catalog_path("E:/data/ws/input/a.shp", workspace) == "input\\a.shp"
        or normalize_catalog_path("E:/data/ws/input/a.shp", workspace) == "input/a.shp"
    )


def test_normalize_strips_legacy_repo_root_prefix() -> None:
    workspace = Path("E:/data/ws")
    assert normalize_catalog_path("ws\\input\\a.shp", workspace) == "input\\a.shp"


def test_normalize_keeps_paths_outside_workspace() -> None:
    workspace = Path("E:/data/ws")
    assert normalize_catalog_path("D:/other/a.shp", workspace) == "D:\\other\\a.shp"
    assert normalize_catalog_path("input\\a.shp", workspace) == "input\\a.shp"
    assert normalize_catalog_path("", workspace) == ""


def test_normalize_is_idempotent() -> None:
    workspace = Path("E:/data/ws")
    once = normalize_catalog_path("E:/data/ws/input/a.shp", workspace)
    assert normalize_catalog_path(once, workspace) == once


# ------------------------------------------------------------------ 落库口径
def test_refresh_writes_workspace_relative_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "input"
    root.mkdir(parents=True)
    _kml(root / "points.kml")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    kernel = _KernelMustNotRun()

    stats = refresh_catalog(catalog, [root], kernel, workspace=workspace)

    assert kernel.calls == 0
    assert stats.errors == []
    record = catalog.get("input\\points.kml")
    assert record is not None, f"落库应是工作区相对写法，实际：{[r.path for r in catalog.list_datasets()]}"
    assert record.path.endswith("points.kml")
    catalog.close()


def test_second_refresh_with_workspace_skips_unchanged(tmp_path: Path) -> None:
    """换了写法也不能把新鲜度判据弄坏：第二次扫描必须全部跳过。"""
    workspace = tmp_path / "workspace"
    root = workspace / "input"
    root.mkdir(parents=True)
    _kml(root / "points.kml")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    kernel = _KernelMustNotRun()

    refresh_catalog(catalog, [root], kernel, workspace=workspace)
    stats = refresh_catalog(catalog, [root], kernel, workspace=workspace)

    assert stats.skipped == 1, stats
    assert stats.scanned == 0
    assert stats.removed == 0
    catalog.close()


# ------------------------------------------------------------------ 合并重复
def test_renormalize_merges_duplicate_spellings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "input"
    root.mkdir(parents=True)
    _kml(root / "points.kml")
    db = workspace / ".gis_agent" / "catalog.db"
    catalog = Catalog(db)

    rel = DatasetRecord(
        path="input\\points.kml",
        kind="vector",
        name="points.kml",
        mtime=1.0,
        size=10,
        feature_count=1,
        fields=[{"name": "name", "type": "String"}],
    )
    stale_abs = DatasetRecord(
        path=str(root / "points.kml"),
        kind="vector",
        name="points.kml",
        mtime=1.0,
        size=10,
        feature_count=1,
        fields=[{"name": "旧字段", "type": "String"}],
    )
    catalog.upsert(rel)
    catalog.upsert(stale_abs)
    # upsert 只写 updated_at=now()，这里手工把「绝对写法那条」改成更早：
    with catalog._lock, catalog._conn:  # noqa: SLF001 - 造数据，直接动库
        catalog._conn.execute(
            "UPDATE datasets SET updated_at = '2026-10-01T00:00:00Z' WHERE path LIKE '%:\\%'"
        )
    assert len(catalog.list_datasets()) == 2, "两条不同写法应当都在库里（这就是待修的症状）"

    merged = catalog.renormalize_paths(workspace)

    assert merged == 1
    rows = catalog.list_datasets()
    assert [row.path for row in rows] == ["input\\points.kml"]
    survivor = catalog.get("input\\points.kml")
    assert survivor is not None
    # 留下的必须是「更新的那条」：相对写法那条的字段
    assert [f["name"] for f in survivor.fields] == ["name"]
    assert catalog.get(str(root / "points.kml")) is None
    assert catalog.renormalize_paths(workspace) == 0, "已经归一过就不该再动"
    catalog.close()


def test_renormalize_normalizes_container(tmp_path: Path) -> None:
    """.gdb 内要素类的 container 也得归一，否则新鲜度按 container 查不到。"""
    workspace = tmp_path / "workspace"
    gdb = workspace / "input" / "result.gdb"
    gdb.mkdir(parents=True)
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    catalog.upsert(
        DatasetRecord(
            path=str(gdb / "county"),
            kind="feature_class",
            container=str(gdb),
            name="county",
            mtime=1.0,
            size=10,
            feature_count=3,
        )
    )

    catalog.renormalize_paths(workspace)

    record = catalog.get("input\\result.gdb\\county")
    assert record is not None
    assert record.container == "input\\result.gdb"
    assert catalog.is_fresh("input\\result.gdb", 1.0, 10), "container 归一后新鲜度要能命中"
    catalog.close()


# ------------------------------------------------------------------ 错分类自愈
def test_refresh_rescans_misclassified_gdb_tables(tmp_path: Path) -> None:
    """旧 catalog 把 .gdb 非空间表标成 feature_class：刷新必须重扫，不能按 mtime 跳过。

    数据一动不动时 ``is_fresh`` 永远返回 True，错分类因此永远自愈不了（DAIR-343）。
    """
    workspace = tmp_path / "workspace"
    root = workspace / "input"
    gdb = root / "result.gdb"
    gdb.mkdir(parents=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"fake")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    path = _rel("input", "result.gdb", "统计")

    # 第一次：模拟旧扫描器（.gdb 成员一律 feature_class，表的 geom_type 是空）
    old = _FakeGdbKernel(gdb, "统计", "feature_class")
    refresh_catalog(catalog, [root], old, workspace=workspace)
    assert catalog.get(path).kind == "feature_class"
    assert catalog.get(path).geom_type == ""

    # 第二次：文件没变，但 kind 与 geom_type 矛盾 → 必须重扫
    new = _FakeGdbKernel(gdb, "统计", "table")
    stats = refresh_catalog(catalog, [root], new, workspace=workspace)
    assert new.calls == 1, "错分类的记录不能被 mtime 新鲜度跳过"
    assert stats.scanned == 1
    record = catalog.get(path)
    assert record.kind == "table"
    assert record.feature_count == 12
    assert record.container == _rel("input", "result.gdb")
    assert [f["name"] for f in record.fields] == ["code"]

    # 第三次：自愈后不再重扫（不是每轮都全量扫）
    again = _FakeGdbKernel(gdb, "统计", "table")
    stats = refresh_catalog(catalog, [root], again, workspace=workspace)
    assert again.calls == 0
    assert stats.skipped == 1
    catalog.close()


def test_refresh_leaves_gdb_table_geom_type_empty(tmp_path: Path) -> None:
    """内核把非空间表的 shapeType 报成 "Table" 时，geom_type 落库必须仍是空。

    落成 ``kind='table' + geom_type='Table'`` 会永久命中 store 的自愈谓词
    ``kind='table' AND COALESCE(geom_type, '') <> ''``：每轮 refresh 都把整个
    .gdb 塞进 stale_kinds 重扫（DAIR-348 审查）。
    """
    workspace = tmp_path / "workspace"
    root = workspace / "input"
    gdb = root / "result.gdb"
    gdb.mkdir(parents=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"fake")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    path = _rel("input", "result.gdb", "统计")

    first = _FakeGdbKernel(gdb, "统计", "table", geom_type="Table")
    refresh_catalog(catalog, [root], first, workspace=workspace)
    record = catalog.get(path)
    assert record.kind == "table"
    assert record.geom_type == "", "表没有几何，shapeType='Table' 不是几何类型"

    second = _FakeGdbKernel(gdb, "统计", "table", geom_type="Table")
    stats = refresh_catalog(catalog, [root], second, workspace=workspace)
    assert second.calls == 0, "类型已归正的记录不该每轮重扫整个 .gdb"
    assert stats.skipped == 1
    catalog.close()


# ------------------------------------------------------------------ 扫描根就是 workspace
def test_refresh_with_workspace_root_cleans_deleted_datasets(tmp_path: Path) -> None:
    """根 == workspace 时必须能清理已删记录（DAIR-336 审查打回的回归）。

    归一后 ``relpath(ws, ws)`` 是 ``"."``，拿它当 ``within`` 前缀永远匹配不上
    ``input/...``，于是已删除的数据集永远留在目录里。兜底根
    （``prewarm._candidate_roots`` / ``runner._candidate_data_roots`` 在没有
    input/data/output 时都给 ``[workspace]``）走的正是这条路。
    """
    workspace = tmp_path / "workspace"
    root = workspace / "input"
    root.mkdir(parents=True)
    kml = _kml(root / "points.kml")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    kernel = _KernelMustNotRun()

    refresh_catalog(catalog, [root], kernel, workspace=workspace)
    assert [r.path for r in catalog.list_datasets()] == [_rel("input", "points.kml")]

    kml.unlink()
    stats = refresh_catalog(catalog, [str(workspace)], kernel, workspace=workspace)

    assert stats.removed == 1, "根就是 workspace 也要把已删的数据集清掉"
    assert catalog.list_datasets() == []
    catalog.close()


def test_refresh_with_workspace_root_keeps_output_records(tmp_path: Path) -> None:
    """全库对账不等于乱删：output 里还在的文件不能被当成「已消失」。"""
    workspace = tmp_path / "workspace"
    (workspace / "input").mkdir(parents=True)
    (workspace / "output").mkdir(parents=True)
    _kml(workspace / "input" / "points.kml")
    _kml(workspace / "output" / "result.kml")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    kernel = _KernelMustNotRun()

    refresh_catalog(catalog, [workspace / "input"], kernel, workspace=workspace)
    refresh_catalog(catalog, [workspace / "output"], kernel, workspace=workspace)
    stats = refresh_catalog(catalog, [str(workspace)], kernel, workspace=workspace)

    assert stats.removed == 0
    assert sorted(r.path for r in catalog.list_datasets()) == [
        _rel("input", "points.kml"),
        _rel("output", "result.kml"),
    ]
    catalog.close()


def test_refresh_with_workspace_root_keeps_external_records(tmp_path: Path) -> None:
    """全库对账只对工作区内负责：外部数据源记录不能因为「这次没扫到」被删。

    根 == workspace 时走 ``within=None``（全库对账）。若不加工作区范围，落在
    workspace 之外的绝对路径记录（外部数据源，归一函数按设计原样保留）也会被
    当成「文件已消失」删掉 —— 文件其实还在，只是不归这次扫描管。
    """
    workspace = tmp_path / "workspace"
    (workspace / "input").mkdir(parents=True)
    _kml(workspace / "input" / "points.kml")
    external = tmp_path / "external"
    external.mkdir()
    outside = _kml(external / "ext.kml")
    catalog = Catalog(workspace / ".gis_agent" / "catalog.db")
    kernel = _KernelMustNotRun()

    refresh_catalog(catalog, [external], kernel, workspace=workspace)
    stats = refresh_catalog(catalog, [str(workspace)], kernel, workspace=workspace)

    assert stats.removed == 0, "工作区外的记录不在本次对账范围内"
    assert str(outside) in {r.path for r in catalog.list_datasets()}
    catalog.close()


# ------------------------------------------------------------------ 引擎即时 refresh
def test_build_engine_refresh_writes_workspace_relative_paths(tmp_path: Path) -> None:
    """web 任务走的即时 refresh 要和 prewarm 同一口径（DAIR-336 审查打回 P1）。

    漏传 ``workspace=`` 时：预热写 ``input/points.kml``，任务启动又用绝对路径写
    一条，库里立刻两份 —— 正是本单要消灭的重复。只在 prewarm 跑起来后才自愈，
    所以必须在源头堵住。
    """
    from gis_cli.engine.runner import build_engine

    workspace = tmp_path / "workspace"
    (workspace / "input").mkdir(parents=True)
    _kml(workspace / "input" / "points.kml")
    (workspace / "config").mkdir()
    (workspace / "config" / "llm_config.json").write_text(
        '{"model": "m", "api_key": "k", "api_base": "http://127.0.0.1:1"}',
        encoding="utf-8",
    )

    bundle = build_engine(workspace=workspace, refresh_catalog=True)
    try:
        paths = [record.path for record in bundle.catalog.list_datasets()]
    finally:
        bundle.close()

    assert paths == [_rel("input", "points.kml")], (
        f"引擎即时 refresh 落库应只有一条相对写法：{paths}"
    )
    assert not [path for path in paths if path[1:2] == ":"], "不能出现绝对路径记录"
