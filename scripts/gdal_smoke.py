# -*- coding: utf-8 -*-
"""GDAL 栈冒烟：离线跑通「读矢量 → 点面计数 → 按面求栅格均值 → 中文图」。

不需要 OSGeo4W，也不需要联网：GDAL 驱动由 pyogrio / rasterio 的 wheel 自带。
跑的是 `.venv-gdal`：

    .venv-gdal/Scripts/python.exe scripts/gdal_smoke.py

三项结果都是**确定值**（几何手算得出），对不上就非零退出 —— 冒烟脚本要能红，
只打印不断言的话，栈坏了也照样"通过"。
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output" / "gdal_smoke"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


#: 必须在 import geopandas/rasterio/pyproj 之前跑完：它们会在导入时读这些变量。
def _drop_foreign_proj_env() -> list[str]:
    """清掉指向 venv 之外的 PROJ/GDAL 数据目录。

    本机装了 PostgreSQL/PostGIS，全局环境里留着 ``PROJ_LIB=E:/PostgreSQL/share/contrib/postgis-3.6/proj``
    与 ``GDAL_DATA=E:/PostgreSQL/gdal-data``。那套 proj.db 的
    layout 是 2，而 wheel 自带的是 6，于是 rasterio 一开数据集就报
    "PROJ: ... from another PROJ installation"。PROJ 只认它先找到的那个 —— 变量
    指向 venv 外就丢掉，回落到 wheel 自带数据。
    """
    dropped: list[str] = []
    for var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
        value = os.environ.get(var)
        if value and not Path(value).as_posix().startswith(REPO.as_posix()):
            os.environ.pop(var, None)
            dropped.append(var)
    return dropped


DROPPED_ENV = _drop_foreign_proj_env()

import geopandas as gpd  # noqa: E402  （必须在上面的环境清理之后）
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.mask import mask  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

#: 三个并排的 0.1°×0.1° 方块，边界取整 → 点阵与栅格列号的期望值可以手算。
ZONES = [
    ("东区", box(100.0, 30.0, 100.1, 30.1)),
    ("中区", box(100.1, 30.0, 100.2, 30.1)),
    ("西区", box(100.2, 30.0, 100.3, 30.1)),
]
#: 每区 4×4 个点，偏移 0.02/0.04/0.06/0.08（四等分，离边界 0.02，不会踩线）。
OFFSETS = (0.02, 0.04, 0.06, 0.08)
POINTS_PER_ZONE = len(OFFSETS) ** 2
#: 栅格 30 列 × 10 行，值 = 列号；每区正好覆盖 10 列 → 均值 4.5 / 14.5 / 24.5。
EXPECTED_MEANS = (4.5, 14.5, 24.5)


def build_inputs() -> tuple[Path, Path, Path]:
    """造三个输入文件（GPKG 面 / GPKG 点 / GeoTIFF），全部落在 output/ 下。"""
    OUT.mkdir(parents=True, exist_ok=True)
    zones_gpkg = OUT / "zones.gpkg"
    points_gpkg = OUT / "points.gpkg"
    dem_tif = OUT / "dem.tif"

    zones = gpd.GeoDataFrame(
        {"name": [name for name, _ in ZONES]},
        geometry=[geom for _, geom in ZONES],
        crs="EPSG:4326",
    )
    zones.to_file(zones_gpkg, driver="GPKG", layer="zones")

    pts: list[Point] = []
    for _, geom in ZONES:
        x0 = geom.bounds[0]
        pts += [Point(x0 + dx, 30.0 + dy) for dx in OFFSETS for dy in OFFSETS]
    # 再丢 3 个落在区外的点：计数必须是"区内的"，全算进去就说明空间谓词没用上。
    pts += [Point(99.97, 30.05), Point(100.33, 30.05), Point(100.05, 30.13)]
    points = gpd.GeoDataFrame({"pid": range(len(pts))}, geometry=pts, crs="EPSG:4326")
    points.to_file(points_gpkg, driver="GPKG", layer="points")

    ncol, nrow = 30, 10
    data = np.tile(np.arange(ncol, dtype="float32"), (nrow, 1))
    transform = rasterio.transform.from_origin(100.0, 30.1, 0.01, 0.01)
    with rasterio.open(
        dem_tif,
        "w",
        driver="GTiff",
        height=nrow,
        width=ncol,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-1.0,
    ) as ds:
        ds.write(data, 1)

    return zones_gpkg, points_gpkg, dem_tif


def count_points_in_zones(zones_gpkg: Path, points_gpkg: Path) -> dict[str, int]:
    """读回两个 GPKG（pyogrio 驱动），数每个面里有几个点。"""
    zones = gpd.read_file(zones_gpkg, layer="zones", engine="pyogrio")
    points = gpd.read_file(points_gpkg, layer="points", engine="pyogrio")
    joined = gpd.sjoin(points, zones, predicate="within", how="inner")
    return joined["name"].value_counts().to_dict()


def zonal_means(zones_gpkg: Path, dem_tif: Path) -> dict[str, float]:
    """读回栅格，按面掩膜求均值（rasterio.mask，不依赖 rasterstats）。"""
    zones = gpd.read_file(zones_gpkg, layer="zones", engine="pyogrio")
    means: dict[str, float] = {}
    with rasterio.open(dem_tif) as ds:
        for name, geom in zip(zones["name"], zones.geometry):
            arr, _ = mask(ds, [geom], crop=True, filled=False)
            means[name] = float(arr.mean())  # masked array：均值自动跳过无效像元
    return means


def make_map(zones_gpkg: Path, points_gpkg: Path, dem_tif: Path,
             counts: dict[str, int], means: dict[str, float], png: Path) -> str:
    """画一张带中文标题的 PNG，返回实际用到的字体名（回退结果要能看见）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next(
        (name for name in ("Microsoft YaHei", "SimHei", "SimSun") if name in installed),
        "DejaVu Sans",  # 没有中文字体就退到默认：图能出，字可能是方框
    )
    plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    zones = gpd.read_file(zones_gpkg, layer="zones", engine="pyogrio")
    points = gpd.read_file(points_gpkg, layer="points", engine="pyogrio")

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    with rasterio.open(dem_tif) as ds:
        img = ds.read(1)
        extent = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)
    im = ax.imshow(img, extent=extent, cmap="viridis", alpha=0.75)
    fig.colorbar(im, ax=ax, shrink=0.85, label="列号（模拟高程）")

    zones.boundary.plot(ax=ax, color="white", linewidth=1.6)
    points.plot(ax=ax, color="red", markersize=12, edgecolor="white", linewidth=0.4)
    for _, row in zones.iterrows():
        name = row["name"]
        cx, cy = row.geometry.centroid.x, row.geometry.bounds[3] - 0.018
        ax.annotate(name, (cx, cy), ha="center", va="center", fontsize=9, color="black")

    # 三行的计数/均值放到图右上角：0.1° 宽的小块里塞不下中文标注，还互相压。
    summary = "\n".join(
        f"{name}：点 {counts.get(name, 0)} 个，均值 {means[name]:.1f}" for name, _ in ZONES
    )
    _, top = ax.get_ylim()
    ax.set_ylim(ax.get_ylim()[0], top + 0.06)  # 给汇总框腾出空白带
    ax.text(0.985, 0.965, summary, transform=ax.transAxes, fontsize=8.5, va="top",
            ha="right", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))

    ax.set_xlabel("经度 / °")
    ax.set_ylabel("纬度 / °")
    ax.set_title("GDAL 冒烟测试：分区点计数与栅格均值", fontsize=13)
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)
    return family


def main() -> int:
    t0 = time.perf_counter()
    zones_gpkg, points_gpkg, dem_tif = build_inputs()

    counts = count_points_in_zones(zones_gpkg, points_gpkg)
    means = zonal_means(zones_gpkg, dem_tif)
    png = OUT / "gdal_smoke.png"
    family = make_map(zones_gpkg, points_gpkg, dem_tif, counts, means, png)

    if DROPPED_ENV:
        print(f"（已忽略外部 PROJ/GDAL 环境变量：{'、'.join(DROPPED_ENV)}）")
    print("[1/3] 点落在面内计数：" + "，".join(f"{n}={counts.get(n, 0)}" for n, _ in ZONES))
    print("[2/3] 按矢量分区栅格均值：" + "，".join(f"{n}={means[n]:.3f}" for n, _ in ZONES))
    print(f"[3/3] PNG：{png}（字体 {family}）")

    for name, _ in ZONES:
        assert counts.get(name, 0) == POINTS_PER_ZONE, (
            f"{name} 点数应为 {POINTS_PER_ZONE}，实际 {counts.get(name, 0)}"
        )
    for (name, _), expected in zip(ZONES, EXPECTED_MEANS):
        assert abs(means[name] - expected) < 1e-6, f"{name} 均值应为 {expected}，实际 {means[name]}"
    assert png.is_file() and png.stat().st_size > 0, f"PNG 没写出来：{png}"

    print(f"OK 耗时 {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
