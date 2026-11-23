# -*- coding: utf-8 -*-
"""任务四a/四b 的离线基线：核密度分级、逐年分布中心、羚羊×土地覆盖比例、长颈鹿聚集性。

用**自己的实现**算一遍（不依赖 agent 产出），作为断言区间来源。

赛题要点：
- 4a：每 5 年一期核密度（1982-1986 / 1987-1991 / 1992-1996 / 1997-2001），
  搜索邻域面积 100 km²，4 张密度图统一图例；逐年分布中心（20 个）→ 迁移图。
- 4b：长颈鹿分布是否聚集（平均最近邻）；羚羊类（Gazelle;Impala）逐年 × 土地覆盖比例表。

输出：benchmark/baselines/gis14_task4a.json、gis14_task4b.json
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

import arcpy
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(r"E:\BST_lab\GIS_CLI")
INPUTS = REPO / "workspace/input/第14届真题数据"
POINTS = INPUTS / "animal_points.shp"
LANDCOVER = INPUTS / "landcover_clip.tif"
BASELINE_DIR = REPO / "benchmark/baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

arcpy.env.overwriteOutput = True
UTM10 = arcpy.SpatialReference(32610)
AREA_KM2 = 100.0
RADIUS_M = float(np.sqrt(AREA_KM2 / np.pi)) * 1000.0   # 100 km² 圆的半径（≈5642 m）
PERIODS = [(1982, 1986), (1987, 1991), (1992, 1996), (1997, 2001)]
GAZELLE = {"Gazelle", "Impala"}

print(f"核密度搜索半径：√(100/π) = {RADIUS_M:.1f} m")

# ---- 读点位 ----
fields = [f.name for f in arcpy.ListFields(str(POINTS))]
print("点位字段:", fields)
rows = []
with arcpy.da.SearchCursor(str(POINTS), ["YEAR", "NAME", "SHAPE@X", "SHAPE@Y"]) as cursor:
    for year, name, x, y in cursor:
        rows.append((int(year), str(name or "").strip(), float(x), float(y)))
print("点数:", len(rows), "| 年份:", min(r[0] for r in rows), "-", max(r[0] for r in rows))

# ---- 4a：逐年分布中心 ----
years = sorted({r[0] for r in rows})
centers = {}
for year in years:
    pts = [(r[2], r[3]) for r in rows if r[0] == year]
    if not pts:
        continue
    arr = np.asarray(pts, dtype="float64")
    centers[year] = [round(float(arr[:, 0].mean()), 2), round(float(arr[:, 1].mean()), 2), len(pts)]
print("逐年中心（前 3 / 后 3）:")
for year in years[:3] + years[-3:]:
    if year in centers:
        x, y, n = centers[year]
        print(f"   {year}: ({x:.0f}, {y:.0f}) n={n}")

# 迁移距离（相邻年）
segments = []
for i in range(len(years) - 1):
    a, b = years[i], years[i + 1]
    if a in centers and b in centers:
        dx = centers[b][0] - centers[a][0]
        dy = centers[b][1] - centers[a][1]
        segments.append({"from_year": a, "to_year": b, "distance_m": round(float(np.hypot(dx, dy)), 1)})
print("迁移距离合计: %.0f m，最大段 %.0f m" % (
    sum(s["distance_m"] for s in segments), max(s["distance_m"] for s in segments)))

# ---- 4a：每 5 年一期的核密度（用 arcpy.sa.KernelDensity，与 agent 同工具，值可比） ----
import tempfile  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="gis14_4ab_"))
GDB = WORK / "base4.gdb"
arcpy.management.CreateFileGDB(str(WORK), GDB.name)
period_stats = []
kde_rasters = []
for start, end in PERIODS:
    subset = [(r[2], r[3]) for r in rows if start <= r[0] <= end]
    if not subset:
        continue
    fc = GDB / f"pts_{start}_{end}"
    arcpy.management.CreateFeatureclass(str(GDB), f"pts_{start}_{end}", "POINT", spatial_reference=UTM10)
    with arcpy.da.InsertCursor(str(fc), ["SHAPE@XY"]) as cursor:
        for xy in subset:
            cursor.insertRow([xy])
    out_raster = str(GDB / f"kde_{start}_{end}")
    # 实测：KernelDensity 不接受 out_raster 关键字参数，必须接住返回的 Raster 再 save
    density = arcpy.sa.KernelDensity(
        in_features=str(fc), population_field="NONE", cell_size=100.0,
        search_radius=RADIUS_M, area_unit_scale_factor="SQUARE_KILOMETERS",
    )
    density.save(out_raster)
    kde_rasters.append(out_raster)
    arr = arcpy.RasterToNumPyArray(out_raster, nodata_to_value=np.nan).astype("float64")
    period_stats.append({
        "period": f"{start}-{end}",
        "points": len(subset),
        "max": round(float(np.nanmax(arr)), 4),
        "p95": round(float(np.nanpercentile(arr, 95)), 4),
        "mean": round(float(np.nanmean(arr)), 4),
    })
    print(f"   核密度 {start}-{end}: n={len(subset):<5} max={period_stats[-1]['max']:.3f} P95={period_stats[-1]['p95']:.3f}")

# 统一分级建议：取各期最大值的上界，做整数步长
overall_max = max(s["max"] for s in period_stats)
step = 10 ** int(np.floor(np.log10(max(overall_max / 5.0, 1e-6))))
for candidate in (1, 2, 2.5, 5, 10):
    if overall_max / (step * candidate) <= 6:
        step = step * candidate
        break
bounds = [0.0]
while bounds[-1] < overall_max:
    bounds.append(round(bounds[-1] + step, 6))
bounds = [round(b, 4) for b in bounds]
print(f"统一分级建议：边界 {bounds}（步长 {step:g}，覆盖最大 {overall_max:.3f}）")

task4a = {
    "task": "gis14_am_task4a",
    "params": {"neighborhood_area_km2": AREA_KM2, "search_radius_m": round(RADIUS_M, 1),
               "cell_m": 100.0, "periods": [f"{a}-{b}" for a, b in PERIODS]},
    "yearly_centers": {str(year): centers[year] for year in years if year in centers},
    "migration_segments": segments,
    "period_kde_stats": period_stats,
    "suggested_bounds": bounds,
    "suggested_step": step,
    "notes": "核密度值单位：点密度（点/km²，按 area_unit_scale_factor=SQUARE_KILOMETERS）",
}
(BASELINE_DIR / "gis14_task4a.json").write_text(json.dumps(task4a, ensure_ascii=False, indent=2), encoding="utf-8")
print("4a 基线已写入")

# ---- 4b-1：长颈鹿逐年平均最近邻（numpy 实现，等价 arcpy.stats.AverageNearestNeighbor） ----
def nn_ratio(points: np.ndarray, area_km2: float) -> dict:
    """平均最近邻比率 R = 观测平均最近邻距离 / 期望距离(1/(2√(n/A)))，含 Z 检验。"""
    n = points.shape[0]
    if n < 3:
        return {"n": n, "observed": None, "expected": None, "ratio": None, "z": None}
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    observed = float(nearest.mean())
    density = n / (area_km2 * 1e6)                 # 点/m²
    expected = 1.0 / (2.0 * np.sqrt(density))
    se = 0.26136 / np.sqrt(n * n / (area_km2 * 1e6))
    z = (observed - expected) / se if se else 0.0
    return {"n": n, "observed": round(observed, 2), "expected": round(expected, 2),
            "ratio": round(observed / expected, 4) if expected else None, "z": round(float(z), 3)}


study_area = arcpy.Describe(str(INPUTS / "study_area.shp")).extent
study_km2 = (study_area.XMax - study_area.XMin) * (study_area.YMax - study_area.YMin) / 1e6
giraffe = np.asarray([(r[2], r[3]) for r in rows if r[1] == "Giraffe"], dtype="float64")
giraffe_all = nn_ratio(giraffe, study_km2)
print(f"\n长颈鹿（全部年份 n={giraffe_all['n']}）: R={giraffe_all['ratio']} Z={giraffe_all['z']} "
      f"（<1 聚集，>1 离散）")
giraffe_by_year = {}
for year in years:
    subset = np.asarray([(r[2], r[3]) for r in rows if r[1] == "Giraffe" and r[0] == year], dtype="float64")
    result = nn_ratio(subset, study_km2)
    giraffe_by_year[str(year)] = result
    if subset.shape[0] >= 3:
        print(f"   {year}: n={result['n']:<4} R={result['ratio']} Z={result['z']}")

# ---- 4b-2：羚羊类逐年 × 土地覆盖比例 ----
lc = arcpy.RasterToNumPyArray(str(LANDCOVER), nodata_to_value=-9999).astype("int32")
lc_desc = arcpy.Describe(str(LANDCOVER))
lc_extent = lc_desc.extent
lc_cell = lc_desc.meanCellWidth
NAMES = {1: "水域", 2: "林地", 4: "淹没植被区", 5: "耕地", 7: "建筑区", 8: "裸地",
         9: "雪/冰", 10: "云", 11: "牧场"}


def sample_class(x: float, y: float) -> int:
    col = int((x - lc_extent.XMin) / lc_cell)
    row = int((lc_extent.YMax - y) / lc_cell)
    if 0 <= row < lc.shape[0] and 0 <= col < lc.shape[1]:
        return int(lc[row, col])
    return -9999


proportions: dict[str, dict] = {}
for year in years:
    subset = [r for r in rows if r[1] in GAZELLE and r[0] == year]
    if not subset:
        continue
    codes = Counter(sample_class(r[2], r[3]) for r in subset)
    total = sum(v for k, v in codes.items() if k > 0)
    outside = codes.get(-9999, 0)
    proportions[str(year)] = {
        "total": len(subset),
        "valid": total,
        "outside": outside,
        "by_class": {
            NAMES.get(code, str(code)): round(count / total, 4)
            for code, count in sorted(codes.items()) if code > 0 and total
        },
    }
print("\n羚羊类逐年分布比例（示例）:")
for year in [y for y in years if str(y) in proportions][:3] + [y for y in years if str(y) in proportions][-2:]:
    entry = proportions[str(year)]
    top = sorted(entry["by_class"].items(), key=lambda kv: -kv[1])[:3]
    print(f"   {year}: n={entry['total']:<4} 有效={entry['valid']:<4} 主要分布={top}")

task4b = {
    "task": "gis14_am_task4b",
    "study_area_km2": round(study_km2, 2),
    "giraffe_nn_all": giraffe_all,
    "giraffe_nn_by_year": giraffe_by_year,
    "gazelle_proportions_by_year": proportions,
    "gazelle_species": sorted(GAZELLE),
    "notes": "比例按‘羚羊类点所在土地覆盖类别计数 / 该年有效点数’计算；R<1 且 Z<-1.96 判聚集",
}
(BASELINE_DIR / "gis14_task4b.json").write_text(json.dumps(task4b, ensure_ascii=False, indent=2), encoding="utf-8")
print("4b 基线已写入")
print("工作目录（中间栅格）:", WORK)
