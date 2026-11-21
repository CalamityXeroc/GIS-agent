# -*- coding: utf-8 -*-
"""任务三b 的离线基线：影像粗匹配 → 归一化 → MNDWI/BU 提取 → 更新土地覆盖 → 分类面积。

用**自己的实现**算一遍（不依赖 agent 产出），作为断言区间来源。

粗匹配的做法（关键经验）：
Landsat **无空间参考**，其数值范围是像素坐标（0~6000）。若只 DefineProjection 再
ProjectRaster，范围仍是 0~6000（被当成米），研究区（UTM 三四十万米）根本不在其中，
ClippingGeometry 会报 ERROR 001566“裁剪要素超出栅格范围”。
因此这里按“整幅影像 ↔ 目标范围线性对应”的假设做**显式重采样**（最近邻），
即赛题所说的“缺乏控制点时的粗匹配”。

输出：benchmark/baselines/gis14_task3b.json
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import arcpy
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(r"E:\BST_lab\GIS_CLI")
LANDSAT = Path(
    r"E:\GIS_documents\小火的竞赛资料\14第十四届全国大学生GIS应用技能大赛"
    r"\14第十四届全国大学生GIS应用技能大赛\A上午\A上午\data\影像数据\Landsat.tif"
)
INPUTS = REPO / "workspace/input/第14届真题数据"
STUDY_AREA = INPUTS / "study_area.shp"
LANDCOVER = INPUTS / "landcover_clip.tif"
BASELINE_DIR = REPO / "benchmark/baselines"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
WORK = Path(tempfile.mkdtemp(prefix="gis14_3b_"))

arcpy.env.overwriteOutput = True
UTM10 = arcpy.SpatialReference(32610)
CELL = 15.0
BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
NAMES = {1: "水域", 2: "林地", 4: "淹没植被区", 5: "耕地", 7: "建筑区", 8: "裸地",
         9: "雪/冰", 10: "云", 11: "牧场"}

study = arcpy.Describe(str(STUDY_AREA))
extent = study.extent
width_m = extent.XMax - extent.XMin
height_m = extent.YMax - extent.YMin
target_cols = int(np.ceil(width_m / CELL))
target_rows = int(np.ceil(height_m / CELL))
print(f"研究区 {width_m/1000:.2f} x {height_m/1000:.2f} km → 目标网格 {target_cols} x {target_rows} @ {CELL:g} m")

# 注意：无空间参考的栅格用 Describe 拿不到 width/height（实测报“方法不存在”），
# 尺寸只能从数组本身拿。
# 多波段读取：用子数据集名 "xxx.tif/Band_2"（RasterToNumPyArray 没有 band_index 参数）
# NoData：无参考 TIFF 用 -1.79e308 之类哨兵表示，必须转成 NaN，否则 Min/Max 归一化会被毁掉
_probe = arcpy.RasterToNumPyArray(f"{LANDSAT}/Band_1", nodata_to_value=np.nan)
src_rows, src_cols = _probe.shape
print(f"Landsat 原始网格 {src_cols} x {src_rows}（无空间参考），有效值占比 "
      f"{np.count_nonzero(~np.isnan(_probe)) / _probe.size:.1%}")
del _probe


def read_band(band: int) -> np.ndarray:
    return arcpy.RasterToNumPyArray(f"{LANDSAT}/Band_{band}", nodata_to_value=np.nan).astype("float64")


def coarse_match(arr: np.ndarray) -> np.ndarray:
    """整幅 ↔ 目标范围的线性对应重采样（最近邻）。"""
    rows = np.clip(np.round((np.arange(target_rows) + 0.5) * (arr.shape[0] / target_rows) - 0.5).astype(int),
                   0, arr.shape[0] - 1)
    cols = np.clip(np.round((np.arange(target_cols) + 0.5) * (arr.shape[1] / target_cols) - 0.5).astype(int),
                   0, arr.shape[1] - 1)
    return arr[np.ix_(rows, cols)]


def write_raster(array: np.ndarray, out: Path, *, dtype: str = "float32") -> str:
    origin = arcpy.Point(extent.XMin, extent.YMin)
    raster = arcpy.NumPyArrayToRaster(array.astype(dtype), origin, CELL, CELL, value_to_nodata=-9999)
    raster.save(str(out))
    arcpy.management.DefineProjection(str(out), UTM10)
    return str(out)


print("\n读取并粗匹配 6 个波段…")
bands: dict[str, np.ndarray] = {}
for index, name in enumerate(BANDS, start=1):
    raw = read_band(index)
    bands[name] = coarse_match(raw)
    print(f"   {name}: 有效范围 [{np.nanmin(raw):.4f}, {np.nanmax(raw):.4f}] → 匹配后 {bands[name].shape}")
    del raw

print("\n按观测区 Min/Max 归一化…")
norm: dict[str, np.ndarray] = {}
ranges: dict[str, list[float]] = {}
for name, arr in bands.items():
    low, high = float(np.nanmin(arr)), float(np.nanmax(arr))
    ranges[name] = [round(low, 2), round(high, 2)]
    span = high - low if high > low else 1.0
    norm[name] = (arr - low) / span

water_index = (norm["green"] - norm["swir2"]) / (norm["green"] + norm["swir2"] + 1e-9)
builtup_index = (norm["swir1"] + norm["swir2"]) / (norm["red"] + norm["nir"] + 1e-9)
cell_area_km2 = (CELL * CELL) / 1e6          # 15 m 网格
LC_CELL_AREA_KM2 = (10.0 * 10.0) / 1e6        # 10 m 土地覆盖网格


def area_filter(mask: np.ndarray, min_area_km2: float) -> tuple[np.ndarray, int, int]:
    """8 邻域连通域 + 按面积过滤（scipy.label，等价 RegionGroup + Con(COUNT)）。"""
    from scipy import ndimage  # type: ignore

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return mask.astype("uint8"), 0, 0
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep = sizes * cell_area_km2 >= min_area_km2
    keep_labels = np.nonzero(keep)[0] + 1
    return np.isin(labels, keep_labels).astype("uint8"), int(count), int(keep.sum())


water_mask, water_total, water_keep = area_filter(np.nan_to_num(water_index, nan=0.0) > 0.4, 0.1)
builtup_mask, builtup_total, builtup_keep = area_filter(np.nan_to_num(builtup_index, nan=0.0) > 3.0, 1.0)
water_km2 = float(water_mask.sum() * cell_area_km2)
builtup_km2 = float(builtup_mask.sum() * cell_area_km2)
print(f"\n水体: 连通块 {water_total} → 保留 {water_keep}，面积 {water_km2:.3f} km²")
print(f"建成区: 连通块 {builtup_total} → 保留 {builtup_keep}，面积 {builtup_km2:.3f} km²")

# BU 分布诊断：赛题阈值 3 在归一化空间里是否可达（决定基线区间怎么写）
_bu = np.nan_to_num(builtup_index, nan=0.0).ravel()
for _q in (50, 90, 99, 99.9):
    print('   BU 分位 P%s: %.3f' % (_q, np.percentile(_bu, _q)))
for _th in (1.0, 1.5, 2.0, 3.0):
    _n = int((_bu > _th).sum())
    print('   BU > %.1f: %d 像元 (%.3f km2)' % (_th, _n, _n * cell_area_km2))
del _bu

# 保存中间栅格供人工核对（水体/建成区/归一化波段）
write_raster(water_mask.astype("float64"), WORK / "water_mask.tif")
write_raster(builtup_mask.astype("float64"), WORK / "builtup_mask.tif")
write_raster(water_index, WORK / "mndwi.tif")
write_raster(builtup_index, WORK / "bu.tif")

# 更新土地覆盖：水→1、建成→7
landcover_arr = arcpy.RasterToNumPyArray(str(LANDCOVER), nodata_to_value=-9999).astype("int32")
lc_rows, lc_cols = landcover_arr.shape


def resample_mask(mask: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    """把 15 m 掩膜最近邻重采样到 10 m 土地覆盖网格（赛题注明分辨率损失可忽略）。"""
    rows = np.clip(np.round((np.arange(target_rows) + 0.5) * (mask.shape[0] / target_rows) - 0.5).astype(int),
                   0, mask.shape[0] - 1)
    cols = np.clip(np.round((np.arange(target_cols) + 0.5) * (mask.shape[1] / target_cols) - 0.5).astype(int),
                   0, mask.shape[1] - 1)
    return mask[np.ix_(rows, cols)]


water_on_lc = resample_mask(water_mask, lc_rows, lc_cols)
builtup_on_lc = resample_mask(builtup_mask, lc_rows, lc_cols)
print('')
print('掩膜已重采样到土地覆盖网格 %dx%d（10 m）' % (lc_rows, lc_cols))
updated = landcover_arr.copy()
updated[(updated != -9999) & (builtup_on_lc == 1)] = 7
updated[(updated != -9999) & (water_on_lc == 1)] = 1
before = Counter(int(v) for v in landcover_arr.ravel() if v > 0)
after = Counter(int(v) for v in updated.ravel() if v > 0)

landcover_rows = []
for code in sorted(set(before) | set(after)):
    landcover_rows.append({
        "code": code,
        "name": NAMES.get(code, "?"),
        "before_km2": round(before.get(code, 0) * LC_CELL_AREA_KM2, 3),
        "after_km2": round(after.get(code, 0) * LC_CELL_AREA_KM2, 3),
    })

baseline = {
    "task": "gis14_am_task3b",
    "params": {"cell_m": CELL, "water_threshold": 0.4, "builtup_threshold": 3.0,
               "water_min_km2": 0.1, "builtup_min_km2": 1.0,
               "match_assumption": "整幅 Landsat ↔ 研究区范围线性对应（最近邻）"},
    "grid": {"cols": target_cols, "rows": target_rows, "cell_m": CELL},
    "band_ranges_raw": ranges,
    "mndwi_range": [round(float(np.nanmin(water_index)), 3), round(float(np.nanmax(water_index)), 3)],
    "bu_range": [round(float(np.nanmin(builtup_index)), 3), round(float(np.nanmax(builtup_index)), 3)],
    "water_km2": round(water_km2, 3),
    "water_components": {"total": water_total, "kept": water_keep},
    "builtup_km2": round(builtup_km2, 3),
    "builtup_components": {"total": builtup_total, "kept": builtup_keep},
    "landcover_update": landcover_rows,
    "total_km2_before": round(sum(before.values()) * LC_CELL_AREA_KM2, 2),
    "total_km2_after": round(sum(after.values()) * LC_CELL_AREA_KM2, 2),
}
(BASELINE_DIR / "gis14_task3b.json").write_text(
    json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("\n基线已写入:", BASELINE_DIR / "gis14_task3b.json")
print("分类面积（前→后，km²）:")
for row in landcover_rows:
    mark = "  ← 更新" if abs(row["after_km2"] - row["before_km2"]) > 0.001 else ""
    print(f"   {row['code']:>2} {row['name']:<6} {row['before_km2']:>9.3f} → {row['after_km2']:>9.3f}{mark}")
print(f"合计 {baseline['total_km2_before']} → {baseline['total_km2_after']} km²")
print("中间栅格（人工核对）:", WORK)
