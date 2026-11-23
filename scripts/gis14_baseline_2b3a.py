# -*- coding: utf-8 -*-
"""任务二b/三a 的离线基线（快速路径）：先裁小再投影，避免全图投影。

产出（写入临时目录）：
- DEM 裁切到研究区后的 min/max/mean（UTM 10N，30 m）
- 土地覆盖裁切后的类别像元数与面积、云像元数（UTM 10N，10 m）
用于校准断言区间并独立核验 agent 数值。
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import arcpy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA = Path(
    r"E:\GIS_documents\小火的竞赛资料\14第十四届全国大学生GIS应用技能大赛"
    r"\14第十四届全国大学生GIS应用技能大赛\A上午\A上午\data"
)
OUT = Path(tempfile.gettempdir()) / "gis14_baseline2b"
OUT.mkdir(parents=True, exist_ok=True)
GDB = OUT / "base.gdb"
arcpy.env.overwriteOutput = True
if not arcpy.Exists(str(GDB)):
    arcpy.management.CreateFileGDB(str(OUT), "base.gdb")

UTM10 = arcpy.SpatialReference(32610)
WGS84 = arcpy.SpatialReference(4326)
# 任务一离线基线算出的研究区（UTM 10N）
RECT = (325820.6, 4533393.7, 359476.3, 4596654.9)

pts = arcpy.Array([arcpy.Point(RECT[0], RECT[1]), arcpy.Point(RECT[2], RECT[1]),
                   arcpy.Point(RECT[2], RECT[3]), arcpy.Point(RECT[0], RECT[3])])
study = GDB / "study_area"
arcpy.management.CreateFeatureclass(str(GDB), "study_area", "POLYGON", spatial_reference=UTM10)
with arcpy.da.InsertCursor(str(study), ["SHAPE@"]) as cur:
    cur.insertRow([arcpy.Polygon(pts, UTM10)])

# 研究区经纬度外包框（用于在原生坐标系下先裁小）
study_wgs = arcpy.management.Project(str(study), str(GDB / "study_wgs"), WGS84)
e = arcpy.Describe(str(study_wgs)).extent
clip_rect = f"{e.XMin} {e.YMin} {e.XMax} {e.YMax}"

# ---------- DEM ----------
mosaic_wgs = GDB / "dem_mosaic_wgs"
if not arcpy.Exists(str(mosaic_wgs)):
    arcpy.management.MosaicToNewRaster(
        [str(DATA / "dem" / f) for f in ("demA.tif", "demB.tif", "demC.tif", "demD.tif")],
        str(GDB), "dem_mosaic_wgs", WGS84, "32_BIT_FLOAT", 0.0003, 1, "FIRST",
    )
dem_utm = GDB / "dem_utm"
if not arcpy.Exists(str(dem_utm)):
    arcpy.management.ProjectRaster(str(mosaic_wgs), str(dem_utm), UTM10, "BILINEAR", 30)
dem_clip = GDB / "dem_clip"
arcpy.management.Clip(in_raster=str(dem_utm), out_raster=str(dem_clip),
                       rectangle="", in_template_dataset=str(study),
                       nodata_value="", clipping_geometry="ClippingGeometry")
dem = {}
for stat in ("MINIMUM", "MAXIMUM", "MEAN"):
    dem[stat.lower()] = float(arcpy.management.GetRasterProperties(str(dem_clip), stat).getOutput(0))
print("DEM 裁切后统计:", dem)

# ---------- 土地覆盖 ----------
lc_wgs_clip = GDB / "lc_wgs_clip"
arcpy.management.Clip(in_raster=str(DATA / "土地覆盖" / "landcover.tif"),
                       out_raster=str(lc_wgs_clip), rectangle=clip_rect,
                       in_template_dataset="", nodata_value="", clipping_geometry="NONE")
lc_utm = GDB / "landcover_clip"
arcpy.management.ProjectRaster(str(lc_wgs_clip), str(lc_utm), UTM10, "NEAREST", 10)
arr = arcpy.RasterToNumPyArray(str(lc_utm))
counts = Counter(int(v) for v in arr.ravel())
names = {1: "水域", 2: "林地", 4: "淹没植被", 5: "耕地", 7: "建筑区", 8: "裸地", 9: "雪/冰", 10: "云", 11: "牧场"}
total = sum(counts.values())
print(f"\n土地覆盖裁切：像元总数 {total}，10 m → {total * 100 / 1e6:.2f} km²")
rows = []
for code in sorted(counts):
    km2 = counts[code] * 100 / 1e6
    rows.append({"code": code, "name": names.get(code, "?"), "pixels": counts[code], "km2": round(km2, 3)})
    print(f"  {code:>2} {names.get(code, '?'):<6} {counts[code]:>9} 像元 {km2:>8.3f} km²")

(OUT / "baseline_2b3a.json").write_text(
    json.dumps({"dem": dem, "landcover": rows, "pixels": total}, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("\n已写入", OUT / "baseline_2b3a.json")
