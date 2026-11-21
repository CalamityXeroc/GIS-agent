# -*- coding: utf-8 -*-
"""第 14 届上午题 任务二b/三 的参考基线：DEM 拼接裁切、土地覆盖裁切与分类面积。

产出：
- 裁切后 DEM 的最小/最大/均值高程（用于断言区间）
- 研究区内土地覆盖各类别像元数与面积（用于断言区间），以及 Clouds 像元数（修补前基线）
写到临时目录，不动原始数据。
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
OUT = Path(tempfile.gettempdir()) / "gis14_baseline"
OUT.mkdir(parents=True, exist_ok=True)

UTM10 = arcpy.SpatialReference(32610)
# 任务一基线算出的研究区矩形（UTM 10N）
RECT = (325820.6, 4533393.7, 359476.3, 4596654.9)

arcpy.env.overwriteOutput = True
arcpy.env.workspace = str(OUT)
tmp = OUT / "work.gdb"
if not arcpy.Exists(str(tmp)):
    arcpy.management.CreateFileGDB(str(OUT), "work.gdb")

pts = arcpy.Array([arcpy.Point(RECT[0], RECT[1]), arcpy.Point(RECT[2], RECT[1]),
                   arcpy.Point(RECT[2], RECT[3]), arcpy.Point(RECT[0], RECT[3])])
study = tmp / "study_area"
arcpy.management.CreateFeatureclass(str(tmp), "study_area", "POLYGON", spatial_reference=UTM10)
with arcpy.da.InsertCursor(str(study), ["SHAPE@"]) as cur:
    cur.insertRow([arcpy.Polygon(pts, UTM10)])
print("研究区已建：", study)

# ---------- DEM 拼接 + 裁切 ----------
dem_out = tmp / "dem_clip"
arcpy.management.MosaicToNewRaster(
    [str(DATA / "dem" / f) for f in ("demA.tif", "demB.tif", "demC.tif", "demD.tif")],
    str(tmp), "dem_mosaic", UTM10, "32_BIT_FLOAT", 30, 1, "FIRST",
)
print("DEM 拼接完成（30 m，UTM10）")
arcpy.management.Clip(str(tmp / "dem_mosaic"), str(dem_out), str(study))
info = {}
for stat in ("MINIMUM", "MAXIMUM", "MEAN"):
    info[stat.lower()] = float(arcpy.management.GetRasterProperties(str(dem_out), stat).getOutput(0))
print("裁切后 DEM：", info)
d = arcpy.Describe(str(dem_out))
print(f"  像元 {d.meanCellWidth} m，范围 ({d.extent.XMin:.0f}, {d.extent.YMin:.0f}) ~ ({d.extent.XMax:.0f}, {d.extent.YMax:.0f})")

# ---------- 土地覆盖裁切 ----------
lc_out = tmp / "landcover_clip"
arcpy.management.ProjectRaster(str(DATA / "土地覆盖" / "landcover.tif"), str(tmp / "lc_utm"), UTM10,
                               "NEAREST", 10)
arcpy.management.Clip(str(tmp / "lc_utm"), str(lc_out), str(study), "", "", "NONE")
arr = arcpy.RasterToNumPyArray(str(lc_out))
counts = Counter(int(v) for v in arr.ravel())
total = sum(counts.values())
print(f"\n研究区土地覆盖像元总数 {total}，像元 10 m -> 面积 {total * 100 / 1e6:.2f} km²")
names = {1: "水域", 2: "林地", 4: "淹没植被", 5: "耕地", 7: "建筑区", 8: "裸地", 9: "雪/冰", 10: "云", 11: "牧场"}
rows = []
for code in sorted(counts):
    km2 = counts[code] * 100 / 1e6
    rows.append({"code": code, "name": names.get(code, "?"), "pixels": counts[code], "km2": round(km2, 3)})
    print(f"  {code:>2} {names.get(code, '?'):<6} {counts[code]:>9} 像元  {km2:>8.3f} km²")
(OUT / "baseline_task2b.json").write_text(
    json.dumps({"dem": info, "landcover": rows, "pixels": total}, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("\n基线写入", OUT / "baseline_task2b.json")
