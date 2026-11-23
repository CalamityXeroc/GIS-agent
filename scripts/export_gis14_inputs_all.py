# -*- coding: utf-8 -*-
"""导出第 14 届任务的显式输入，让每个任务都能独立重跑（不依赖临时运行目录）。

产出（workspace/input/第14届真题数据/）：
- animal_points.shp     清洗并投影后的野生动物点位（UTM 10N，含 YEAR/COUNT/NAME/LON/LAT）
- study_area.shp        研究区矩形（UTM 10N）
- landcover_clip.tif    已裁切到研究区的土地覆盖（10 m，UTM 10N）
- dem_clip.tif          拼接+裁切后的 DEM（30 m，UTM 10N）
- README.md             输入清单说明
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import arcpy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(r"E:\BST_lab\GIS_CLI")
CURATED = REPO / "workspace/成果展示/第14届真题实测"
DST = REPO / "workspace/input/第14届真题数据"
DST.mkdir(parents=True, exist_ok=True)
arcpy.env.overwriteOutput = True

SOURCES = {
    "animal_points": CURATED / "任务一_清洗投影与研究区/result_data.gdb/animal_points",
    "study_area": CURATED / "任务一_清洗投影与研究区/result_data.gdb/study_area",
    "landcover_clip": CURATED / "任务二b_DEM拼接与土地覆盖/result_data.gdb/landcover_clip",
    "dem_clip": CURATED / "任务二b_DEM拼接与土地覆盖/result_data.gdb/dem_clip",
}

for name, source in SOURCES.items():
    target = DST / f"{name}.shp" if name != "animal_points" and name != "study_area" else DST / f"{name}.shp"
    if name not in ("animal_points", "study_area"):
        target = DST / f"{name}.tif"
    if not arcpy.Exists(str(source)):
        print(f"缺失源: {source}")
        continue
    if target.exists():
        print(f"已存在，跳过: {target.name}")
    elif target.suffix == ".tif":
        arcpy.management.CopyRaster(str(source), str(target))
    else:
        arcpy.management.CopyFeatures(str(source), str(target))
    desc = arcpy.Describe(str(target))
    extra = ""
    if target.suffix == ".shp":
        count = arcpy.management.GetCount(str(target))[0]
        fields = [f.name for f in arcpy.ListFields(str(target)) if not f.name.startswith("Shape")]
        extra = f" 要素数={count} 字段={fields}"
    else:
        extra = f" {desc.width}x{desc.height} 像元={desc.meanCellWidth:.1f}m"
    print(f"OK  {target.name}: CRS={desc.spatialReference.factoryCode}{extra}")

readme = DST / "README.md"
readme.write_text(
    "# 第14届（2025）上午题 · 基准任务显式输入\n\n"
    "这些文件从任务一到任务二b 的成果中导出，目的是让**每个任务都能独立重跑**\n"
    "（不依赖上一次运行的临时工作区）。\n\n"
    "| 文件 | 内容 | 坐标系/规格 |\n"
    "|---|---|---|\n"
    "| `animal_points.shp` | 清洗、投影后的野生动物观测点（YEAR/COUNT/NAME/LON/LAT） | UTM 10N（EPSG:32610）|\n"
    "| `study_area.shp` | 研究区矩形（观测点包络外扩 2 km） | UTM 10N |\n"
    "| `landcover_clip.tif` | 已裁切到研究区的土地覆盖（编码 1~11） | UTM 10N，10 m |\n"
    "| `dem_clip.tif` | 4 幅 FABDEM 拼接并裁切后的高程 | UTM 10N，30 m |\n\n"
    "Landsat.tif（6 波段、无空间参考、557 MB）不复制，任务三b 直接引用外部原始路径。\n",
    encoding="utf-8",
)
print("已写 README.md")
