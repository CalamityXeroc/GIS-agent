# -*- coding: utf-8 -*-
"""把中间成果导出为独立输入文件，让基准任务可独立重跑（不依赖临时运行目录）。

导出：
- landcover_clip.tif（任务三a 的输入：已裁切到研究区的土地覆盖）
- dem_clip.tif（DEM 裁切结果，留作参考/后续任务用）
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import arcpy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(r"E:\BST_lab\GIS_CLI")
BASE = REPO / "workspace/成果展示/第14届真题实测"
DST = REPO / "workspace/input/第14届真题数据"
DST.mkdir(parents=True, exist_ok=True)
arcpy.env.overwriteOutput = True

pairs = [
    (BASE / "任务二b_DEM拼接与土地覆盖/result_data.gdb/landcover_clip", DST / "landcover_clip.tif"),
    (BASE / "任务二b_DEM拼接与土地覆盖/result_data.gdb/dem_clip", DST / "dem_clip.tif"),
]
for src, dst in pairs:
    if not arcpy.Exists(str(src)):
        print("缺失:", src)
        continue
    arcpy.management.CopyRaster(str(src), str(dst))
    desc = arcpy.Describe(str(dst))
    print(f"已导出 {dst.name}: {desc.width}x{desc.height} 像元 {desc.meanCellWidth:.1f} m "
          f"CRS={desc.spatialReference.factoryCode}")
