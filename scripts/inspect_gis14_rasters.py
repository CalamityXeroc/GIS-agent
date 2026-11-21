# -*- coding: utf-8 -*-
"""只读描述：第 14 届上午题 DEM / 土地覆盖 / Landsat 影像元数据。"""
import io
import sys
from pathlib import Path

import arcpy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA = Path(
    r"E:\GIS_documents\小火的竞赛资料\14第十四届全国大学生GIS应用技能大赛"
    r"\14第十四届全国大学生GIS应用技能大赛\A上午\A上午\data"
)
targets = [
    DATA / "dem" / "demA.tif",
    DATA / "dem" / "demB.tif",
    DATA / "dem" / "demC.tif",
    DATA / "dem" / "demD.tif",
    DATA / "土地覆盖" / "landcover.tif",
    DATA / "影像数据" / "Landsat.tif",
]
for path in targets:
    d = arcpy.Describe(str(path))
    sr = getattr(d, "spatialReference", None)
    ext = d.extent
    bands = getattr(d, "bandCount", None)
    print(f"--- {path.name}")
    print(f"    raster={getattr(d, 'isRaster', None)} bands={bands} format={getattr(d, 'format', '')}")
    print(f"    cell={d.meanCellWidth} x {d.meanCellHeight}")
    print(f"    CRS={getattr(sr, 'name', '') if sr else '<无>'} wkid={getattr(sr, 'factoryCode', None) if sr else None}")
    print(f"    extent=({ext.XMin:.1f}, {ext.YMin:.1f}) ~ ({ext.XMax:.1f}, {ext.YMax:.1f})")
    try:
        rng = arcpy.management.GetRasterProperties(str(path), "MINIMUM").getOutput(0)
        print(f"    min={rng}", end="")
        print(f" max={arcpy.management.GetRasterProperties(str(path), 'MAXIMUM').getOutput(0)}")
    except Exception as exc:
        print(f"    统计读取失败: {exc}")
