# -*- coding: utf-8 -*-
"""能力补齐前置探测：ArcPy 扩展许可 + 工具可用性 + PM（下午卷）数据画像。

输出一份清单，用于决定哪些配方可直接用 ArcPy 工具、哪些需要降级实现。
"""
import sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

arcpy.env.overwriteOutput = True

print("=" * 70)
print("一、许可与产品")
print("=" * 70)
print("ProductInfo:", arcpy.ProductInfo())
for ext in ("Spatial", "3D", "Network", "DataInteroperability", "GeoStats", "JTX"):
    try:
        code = arcpy.CheckExtension(ext)
        print(f"  CheckExtension({ext:22s}) = {code}")
    except Exception as exc:
        print(f"  CheckExtension({ext:22s}) 异常: {str(exc)[:60]}")

print()
print("=" * 70)
print("二、关键工具可用性（存在即可用于探测）")
print("=" * 70)
TOOLS = [
    ("overlay   Clip",                 lambda: arcpy.analysis.Clip),
    ("overlay   Intersect",            lambda: arcpy.analysis.Intersect),
    ("overlay   Erase",                lambda: arcpy.analysis.Erase),
    ("overlay   Union",                lambda: arcpy.analysis.Union),
    ("geometry  FeatureToPoint",       lambda: arcpy.management.FeatureToPoint),
    ("geometry  MultipartToSinglepart", lambda: arcpy.management.MultipartToSinglepart),
    ("geometry  MinBoundingGeometry",  lambda: arcpy.management.MinimumBoundingGeometry),
    ("geometry  Dissolve",             lambda: arcpy.management.Dissolve),
    ("geometry  SimplifyPolygon(CT)",  lambda: arcpy.cartography.SimplifyPolygon),
    ("geometry  Generalize(edit)",     lambda: arcpy.edit.Generalize),
    ("select    Select",               lambda: arcpy.analysis.Select),
    ("table     Frequency",            lambda: arcpy.analysis.Frequency),
    ("table     Statistics",           lambda: arcpy.analysis.Statistics),
    ("raster    PolygonToRaster",      lambda: arcpy.conversion.PolygonToRaster),
    ("raster    RasterToPolygon",      lambda: arcpy.conversion.RasterToPolygon),
    ("raster    ZonalStatisticsAsTable", lambda: arcpy.sa.ZonalStatisticsAsTable),
    ("raster    ZonalStatistics",      lambda: arcpy.sa.ZonalStatistics),
    ("raster    ExtractByMask",        lambda: arcpy.sa.ExtractByMask),
    ("raster    ExtractValuesToPoints", lambda: arcpy.sa.ExtractValuesToPoints),
    ("raster    Con / IsNull / SetNull", lambda: (arcpy.sa.Con, arcpy.sa.IsNull, arcpy.sa.SetNull)),
    ("raster    Slope/Aspect/Hillshade", lambda: (arcpy.sa.Slope, arcpy.sa.Aspect, arcpy.sa.Hillshade)),
    ("raster    IDW / Kriging / Spline", lambda: (arcpy.sa.Idw, arcpy.sa.Kriging, arcpy.sa.Spline)),
    ("raster    Reclassify",           lambda: arcpy.sa.Reclassify),
    ("raster    RasterCalculator(sa)", lambda: arcpy.sa.RasterCalculator),
    ("raster    ia.RasterCalculator",  lambda: arcpy.ia.RasterCalculator),
    ("analysis  GenerateNearTable",    lambda: arcpy.analysis.GenerateNearTable),
    ("analysis  Near",                 lambda: arcpy.analysis.Near),
    ("analysis  PointDistance",        lambda: arcpy.analysis.PointDistance),
]
for label, getter in TOOLS:
    try:
        getter()
        print(f"  [OK]   {label}")
    except AttributeError as exc:
        print(f"  [MISS] {label}  ({str(exc)[:50]})")
    except Exception as exc:
        print(f"  [ERR]  {label}  ({str(exc)[:50]})")

print()
print("=" * 70)
print("三、PM（下午卷）数据画像")
print("=" * 70)
DATA = glob.glob(r"E:\BST_lab\GIS_CLI\workspace\input\**\B下午\B下午\数据", recursive=True)
DATA = DATA[0] if DATA else ""
print("数据目录:", DATA)
if DATA:
    for name in ("district.shp", "community.shp", "green_land.shp", "buildings.shp",
                 "hospital.shp", "residental_areas.shp", "road.shp"):
        p = os.path.join(DATA, name)
        if not arcpy.Exists(p):
            print(f"\n[{name}] 不存在")
            continue
        d = arcpy.Describe(p)
        flds = [(f.name, f.type) for f in arcpy.ListFields(p) if f.type not in ("Geometry", "OID")]
        cnt = int(arcpy.management.GetCount(p)[0])
        print(f"\n[{name}] {cnt} 要素 | CRS={d.spatialReference.name} (wkid={d.spatialReference.factoryCode})")
        print("  字段:", ", ".join(f"{n}:{t[:3]}" for n, t in flds[:18]))
        # 关键字段样本
        shown = [n for n, t in flds if t in ("String", "Double", "Long", "SmallInteger", "Single")][:5]
        if shown:
            with arcpy.da.SearchCursor(p, shown) as cur:
                for i, row in enumerate(cur):
                    if i >= 2:
                        break
                    print("   样本:", dict(zip(shown, [str(v)[:28] for v in row])))

    tif = os.path.join(DATA, "pop_density.tif")
    if arcpy.Exists(tif):
        r = arcpy.Raster(tif)
        d = arcpy.Describe(tif)
        print(f"\n[pop_density.tif] 尺寸={r.width}x{r.height} 波段={r.bandCount} 像元={r.meanCellWidth:.3f}x{r.meanCellHeight:.3f}"
              f" | dtype={d.pixelType} | CRS={d.spatialReference.name} (wkid={d.spatialReference.factoryCode})")
        print(f"  范围: {r.minimum:.4f} ~ {r.maximum:.4f} | NoData={r.noDataValue} | 像元数={r.width * r.height}")
    else:
        print("\n[pop_density.tif] 不存在")