# -*- coding: utf-8 -*-
"""Task3 关键能力探针: 空间连接SUM聚合 + Getis-Ord Gi* 热点分析。"""
import sys, io, os, glob, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

arcpy.env.overwriteOutput = True

runs = sorted(glob.glob(r"E:\BST_lab\GIS_CLI\benchmark\runs\2026*"))
county_pop = os.path.join(runs[-1], "am_task2b_neighbor_highlight", "workspace", "output", "result_data.gdb", "county_pop")
assert arcpy.Exists(county_pop), county_pop
base = glob.glob(r"E:\BST_lab\GIS_CLI\workspace\input\**\B上午\数据", recursive=True)[0]
city_src = os.path.join(base, "city.shp")
print("county_pop:", arcpy.Describe(county_pop).spatialReference.factoryCode)
print("city:", arcpy.Describe(city_src).spatialReference.name)

tmp_dir = tempfile.mkdtemp(prefix="t3probe_")
tmp_gdb = os.path.join(tmp_dir, "probe.gdb")
arcpy.management.CreateFileGDB(tmp_dir, "probe.gdb")

city = os.path.join(tmp_gdb, "city")
sr = arcpy.SpatialReference(4508)
arcpy.management.Project(city_src, city, sr)
print("city projected ->", arcpy.Describe(city).spatialReference.factoryCode)

years = [f"Y{y}" for y in range(2011, 2021)]
fm = ";".join(f"{f} SUM" for f in years)
city_pop = os.path.join(tmp_gdb, "city_pop")
try:
    arcpy.analysis.SpatialJoin(
        target_features=city, join_features=county_pop, out_feature_class=city_pop,
        join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL", match_option="INTERSECT",
        field_mapping=fm,
    )
    flds = [f.name for f in arcpy.ListFields(city_pop)]
    got = [f for f in years if f in flds]
    print("SUM shorthand OK, year fields present:", len(got), "of", len(years))
    with arcpy.da.SearchCursor(city_pop, ["name_1", "Y2020", "Join_Count"]) as cur:
        vals = [(r[0], r[1], r[2]) for r in cur][:4]
    print("sample (city, sum2020, n_counties):", vals)
    print("all fields:", [f.name for f in arcpy.ListFields(city_pop)][:18])
except Exception as e:
    print("SUM shorthand FAILED:", str(e)[:300])

try:
    hs = os.path.join(tmp_gdb, "city_hot")
    arcpy.stats.HotSpots(
        Input_Feature_Class=city_pop, Input_Field="Y2011",
        Output_Feature_Class=hs, Conceptualization_of_Spatial_Relationships="CONTIGUITY_EDGES_CORNERS",
        Standardization="ROW",
    )
    flds = [f.name for f in arcpy.ListFields(hs)]
    bins = {}
    with arcpy.da.SearchCursor(hs, ["Gi_Bin"]) as cur:
        for r in cur:
            bins[r[0]] = bins.get(r[0], 0) + 1
    print("HotSpots OK; Gi fields:", [f for f in flds if f.startswith("Gi")], "bins:", bins)
except Exception as e:
    print("HotSpots FAILED:", str(e)[:300])
