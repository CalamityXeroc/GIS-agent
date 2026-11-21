# -*- coding: utf-8 -*-
import sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

base = r"E:\BST_lab\GIS_CLI\workspace\input\13第十三届全国大学生GIS应用技能大赛（兰州交通大学）\数据\B上午\B上午\数据"
if not os.path.isdir(base):
    cands = glob.glob(r"E:\BST_lab\GIS_CLI\workspace\input\**\B上午\数据", recursive=True)
    base = cands[0] if cands else base
print("base:", base)
for name in ("city.shp", "county.shp"):
    p = os.path.join(base, name)
    if arcpy.Exists(p):
        flds = [(f.name, f.type) for f in arcpy.ListFields(p) if f.type not in ("Geometry", "OID")]
        print(f"[{name}] count={int(arcpy.management.GetCount(p)[0])} fields={flds[:14]}")

csv_path = os.path.join(base, "population_census.csv")
print("\n[census csv] exists:", os.path.exists(csv_path))
if os.path.exists(csv_path):
    with open(csv_path, "rb") as fh:
        raw = fh.read(600)
    for enc in ("utf-8-sig", "gbk"):
        try:
            head = raw.decode(enc).splitlines()[:3]
            print(f"  encoding={enc}:")
            for line in head:
                print("   ", line)
            break
        except Exception:
            continue

# 许可检查
print("\n[license]")
for prod in ("ArcInfo", "Advanced"):
    try:
        print(" has", prod, "=", arcpy.ProductInfo())
        break
    except Exception as e:
        print(" license check error", e)
for tool in ("HotSpots", "ClusterAndOutlierAnalysis"):
    try:
        print(" tool exists:", tool, hasattr(arcpy.stats, tool))
    except Exception as e:
        print(" tool check error", e)
