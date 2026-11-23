# -*- coding: utf-8 -*-
import sys, io, os, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

ws = sorted(glob.glob(r"C:\Users\cheng\AppData\Local\Temp\bench3a_*"))[-1]
gdb = os.path.join(ws, "output", "result_data.gdb")
county_pop = os.path.join(gdb, "county_pop")
city = os.path.join(gdb, "city_4508")

def bbox(path, where=None, sub=None):
    with arcpy.da.SearchCursor(path, ["SHAPE@"], None if not where else None) as cur:
        xs = []
        for (g,) in cur:
            if g is None:
                continue
            e = g.extent
            xs.append((e.XMin, e.YMin, e.XMax, e.YMax))
    if not xs:
        return None
    return (min(x[0] for x in xs), min(x[1] for x in xs), max(x[2] for x in xs), max(x[3] for x in xs))

print("county_pop extent:", [round(v) for v in arcpy.Describe(county_pop).extent.JSON.values()] if False else [round(v) for v in (arcpy.Describe(county_pop).extent.XMin, arcpy.Describe(county_pop).extent.YMin, arcpy.Describe(county_pop).extent.XMax, arcpy.Describe(county_pop).extent.YMax)])
print("city extent:", [round(v) for v in (arcpy.Describe(city).extent.XMin, arcpy.Describe(city).extent.YMin, arcpy.Describe(city).extent.XMax, arcpy.Describe(city).extent.YMax)])

# 娄底市 4313xx 的区县 bbox
def county_bbox_by_prefix(prefix):
    xs = []
    with arcpy.da.SearchCursor(county_pop, ["SHAPE@", "code"]) as cur:
        for g, code in cur:
            if str(code).startswith(prefix) and g is not None:
                e = g.extent
                xs.append((e.XMin, e.YMin, e.XMax, e.YMax))
    if not xs:
        return None
    return (round(min(x[0] for x in xs)), round(min(x[1] for x in xs)), round(max(x[2] for x in xs)), round(max(x[3] for x in xs)))

def city_bbox_by_name(name):
    with arcpy.da.SearchCursor(city, ["SHAPE@", "name"]) as cur:
        for g, n in cur:
            if n == name and g is not None:
                e = g.extent
                return (round(e.XMin), round(e.YMin), round(e.XMax), round(e.YMax))
    return None

for pref, cname in (("4313", "娄底市"), ("4309", "益阳市"), ("4213", "随州市"), ("4106", "鹤壁市"), ("4101", "郑州市")):
    print(f"{cname}: 区县bbox={county_bbox_by_prefix(pref)} 市面bbox={city_bbox_by_name(cname)}")
