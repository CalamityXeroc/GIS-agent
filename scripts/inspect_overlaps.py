# -*- coding: utf-8 -*-
"""Detail the residual overlapping pairs in county_1."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import arcpy

p = r"E:\BST_lab\GIS_CLI\benchmark\runs\20260909_144614\am_task1c_topology\workspace\output\result_data.gdb\county_1"
oids = [(o, s, n, c) for o, s, n, c in arcpy.da.SearchCursor(p, ["OID@", "SHAPE@", "name", "code"])]
print("total:", len(oids))
pairs = []
for i in range(len(oids)):
    for j in range(i + 1, len(oids)):
        g1, g2 = oids[i][1], oids[j][1]
        if g1.equals(g2) or g1.overlaps(g2) or g1.contains(g2) or g1.within(g2):
            if g1.equals(g2):
                rel = "equal"
            elif g1.contains(g2):
                rel = "contains"
            elif g1.within(g2):
                rel = "within"
            else:
                rel = "overlaps"
            inter = g1.intersect(g2, 4).area
            pairs.append((oids[i][0], oids[i][2], oids[i][3], oids[j][0], oids[j][2], oids[j][3], rel, round(inter, 1)))
print("残留重叠对:")
for x in pairs:
    print("  OID%s(%s/%s) vs OID%s(%s/%s) rel=%s inter_area=%s" % x)
