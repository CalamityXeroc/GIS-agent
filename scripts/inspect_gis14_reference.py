# -*- coding: utf-8 -*-
"""Inspect the 14th-competition reference result GDBs (read-only)."""
import sys
import io
import os
import arcpy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = r"E:\GIS_documents\小火的竞赛资料\14第十四届全国大学生GIS应用技能大赛\14第十四届全国大学生GIS应用技能大赛\A上午\A上午\Result\14am"

for gdb in ("resule_data.gdb", "temp_data.gdb", "14am.gdb"):
    path = os.path.join(BASE, gdb)
    print("=" * 74)
    print(gdb, "exists=", arcpy.Exists(path))
    try:
        arcpy.env.workspace = path
        datasets = arcpy.ListDatasets() or []
        print(" datasets:", datasets or "-")
        print(" fcs     :", arcpy.ListFeatureClasses() or "-")
        print(" rasters :", arcpy.ListRasters() or "-")
        print(" tables  :", arcpy.ListTables() or "-")
        for ds in datasets:
            try:
                fcs = arcpy.ListFeatureClasses(feature_dataset=ds)
            except Exception:
                fcs = []
            print(f"   [{ds}] fcs={fcs}")
    except Exception as exc:
        print(" ERROR:", exc)
