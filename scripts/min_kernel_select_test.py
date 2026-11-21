# -*- coding: utf-8 -*-
"""验证安全替代：arcpy.analysis.Select / FeatureClassToFeatureClass 在持久内核里是否稳定。"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")
from pathlib import Path
from gis_cli.runtime.code_runner import CodeRunner

WS = Path(r"C:\Users\cheng\AppData\Local\Temp\min_ws")
r = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "k2")

for label, code in [
    ("arcpy.analysis.Select", "\n".join([
        "import arcpy, os",
        "gdb = os.path.join(r'" + str(WS) + "', 't.gdb')",
        "fc = os.path.join(gdb, 'pts')",
        "out = os.path.join(gdb, 'sel')",
        "if arcpy.Exists(out):",
        "    arcpy.management.Delete(out)",
        "arcpy.analysis.Select(fc, out, 'OBJECTID = 1')",
        "set_result({'n': int(arcpy.management.GetCount(out)[0])})",
    ])),
    ("FeatureClassToFeatureClass(where)", "\n".join([
        "import arcpy, os",
        "gdb = os.path.join(r'" + str(WS) + "', 't.gdb')",
        "fc = os.path.join(gdb, 'pts')",
        "out = os.path.join(gdb, 'sel2')",
        "if arcpy.Exists(out):",
        "    arcpy.management.Delete(out)",
        "arcpy.conversion.FeatureClassToFeatureClass(fc, gdb, 'sel2', 'NAME = \\'a\\'')",
        "set_result({'n': int(arcpy.management.GetCount(out)[0])})",
    ])),
]:
    t0 = time.time()
    res = r.run(code, timeout=120, workspace=str(WS))
    print(f"[{'OK ' if res.ok else 'FAIL'}] {label:34s} {time.time()-t0:6.1f}s | {res.result} | {str((res.error or {}).get('message',''))[:90]}")
r.shutdown()