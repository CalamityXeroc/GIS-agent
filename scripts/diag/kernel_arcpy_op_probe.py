"""DAIR-246 复测：在引擎持久内核里做一次**真实 arcpy 操作**（建 gdb + 要素类 + 插入点）。

证明的不只是 `import arcpy` 能返回，而是许可/初始化链路真的通。

用法:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/diag/kernel_arcpy_op_probe.py <workspace_dir>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from gis_cli.runtime.kernel import ArcpyKernel

CODE_TMPL = r"""
import os, arcpy
ws = r"{ws}"
os.makedirs(ws, exist_ok=True)
gdb = os.path.join(ws, "probe.gdb")
if not arcpy.Exists(gdb):
    arcpy.management.CreateFileGDB(ws, "probe.gdb")
fc = os.path.join(gdb, "probe_pt")
if arcpy.Exists(fc):
    arcpy.management.Delete(fc)
arcpy.management.CreateFeatureclass(gdb, "probe_pt", "POINT", spatial_reference=arcpy.SpatialReference(4326))
with arcpy.da.InsertCursor(fc, ["SHAPE@XY"]) as cur:
    cur.insertRow([(116.39, 39.91)])
print("exists=", arcpy.Exists(fc))
print("count=", int(arcpy.management.GetCount(fc)[0]))
print("sr=", arcpy.Describe(fc).spatialReference.name)
print("version=", arcpy.GetInstallInfo().get("Version"))
"""


def main() -> int:
    ws = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark/reports/dair-246_arcpy_hang/arcpy_ws").resolve()
    ws.mkdir(parents=True, exist_ok=True)
    k = ArcpyKernel()
    t0 = time.time()
    k.start()
    start_s = time.time() - t0
    print(f"[probe] kernel ready in {start_s:.2f}s", flush=True)

    t1 = time.time()
    res = k.execute(CODE_TMPL.format(ws=str(ws).replace("\\", "\\\\")), timeout=180)
    exec_s = time.time() - t1
    k.shutdown()

    print(f"[probe] execute ok={res.ok} in {exec_s:.2f}s", flush=True)
    print(f"[probe] stdout:\n{(res.stdout or '').strip()}", flush=True)
    if res.error:
        print(f"[probe] error: {res.error}", flush=True)
    print(f"[probe] SUMMARY kernel_start_s={start_s:.2f} execute_s={exec_s:.2f} ok={res.ok}", flush=True)
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
