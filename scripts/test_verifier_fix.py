# -*- coding: utf-8 -*-
"""Test the fixed _VERIFY_CODE against the am_task1a workspace."""
import sys, io, json
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

from gis_cli.engine.verifier import Verifier
from gis_cli.runtime.code_runner import CodeRunner

ws = Path(r"E:\BST_lab\GIS_CLI\benchmark\runs\20260909_120306\am_task1a_projection\workspace")
runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel")
v = Verifier(workspace=ws, code_runner=runner)

print("=== 工作区 output 实际内容 ===")
for p in sorted((ws / "output").rglob("*")):
    if p.is_dir() and p.name.endswith(".gdb"):
        print(" [gdb]", p.relative_to(ws))
    elif p.is_file() and p.suffix.lower() in {".shp", ".tif", ".csv"}:
        print(" [file]", p.relative_to(ws))

print()
print("=== structural_check（修复后的内核路径）===")
artifacts = v._discover_outputs()
print("auto-discovered artifacts:", [Path(a).name for a in artifacts])
records = v.structural_check(artifacts)
for r in records:
    print(f"  exists={r.get('exists')} n={r.get('feature_count')} wkid={r.get('wkid')} ext={r.get('extent')} <- {Path(r['path']).name}")
runner.shutdown()
