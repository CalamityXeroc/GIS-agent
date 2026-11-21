# -*- coding: utf-8 -*-
"""Offline test: map generation from scratch (no existing .aprx)."""
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

from gis_cli.arcpy_bridge import build_graduated_colors_code
from gis_cli.runtime.code_runner import CodeRunner

ws = Path(sys.argv[1])
in_fc = sys.argv[2]
out_jpg = sys.argv[3]
field = sys.argv[4]

runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel_maptest")
code = build_graduated_colors_code(
    input_path=in_fc,
    field_name=field,
    output_path=out_jpg,
    title="测试专题图",
)
res = runner.run(code, timeout=600, workspace=str(ws))
print("ok=", res.ok)
print("stdout tail:", (res.stdout or "")[-500:])
if res.error:
    print("error:", str(res.error)[:600])
runner.shutdown()
p = Path(out_jpg)
print("jpg exists:", p.exists(), "size:", p.stat().st_size if p.exists() else 0)
