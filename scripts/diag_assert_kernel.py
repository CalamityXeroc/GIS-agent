# -*- coding: utf-8 -*-
"""诊断：run 结束后断言评估时内核调用为何返回 None。

复现 field_stats_between 的执行路径，打印 code_runner 的原始返回。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WS = Path(r"E:\BST_lab\GIS_CLI\benchmark\runs\20260922_152144\gis14_am_task1_animal_points\workspace")
FC = str(WS / "output" / "result_data.gdb" / "animal_points")

from gis_cli.engine.runner import build_engine

bundle = build_engine(workspace=WS, refresh_catalog=False)
runner = bundle.recipes.code_runner
print("code_runner 类型:", type(runner).__name__)
print("code_runner.kernel?", getattr(runner, "kernel", None) is not None)

code_describe = (
    "import arcpy\n"
    f"_p = {FC!r}\n"
    "_rec = {'exists': arcpy.Exists(_p)}\n"
    "set_result(_rec)\n"
)
out1 = runner.run(code_describe, timeout=120)
print("describe 调用: ok=", out1.ok, "result=", out1.result, "error=", str(out1.error)[:120])

code_stat = (
    "import arcpy, statistics as _st\n"
    f"_p = {FC!r}\n_f = 'LAT'\n"
    "_vals = []\n_n = 0\n_null = 0\n"
    "with arcpy.da.SearchCursor(_p, [_f]) as cur:\n"
    "    for row in cur:\n"
    "        _n += 1\n"
    "        _v = row[0]\n"
    "        if _v is None:\n"
    "            _null += 1\n"
    "            continue\n"
    "        _vals.append(float(_v))\n"
    "set_result({'min': min(_vals) if _vals else None, 'max': max(_vals) if _vals else None,\n"
    "            'count': _n, 'null': _null})\n"
)
out2 = runner.run(code_stat, timeout=180)
print("SearchCursor 调用: ok=", out2.ok, "result=", out2.result, "error=", str(out2.error)[:200])

bundle.close()
