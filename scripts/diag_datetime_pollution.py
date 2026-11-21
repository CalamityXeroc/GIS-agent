# -*- coding: utf-8 -*-
"""查出是谁把调用方命名空间里的 datetime 从类改成了模块。"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

from datetime import datetime

print("1) 导入前:", type(datetime).__name__)

import arcpy  # noqa: E402

print("2) import arcpy 后:", type(datetime).__name__)

from gis_cli.engine.runner import build_engine  # noqa: E402

print("3) 导入 build_engine 后:", type(datetime).__name__)

# 模拟 benchmark runner：建引擎（会用进程内 arcpy 做目录扫描等）
from pathlib import Path  # noqa: E402

ws = Path(r"C:\Users\cheng\AppData\Local\Temp\diag_dt_ws")
ws.mkdir(parents=True, exist_ok=True)
(ws / "input").mkdir(exist_ok=True)
(ws / "output").mkdir(exist_ok=True)
bundle = build_engine(workspace=ws, refresh_catalog=False)
print("4) build_engine 后:", type(datetime).__name__)
try:
    print("   datetime.now() ->", datetime.now().isoformat())
except Exception as exc:
    print("   失败:", exc)
print("   sys.modules['datetime'] =", sys.modules.get("datetime"))
import gis_cli.engine.runner as er

print("5) gis_cli.engine.runner 里的 datetime:", getattr(er, "datetime", None))
bundle.close()