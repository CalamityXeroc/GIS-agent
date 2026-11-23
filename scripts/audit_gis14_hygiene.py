# -*- coding: utf-8 -*-
"""盘点四个 gis14 任务交付目录的卫生状况（用本次新增的 hygiene 模块）。"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\BST_lab\GIS_CLI")
sys.path.insert(0, str(REPO / "src"))

from gis_cli.runtime.hygiene import dir_inventory, hygiene_report  # noqa: E402

BASE = REPO / "workspace/成果展示/第14届真题实测"
TASKS = [
    "任务一_清洗投影与研究区",
    "任务二a_气象插值与精度检验",
    "任务二b_DEM拼接与土地覆盖",
    "任务三a_云区修补与分类面积",
]

for task in TASKS:
    target = str(BASE / task)
    inventory, err = dir_inventory(None, target)
    if inventory is None:
        print(f"--- {task}: 读取失败 {err}")
        continue
    problems, warnings, detail = hygiene_report(inventory, ratio_limit=5.0)
    print(f"--- {task}")
    for name, info in (inventory.get("gdb") or {}).items():
        items = info["items"]
        print(f"    {name}: {len(items)} 项 / {info['bytes'] / 1e6:.0f} MB")
        for item in items:
            print(f"        - {item}")
    print(f"    散文件: {inventory.get('files')}")
    print(f"    问题: {problems or '无'}")
    print(f"    警告: {warnings or '无'}")
