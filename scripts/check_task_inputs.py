# -*- coding: utf-8 -*-
"""自检：benchmark/tasks/*.yaml 里声明的输入是否都存在。

背景：基准任务曾经靠 carry_from 承接上一步的临时工作区，运行目录一清理任务就缺数据
（实测：任务二b 从零开始、缺研究区）。现在任务都应声明**显式输入**，这个脚本负责守门。

用法: python scripts/check_task_inputs.py [任务名过滤]
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmark"))

from runner import load_task  # noqa: E402

SHAPEFILE_SIDECARS = (".dbf", ".shx", ".prj")


def check(pattern: str = "") -> int:
    problems = 0
    tasks = sorted((REPO / "benchmark/tasks").glob("*.yaml"))
    for path in tasks:
        if pattern and pattern not in path.name:
            continue
        try:
            task = load_task(path)
        except Exception as exc:
            print(f"[解析失败] {path.name}: {exc}")
            problems += 1
            continue
        inputs = task.get("inputs") or []
        carried = task.get("carry_from")
        marks: list[str] = []
        for item in inputs:
            raw = str(item.get("source") or "")
            target = Path(raw) if Path(raw).is_absolute() else REPO / raw
            if not target.exists():
                marks.append(f"缺失 {raw}")
                problems += 1
                continue
            if target.suffix.lower() == ".shp":
                missing = [ext for ext in SHAPEFILE_SIDECARS if not target.with_suffix(ext).exists()]
                if missing:
                    marks.append(f"shapefile 缺伴生文件 {missing}: {raw}")
                    problems += 1
        note = ""
        if carried:
            note = f"（承接 {carried}）"
        status = "OK " if not marks else "!! "
        detail = "；".join(marks) if marks else f"{len(inputs)} 项输入"
        print(f"{status}{task['id']:<44} {detail}{note}")
    print("")
    print("全部输入就绪 ✓" if problems == 0 else f"发现 {problems} 个问题 ✗")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(check(sys.argv[1] if len(sys.argv) > 1 else ""))
