# -*- coding: utf-8 -*-
"""对某次运行的产物复核基准断言（runner 报告缺失/崩溃时用）。

用法: python scripts/eval_run_assertions.py <task_id> [run_dir]
不带 run_dir 时取最近一次包含该任务的运行。
"""
import sys, io, glob
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

task_id = sys.argv[1] if len(sys.argv) > 1 else ""
if not task_id:
    raise SystemExit("用法: eval_run_assertions.py <task_id> [run_dir]")
task_yaml = REPO / "benchmark" / "tasks" / f"{task_id}.yaml"
if not task_yaml.exists():
    candidates = list((REPO / "benchmark" / "tasks").glob(f"{task_id}*.yaml"))
    if not candidates:
        raise SystemExit(f"找不到任务定义: {task_id}")
    task_yaml = candidates[0]

if len(sys.argv) > 2:
    run_dir = Path(sys.argv[2])
else:
    runs = sorted(glob.glob(str(REPO / "benchmark" / "runs" / "*" / task_id / "workspace")), reverse=True)
    if not runs:
        raise SystemExit(f"找不到 {task_id} 的运行工作区")
    run_dir = Path(runs[0]).parent.parent

ws = run_dir / task_id / "workspace"
task = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
print(f"任务: {task.get('id')} | 工作区: {ws}")

from gis_cli.recipes.store import RecipeLibrary  # noqa: E402
from gis_cli.runtime.code_runner import CodeRunner  # noqa: E402

runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel_eval")
lib = RecipeLibrary(code_runner=runner)
lib.load()

assertions = []
for raw in task.get("assertions", []):
    item = dict(raw)
    if item.get("path") and not Path(str(item["path"])).is_absolute():
        item["path"] = str(ws / str(item["path"]))
    assertions.append(item)

results = lib.run_assertions(assertions)
ok = 0
for r in results:
    ok += 1 if r.ok else 0
    print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.kind:<24} {r.detail[:80]}")
print(f"断言 {ok}/{len(results)}")
runner.shutdown()