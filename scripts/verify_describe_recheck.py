# -*- coding: utf-8 -*-
"""实证：同会话 Describe 假阴性 → Pro 子进程复核后判真（DAIR-16 Bug 1）。

用法:
    python scripts/verify_describe_recheck.py [workspace]

默认用 DAIR-15 探针的 workspace（``real_{a,b}.gdb`` 还在）。内核用"永远说不存在"
的桩来模拟脏会话——DAIR-15 那次就是这个症状：trace seq 104 的 hygiene 两次都判
``real_*.gdb: 0MB/0项``，而全新 Pro 进程里 ``exists=True``、``count=3``。
对照组把复核关掉（``_describe_via_pro → None``），同一份代码立刻回到 0/4。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # type: ignore  # noqa: E402

from gis_cli.recipes.store import RecipeLibrary  # noqa: E402

WS = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else REPO / "benchmark/runs/20261023_145529/probe_fake_gdb_guard/workspace"
)
TASK = REPO / "benchmark/tasks/probe_fake_gdb_guard.yaml"


class _DirtySession:
    """会话窗口永远说"不存在"（DAIR-15 实测症状）。"""

    def run(self, code, timeout=None, workspace=None):
        return SimpleNamespace(
            ok=True, stdout="", stderr="", error=None, display_images=[],
            result={"exists": False, "feature_count": None},
        )


def _gdb_assertions() -> list[dict]:
    task = yaml.safe_load(TASK.read_text(encoding="utf-8")) or {}
    out = []
    for raw in task.get("assertions", []):
        if str(raw.get("type")) not in {"vector_exists", "feature_count_between"}:
            continue
        item = dict(raw)
        item["path"] = str(WS / str(item["path"]))
        out.append(item)
    return out


def _evaluate(lib: RecipeLibrary, label: str) -> int:
    assertions = _gdb_assertions()
    started = time.time()
    results = lib.run_assertions(assertions)
    passed = sum(1 for r in results if r.ok)
    print(f"\n=== {label}：{passed}/{len(results)} 通过（{time.time() - started:.1f}s）")
    for r in results:
        print(f"  [{'OK  ' if r.ok else 'FAIL'}] {r.kind:<22} {r.detail[:80]}")
    return passed


def main() -> int:
    print(f"workspace: {WS}")
    if not WS.exists():
        print("workspace 不存在；请显式传入一个还留着 real_*.gdb 的运行目录")
        return 2

    lib = RecipeLibrary(code_runner=_DirtySession())
    lib._describe_via_pro = lambda path: None  # type: ignore[method-assign]
    before = _evaluate(lib, "对照：关掉复核（旧行为）")
    after_lib = RecipeLibrary(code_runner=_DirtySession())
    after = _evaluate(after_lib, "修复后：同会话说没有 → Pro 子进程复核")
    print(f"\n结论：{before}/4 → {after}/4")
    return 0 if (before == 0 and after == 4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
