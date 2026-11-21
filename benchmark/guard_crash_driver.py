# -*- coding: utf-8 -*-
"""异常退出路径的回归驱动（DAIR-78）：在假仓库里跑真实的 ``runner.main()``。

被测进程（stub 掉的 ``loop.run``）先往宿主 ``src/`` 写影子包再抛异常，用来验证
"异常退出也要落闸门判定 + 退出码 2"。必须起子进程：要验的是**进程退出码**，
同进程里 try 住 ``KeyboardInterrupt`` / ``SystemExit`` 会把退出码吃掉。

只被 ``guard_selftest.py`` 调用，不参与正常跑测。

用法：``python benchmark/guard_crash_driver.py <repo> <clean|crash|kb|sysexit|runtime|poison>``
退出码 = ``runner.main()`` 的退出码。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

import runner as runner_mod  # noqa: E402  (benchmark/ 同目录)


def _install_stub(repo: Path, mode: str, trace: Path) -> None:
    class _Loop:
        recorder = SimpleNamespace(jsonl_path=trace)

        def run(self, prompt, requirements=None):  # noqa: ANN001
            # ASCII 标记：自测靠它判断"这一轮到底开跑了没有"
            print("[stub-loop] run called")
            if mode == "poison":
                # 只改已跟踪文件：src/ 顶层目录一模一样，只看顶层目录的 precheck 看不出来
                (repo / "src" / "gis_cli" / "__init__.py").write_text(
                    "VERSION = 'poisoned'\n", encoding="utf-8"
                )
                raise RuntimeError("boom: 写脏已跟踪文件后崩溃")
            if mode == "crash":
                # 没碰 src/ 的普通崩溃：不能因此被作废（退出码 1，不是 2）
                raise RuntimeError("boom: 未越界的崩溃")
            if mode != "clean":
                pkg = repo / "src" / "arcpy"  # 与依赖同名 → 影子包
                pkg.mkdir(parents=True, exist_ok=True)
                (pkg / "__init__.py").write_text("x = 1\n", encoding="utf-8")
                if mode == "kb":
                    raise KeyboardInterrupt()
                if mode == "sysexit":
                    raise SystemExit(0)
                raise RuntimeError("boom: 模拟被测引擎崩溃")
            return SimpleNamespace(
                status="completed", turn=1, artifacts=[], methodology=None,
                final_summary="stub", error=None, goal="stub",
            )

    bundle = SimpleNamespace(
        loop=_Loop(),
        llm=SimpleNamespace(usage_summary=lambda: {}),
        recipes=SimpleNamespace(run_assertions=lambda assertions: []),
        close=lambda: None,
    )
    runner_mod.REPO = repo
    runner_mod.build_engine = lambda **kwargs: bundle


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: guard_crash_driver.py <repo> <clean|crash|kb|sysexit|runtime|poison>")
        return 64
    repo, mode = Path(argv[0]).resolve(), argv[1]
    report_dir = repo / "benchmark" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    run_root = repo / "benchmark" / "runs" / "guard_driver"
    run_root.mkdir(parents=True, exist_ok=True)
    trace = run_root / "stub_trace.jsonl"
    trace.write_text("", encoding="utf-8")
    task_path = repo / "benchmark" / "_selftest" / "task.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text("id: guard_exc_task\nprompt: stub\nassertions: []\n", encoding="utf-8")

    _install_stub(repo, mode, trace)
    before = {p.name for p in report_dir.glob("*.json")}
    sys.argv = ["runner.py", "--task", str(task_path), "--no-publish", "--no-carry"]
    rc = runner_mod.main()
    written = [p for p in sorted(report_dir.glob("*.json")) if p.name not in before]
    payload = json.loads(written[-1].read_text(encoding="utf-8")) if written else {}
    guard = payload.get("src_guard") or {}
    print(f"[guard-driver] mode={mode} rc={rc} report={written[-1].name if written else None} "
          f"status={payload.get('status')} invalidated={payload.get('invalidated')} "
          f"reason={payload.get('invalidated_reason')} verdict={guard.get('verdict')} "
          f"evidence={'yes' if guard.get('evidence_dir') else 'no'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
