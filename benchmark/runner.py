# -*- coding: utf-8 -*-
"""Benchmark runner for the GIS engine.

Creates an isolated workspace per task (inputs only, never reference answers),
runs the engine, evaluates structural assertions, and writes reports.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import datetime as dt  # 注意：arcpy 会把调用方命名空间的 datetime 换成模块，不能 `from datetime import datetime`
from pathlib import Path
from typing import Any

import yaml  # type: ignore

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "benchmark" / "runs"
sys.path.insert(0, str(REPO / "src"))

from gis_cli.engine.runner import build_engine  # noqa: E402
from src_guard import SrcGuard, apply_to_record  # noqa: E402  (benchmark/ 同目录)


def load_task(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid task file: {path}")
    data.setdefault("id", path.stem)
    data.setdefault("assertions", [])
    data.setdefault("inputs", [])
    return data


_SHAPEFILE_SIDECARS = (".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx", ".shp.xml", ".qmd")


def review_switch(task: dict[str, Any]) -> bool | None:
    """双复核（W5）开关：顶层 ``review`` 优先，其次看 ``requirements`` 里的标记。

    计划原文是「开关进 requirements，默认关」——交付级任务的两处写法都认：

    - ``review: true``（任务顶层，最简）；
    - ``requirements: [{..., review: true}]``（只对某条交付要求开）。

    返回 ``None`` = 不覆盖 ``engine.review_enabled``（默认关）；
    在 ``build_engine`` 前定下，所以关着的时候复核工具压根不进工具表。
    """
    if task.get("review") is not None:
        return bool(task.get("review"))
    for item in task.get("requirements") or []:
        if isinstance(item, dict) and item.get("review") is not None:
            return bool(item.get("review"))
    return None


def _latest_workspace(task_id: str) -> Path | None:
    """Locate the workspace of the most recent run of a task (for chain tasks).

    按 **mtime** 取最新而不是按目录名排序——不同时期运行目录的时间戳格式不同
    （如 20260921_212830 与 2026_09_21_212933），按名字排序会误选旧目录，
    导致链式任务承接了坏工作区（14 届任务二a 首跑实测踩中）。
    """
    roots = []
    if RUNS_DIR.exists():
        for root in RUNS_DIR.glob("*/" + task_id + "/workspace"):
            if (root / "output").exists():
                try:
                    roots.append((root.stat().st_mtime, root))
                except OSError:
                    continue
    if not roots:
        return None
    return max(roots, key=lambda pair: pair[0])[1]


def carry_workspace(task: dict[str, Any], ws: Path) -> str:
    """Carry outputs and the project log from an earlier task into this workspace.

    Real GIS work (and the exam) is chained: task 2 builds on task 1's layers.
    Copying the previous ``output/`` and ``.gis_agent/PROJECT_LOG.md`` lets the
    agent continue instead of redoing work, and exercises the log mechanism.
    """
    previous_id = str(task.get("carry_from") or "").strip()
    if not previous_id:
        return ""
    source = _latest_workspace(previous_id)
    if source is None:
        print(f"[carry_from] 未找到 {previous_id} 的历史工作区，本次从零开始")
        return ""
    carried = []
    for name in ("output", "input"):
        src = source / name
        if not src.exists():
            continue
        dst = ws / name
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if target.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        carried.append(name)
    log_src = source / ".gis_agent" / "PROJECT_LOG.md"
    if log_src.exists():
        log_dst = ws / ".gis_agent" / "PROJECT_LOG.md"
        log_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_src, log_dst)
        carried.append("PROJECT_LOG.md")
    note = f"[carry_from] 已承接 {previous_id}: {source}（{', '.join(carried)}）"
    print(note)
    return note


def prepare_workspace(task: dict[str, Any], run_root: Path, carry: bool = True) -> Path:
    """Create an isolated workspace containing only the task inputs."""
    ws = run_root / task["id"] / "workspace"
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    (ws / "input").mkdir(parents=True, exist_ok=True)
    (ws / "output").mkdir(parents=True, exist_ok=True)
    for item in task.get("inputs", []):
        source = REPO / str(item["source"])
        if not source.exists():
            raise FileNotFoundError(f"input missing: {source}")
        dest = ws / "input" / source.name
        if source.is_dir():
            shutil.copytree(source, dest)
        elif source.suffix.lower() == ".shp":
            # A shapefile without its sidecars is broken data (.dbf holds ALL
            # attribute values and cannot be recovered from the .shp alone).
            shutil.copy2(source, dest)
            for ext in _SHAPEFILE_SIDECARS:
                sidecar = source.with_suffix(ext)
                if sidecar.exists():
                    shutil.copy2(sidecar, ws / "input" / sidecar.name)
        else:
            shutil.copy2(source, dest)
    # config/llm_config.json must be visible from the workspace
    cfg_dir = ws / "config"
    cfg_dir.mkdir(exist_ok=True)
    src_cfg = REPO / "config" / "llm_config.json"
    if src_cfg.exists():
        shutil.copy2(src_cfg, cfg_dir / "llm_config.json")
    if carry:
        carry_workspace(task, ws)
    return ws


def evaluate(task: dict[str, Any], workspace: Path, bundle: Any) -> list[dict[str, Any]]:
    """Evaluate assertions against the workspace output."""
    assertions = []
    for raw in task.get("assertions", []):
        item = dict(raw)
        target = item.get("path") or item.get("target")
        if target and not Path(str(target)).is_absolute():
            item["path"] = str(workspace / str(target))
        assertions.append(item)
    results = bundle.recipes.run_assertions(assertions)
    return [r.to_dict() for r in results]


PUBLISH_DIR = REPO / "workspace" / "成果展示" / "基准产出"


def publish_outputs(workspace: Path, run_id: str, task_id: str) -> Path | None:
    """把本次运行的产出拷进 workspace，供人查看（过程库 temp_data.gdb + 结果库 + 报告）。

    基准运行本身仍在隔离工作区里跑（每任务只拿到自己的输入、看不到参考答案），
    这里只是把结果"搬运"到 workspace，避免运行目录被删后无处查看。
    """
    src = workspace / "output"
    if not src.exists():
        return None
    dest = PUBLISH_DIR / run_id / task_id
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dest / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        except Exception as exc:  # 发布失败不该影响评测结论
            print(f"[publish] 跳过 {item.name}: {type(exc).__name__}: {exc}")
    print(f"[publish] 产出已发布到: {dest}")
    return dest


def run_task(
    task_path: Path,
    run_root: Path,
    report_dir: Path,
    args_distill: bool = False,
    args_publish: bool = True,
    args_carry: bool = True,
    args_guard: bool = True,
) -> dict[str, Any]:
    task = load_task(task_path)
    task_id = task["id"]
    # record 必须在进 try 之前建好：正常返回 / 普通异常 / Ctrl-C / SystemExit 都要能
    # 拿到它并落盘。DAIR-78：旧代码把闸门判定只放在 finally 的局部变量里，异常一冒泡
    # 就整个丢掉，下一轮还把上一轮的污染当成新基线。
    record: dict[str, Any] = {
        "task_id": task_id,
        "name": task.get("name", ""),
        "status": "runner_error",
        "turns": 0,
        "duration_seconds": 0.0,
        "artifacts": [],
        "methodology": None,
        "summary": None,
        "error": None,
        "traceback": None,
        "assertions": [],
        "assertions_passed": 0,
        "assertions_total": 0,
        "success": False,
        "invalidated": False,
        "invalidated_reason": None,
        "src_guard": None,
        "interrupted": False,
        "published": None,
        "distilled_recipe": None,
        "trace": None,
        "usage": {},
        "timestamp": dt.datetime.now().isoformat(),
    }
    guard = SrcGuard(REPO) if args_guard else None
    guard_started = False
    bundle = None
    started = time.time()
    try:
        # 跑前预检：src/ 顶层影子包 / git 侧已脏 → SrcEscapeError，不在脏基线上开测
        if guard is not None:
            guard.begin()
            guard_started = True
        workspace = prepare_workspace(task, run_root, carry=args_carry)
        print(f"\n=== [{task_id}] {task.get('name', '')} ===")
        print(f"workspace: {workspace}")

        bundle = build_engine(
            workspace=workspace,
            refresh_catalog=True,
            engine_overrides={
                "run_deadline_seconds": task.get("timeout_seconds"),
                # 任务可以自带轮次上限（"四张图+迁移图"这类交付任务 60 轮不够）
                "max_turns": task.get("max_turns"),
                # 交付级任务可在 YAML 写 review: true（或 requirements 里任一条带 review: true）
                # 打开双复核（W5，token 翻倍）；None 不覆盖 engine.review_enabled，默认关
                "review_enabled": review_switch(task),
            },
        )
        state = bundle.loop.run(
            str(task.get("prompt", "")),
            requirements=task.get("requirements") or None,
        )
        try:
            assertions = evaluate(task, workspace, bundle)
        except Exception as exc:  # 断言评估异常不应丢掉整次运行的记录
            import traceback

            print(f"[evaluate] 断言评估异常: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            assertions = [{"kind": "evaluate_error", "ok": False, "detail": str(exc)[:300]}]
        distilled_path = None
        if args_distill and state.status == "completed":
            distilled_path = distill_run(bundle, task, state, report_dir)
        passed = [a for a in assertions if a.get("ok")]
        record.update({
            "status": state.status,
            "turns": state.turn,
            "artifacts": state.artifacts,
            "methodology": state.methodology,
            "summary": state.final_summary,
            "error": state.error,
            "assertions": assertions,
            "assertions_passed": len(passed),
            "assertions_total": len(assertions),
            # 判成功以"验收断言全过"为准：模型常在交付后继续打磨直到轮次/时限用尽，
            # 那时 status 是 max_turns/deadline_exceeded，但产物已全部通过验收（14 届四a 实测）。
            "success": len(assertions) > 0 and len(passed) == len(assertions),
            "completed": state.status == "completed",
            "trace": str(bundle.loop.recorder.jsonl_path),
            # 用量落进报告：A/B（子代理开/关）靠 total token 判胜负，不能只靠 trace
            "usage": bundle.llm.usage_summary(),
            "distilled_recipe": str(distilled_path) if distilled_path else None,
        })
        published_path = publish_outputs(workspace, run_root.name, task_id) if args_publish else None
        record["published"] = str(published_path) if published_path else None
    except BaseException as exc:  # noqa: BLE001 —— KeyboardInterrupt / SystemExit 也要落记录
        import traceback

        print(f"TASK CRASHED {task_id}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        record.update({
            "status": "runner_error",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            # Ctrl-C 之后不该接着跑下一轮（main 据此停批）
            "interrupted": isinstance(exc, KeyboardInterrupt),
        })
        if guard is not None and not guard_started and guard.before is not None:
            # 预检就中止了（没开测）：把跑前快照留在记录里，别让人只看到一句异常
            record["src_guard"] = {
                "ok": False,
                "verdict": "precheck_failed",
                "changes": [],
                "git_new_entries": [],
                "watch_dir": str(guard.watch_dir),
                "evidence_dir": None,
                "reason": str(exc)[:500],
            }
    finally:
        if bundle is not None:
            bundle.close()
        # 采样点必须在被测进程结束之后（loop 结束即 agent 不再写盘）。
        # 判定直接写进 record：异常路径同样要作废，不能只停在局部变量里。
        if guard_started:
            apply_to_record(record, guard.end(), report_dir, run_root.name, task_id, REPO)
        record["duration_seconds"] = round(time.time() - started, 1)

    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_crash" if record["status"] == "runner_error" else ""
    out = report_dir / f"{task_id}_{int(time.time())}{suffix}.json"
    record["report_path"] = str(out)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status={record['status']} assertions={record['assertions_passed']}/{record['assertions_total']} "
          f"success={record['success']} duration={record['duration_seconds']}s")
    if record.get("invalidated"):
        print(f"INVALIDATED(src_escape): 本轮作废，证据 {(record.get('src_guard') or {}).get('evidence_dir')}")
    for a in record["assertions"]:
        mark = "OK " if a.get("ok") else "FAIL"
        print(f"  [{mark}] {a.get('kind')} {a.get('target')} {a.get('detail')}")
    return record


def distill_run(bundle: Any, task: dict[str, Any], state: Any, report_dir: Path) -> Path | None:
    """Distill a successful run into a candidate recipe (human review dir)."""
    try:
        from gis_cli.recipes.distiller import RecipeDistiller

        distiller = RecipeDistiller(bundle.llm, bundle.recipes)
        snippets = RecipeDistiller.collect_successful_code(bundle.loop.recorder.jsonl_path)
        recipe = distiller.distill(
            goal=state.goal,
            methodology=state.methodology,
            final_summary=state.final_summary,
            code_snippets=snippets,
            artifacts=state.artifacts,
        )
        if recipe is None:
            return None
        out_dir = report_dir / "distilled"
        path = distiller.save_candidate(recipe, out_dir, provenance=f"task:{task['id']}")
        print(f"distilled candidate recipe -> {path}")
        return path
    except Exception as exc:
        print(f"distill failed: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GIS engine benchmark")
    parser.add_argument("--task", action="append", help="task id or yaml path (repeatable)")
    parser.add_argument("--all", action="store_true", help="run all tasks in benchmark/tasks")
    parser.add_argument("--keep", action="store_true", help="keep run workspaces")
    parser.add_argument(
        "--no-carry",
        action="store_true",
        help="不承接上一任务的工作区（验证任务是否自洽可独立重跑）",
    )
    parser.add_argument("--distill", action="store_true", help="distill successful runs into candidate recipes")
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="不把产出拷到 workspace/成果展示/基准产出（默认会拷）",
    )
    parser.add_argument(
        "--no-guard",
        action="store_true",
        help="关闭 src/ 越界闸门（仅用于闸门自身排障，正常跑测一律开着）",
    )
    parser.add_argument(
        "--guard-selftest",
        action="store_true",
        help="跑 src/ 闸门自测（假 agent 写 src/arcpy/，验证发现+作废+归档，不留残留）",
    )
    args = parser.parse_args()

    if args.guard_selftest:
        from guard_selftest import run_selftest

        return run_selftest(REPO, REPO / "benchmark" / "reports")

    task_dir = REPO / "benchmark" / "tasks"
    if args.all:
        paths = sorted(task_dir.glob("*.yaml"))
    elif args.task:
        paths = []
        for item in args.task:
            candidate = Path(item)
            if candidate.exists():
                paths.append(candidate)
            else:
                paths.append(task_dir / f"{item}.yaml")
    else:
        parser.error("specify --task or --all")

    run_root = REPO / "benchmark" / "runs" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = REPO / "benchmark" / "reports"
    records = []
    for path in paths:
        if not path.exists():
            print(f"SKIP missing task: {path}")
            continue
        try:
            record = run_task(
                path,
                run_root,
                report_dir,
                args_distill=args.distill,
                args_publish=not args.no_publish,
                args_carry=not args.no_carry,
                args_guard=not args.no_guard,
            )
        except BaseException as exc:  # noqa: BLE001 —— run_task 之外的失败（含 Ctrl-C）也不能丢记录
            import traceback

            print(f"TASK CRASHED {path.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            record = {
                "task_id": path.stem,
                "status": "runner_error",
                "success": False,
                "invalidated": False,
                "invalidated_reason": None,
                "src_guard": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
                "assertions_passed": 0,
                "assertions_total": 0,
                "turns": 0,
                "duration_seconds": 0,
            }
            try:
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / f"{path.stem}_{int(time.time())}_crash.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
        records.append(record)
        if record.get("interrupted"):
            print("[runner] 收到 KeyboardInterrupt：本轮判定已落盘，停止后续任务")
            break

    print("\n=== SUMMARY ===")
    for record in records:
        mark = " INVALIDATED(src_escape)" if record.get("invalidated") else ""
        print(f"{record['task_id']:<28} success={record['success']} "
              f"assertions={record['assertions_passed']}/{record['assertions_total']} "
              f"turns={record['turns']} {record['duration_seconds']}s{mark}")
    if not args.keep:
        shutil.rmtree(run_root, ignore_errors=True)
    if any(r.get("invalidated") for r in records):
        return 2  # 越界作废与"跑失败"分开：退出码 2 专指结果不可信
    return 0 if all(r["success"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
