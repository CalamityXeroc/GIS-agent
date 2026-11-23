# -*- coding: utf-8 -*-
"""``src/`` 闸门的自测（runner.py --guard-selftest）。

三个场景，全部跑在真实仓库上但只动 ``src/arcpy/``（那正是真事故的形状），
每个场景结束都清理，并断言清理后指纹回到基线：

1. clean     —— 走一遍正常 run 会给 src/ 带来的副作用（import 生成 __pycache__、
               读文件），闸门必须**不误报**；
2. escape    —— 起一个独立进程当"假 agent"写 ``src/arcpy/``，闸门必须发现、
               精确到路径、把它写进 report record（invalidated=true）并归档证据；
3. precheck  —— 残留的影子包必须让下一轮**开跑前**就中止（SrcEscapeError）。

再往下一组是 DAIR-78 的异常退出路径（在假仓库里跑真 ``runner.main()``，
要的是**进程退出码**）：KeyboardInterrupt / SystemExit(0) / RuntimeError 带着
越界一起退出时，判定和证据都不能丢；``git status -- src`` 脏的基线不许开测。

用法：``python benchmark/runner.py --guard-selftest``
退出码：0 全部通过，1 有场景失败。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src_guard import SrcGuard, SrcEscapeError, apply_to_record, hash_tree

PROBE_PKG = "arcpy"  # 与 real arcpy 同名，影子风险最高


def _remove_probe(repo: Path) -> None:
    shutil.rmtree(repo / "src" / PROBE_PKG, ignore_errors=True)


def _run_fake_agent(repo: Path) -> None:
    script = Path(__file__).with_name("fake_escape_agent.py")
    proc = subprocess.run(
        [sys.executable, str(script), str(repo)],
        capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
    )
    out = proc.stdout.strip().replace("�", "?")
    print(f"  [fake-agent] rc={proc.returncode} {out}")


def _make_fake_repo(dest: Path) -> Path:
    """假仓库（带 git）：src/ 只有白名单内的 gis_cli + baml_client，基线干净。"""
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "src" / "gis_cli").mkdir(parents=True)
    (dest / "src" / "gis_cli" / "__init__.py").write_text("VERSION = 'clean'\n", encoding="utf-8")
    (dest / "src" / "baml_client").mkdir(parents=True)
    (dest / "src" / "baml_client" / "__init__.py").write_text("ok = 1\n", encoding="utf-8")
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="selftest", GIT_AUTHOR_EMAIL="selftest@local",
        GIT_COMMITTER_NAME="selftest", GIT_COMMITTER_EMAIL="selftest@local",
    )
    for args in (["init", "-q", "."], ["add", "-A"], ["commit", "-q", "-m", "baseline"]):
        proc = subprocess.run(
            ["git", "-C", str(dest), *args], capture_output=True, text=True,
            timeout=60, encoding="utf-8", errors="replace", env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git {args[0]} 失败：{proc.stderr.strip()[:300]}")
    return dest


def _run_crash_driver(repo: Path, mode: str) -> tuple[int, dict, str]:
    """在假仓库里跑真 ``runner.main()`` 子进程，返回 (退出码, 本轮 report, stdout)。"""
    driver = Path(__file__).with_name("guard_crash_driver.py")
    report_dir = repo / "benchmark" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(driver), str(repo), mode], capture_output=True, text=True,
        timeout=300, encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
    )
    # 按 mtime 取最新：同一秒内跑两轮时 report 文件名会撞（{task_id}_{秒}），按名字 diff 会漏
    reports = sorted(report_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    payload = json.loads(reports[-1].read_text(encoding="utf-8")) if reports else {}
    return proc.returncode, payload, proc.stdout


def _run_runner_integration(repo: Path, report_dir: Path) -> tuple[bool, str]:
    """跑一遍真实的 run_task 链路（只 stub 掉 LLM/引擎），确认 record 被作废并落盘。"""
    import json as _json
    from types import SimpleNamespace

    import runner as runner_mod

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    run_root = repo / "benchmark" / "runs" / f"_guard_selftest_{dt.datetime.now().strftime('%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    trace = run_root / "stub_trace.jsonl"
    trace.write_text("", encoding="utf-8")
    task_path = report_dir / "task.yaml"
    task_path.write_text("id: guard_selftest_task\nprompt: stub\nassertions: []\n", encoding="utf-8")

    class _Loop:
        recorder = SimpleNamespace(jsonl_path=trace)

        def run(self, prompt: str, requirements=None):  # noqa: ANN001
            _run_fake_agent(repo)  # 被测进程在 run 窗口内写宿主 src/
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
    original = runner_mod.build_engine
    runner_mod.build_engine = lambda **kwargs: bundle
    try:
        record = runner_mod.run_task(
            task_path, run_root, report_dir,
            args_distill=False, args_publish=False, args_carry=False, args_guard=True,
        )
    finally:
        runner_mod.build_engine = original
        _remove_probe(repo)
        shutil.rmtree(run_root, ignore_errors=True)

    reports = sorted(report_dir.glob("guard_selftest_task_*.json"))
    saved = _json.loads(reports[-1].read_text(encoding="utf-8")) if reports else {}
    ok = (
        record.get("invalidated") is True
        and record.get("success") is False
        and saved.get("invalidated") is True
        and bool((record.get("src_guard") or {}).get("evidence_dir"))
    )
    return ok, (
        f"invalidated={record.get('invalidated')} success={record.get('success')} "
        f"report={reports[-1].name if reports else None} "
        f"evidence={(record.get('src_guard') or {}).get('evidence_dir')}"
    )


def run_selftest(repo: Path, report_root: Path) -> int:
    repo = Path(repo)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(report_root) / f"src_guard_selftest_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"scenario": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(f"\n=== src-guard 自测 · {out_dir} ===")
    _remove_probe(repo)

    # 1) clean：正常 run 的副作用（import 生成 __pycache__）不得误报
    guard = SrcGuard(repo)
    guard.begin()
    sys.path.insert(0, str(repo / "src"))
    import gis_cli  # noqa: F401  —— 生成 src/gis_cli/__pycache__ 等正常副产物
    (repo / "src" / "gis_cli" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "gis_cli" / "__pycache__" / "probe.cpython-312.pyc").write_bytes(b"\x00probe")
    clean = guard.end()
    check("clean/无副作用", clean["ok"], f"changes={len(clean['changes'])}（期望 0）")

    # 2) escape：独立进程写入 src/arcpy/ → 必须发现 + 作废 + 归档
    guard2 = SrcGuard(repo)
    baseline_after_clean = guard2.begin()["tree"]
    _run_fake_agent(repo)
    escape = guard2.end()
    paths = {item["path"] for item in escape["changes"]}
    expect = {f"src/{PROBE_PKG}/__init__.py", f"src/{PROBE_PKG}/mp.py"}
    check("escape/发现路径", expect <= paths and not escape["ok"], f"changes={sorted(paths)}")
    high = [item for item in escape["changes"] if item["severity"] == "high"]
    check("escape/定级 high", len(high) == len(expect), f"{[i['path'] for i in high]}")

    record = {"task_id": "guard_selftest", "success": True, "assertions": []}
    apply_to_record(
        record, escape, out_dir, f"selftest_{stamp}", "guard_selftest", repo,
        {"cleanup_note": "自测场景：证据归档后 `src/arcpy/` 已由自测程序移出（验证过指纹复原）。"},
    )
    evidence = Path(record["src_guard"]["evidence_dir"] or "")
    archived = (evidence / "findings.json").exists() and (evidence / "added" / PROBE_PKG / "mp.py").exists()
    check(
        "escape/作废+归档",
        record.get("invalidated") is True and record["success"] is False and record.get("assertions_validated") is False and archived,
        f"invalidated={record.get('invalidated')} evidence={evidence}",
    )

    # 3) precheck：证据留档后清理现场，指纹必须回到基线
    _remove_probe(repo)
    after_cleanup = hash_tree(repo / "src")
    check(
        "cleanup/指纹复原",
        after_cleanup == baseline_after_clean,
        f"diff={sorted(set(after_cleanup) ^ set(baseline_after_clean))[:5]}",
    )

    # 4) precheck：残留影子包必须开跑前中止
    (repo / "src" / PROBE_PKG).mkdir(parents=True, exist_ok=True)
    (repo / "src" / PROBE_PKG / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    try:
        SrcGuard(repo).begin()
        aborted, detail = False, "begin() 没有中止"
    except SrcEscapeError as exc:
        aborted, detail = True, str(exc)[:200]
    finally:
        _remove_probe(repo)
    check("precheck/开跑前中止", aborted, detail)
    check("cleanup/无残留", not (repo / "src" / PROBE_PKG).exists(), "src/arcpy 已移除")

    # 5) 端到端：run_task 真实链路（stub 掉 LLM/引擎，只让它写 src/）→ report 必须 invalidated
    check("runner/端到端作废", *_run_runner_integration(repo, out_dir / "runner_report"))

    # 6) DAIR-78：异常退出路径也要落判定（旧代码下这五条全红）
    fake_root = repo / "benchmark" / "runs" / f"_guard_selftest_{stamp}_fake"
    try:
        for label, mode in (
            ("exc/KeyboardInterrupt+退出码2", "kb"),
            ("exc/SystemExit(0)+退出码2", "sysexit"),
            ("exc/RuntimeError+crash记录带闸门", "runtime"),
        ):
            rc, crash, _ = _run_crash_driver(_make_fake_repo(fake_root / mode), mode)
            guard = crash.get("src_guard") or {}
            check(
                label,
                rc == 2
                and crash.get("status") == "runner_error"
                and crash.get("invalidated") is True
                and crash.get("invalidated_reason") == "src_escape"
                and guard.get("verdict") == "escaped"
                and bool(guard.get("evidence_dir")),
                f"rc={rc} status={crash.get('status')} invalidated={crash.get('invalidated')} "
                f"reason={crash.get('invalidated_reason')} verdict={guard.get('verdict')} "
                f"evidence={'yes' if guard.get('evidence_dir') else 'no'}",
            )

        # 反向：没碰 src/ 的普通崩溃不能被当成越界（否则退出码 2 失去区分意义）
        rc, crash_only, _ = _run_crash_driver(_make_fake_repo(fake_root / "crash_only"), "crash")
        guard_only = crash_only.get("src_guard") or {}
        check(
            "crash/未越界不作废",
            rc == 1
            and crash_only.get("status") == "runner_error"
            and crash_only.get("invalidated") is False
            and guard_only.get("verdict") == "clean",
            f"rc={rc} status={crash_only.get('status')} invalidated={crash_only.get('invalidated')} "
            f"verdict={guard_only.get('verdict')}",
        )

        # 顶层目录不变、只有已跟踪文件被改：只能靠 git_status_src 看出来
        dirty = _make_fake_repo(fake_root / "dirty_baseline")
        (dirty / "src" / "gis_cli" / "__init__.py").write_text("VERSION = 'poisoned'\n", encoding="utf-8")
        try:
            SrcGuard(dirty).begin()
            aborted, dirty_detail = False, "begin() 没有中止"
        except SrcEscapeError as exc:
            aborted, dirty_detail = True, str(exc)[:240]
        check(
            "precheck/git脏基线中止",
            aborted and "基线已被污染" in dirty_detail and "git status -- src 非空" in dirty_detail,
            dirty_detail,
        )

        # 后果链：本轮改已跟踪文件后 crash（顶层目录不变、判定丢了也没人清现场）
        # → 下一轮必须靠 git_status_src 在开跑前就停
        chain = _make_fake_repo(fake_root / "chain")
        rc1, cycle1, _ = _run_crash_driver(chain, "poison")
        rc2, cycle2, out2 = _run_crash_driver(chain, "clean")
        guard2 = cycle2.get("src_guard") or {}
        check(
            "chain/污染不被当新基线",
            cycle1.get("invalidated") is True
            and "[stub-loop] run called" not in out2
            and cycle2.get("status") == "runner_error"
            and guard2.get("verdict") == "precheck_failed"
            and "基线已被污染" in str(cycle2.get("error")),
            f"cycle1 rc={rc1} invalidated={cycle1.get('invalidated')}；cycle2 rc={rc2} "
            f"verdict={guard2.get('verdict')} 开跑了={'[stub-loop] run called' in out2} "
            f"error={str(cycle2.get('error'))[:80]}",
        )
    finally:
        shutil.rmtree(fake_root, ignore_errors=True)

    failed = [item for item in checks if not item["ok"]]
    (out_dir / "result.json").write_text(
        json.dumps({"all_ok": not failed, "checks": checks, "repo": str(repo)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n=== src-guard 自测结果：{len(checks) - len(failed)}/{len(checks)} 通过 · {out_dir} ===")
    return 0 if not failed else 1
