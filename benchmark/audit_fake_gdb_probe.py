# -*- coding: utf-8 -*-
"""探针任务 probe_fake_gdb_guard 的 trace 审计。

runner 的断言系统没有"负向存在/拦截发生"类断言，假库拦截是否真的发生只能
从 trace 证据判定（trace 里 tool_call 记录完整代码、tool_result 记录 ok/summary，
见 engine/loop.py `_handle_action`）。本脚本把探针的四个关键行为从 trace 里
配对核验：

  (a) 往假库写要素类 → 被 FakeGdbPath 拦（正/反斜杠各一次）。判据是**系统侧**证据：
      auto-repair 的 ``repair_attempt.error`` 或 tool_result 的 summary 里出现"假 .gdb"/
      "FakeGdbPath"。被自动修复掩盖也必须算——修复成功后 tool_result 是 ok=True，
      只看 ok 会把真正的拦截误判成"未拦截"（DAIR-16 实测过两次）。
  (c) shutil.rmtree / Delete 清理假库 → ok=True（清理意图放行，两种写法）
  (b) 按改真库后写入 → ok=True
  (d) 工作区 output/ 下不再有 fake_*.gdb 目录（文件系统检查）

用法：python audit_fake_gdb_probe.py <trace.jsonl> <workspace>
退出码：全部通过 0，否则 1（供 CI/基准汇总直接用）。
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

BS = chr(92)  # 反斜杠（避免源码里写 \f 之类转义歧义）

FAKE_NAMES = ("fake_a.gdb", "fake_b.gdb")
#: 真正"往里写"的调用；按 AST 取被调函数名，不看字符串（总结/记录 md 里提到这些名字
#: 不算写入——以前按子串判，把"第五步写探针记录"那个 cell 也当成了写假库）。
WRITE_CALLS = {"CreateFeatureclass", "CreateFeatureclass_management", "InsertCursor", "CopyFeatures"}
CLEANUP_CALLS = {"rmtree", "remove", "rmdir", "rename", "replace", "Delete", "Delete_management",
                 "DeleteFeatures", "TruncateTable"}
WRITE_MARKERS = ("CreateFeatureclass", "InsertCursor", "CopyFeatures")  # 解析不出 AST 时的退路
CLEANUP_MARKERS = ("rmtree", "Delete_management", ".rmdir", "os.remove")


def _iter_pairs(events: list[dict]) -> list[tuple[dict, dict, list[dict]]]:
    """把 execute_code 的 tool_call 与紧随的 tool_result 配成对（附中间的 auto-repair 事件）。"""
    pairs: list[tuple[dict, dict, list[dict]]] = []
    pending: dict | None = None
    repairs: list[dict] = []
    for evt in events:
        payload = evt.get("payload") or {}
        if evt.get("event") == "tool_call" and payload.get("tool") == "execute_code":
            pending, repairs = evt, []
        elif evt.get("event") == "repair_attempt" and pending is not None:
            repairs.append(evt)
        elif evt.get("event") == "tool_result" and payload.get("tool") == "execute_code" and pending is not None:
            pairs.append((pending, evt, repairs))
            pending, repairs = None, []
    return pairs


def _called_names(code: str) -> set[str]:
    """代码里真正被调用的函数名（AST）；半截代码解析不出时返回空集。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


def _is_write(code: str) -> bool:
    calls = _called_names(code)
    if calls:
        return bool(calls & WRITE_CALLS)
    return any(m in code for m in WRITE_MARKERS)


def _is_cleanup(code: str) -> bool:
    calls = _called_names(code)
    if calls:
        return bool(calls & CLEANUP_CALLS) and not (calls & WRITE_CALLS)
    return any(m in code for m in CLEANUP_MARKERS) and "CreateFeatureclass" not in code


def _guard_hit(repairs: list[dict], result: dict) -> str:
    """拦截证据来自系统侧：auto-repair 拿到过的 error，或最终 tool_result。

    返回 "repair" / "result" / ""（无证据）。被自动修复掩盖（最终 ok=True）也算。
    """
    for evt in repairs:
        err = str((evt.get("payload") or {}).get("error") or "")
        if "假 .gdb" in err or "FakeGdbPath" in err:
            return "repair"
    payload = result.get("payload") or {}
    summary = str(payload.get("summary") or "")
    if not payload.get("ok") and ("假 .gdb" in summary or "FakeGdbPath" in summary):
        return "result"
    return ""


def _code_of(call: dict) -> str:
    return str((call.get("payload") or {}).get("args", {}).get("code", "") or "")


def _path_form(code: str) -> str:
    """该代码段里 fake_*.gdb 的路径写法：forward / back / mixed / none。"""
    fwd = bool(re.search(r"output/fake_[ab]\.gdb", code))
    back = bool(re.search(r"output" + re.escape(BS) + r"+fake_[ab]\.gdb", code))
    if fwd and back:
        return "mixed"
    if fwd:
        return "forward"
    if back:
        return "back"
    return "none"


def audit_trace(trace_path: str | Path) -> dict:
    events = [json.loads(line) for line in Path(trace_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = _iter_pairs(events)

    intercepts: list[dict] = []
    cleanups: list[dict] = []
    real_writes: list[dict] = []

    for call, result, repairs in pairs:
        code = _code_of(call)
        ok = bool((result.get("payload") or {}).get("ok"))
        summary = str((result.get("payload") or {}).get("summary", "") or "")
        form = _path_form(code)
        touches_fake = any(name in code for name in FAKE_NAMES)
        cleanup = touches_fake and _is_cleanup(code)
        write = touches_fake and _is_write(code) and not cleanup
        touches_real = ("real_a.gdb" in code or "real_b.gdb" in code) and _is_write(code)

        if write:
            hit = _guard_hit(repairs, result)
            intercepts.append(
                {
                    "seq": call.get("seq"),
                    "form": form,
                    "intercepted": bool(hit),
                    "via": hit,
                    "ok": ok,
                    "summary_head": summary[:160],
                }
            )
        elif cleanup:
            cleanups.append({"seq": call.get("seq"), "form": form, "allowed": ok, "ok": ok, "summary_head": summary[:160]})
        if touches_real:
            real_writes.append({"seq": call.get("seq"), "form": form, "ok": ok, "summary_head": summary[:160]})

    return {"intercepts": intercepts, "cleanups": cleanups, "real_writes": real_writes}


def audit_workspace(workspace: str | Path) -> dict:
    out = Path(workspace) / "output"
    fake_dirs = sorted(p.name for p in out.glob("fake_*.gdb")) if out.exists() else []
    real_ok = {name: (Path(workspace) / "output" / name / "gdb").exists() for name in ("real_a.gdb", "real_b.gdb")}
    return {"fake_dirs_left": fake_dirs, "real_gdb_marker": real_ok}


def verdict(report: dict) -> list[str]:
    """返回失败原因列表；空列表 = 全部通过。"""
    problems: list[str] = []

    def _has(seq_list: list[dict], form: str, want: bool) -> bool:
        return any(item.get("form") == form and bool(item.get("intercepted" if "intercepted" in item else "allowed")) == want for item in seq_list)

    for form in ("forward", "back"):
        hits = [i for i in report["intercepts"] if i["form"] == form]
        if not hits:
            problems.append(f"(a) 假库写入-{form}：trace 里没有找到写假库的 execute_code 调用")
        elif not any(i["intercepted"] for i in hits):
            problems.append(f"(a) 假库写入-{form}：调用存在但未被拦截（ok/summary 无 拦截证据）：{hits}")
    for form in ("forward", "back"):
        cleans = [c for c in report["cleanups"] if c["form"] == form]
        if not cleans:
            problems.append(f"(c) 清理-{form}：trace 里没有找到 rmtree 假库的 execute_code 调用")
        elif not any(c["allowed"] for c in cleans):
            problems.append(f"(c) 清理-{form}：清理调用被拦截（清理意图应放行）：{cleans}")
    if not report["real_writes"] or not any(w["ok"] for w in report["real_writes"]):
        problems.append(f"(b) 真库写入：没有成功的真库写入调用：{report['real_writes']}")
    ws = report.get("workspace") or {}
    if ws.get("fake_dirs_left"):
        problems.append(f"(d) 交付清单含假库目录：{ws['fake_dirs_left']}")
    for name, ok in (ws.get("real_gdb_marker") or {}).items():
        if not ok:
            problems.append(f"(b) {name} 缺少真 GDB 标记文件（gdb）")
    return problems


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    trace_path, workspace = sys.argv[1], sys.argv[2]
    report = audit_trace(trace_path)
    report["workspace"] = audit_workspace(workspace)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    problems = verdict(report)
    if problems:
        print("\n== FAIL ==")
        for p in problems:
            print(" -", p)
        return 1
    print("\n== PASS == 拦截(2/2)、清理放行(2/2)、真库写入成功、无假库残留")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
