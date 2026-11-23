"""DAIR-246 复测：走真实引擎（build_engine → loop.run）跑一次含 execute_code 的任务。

不经过 web 层、用独立 workspace，避免与正在跑的 DAIR-247 服务抢同一个 workspace/内核。

用法:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/diag/run_execute_code_probe.py <workspace_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from gis_cli.engine.runner import build_engine

GOAL = "只用一次 execute_code 计算 2+3，然后用一句话回答结果（不要写文件）。"


def main() -> int:
    ws = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark/reports/dair-246_arcpy_hang/ws").resolve()
    ws.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []
    t0 = time.time()

    def emit(name: str, payload: dict | None = None) -> None:
        data = payload if isinstance(payload, dict) else {}
        events.append({"t": round(time.time() - t0, 2), "event": name, "data": data})
        if name in {"run_start", "tool_call", "tool_result", "run_end"}:
            brief = {k: data.get(k) for k in ("run_id", "tool", "ok", "summary", "status", "error") if k in data}
            if "summary" in brief and brief["summary"]:
                brief["summary"] = str(brief["summary"])[:160]
            print(f"[probe] +{time.time() - t0:6.1f}s {name} {brief}", flush=True)

    bundle = build_engine(
        workspace=ws,
        config_path=None,
        refresh_catalog=False,
        on_event=emit,
        engine_overrides={"max_turns": 10},
        plan_gate=None,
    )
    state = {}
    try:
        st = bundle.loop.run(GOAL)
        state = dict(st.to_dict() if hasattr(st, "to_dict") else st)
    finally:
        try:
            bundle.close()
        except Exception as exc:  # pragma: no cover
            print(f"[probe] close failed: {exc}", flush=True)

    total = round(time.time() - t0, 2)
    tool_calls = [e for e in events if e["event"] == "tool_call"]
    tool_results = [e for e in events if e["event"] == "tool_result"]
    run_id = next((e["data"].get("run_id") for e in events if e["event"] == "run_start"), None)
    out = {
        "workspace": str(ws),
        "run_id": run_id,
        "total_s": total,
        "status": state.get("status"),
        "error": state.get("error"),
        "turns": state.get("turn"),
        "tool_calls": [{"t": c["t"], "tool": c["data"].get("tool")} for c in tool_calls],
        "tool_results": [
            {"t": c["t"], "tool": c["data"].get("tool"), "ok": c["data"].get("ok"),
             "summary": (c["data"].get("summary") or "")[:300]}
            for c in tool_results
        ],
        "final_summary": (state.get("final_summary") or "")[:500],
    }
    print("[probe] JSON " + json.dumps(out, ensure_ascii=False), flush=True)
    return 0 if state.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
