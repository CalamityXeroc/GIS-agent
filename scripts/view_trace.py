# -*- coding: utf-8 -*-
"""查看引擎 trace JSONL 的最后若干事件（人类可读）。

用法: python scripts/view_trace.py <trace.jsonl> [尾部条数=12]
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    path = Path(sys.argv[1])
    tail_n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    lines = path.read_text(encoding="utf-8").splitlines()
    print(f"trace 共 {len(lines)} 行；显示最后 {tail_n} 条")
    print()
    for line in lines[-tail_n:]:
        try:
            e = json.loads(line)
        except Exception:
            continue
        kind = str(e.get("event") or e.get("type") or "?")
        data = e.get("payload") if isinstance(e.get("payload"), dict) else {}
        if not data:
            data = {k: v for k, v in e.items() if k not in ("event", "type", "ts", "seq")}
        if kind == "llm_result":
            calls = data.get("tool_calls") or data.get("actions") or []
            names = [str(c.get("tool") or c.get("name")) for c in calls if isinstance(c, dict)]
            print(f"[llm ] model={str(data.get('model') or '?')[:26]:26s} actions={names}")
        elif kind == "tool_result":
            summary = str(data.get("summary") or "")[:150].replace("\n", " | ")
            print(f"[tool] ok={data.get('ok')} {summary}")
        elif kind == "tool_call":
            print(f"[call] {data.get('name')} {str(data.get('arguments'))[:130]}")
        elif kind == "turn":
            print(f"--- turn {data.get('turn')} ---")
        elif kind == "run_start":
            print(f"[start] {str(data.get('goal'))[:110]}")
        elif kind in ("deadline_exceeded", "llm_error", "error"):
            print(f"[!!{kind}] {str(data)[:180]}")
        else:
            print(f"[{kind}] {str(data)[:130]}")


if __name__ == "__main__":
    main()
