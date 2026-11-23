"""DAIR-246 复测：通过 `gis-agent web` 的 HTTP API 真跑一次 execute_code 任务。

服务端由调用方启动（见 report.md 第 3.5 节），本脚本只做：等健康 → 提交 → 轮询 → 打印。
用法:
    python scripts/diag/web_run_probe.py <base_url> <token> <goal>
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE, TOKEN, GOAL = sys.argv[1], sys.argv[2], sys.argv[3]


def call(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{BASE}{path}{'&' if '?' in path else '?'}token={TOKEN}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    for _ in range(60):
        try:
            call("GET", "/api/metrics/health")
            break
        except Exception:
            time.sleep(1)
    else:
        print("[probe] server never became healthy", flush=True)
        return 2
    print("[probe] server healthy", flush=True)

    t0 = time.time()
    job = call("POST", "/api/runs", {"goal": GOAL, "plan_gate": False, "max_turns": 12})
    job_id = job.get("id") or job.get("job_id")
    print(f"[probe] submitted job={job_id}", flush=True)

    last = None
    while time.time() - t0 < 240:
        cur = call("GET", f"/api/runs/{job_id}")
        status = cur.get("status")
        if status != last:
            print(f"[probe] +{time.time() - t0:6.1f}s status={status} tool={cur.get('current_tool')} turns={cur.get('turns')}", flush=True)
            last = status
        if status in {"succeeded", "failed", "stopped", "interrupted", "max_turns"}:
            out = {
                "job_id": job_id,
                "status": status,
                "engine_run_id": cur.get("engine_run_id"),
                "duration_s": cur.get("duration_s"),
                "turns": cur.get("turns"),
                "error": (cur.get("error") or "")[:400],
                "verdict": cur.get("verdict"),
            }
            print("[probe] JSON " + json.dumps(out, ensure_ascii=False), flush=True)
            return 0 if status == "succeeded" else 1
        time.sleep(2)
    print("[probe] timeout waiting for run", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
