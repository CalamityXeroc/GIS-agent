"""DAIR-246 诊断脚本：验证引擎持久内核 + execute_code 链路（不跑 LLM）。

用法:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/diag/kernel_exec_probe.py
"""
from __future__ import annotations

import json
import time

from gis_cli.runtime.kernel import ArcpyKernel as Kernel


def main() -> int:
    t0 = time.time()
    k = Kernel()
    print(f"[probe] python_executable = {k.python_executable}", flush=True)
    print(f"[probe] kernel.start() ...", flush=True)
    k.start()
    print(f"[probe] kernel ready in {time.time() - t0:.2f}s (includes arcpy import)", flush=True)

    cases = [
        ("import arcpy", "import arcpy; print('arcpy', arcpy.GetInstallInfo()['Version'])"),
        ("execute_code", "import arcpy\nprint('workspace=', arcpy.env.workspace)\nprint('describe ok', bool(arcpy.Describe))\n"),
        ("state_kept", "x = 41\nprint('x set')"),
        ("state_reuse", "print('x+1 =', x + 1)"),
    ]
    out = {"kernel_start_s": None, "cases": []}
    for name, code in cases:
        ts = time.time()
        res = k.execute(code, timeout=180)
        dt = time.time() - ts
        out["cases"].append(
            {"case": name, "ok": bool(res.ok), "seconds": round(dt, 2),
             "stdout": (res.stdout or "").strip()[-300:], "error": res.error}
        )
        print(f"[probe] {name}: ok={res.ok} {dt:.2f}s stdout={(res.stdout or '').strip()[:120]!r}", flush=True)

    out["kernel_start_s"] = round(time.time() - t0, 2)
    k.shutdown()
    print("[probe] JSON " + json.dumps(out, ensure_ascii=False), flush=True)
    return 0 if all(c["ok"] for c in out["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
