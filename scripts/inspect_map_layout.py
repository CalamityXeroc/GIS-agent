# -*- coding: utf-8 -*-
"""版面体检 CLI：量测 .aprx（可选配合导出的 JPG）。

用法:
    python scripts/inspect_map_layout.py <工程.aprx> [图.jpg]
退出码 0=通过，1=有不通过项（便于挂进 CI / 基准）。
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gis_cli.cartography.qc import check_layout, summarize  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    aprx = sys.argv[1]
    image = sys.argv[2] if len(sys.argv) > 2 else ""
    spec = None
    spec_path = Path(aprx).with_suffix(".spec.json")
    if spec_path.exists():
        try:
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
            spec = payload.get("spec")
        except Exception:
            spec = None
    result = check_layout(aprx, spec=spec, image_path=image)
    print(f"工程: {aprx}")
    if image:
        print(f"图片: {image}")
    for check in result.get("checks") or []:
        mark = "OK  " if check.get("ok") else "FAIL"
        print(f"  [{mark}] {check.get('name'):<26} {str(check.get('detail'))[:110]}")
    print("-" * 70)
    print(f"结论: {'全部通过' if result.get('ok') else summarize(result)}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
