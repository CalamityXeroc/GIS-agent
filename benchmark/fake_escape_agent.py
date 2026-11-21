# -*- coding: utf-8 -*-
"""假 agent：模拟"被测进程往宿主 src/ 写影子包"（只被 guard_selftest 调用）。

真事故（DAIR-62）的形状：``src/arcpy/{__init__.py,mp.py}``，与依赖同名。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo = Path(sys.argv[1])
    pkg = repo / "src" / "arcpy"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text('"""fake arcpy bridge"""\n__bridged__ = True\n', encoding="utf-8")
    (pkg / "mp.py").write_text("BRIDGED = True\n", encoding="utf-8")
    print(f"[fake-agent] wrote {pkg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
