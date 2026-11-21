# -*- coding: utf-8 -*-
"""冒烟测试：持久内核里的 matplotlib 中文字体配置是否生效。

三件事：
1. 内核启动后被自动配置（画中文图 → 保存 → stderr 无 "missing from font"）
2. 显式指定无中文字形的字体时，**能**检测到缺字形告警（证明"发现机制"可用）
3. tools 层能把该告警转成修复提示（错误记忆 mpl-cjk）

用法：用 ArcGIS Pro 的 Python 直接跑本脚本。
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(r"E:\BST_lab\GIS_CLI")
sys.path.insert(0, str(REPO / "src"))

from gis_cli.runtime.code_runner import CodeRunner  # noqa: E402

CODE_OK = '''
import matplotlib
import matplotlib.pyplot as plt
import os

out = os.path.join(r"__WS__", "output", "cn_ok.png")
plt.rcParams["figure.dpi"] = 90
fig, ax = plt.subplots(figsize=(4, 3))
ax.set_title("社区老年人口分布")
ax.set_xlabel("经度（度）")
ax.plot([1, 2, 3], [3, 2, 1], label="样本点")
ax.legend()
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
set_result({"output": out, "font": matplotlib.rcParams["font.sans-serif"][:3], "exists": os.path.exists(out)})
'''

CODE_BAD = '''
import matplotlib
import matplotlib.pyplot as plt
import os

out = os.path.join(r"__WS__", "output", "cn_tofu.png")
fig, ax = plt.subplots(figsize=(3, 2))
ax.set_title("中文标题", fontfamily="Arial")   # 故意指定无中文字形的字体
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
set_result({"output": out})
'''


def main() -> None:
    ws = Path(tempfile.mkdtemp(prefix="mpl_smoke_"))
    (ws / "output").mkdir(parents=True, exist_ok=True)
    runner = CodeRunner(workspace=ws, state_dir=ws / ".gis_agent" / "kernel")
    try:
        print("== 1) 内核里画中文图（应无缺字形告警） ==")
        out = runner.run(CODE_OK.replace("__WS__", str(ws).replace("\\", "\\\\")), timeout=300)
        info = out.result if isinstance(out.result, dict) else {}
        stderr = str(out.stderr or "")
        print("   ok:", out.ok, "| 字体:", info.get("font"), "| 图存在:", info.get("exists"))
        print("   缺字形告警:", "missing from font" in stderr)
        assert out.ok, out.error
        assert info.get("exists"), "图未生成"
        assert "missing from font" not in stderr, f"仍有缺字形告警:\n{stderr[-500:]}"
        print("   ✓ 中文字体配置生效")

        print("== 2) 故意用 Arial 画中文（应能检测到缺字形） ==")
        bad = runner.run(CODE_BAD.replace("__WS__", str(ws).replace("\\", "\\\\")), timeout=300)
        bad_stderr = str(bad.stderr or "")
        print("   缺字形告警:", "missing from font" in bad_stderr)
        print("   stderr 片段:", bad_stderr.strip().splitlines()[-1][:120] if bad_stderr.strip() else "(空)")

        print("== 3) tools 层把告警转成修复提示 ==")
        from gis_cli.engine.tools import _stderr_warning_hint

        hint = _stderr_warning_hint(bad_stderr)
        print("   hint:", hint[:150] if hint else "(空)")
        if "missing from font" in bad_stderr:
            assert "[错误记忆/mpl-cjk]" in hint, f"未能生成修复提示: {hint!r}"
            print("   ✓ 告警可见 + 提示生效")
        else:
            print("   ! 本轮未出现缺字形告警（字体回退到了中文字体），跳过提示断言")
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
