# -*- coding: utf-8 -*-
"""界面预览：把一整套 UI 元素渲染出来（用于肉眼验收与回归）。

用法::

    # 看彩色效果（在支持颜色的终端里）
    PYTHONPATH=src python scripts/preview_ui.py

    # 看纯文本版（老终端/重定向到文件）
    PYTHONPATH=src python scripts/preview_ui.py --no-color --width 100 > preview.txt

演示内容：启动横幅 → 工具动作/结果 → 修复 → 验收 → 结果卡片 → 状态行。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gis_cli.ui import events as ev  # noqa: E402
from gis_cli.ui import render as rd  # noqa: E402
from gis_cli.ui import theme as th  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="system", choices=["system", "dark", "light"])
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--width", type=int, default=0)
    args = parser.parse_args()

    console = th.make_console(args.mode, no_color=args.no_color, width=args.width or None)

    console.print(
        rd.banner(
            model="glm-5.3-flash",
            api_base="https://tokenrhythm.studio/v1",
            workspace="./workspace",
            catalog="2 图层 · 3596 行属性",
            recipes=47,
            session_id="20261003-1045",
            kernel="ready",
            version="0.7.1",
        )
    )
    console.print(rd.render_unit(ev.RenderUnit("note", "抽取验收需求 3 条", lines=["r1 点位清洗", "r2 核密度", "r3 出图"])))
    console.print()
    console.print(rd.render_unit(ev.RenderUnit("step", "catalog_query", lines=["keyword=animal_points"]), elapsed=0.3))
    console.print(rd.render_unit(ev.RenderUnit("result", "数据目录查询：2 条匹配", ok=True)))
    console.print(rd.render_unit(ev.RenderUnit("step", "run_recipe", lines=["recipe_id=kernel_density periods=1982-1986,…"]), elapsed=12.4))
    console.print(rd.render_unit(ev.RenderUnit("result", "4 期核密度 → output/result_data.gdb/kde_*", ok=True)))
    console.print(rd.render_unit(ev.RenderUnit("step", "graduated_colors_map", lines=["class_bounds=0,2,4,6,8"]), elapsed=8.1))
    console.print(rd.render_unit(ev.RenderUnit("warn", "自动修复第 1 次", lines=["KeyError '_oh' → 改用显式变量"])))
    console.print(rd.render_unit(ev.RenderUnit("result", "密度_1997_2001.jpg（图例 0-2/2-4/4-6/6-8）", ok=True)))
    console.print(rd.render_unit(ev.RenderUnit("note", "附上成图自检（1 张）", lines=["密度_1997_2001.jpg"])))
    console.print(rd.render_unit(ev.RenderUnit("verify", "验收通过（7/7 断言）", ok=True)))
    console.print()
    console.print(
        rd.result_card(
            {
                "status": "completed",
                "turn": 24,
                "tasklist_digest": "- [x] 四期核密度\n- [x] 统一图例出图\n- [x] 自检与验收",
                "artifacts": [
                    "output/成果图/密度_1982_1986.jpg",
                    "output/成果图/密度_1997_2001.jpg",
                    "output/成果图/密度_1997_2001.aprx",
                ],
                "methodology": "核密度（邻域 100km² → 半径 √(A/π)≈5642m，像元 100m）；四图共用分级 0/2/4/6/8",
                "final_summary": "已交付 4 张密度图（中文与比例尺可读），共用同一图例。",
            }
        )
    )

    tracker = ev.StatusTracker(model="glm-5.3-flash", status="running", turn=24, tool_count=11, tokens_in=31800, tokens_out=6420)
    console.print(rd.footer_text(tracker, width=args.width))
    console.print(rd.hint_line(busy=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
