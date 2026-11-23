# -*- coding: utf-8 -*-
"""转录预览：把一段真实形态的事件流渲染出来，用于肉眼验收与快照对比。

用法::

    PYTHONPATH=src python scripts/preview_transcript.py            # 彩色/纯文本
    PYTHONPATH=src python scripts/preview_transcript.py --png out.png   # 渲染成图片（用 ArcGIS 解释器跑）

事件序列刻意还原用户实际遇到过的场景：
连续 12 次目录查询、一次长推理、代码执行、自动修复（长根因）、验收与最终答复。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gis_cli.ui import events as ev  # noqa: E402
from gis_cli.ui import render as rd  # noqa: E402
from gis_cli.ui import theme as th  # noqa: E402

REASONING = (
    "There are task documents in the input folder. Let me read the B上午试题.pdf and "
    "B下午试题.pdf, plus the README, to understand what the user wants.\n"
    "Then I will check the data catalog for the layers mentioned in the task "
    "(study_area, community, green_land, hospital, pop_density) and plan the workflow."
)


def emit_events() -> list[tuple[str, dict]]:
    items = [
        ("run_start", {"goal": "统计 animal_points.shp 里 NAME 字段各物种的点位数，并写出 CSV", "run_id": "run_demo"}),
        ("turn", {"turn": 1, "tasklist": ""}),
        (
            "llm_result",
            {
                "model": "deepseek-flash",
                "actions": [{"tool": "update_tasklist", "reasoning": REASONING, "content": ""}],
                "usage": {"prompt_tokens": 8787, "completion_tokens": 628},
            },
        ),
        ("tool_call", {"tool": "update_tasklist", "args": {"items": "3"}}),
        ("tool_result", {"tool": "update_tasklist", "ok": True, "summary": "任务清单已更新"}),
        ("tool_call", {"tool": "catalog_query", "args": {"keyword": "animal_points"}}),
        ("tool_result", {"tool": "catalog_query", "ok": True, "summary": "数据目录查询：2 条匹配（关键词: animal_points）"}),
    ]
    # 用户实际遇到过的"连打 12 次目录查询"——现在应该合并成一行 ×11
    for i in range(11):
        items.append(("tool_call", {"tool": "catalog_query", "args": {"keyword": f"kw{i}"}}))
        items.append(
            ("tool_result", {"tool": "catalog_query", "ok": True, "summary": f"数据目录查询：{i + 3} 条匹配（关键词: kw{i}）"})
        )
    items += [
        ("turn", {"turn": 2, "tasklist": ""}),
        (
            "llm_result",
            {
                "model": "deepseek-flash",
                "actions": [
                    {
                        "tool": "execute_code",
                        "reasoning": "Now count points per species with a SearchCursor and write a UTF-8 CSV.",
                        "content": "",
                    }
                ],
                "usage": {"prompt_tokens": 12014, "completion_tokens": 785},
            },
        ),
        ("tool_call", {"tool": "execute_code", "args": {"code": "import arcpy\n", "description": "按 NAME 分组统计"}}),
        (
            "tool_result",
            {"tool": "execute_code", "ok": True, "summary": "执行成功（22973 ms）：按 NAME 分组统计并写出 UTF-8 CSV；产出 1 个文件"},
        ),
        (
            "repair_attempt",
            {
                "attempt": 2,
                "error": (
                    "报错的根因：arcpy.management.GetRasterProperties 的 property_type 参数不支持 'SUM'"
                    "（只支持 MAXIMUM/MINIMUM/MEAN/STD/TOP/... 等），因此 ERROR 000800，需改用 "
                    "RasterToNumPyArray 后自行求和，同时注意 nodata 处理与坐标系一致性。"
                ),
            },
        ),
        ("turn", {"turn": 3, "tasklist": ""}),
        (
            "llm_result",
            {
                "model": "deepseek-flash",
                "actions": [{"tool": "finish", "reasoning": "Everything verified; deliver the answer.", "content": ""}],
                "usage": {"prompt_tokens": 15339, "completion_tokens": 3923},
            },
        ),
        ("verification", {"pass": True, "summary": "验收通过（7/7 断言）"}),
        (
            "tool_result",
            {
                "tool": "finish",
                "ok": True,
                "summary": (
                    "验收通过。已完成 NAME 字段物种点位数统计并写出 output/物种点位数.csv。\n"
                    "【前三个物种】1. Giraffe — 1792  2. Gazelle — 1159  3. Impala — 361\n"
                    "【口径】投影坐标系 EPSG:32610，空值单独计数，非空合计+空值=要素总数。"
                ),
            },
        ),
    ]
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="system", choices=["system", "dark", "light"])
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--png", default="", help="渲染成 PNG（需要 PIL，用 ArcGIS 解释器）")
    args = parser.parse_args()

    console = th.make_console(
        args.mode,
        no_color=args.no_color,
        color=True if args.color else None,
        width=args.width,
    )
    tracker = ev.StatusTracker(model="deepseek-flash", workspace="workspace")
    store = rd.ToolOutputStore()
    view = rd.TranscriptView(console, tracker, store)

    if not args.png:
        console.print(
            rd.banner(
                model="deepseek-flash",
                api_base="https://api.deepseek.com",
                workspace="workspace",
                catalog="vector 24、table 27",
                recipes=47,
                session_id="20261011-121134",
                kernel="ready",
                version=th.engine_version(),
            )
        )
        console.print(rd.user_block("统计 animal_points 里各物种点位数"))
        console.print()

    for name, payload in emit_events():
        tracker.apply(name, payload)
        view.handle(name, payload)
    view.flush()
    if not args.png:
        console.print()
        console.print(rd.footer_text(tracker, width=args.width))
        console.print(rd.hint_line())

    if args.png:
        html = console.export_html(inline_styles=True)
        Path(args.png).with_suffix(".html").write_text(html, encoding="utf-8")
        print(f"已导出 {Path(args.png).with_suffix('.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
