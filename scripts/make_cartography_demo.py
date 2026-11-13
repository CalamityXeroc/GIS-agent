# -*- coding: utf-8 -*-
"""制图规则引擎端到端示例：5 种图种各出一张（自动版式 + 版面体检）。

用法: python scripts/make_cartography_demo.py [--only 1,3]
输出: workspace/成果展示/制图设计示例/
"""
import argparse
import glob
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import arcpy  # noqa: E402

arcpy.env.overwriteOutput = True

from gis_cli.cartography import create_map  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "workspace" / "成果展示" / "制图设计示例"


def latest(workspace_glob: str) -> Path:
    found = sorted(glob.glob(str(REPO / workspace_glob)))
    if not found:
        raise SystemExit(f"找不到数据: {workspace_glob}")
    return Path(found[-1])


def prepare_hospital_prj(gdb: str) -> str:
    """把医院点投影到 4548（与社区同坐标系），用于"点位+面"叠加图。"""
    source = glob.glob(str(REPO / "workspace/input/**/B下午/B下午/数据/hospital.shp"), recursive=True)
    if not source:
        return ""
    target = gdb + "\\hospital_prj"
    if not arcpy.Exists(target):
        arcpy.management.Project(source[0], target, arcpy.SpatialReference(4548))
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="只跑指定序号，如 1,3")
    args = parser.parse_args()
    only = {int(x) for x in args.only.split(",") if x.strip().isdigit()}

    task1 = latest("benchmark/runs/*/pm_task1_elderly_raster/workspace")
    task2b = latest("benchmark/runs/*/pm_task2b_service_value_rank/workspace")
    task2c = latest("benchmark/runs/*/pm_task2c_map_eval_communities/workspace")
    gdb2b = str(task2b / "output" / "result_data.gdb")
    gdb2c = str(task2c / "output" / "result_data.gdb")

    cases = [
        {
            "id": 1, "name": "分级设色·社区老年人口",
            "layers": [gdb2b + "\\community_service"],
            "intent": {"theme": "老年人口", "scale": "社区尺度", "field": "ELDER_COMM",
                       "legend_labels": "semantic"},
        },
        {
            "id": 2, "name": "分类设色·社区绿地评价",
            "layers": [gdb2c + "\\community_eval"],
            "intent": {"theme": "社区绿地服务评价", "field": "类别",
                       "category_colors": "标杆社区:#1F77B4;需整改社区:#2CA02C",
                       "legend_title": "社区类型"},
        },
        {
            "id": 3, "name": "栅格·老年人口分布",
            "layers": [gdb2b + "\\elder_fill"],
            "intent": {"theme": "老年人口", "color_ramp": "YlOrRd"},
        },
        {
            "id": 4, "name": "点位+面·医院与社区",
            "layers": [gdb2b + "\\community_prj", prepare_hospital_prj(gdb2b)],
            "intent": {"theme": "医疗设施与社区分布", "mode": "multi_layer", "title": "医院点位与社区分布图"},
        },
        {
            "id": 5, "name": "单一符号·社区服务圈",
            "layers": [gdb2b + "\\comm_buffer_500"],
            "intent": {"theme": "社区500米服务圈", "mode": "single"},
        },
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in cases:
        if only and case["id"] not in only:
            continue
        layers = [path for path in case["layers"] if path and arcpy.Exists(path)]
        if not layers:
            print(f"[SKIP] {case['name']}: 数据不存在")
            continue
        output = str(OUT_DIR / f"图{case['id']}_{case['name'].split('·')[-1]}.jpg")
        print(f"\n=== [{case['id']}] {case['name']} ===")
        started = time.time()
        try:
            result = create_map(layers, case["intent"], output_path=output)
        except Exception as exc:
            print(f"  [FAIL] {type(exc).__name__}: {str(exc)[:200]}")
            summary.append({"id": case["id"], "name": case["name"], "ok": False, "error": str(exc)[:200]})
            continue
        elapsed = time.time() - started
        print(f"  输出: {result['output']} ({result.get('size', 0) / 1024:.0f} KB, {elapsed:.1f}s)")
        print(f"  工程: {result['aprx']}")
        print("  设计:" + "\n    ".join(str(result.get("design_note") or "").split("｜")))
        print(f"  体检: {'通过' if result.get('qc_ok') else '未通过'} | {result.get('qc_summary')}")
        print(f"  修复轮次: {result.get('repair_rounds')}")
        for note in (result.get("notes") or [])[:10]:
            print(f"    · {note}")
        spec_path = Path(result["output"]).with_suffix(".spec.json")
        spec_path.write_text(json.dumps(
            {"spec": result.get("spec"), "qc": result.get("qc"), "notes": result.get("notes")},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  规格: {spec_path.name}")
        summary.append({
            "id": case["id"], "name": case["name"], "ok": bool(result.get("qc_ok")),
            "output": result["output"], "aprx": result["aprx"],
            "design_note": result.get("design_note"), "qc_summary": result.get("qc_summary"),
            "repair_rounds": result.get("repair_rounds"), "seconds": round(elapsed, 1),
        })
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 汇总 ===")
    for item in summary:
        print(f"  [{('OK ' if item.get('ok') else 'BAD')}] 图{item['id']} {item['name']}: {item.get('design_note', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())