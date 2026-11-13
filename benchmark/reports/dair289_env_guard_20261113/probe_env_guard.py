# -*- coding: utf-8 -*-
"""DAIR-289 复验：跑前 env 非空时，配方第 1 步 ``CopyFeatures`` 不再被带偏。

被验版本：HEAD（DAIR-289 改动）。对照版本：``d643274a``（修复前，必须红）。
环境：Windows + ArcGIS Pro（arcpy）+ 引擎 venv；真数据在
``benchmark/reports/p0-1c_dual_backend_20261102/work``（county4508.shp × rasterA_4508.tif，
两者都是 EPSG:4508，栅格 cell=1000）。

断言（errors 为空才 exit 0）：

1. **预置**：绕过内核的按次隔离（走 ``_execute_raw``）把
   ``outputCoordinateSystem = EPSG:4326``、``extent = (0,0,1000,1000)``（与 county4508
   的 EPSG:4508 范围完全不交）塞进内核 —— 模拟"别的任务留下的状态"；
2. **负向（红→绿）**：这个预置态下跑 ``zone_raster_sum``（county4508 × rasterA_4508，
   ``zid``）。修复前第 1 步 ``CopyFeatures`` 就被重投影+裁空，最后报
   ``ValueError: zero-size array to reduction operation maximum``（DAIR-288 原文）；
   修复后必须成功；（另一类预置 —— 例如 32610 那种投影后完全不相交的 —— 在 Pro 3.7 上
   会在 CopyFeatures 直接报 ERROR 000210/080467，同样是第 1 步被带偏，见 README）
3. 预置态下的 ``zone_count`` / ``total_value`` 与干净 env 下**逐值相同**；
4. 跑后 env 逐字段回到**预置值**（不是清空，也不是留成干净基线）；
5. **正向**：干净 env 下跑完 env 仍逐字段是 None，紧接着同内核跑
   ``zonal_statistics``（county4508 × rasterA_4508、``zid``、``SUM;MEAN``）必须成功、
   ``record_count = 385``（= DAIR-134 R1 / DAIR-288 验过的那条链）；
6. **防假阴性**：打印并断言 ``gis_cli.__file__`` 落在被验那棵树里。
   DAIR-288 踩过：临时树里相对 ``PYTHONPATH`` 会 import 到仓库 editable 安装的那份，
   用例全过是假阳性。

命令::

  cd <树>；PYTHONPATH=<树>/src .venv/Scripts/python.exe <树>/benchmark/reports/dair289_env_guard_20261113/probe_env_guard.py

对照树（``d643274a``）用 ``git archive`` 解出来跑；那棵树里 shp/tif 没入版本库，
得把 ``DAIR289_DATA`` 指回工作树的 ``benchmark/reports/p0-1c_dual_backend_20261102/work``。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = Path(
    os.environ.get(
        "DAIR289_DATA",
        str(REPO / "benchmark" / "reports" / "p0-1c_dual_backend_20261102" / "work"),
    )
)
WORK = Path(
    os.environ.get("DAIR289_WORK", r"C:/Users/cheng/AppData/Local/Temp/dair289_env_guard/work")
).resolve()
OUT = WORK / "out"
EVIDENCE = Path(os.environ.get("DAIR289_EVIDENCE", str(HERE / "evidence_env_guard.json")))

sys.path.insert(0, str(REPO / "src"))

import gis_cli  # noqa: E402
from gis_cli.arcpy_bridge import discover_arcgis_pro_python  # noqa: E402
from gis_cli.recipes.store import RecipeLibrary  # noqa: E402
from gis_cli.runtime.code_runner import CodeRunner  # noqa: E402

#: 读 env：走 ``_execute_raw``（**绕过** execute 的按次隔离），读到的才是内核真实状态
READ_CODE = r'''
import arcpy, json as _json
_e = arcpy.env.extent
def _brief(v):
    return None if v is None else str(v)
def _cs(v):
    # str(SpatialReference) 是 "<geoprocessing spatial reference object object at 0x…>"，
    # 带地址的字符串没法跨次比，取 wkid。
    if v is None:
        return None
    try:
        return int(v.factoryCode)
    except Exception:
        return str(v)
print("ENVJSON" + _json.dumps({
    "extent": None if _e is None else [round(float(v), 3) for v in
                                       (_e.XMin, _e.YMin, _e.XMax, _e.YMax)],
    "snapRaster": _brief(arcpy.env.snapRaster),
    "outputCoordinateSystem": _cs(arcpy.env.outputCoordinateSystem),
    "cellSize": _brief(arcpy.env.cellSize),
    "mask": _brief(arcpy.env.mask),
    "overwriteOutput": bool(arcpy.env.overwriteOutput),
}))
'''

#: 预置"别的任务留下的" env：另一套 CRS（WGS84 地理坐标）+ 与 county4508 完全不相交的范围。
#: 这组预置在 d643274a 上复现出 DAIR-288 的原文错误：
#: ``ValueError: zero-size array to reduction operation maximum``。
PRESET_CODE = r'''
import arcpy
arcpy.env.extent = arcpy.Extent(0.0, 0.0, 1000.0, 1000.0)
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(4326)
print("PRESET-OK")
'''

CLEAN_CODE = r'''
import arcpy
arcpy.env.extent = None
arcpy.env.cellSize = None
arcpy.env.snapRaster = None
arcpy.env.mask = None
arcpy.env.outputCoordinateSystem = None
print("CLEAN-OK")
'''

#: 干净基线：extent/cellSize/snapRaster/mask/outputCoordinateSystem 全 None
_CLEAN_ENV = {
    "extent": None,
    "snapRaster": None,
    "outputCoordinateSystem": None,
    "cellSize": None,
    "mask": None,
}


def main() -> int:
    pro = discover_arcgis_pro_python()
    OUT.mkdir(parents=True, exist_ok=True)
    zone_path = DATA / "county4508.shp"
    raster_path = DATA / "rasterA_4508.tif"
    gdb = OUT / "chain.gdb"

    runner = CodeRunner(workspace=WORK, python_executable=pro.python_executable, exec_timeout=900)
    kernel = runner._ensure_kernel()  # 预置/读 env 要用裸执行通道（见模块说明）
    lib = RecipeLibrary(code_runner=runner)
    lib.load()
    ctx = SimpleNamespace(workspace=WORK, code_runner=runner, exec_timeout=900)

    state: dict = {
        "arcpy_python": pro.python_executable,
        "gis_cli_file": str(Path(gis_cli.__file__).resolve()),
        "zone_path": str(zone_path),
        "raster_path": str(raster_path),
        "errors": [],
    }

    def check(cond: bool, message: str) -> None:
        if not cond:
            state["errors"].append(message)
            print(f"[FAIL] {message}")
        else:
            print(f"[ ok ] {message}")

    def raw(code: str) -> str:
        out = kernel._execute_raw(code, timeout=600)
        if not out.ok:
            raise RuntimeError(f"裸执行失败: {out.error}")
        return out.stdout or ""

    def env_state() -> dict:
        line = [ln for ln in raw(READ_CODE).splitlines() if ln.startswith("ENVJSON")][-1]
        return json.loads(line[len("ENVJSON"):])

    def run_zone(out_name: str) -> dict:
        obs = lib.run(
            "zone_raster_sum",
            {
                "zone_path": str(zone_path),
                "zone_field": "zid",
                "raster_path": str(raster_path),
                "output_table": str(gdb / out_name),
            },
            ctx,
            timeout=900,
        )
        return {
            "ok": obs.ok,
            "summary": (obs.summary or "")[-600:],
            "result": (obs.data or {}).get("result") or {},
        }

    try:
        # 6) 防假阴性：确认跑在被验的那棵树上
        check(Path(state["gis_cli_file"]).is_relative_to(REPO / "src"),
              f"gis_cli 来自被验树 {REPO / 'src'}（实际 {state['gis_cli_file']}）")
        check(zone_path.exists() and raster_path.exists(), f"派生输入存在 {DATA}")

        if not gdb.exists():
            made = runner.run(
                f"import arcpy\narcpy.management.CreateFileGDB({str(OUT)!r}, 'chain.gdb')\n"
                "set_result({'ok': True})\n",
                timeout=300,
            )
            assert made.ok, made.error

        # 1) 干净起点
        raw(CLEAN_CODE)
        state["env_start"] = env_state()

        # 2) 预置污染态（绕过按次隔离）
        raw(PRESET_CODE)
        state["env_preset"] = env_state()
        check(state["env_preset"]["extent"] == [0.0, 0.0, 1000.0, 1000.0]
              and state["env_preset"]["outputCoordinateSystem"] == 4326,
              f"预置生效（与 county4508 不相交的 (0,0,1000,1000) + EPSG:4326）: {state['env_preset']}")

        # 3) 负向：预置态下跑 zone_raster_sum（修复前正是在这里红）
        t0 = time.time()
        state["run_poisoned"] = run_zone("zrs_poisoned")
        state["run_poisoned"]["wall_s"] = round(time.time() - t0, 1)
        check(state["run_poisoned"]["ok"],
              f"预置污染态下 zone_raster_sum 成功（修复前报 zero-size array）: {state['run_poisoned']['summary']}")

        # 4) 跑后 env 必须回到预置值
        state["env_after_poisoned_run"] = env_state()
        check(state["env_after_poisoned_run"] == state["env_preset"],
              f"跑后 env 逐字段回到预置值（不是清空）: {state['env_after_poisoned_run']}")

        # 5) 干净 env 下的基线（同一个配方、同一输入）
        raw(CLEAN_CODE)
        state["env_clean"] = env_state()
        check(all(state["env_clean"][k] is None for k in _CLEAN_ENV),
              f"干净基线全 None: {state['env_clean']}")
        state["run_clean"] = run_zone("zrs_clean")
        check(state["run_clean"]["ok"],
              f"干净 env 下 zone_raster_sum 成功: {state['run_clean']['summary']}")

        # 6) 预置态 vs 干净 env：逐值一致
        p, c = state["run_poisoned"]["result"], state["run_clean"]["result"]
        state["compare"] = {
            "zone_count": [p.get("zone_count"), c.get("zone_count")],
            "total_value": [p.get("total_value"), c.get("total_value")],
            "cell_size": [p.get("cell_size"), c.get("cell_size")],
        }
        check(p.get("zone_count") == c.get("zone_count") and p.get("zone_count"),
              f"zone_count 与干净 env 一致: {state['compare']}")
        check(p.get("total_value") is not None
              and abs(float(p["total_value"]) - float(c["total_value"])) <= 1e-6 * max(
                  1.0, abs(float(c["total_value"]))),
              f"total_value 与干净 env 一致: {state['compare']}")

        # 7) 正向：干净 env 跑完仍是全 None，紧接 zonal_statistics 仍 = 385
        state["env_after_clean_run"] = env_state()
        check(all(state["env_after_clean_run"][k] is None for k in _CLEAN_ENV),
              f"干净 env 跑完 env 仍逐字段是 None: {state['env_after_clean_run']}")
        zonal = lib.run(
            "zonal_statistics",
            {
                "zone_path": str(zone_path),
                "zone_field": "zid",
                "raster_path": str(raster_path),
                "statistics": "SUM;MEAN",
                "output_table": str(gdb / "zonal_after_zrs"),
            },
            ctx,
            timeout=900,
        )
        state["zonal_after"] = {
            "ok": zonal.ok,
            "summary": (zonal.summary or "")[-600:],
            "result": (zonal.data or {}).get("result") or {},
        }
        check(zonal.ok, f"同内核紧跟的 zonal_statistics 成功: {state['zonal_after']['summary']}")
        rec = state["zonal_after"]["result"].get("record_count")
        state["zonal_record_count"] = rec
        check(rec == 385, f"zonal record_count={rec}，期望 385")
    except Exception as exc:  # noqa: BLE001
        import traceback

        state["errors"].append(f"probe 崩溃: {exc}")
        state["traceback"] = traceback.format_exc()[-3000:]
    finally:
        try:
            runner.shutdown()
        except Exception:  # noqa: BLE001
            pass

    state["ok"] = not state["errors"]
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in state.items() if k != "traceback"},
                     ensure_ascii=False, indent=1, default=str))
    return 0 if state["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
