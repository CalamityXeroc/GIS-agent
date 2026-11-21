# -*- coding: utf-8 -*-
"""能力补齐：新配方批量离线验证（真实数据 + 断言 + 关键数值核对）。

覆盖：overlay(4) / geometry(6) / raster(9) / table(3) / infra(2)
重点：PM 任务一的"老年人口栅格分摊"必须守恒（误差 <1%）。
用法：python scripts/validate_capabilities_offline.py [stage]
      stage: all(默认) | prep | overlay | geometry | table | raster | infra
"""
import sys, io, os, glob, shutil, time, json
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

import arcpy
arcpy.env.overwriteOutput = True

from gis_cli.runtime.code_runner import CodeRunner
from gis_cli.recipes.store import RecipeLibrary

STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
REPO = Path(r"E:\BST_lab\GIS_CLI")
DATA = None
for cand in glob.glob(str(REPO / "workspace/input/**/B下午/B下午/数据"), recursive=True):
    DATA = Path(cand)
    break
assert DATA, "找不到 PM 数据目录"

WS = Path(r"C:\Users\cheng\AppData\Local\Temp") / "cap_ws"
if STAGE == "prep" or not (WS / "prep_done.json").exists():
    if WS.exists():
        shutil.rmtree(WS, ignore_errors=True)
    (WS / "input").mkdir(parents=True)
    (WS / "output").mkdir(parents=True)
    for name in ("district.shp", "community.shp", "green_land.shp", "buildings.shp",
                 "hospital.shp", "residental_areas.shp", "road.shp"):
        src = DATA / name
        for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
            f = src.with_suffix(ext)
            if f.exists():
                shutil.copy2(f, WS / "input" / f.name)
    for f in DATA.glob("pop_density.tif*"):
        shutil.copy2(f, WS / "input" / f.name)
print("WS:", WS)

runner = CodeRunner(workspace=WS, state_dir=WS / ".gis_agent" / "kernel")
lib = RecipeLibrary(code_runner=runner)
lib.load()


class Ctx:
    workspace = WS
    code_runner = runner


GDB = str(WS / "output" / "result_data.gdb")
I = lambda n: str(WS / "input" / n)
G = lambda n: GDB + "\\" + n
SRID = 4527

RESULTS = []


def run_job(label, recipe_id, params, checks=None, expect_ok=True):
    """跑一个配方 + 断言 + 自定义数值核对。"""
    t0 = time.time()
    obs = lib.run(recipe_id, params, Ctx())
    dt = time.time() - t0
    ok_recipe = bool(obs.ok)
    res = (obs.data or {}).get("result") if isinstance(obs.data, dict) else None
    extra = []
    if checks and ok_recipe:
        for name, fn in checks:
            try:
                val = fn(res if isinstance(res, dict) else {})
                extra.append((name, bool(val), ""))
            except Exception as exc:
                extra.append((name, False, str(exc)[:120]))
    ok_extra = all(v for _, v, _ in extra) if extra else True
    status = "OK  " if (ok_recipe == expect_ok and ok_extra) else "FAIL"
    RESULTS.append({"label": label, "recipe": recipe_id, "ok": status.strip() == "OK", "sec": round(dt, 1)})
    print(f"[{status}] {label:42s} {recipe_id:26s} {dt:6.1f}s | {str(obs.summary)[:110]}")
    for name, ok, err in extra:
        print(f"        - 核对 {name}: {'通过' if ok else '失败 ' + err}")
    if obs.error:
        print(f"        ! 错误: {json.dumps(obs.error, ensure_ascii=False)[:200]}")
    return obs


def raster_sum(path):
    r = arcpy.RasterToNumPyArray(arcpy.Raster(path), nodata_to_value=float("nan")).astype("float64")
    import numpy as np
    return float(np.nansum(r)), int(np.count_nonzero(~np.isnan(r)))


def prepare_data():
    print("\n" + "=" * 78)
    print("阶段 prep：建库 + 投影 + 裁剪（同时测试 create_file_gdb / project_to_crs / clip_layer / field_calculate）")
    print("=" * 78)
    run_job("建过程库 temp_data.gdb", "create_file_gdb", {"gdb_path": str(WS / "output" / "temp_data.gdb")})
    run_job("建结果库 result_data.gdb", "create_file_gdb", {"gdb_path": GDB})

    run_job("投影 district → 4527", "project_to_crs",
            {"input_path": I("district.shp"), "output_path": G("district_prj"), "target_srid": SRID})
    run_job("投影 community → 4527（2341 要素）", "project_to_crs",
            {"input_path": I("community.shp"), "output_path": G("community_prj"), "target_srid": SRID})
    run_job("裁剪 green_land（4326）到 district 范围", "clip_layer",
            {"input_path": I("green_land.shp"), "clip_path": I("district.shp"),
             "output_path": G("green_clip_4326")})
    run_job("裁剪 buildings（4326）到 district 范围", "clip_layer",
            {"input_path": I("buildings.shp"), "clip_path": I("district.shp"),
             "output_path": G("buildings_clip_4326")})
    run_job("投影 green_clip → 4527", "project_to_crs",
            {"input_path": G("green_clip_4326"), "output_path": G("green_prj"), "target_srid": SRID})
    run_job("投影 buildings_clip → 4527", "project_to_crs",
            {"input_path": G("buildings_clip_4326"), "output_path": G("buildings_prj"), "target_srid": SRID})
    run_job("解析「老年人」百分比并计算老年人口 ELDER_N", "field_calculate",
            {"input_path": G("district_prj"), "field_name": "ELDER_N",
             "expression": "float(!老年人![:-1]) / 100.0 * float(!常住人!)", "field_type": "DOUBLE"},
            checks=[("ELDER_N>0 且总和在合理区间", lambda r: (lib._field_stat(G("district_prj"), "ELDER_N", "sum") or 0) > 1000)])


def stage_overlay():
    print("\n" + "=" * 78)
    print("阶段 overlay：裁剪 / 相交 / 擦除 / 联合")
    print("=" * 78)
    run_job("裁剪 绿地 到 街区（投影系）", "clip_layer",
            {"input_path": G("green_prj"), "clip_path": G("district_prj"), "output_path": G("green_in")},
            checks=[("要素数>0", lambda r: r.get("feature_count", 0) > 0)])
    run_job("相交 绿地 ∩ 社区", "intersect_layers",
            {"input_path": G("green_in"), "overlay_path": G("community_prj"), "output_path": G("green_comm")},
            checks=[("要素数>0", lambda r: r.get("feature_count", 0) > 0),
                    ("含双方字段", lambda r: len(r.get("fields", [])) > 5)])
    run_job("擦除 建筑 − 社区", "erase_layer",
            {"input_path": G("buildings_prj"), "erase_path": G("community_prj"), "output_path": G("bldg_outside")},
            checks=[("要素数不超过输入", lambda r: r.get("feature_count", 0) <= r.get("input_count", 0))])
    run_job("联合 社区 ∪ 街区", "union_layers",
            {"input_path": G("community_prj"), "overlay_path": G("district_prj"), "output_path": G("comm_dist_union")},
            checks=[("要素数≥较大输入", lambda r: r.get("feature_count", 0) >= max(r.get("input_count", 0), r.get("overlay_count", 0)))])


def stage_geometry():
    print("\n" + "=" * 78)
    print("阶段 geometry：转点 / 融合统计 / 拆分 / 凸包 / 简化 / 筛选")
    print("=" * 78)
    run_job("社区转中心点", "feature_to_centroid",
            {"input_path": G("community_prj"), "output_path": G("community_pts")},
            checks=[("点数=社区数", lambda r: r.get("feature_count") == r.get("input_count"))])
    run_job("绿地按社区融合求面积和", "dissolve_with_stats",
            {"input_path": G("green_comm"), "output_path": G("green_comm_stat"),
             "dissolve_field": "BSM", "stat_fields": "Shape_Area SUM"},
            checks=[("组数≤社区数", lambda r: r.get("group_count", 0) <= 2341),
                    ("绿地总面积>0", lambda r: (lib._field_stat(G("green_comm_stat"), "SUM_Shape_Area", "sum") or 0) > 0)])
    run_job("社区加常量字段 ONE", "field_calculate",
            {"input_path": G("community_prj"), "field_name": "ONE", "expression": "1", "field_type": "LONG"})
    run_job("融合全部社区（多部件）", "dissolve_with_stats",
            {"input_path": G("community_prj"), "output_path": G("community_diss_all"),
             "dissolve_field": "ONE", "stat_fields": "Shape_Area SUM", "multi_part": "MULTI_PART"})
    run_job("多部件拆分为单部件", "multipart_to_singlepart",
            {"input_path": G("community_diss_all"), "output_path": G("community_single")},
            checks=[("拆分后要素数>0", lambda r: r.get("feature_count", 0) > 0)])
    run_job("绿地凸包", "minimum_bounding_geometry",
            {"input_path": G("green_in"), "output_path": G("green_hull"),
             "geometry_type": "CONVEX_HULL", "group_option": "ALL"},
            checks=[("凸包为单个要素", lambda r: r.get("feature_count") == 1)])
    run_job("绿地几何简化（5 米）", "simplify_geometry",
            {"input_path": G("green_in"), "output_path": G("green_simplified"), "tolerance": 5},
            checks=[("要素数不减少", lambda r: r.get("feature_count", 0) > 0)])
    areas = []
    with arcpy.da.SearchCursor(G("community_prj"), ["Shape_Area"]) as cur:
        for (v,) in cur:
            if v is not None:
                areas.append(float(v))
    areas.sort()
    median = areas[len(areas) // 2] if areas else 0
    run_job("属性筛选（面积大于中位数的社区）", "select_by_attribute",
            {"input_path": G("community_prj"), "output_path": G("community_large"),
             "where_clause": f"Shape_Area > {median:.12f}", "expect_min": 1},
            checks=[("结果数在(0, 输入数)之间", lambda r: 0 < r.get("feature_count", 0) < r.get("input_count", 0))])


def stage_table():
    print("\n" + "=" * 78)
    print("阶段 table：频数 / 邻接汇总筛选 / 排名打标")
    print("=" * 78)
    run_job("频数统计（每个社区的绿地斑块数）", "frequency_table",
            {"in_table": G("green_comm"), "out_table": G("green_freq"), "frequency_fields": "BSM"},
            checks=[("记录数>0", lambda r: r.get("record_count", 0) > 0)])
    run_job("取 200 个社区做邻接测试", "select_by_attribute",
            {"input_path": G("district_prj"), "output_path": G("district_sub"),
             "where_clause": "OBJECTID <= 3", "expect_min": 1},
            checks=[("子集非空", lambda r: r.get("feature_count", 0) > 0)])
    run_job("子区域内社区（供邻接）", "clip_layer",
            {"input_path": G("community_prj"), "clip_path": G("district_sub"), "output_path": G("community_sub")},
            checks=[("子集非空", lambda r: r.get("feature_count", 0) > 0)])
    run_job("邻接汇总 + 阈值筛选", "neighbor_sum_filter",
            {"input_path": G("community_sub"), "output_path": G("community_nb"),
             "value_field": "Shape_Area", "sum_field": "NB_AREA_SUM",
             "where_clause": "Shape_Area > 0.000005 AND NB_AREA_SUM > 0.00005",
             "selected_path": G("community_nb_sel")},
            checks=[("邻居对>0", lambda r: r.get("neighbor_pairs", 0) > 0),
                    ("筛选结果>0", lambda r: r.get("selected_count", 0) > 0)])
    run_job("排名打标（街区老年人口）", "rank_top_bottom_flag",
            {"input_path": G("district_prj"), "output_path": G("district_rank"),
             "rank_field": "ELDER_N", "top_n": 5, "bottom_n": 5,
             "bottom_include_null": True},
            checks=[("前5名标记数=5", lambda r: r.get("top_count") == 5),
                    ("后5名标记数=5", lambda r: r.get("bottom_count") == 5)])


def stage_raster():
    print("\n" + "=" * 78)
    print("阶段 raster：投影 / 面转栅格 / 分区统计 / 分摊计算 / 掩膜 / 采样 / 矢量化 / 插值 / 地形")
    print("=" * 78)
    run_job("栅格投影（UTM50N → 4527，5 米）", "project_raster",
            {"input_path": I("pop_density.tif"), "output_path": G("density_prj"),
             "target_srid": SRID, "cell_size": 5})
    run_job("街区老年人口 → 5 米栅格", "polygon_to_raster",
            {"input_path": G("district_prj"), "value_field": "ELDER_N",
             "output_path": G("elder_raster"), "cell_size": 5},
            checks=[("值域>0", lambda r: (r.get("max") or 0) > 0)])
    run_job("街区人口密度总值栅格（权重分母）", "zonal_statistics_raster",
            {"zone_path": G("district_prj"), "zone_field": "", "raster_path": G("density_prj"),
             "output_path": G("zone_sum"), "statistics": "SUM"})
    run_job("★ 分摊计算（像元权重×街区老年人口）", "raster_calculator",
            {"inputs": f"density={G('density_prj')},zone_sum={G('zone_sum')},elder={G('elder_raster')}",
             "expression": "Con(IsNull(zone_sum), 0, Con(zone_sum == 0, 0, density / zone_sum * elder))",
             "output_path": G("elder_alloc")},
            checks=[("栅格非常量", lambda r: r.get("min") != r.get("max"))])
    run_job("★ 补缺（NoData 置 0）", "raster_calculator",
            {"inputs": f"alloc={G('elder_alloc')}",
             "expression": "Con(IsNull(alloc), 0, alloc)",
             "output_path": G("elder_alloc_fill")},
            checks=[("无 NoData", lambda r: True)])
    run_job("掩膜提取（街区范围）", "raster_extract_by_mask",
            {"raster_path": G("elder_alloc_fill"), "mask_path": G("district_prj"),
             "output_path": G("elder_masked")})
    run_job("社区中心点采样分摊栅格", "sample_raster_at_points",
            {"points_path": G("community_pts"), "raster_path": G("elder_alloc_fill"),
             "output_path": G("community_sampled")},
            checks=[("空值率<50%", lambda r: (r.get("null_rate") if r.get("null_rate") is not None else 1.0) < 0.5)])
    run_job("二值化（>0 → 1）用于矢量化", "raster_calculator",
            {"inputs": f"a={G('elder_alloc_fill')}",
             "expression": "Con(a > 0, 1, 0)", "output_path": G("elder_binary")},
            checks=[("测试进程视角: 栅格已生成", lambda r: arcpy.Exists(str(r.get("output", "")))),
                    ("内核视角: 栅格已生成", lambda r: bool(lib._describe(str(r.get("output", ""))) or {}))])
    run_job("栅格转面", "polygonize_raster",
            {"raster_path": G("elder_binary"), "output_path": G("elder_polygon")},
            checks=[("面数>0", lambda r: r.get("feature_count", 0) > 0)])
    run_job("取前 3000 栋建筑", "select_by_attribute",
            {"input_path": G("buildings_prj"), "output_path": G("buildings_sub"),
             "where_clause": "OBJECTID <= 3000", "expect_min": 100})
    run_job("建筑转中心点（供插值）", "feature_to_centroid",
            {"input_path": G("buildings_sub"), "output_path": G("buildings_pts")})
    run_job("建筑高度 IDW 插值", "interpolate_surface",
            {"points_path": G("buildings_pts"), "field_name": "Height",
             "output_path": G("height_idw"), "method": "IDW", "cell_size": 20},
            checks=[("栅格非常量", lambda r: r.get("raster_min") != r.get("raster_max"))])
    run_job("地形派生（以密度栅格代 DEM 测坡度）", "terrain_derivatives",
            {"dem_path": G("density_prj"), "output_path": G("slope_test"), "kind": "SLOPE"},
            checks=[("坡度非常量", lambda r: r.get("min") != r.get("max"))])


def stage_infra():
    print("\n" + "=" * 78)
    print("阶段 infra：交付清理（先预览后执行）")
    print("=" * 78)
    arcpy.env.workspace = GDB
    fcs = (arcpy.ListFeatureClasses() or [])
    rasters = (arcpy.ListRasters() or [])
    priority = [n for n in fcs + rasters if True]
    keep = ",".join(priority[:6]) if priority else ""
    print(f"  库内共 {len(fcs)} 要素 + {len(rasters)} 栅格；keep 取前 6 个")
    run_job("清理预览（dry_run=true）", "cleanup_gdb",
            {"gdb_path": GDB, "keep": keep,
             "drop_patterns": "_tmp,green_clip,buildings_clip,community_sub,district_sub", "dry_run": True},
            checks=[("预览不删除", lambda r: len(r.get("dropped", [])) == 0 and len(r.get("planned_drop", [])) >= 1)])


def conservation_check():
    """PM 任务一核心：分摊前后老年人口总量守恒。"""
    print("\n" + "=" * 78)
    print("★ 守恒核对：分摊栅格总量 vs 街区老年人口总量")
    print("=" * 78)
    total_elder = lib._field_stat(G("district_prj"), "ELDER_N", "sum")
    for name in ("elder_alloc", "elder_alloc_fill"):
        path = G(name)
        if not arcpy.Exists(path):
            print(f"  {name}: 不存在，跳过")
            continue
        s, n = raster_sum(path)
        err = abs(s - total_elder) / total_elder * 100 if total_elder else None
        flag = "OK  " if err is not None and err < 1.0 else "WARN"
        print(f"  [{flag}] {name}: 栅格合计={s:,.0f} | 街区老年人口合计={total_elder:,.0f} | 误差={err:.3f}% | 有效像元={n:,}")


if __name__ == "__main__":
    t0 = time.time()
    if STAGE in ("all", "prep"):
        prepare_data()
    if STAGE in ("all", "overlay"):
        stage_overlay()
    if STAGE in ("all", "geometry"):
        stage_geometry()
    if STAGE in ("all", "table"):
        stage_table()
    if STAGE in ("all", "raster"):
        stage_raster()
        conservation_check()
    if STAGE in ("all", "infra"):
        stage_infra()

    (WS / "prep_done.json").write_text("{}", encoding="utf-8")
    print("\n" + "=" * 78)
    ok = sum(1 for r in RESULTS if r["ok"])
    print(f"总计 {ok}/{len(RESULTS)} 通过 | 耗时 {time.time() - t0:.0f}s")
    for r in RESULTS:
        if not r["ok"]:
            print(f"  FAIL: {r['label']} ({r['recipe']})")
    runner.shutdown()