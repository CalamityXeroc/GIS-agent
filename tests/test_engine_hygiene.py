# -*- coding: utf-8 -*-
"""交付卫生与类别命名测试（不需要 ArcGIS）。

对应两处实测问题（第 14 届基准）：
1. 过程数据占交付量 96%：temp_data.gdb 里 4 份 dem_mosaic* 重做残留 + output 里的 .pkl；
2. 类别编码被当图例名用（土地覆盖 1/2/5/7/8/11 应显示 水域/林地/耕地/建筑区/裸地/牧场）。
"""
from __future__ import annotations

from gis_cli.runtime.hygiene import hygiene_report
from gis_cli.runtime.mpl_setup import CJK_FAMILIES, CJK_FONT_FILES, available_font_files


def _inventory(temp_items, result_items=("landcover_fixed",), files=(), temp_bytes=50_000_000, result_bytes=5_000_000):
    return {
        "gdb": {
            "temp_data.gdb": {"items": list(temp_items), "bytes": temp_bytes},
            "result_data.gdb": {"items": list(result_items), "bytes": result_bytes},
        },
        "files": list(files),
    }


def test_hygiene_flags_duplicate_families_in_temp():
    """同前缀的多个中间件（重做残留）必须被判为问题。"""
    problems, warnings, detail = hygiene_report(_inventory(["dem_mosaic_wgs84", "dem_mosaic", "dem_mosaic_utm"]))
    assert any("重做残留" in p and "dem_mosaic" in p for p in problems), problems
    assert "GDB=2" in detail


def test_hygiene_allows_multiyear_names_in_result():
    """结果库里的 city_pop_2010 / city_pop_2020 属于合理多期成果，不得误报。"""
    problems, _warnings, _detail = hygiene_report(
        _inventory(["landcover_utm10"], result_items=["city_pop_2010", "city_pop_2020", "city_pop_1990"])
    )
    assert not problems, problems


def test_hygiene_flags_process_files_and_ratio_warning():
    problems, warnings, _detail = hygiene_report(
        _inventory(["dem_mosaic"], files=["清洗报告.md", "_meteo_clean.pkl"], temp_bytes=159_000_000, result_bytes=4_400_000),
        ratio_limit=5.0,
    )
    assert any(".pkl" in p for p in problems), problems
    assert any("倍" in w for w in warnings), warnings


def test_hygiene_flags_probe_artifacts():
    """回归：模型“试一下”留下的探查件要判**问题**（不只是提醒）。

    实测（14 届 4a-2）：为验证 create_map 能不能用，模型在成果目录里建了
    ``_test.jpg`` / ``_test1.jpg`` / ``_test2.jpg``，出完图忘了删。
    考卷明确要求“提交时删除非必要过程数据”。

    以前只记 warning，而基准断言只把 problems 判失败 —— 结果 14 届四a
    那次交付目录里躺着 14 个 ``_m0.aprx`` / ``_q1.jpg`` / ``_r1.jpg``，
    卫生检查照样报“无问题”。下划线开头不是正常交付物的命名习惯，判问题。

    真实数据名不能误伤（test_area.shp 不报）。
    """
    problems, warnings, _detail = hygiene_report(
        _inventory(["year_centers"], files=[
            "分布中心迁移.jpg", "_test1.jpg", "_test2.jpg", "test_area.shp",
        ])
    )
    assert any("_test1.jpg" in p for p in problems), problems
    assert not any("test_area" in p for p in problems), ("真数据不该被误报", problems)
    assert not any("test_area" in w for w in warnings), ("真数据不该被误报", warnings)


def test_hygiene_ok_for_clean_delivery():
    problems, warnings, _detail = hygiene_report(
        _inventory(["landcover_utm10"], files=["分类面积统计.md"], temp_bytes=6_000_000, result_bytes=5_000_000)
    )
    assert not problems and not warnings, (problems, warnings)


def test_deliverable_hygiene_assertion_reads_runner_inventory():
    """断言层：通过 code_runner 拿到目录清单后按同样判据判定。"""
    from gis_cli.recipes.store import RecipeLibrary

    class _Outcome:
        ok = True
        error = ""

        def __init__(self, result):
            self.result = result

    class _Runner:
        def __init__(self, result):
            self.result = result

        def run(self, code, timeout=0):
            return _Outcome(self.result)

    lib = RecipeLibrary()
    lib.code_runner = _Runner(_inventory(["dem_mosaic", "dem_mosaic_utm"], files=["_meteo_clean.pkl"]))
    bad = lib.run_assertions([{"type": "deliverable_hygiene_ok", "path": "output"}])[0]
    assert bad.ok is False
    assert "重做残留" in bad.detail or ".pkl" in bad.detail

    lib.code_runner = _Runner(_inventory(["landcover_utm10"], files=["分类面积统计.md"]))
    good = lib.run_assertions([{"type": "deliverable_hygiene_ok", "path": "output"}])[0]
    assert good.ok is True, good.detail


def test_category_names_parsing_and_spec():
    """值→显示名映射的解析，以及它进入渲染规格（供 apply 层设图例标签）。"""
    from gis_cli.cartography.apply import _parse_name_map

    assert _parse_name_map("1:水域;2:林地") == {"1": "水域", "2": "林地"}
    assert _parse_name_map("1=水域") == {"1": "水域"}
    assert _parse_name_map({"1": "水域"}) == {"1": "水域"}
    assert _parse_name_map("") == {}

    from gis_cli.cartography.design import design_layout

    facts = {
        "layers": [
            {
                "path": "x/landcover_fixed",
                "name": "landcover_fixed",
                "geometry": "Raster",
                "feature_count": None,
                "fields": ["Value"],
                "crs_wkid": 32610,
                "extent": {"xmin": 0, "ymin": 0, "xmax": 1000, "ymax": 1000},
                "field_stats": {"Value": {"distinct": 3, "null_rate": 0.0}},
                "renderer_hint": "unique",
            }
        ],
        "extent": {"xmin": 0, "ymin": 0, "xmax": 1000, "ymax": 1000},
        "crs_wkid": 32610,
    }
    spec = design_layout(facts, {"renderer": {"mode": "unique", "field": "Value"}, "category_names": "1:水域;2:林地"})
    assert "category_names" in spec["renderer"] or "category_names" in str(spec)


def test_mpl_font_candidates_are_sane():
    """字体候选里不能出现 SimSun-ExtB（只有生僻字，会整片变方框）。"""
    assert all("simsunb" not in path.lower() for path in CJK_FONT_FILES)
    assert "Microsoft YaHei" in CJK_FAMILIES
    assert isinstance(available_font_files(), list)


def test_hygiene_does_not_flag_distinct_purposes_with_shared_prefix():
    """同前缀但用途不同的中间件不得误报（实测任务二a：tmp_idw_train / tmp_idw_mask）。"""
    problems, _warnings, _detail = hygiene_report(
        _inventory(["tmp_idw_train", "tmp_idw_mask", "tmp_idw2", "tmp_idw2_mask", "train_pts", "check_pts"])
    )
    assert not problems, problems


def test_hygiene_flags_prefix_chain_as_residue():
    """真正的重做残留：基础名是其余名字的前缀（dem_mosaic ⊂ dem_mosaic_wgs84 ⊂ dem_mosaic_utm）。"""
    problems, _warnings, _detail = hygiene_report(
        _inventory(["dem_mosaic", "dem_mosaic_wgs84", "dem_mosaic_utm", "landcover_utm10"])
    )
    assert any("dem_mosaic" in p and "重做残留" in p for p in problems), problems
    assert not any("landcover_utm10" in p for p in problems), problems


def test_image_input_builds_data_url_without_pillow(tmp_path):
    """小图直接编码为 data URL（无需 Pillow）；非图片与空值返回空串。"""
    from gis_cli.runtime.image_input import as_data_url, is_image_path

    png = tmp_path / "tiny.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    url = as_data_url(str(png))
    assert url.startswith("data:image/png;base64,")
    assert as_data_url("") == ""
    assert as_data_url(str(tmp_path / "nope.jpg")) == ""
    assert as_data_url(str(tmp_path)) == ""
    assert is_image_path("a.JPG") and not is_image_path("a.txt")


class _VisionLLM:
    """假识图模型：固定返回乱码判定。"""

    class _Cfg:
        vision = True
        model = "fake-vision"

    config = _Cfg()

    def chat(self, messages, **kwargs):
        from gis_cli.engine.llm_client import LLMResponse

        multimodal = [m for m in messages if isinstance(m.get("content"), list)]
        assert multimodal, "识图消息必须包含多模态内容"
        assert any(part.get("type") == "image_url" for part in multimodal[0]["content"])
        return LLMResponse(
            content='{"text_ok": false, "issues": ["中文显示为方框"], "summary": "图面文字不可读"}',
            model="fake-vision",
        )


def test_verifier_vision_check_warns_or_blocks(tmp_path):
    """识图质检：warn 只提醒，block 计入验收失败。"""
    from gis_cli.engine.verifier import Verifier

    img = tmp_path / "map.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 300)

    warn = Verifier(workspace=tmp_path, llm_client=_VisionLLM(), vision_mode="warn")
    verdict = warn.vision_check_image(img)
    assert verdict["ok"] is False and "方框" in " ".join(verdict["issues"])

    off = Verifier(workspace=tmp_path, llm_client=_VisionLLM(), vision_mode="off")
    assert off.vision_check_image(img).get("skipped") is True


def test_shared_legend_bounds_reads_decisions_and_notes(tmp_path):
    """回归：分级边界写在 decisions/notes 里时也要能认出来。

    实测（14 届 4a-1 实测）：制图配方写出的 .spec.json 顶层**没有** renderer 键，
    分级藏在 decisions（explicit_bounds）与 notes（raster_classify: [...]）里。
    检查器只认 renderer.applied_bounds，于是四张图明明是统一图例却报
    “这些图没有分级信息（未统一图例）”——而这正是 benchmark 里那条断言的判据。
    """
    import json

    from gis_cli.recipes.store import RecipeLibrary

    specs = tmp_path / "制图"
    specs.mkdir()
    # 一份走 decisions，一份走 notes：两种写法都要能认出
    (specs / "图1.spec.json").write_text(
        json.dumps(
            {"decisions": [{"key": "explicit_bounds", "value": [0, 2, 4, 6, 8]}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (specs / "图2.spec.json").write_text(
        json.dumps(
            {"notes": ["raster_classify(统一图例): [0.0, 2.0, 4.0, 6.0, 8.0]"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lib = RecipeLibrary()
    good = lib.run_assertions(
        [{"type": "shared_legend_bounds", "path": str(specs), "value": {"min_files": 2}}]
    )[0]
    assert good.ok is True, good.detail
    assert "2 份图共用 1 套分级" in good.detail, good.detail


def test_map_layout_ok_routes_through_code_runner(tmp_path):
    """回归：版面体检要交给 code_runner（内核）跑，而不是在 venv 里 import arcpy。

    实测（14 届 4a-2）：benchmark/CLI 跑在没有 arcpy 的 venv 里，check_layout
    直接报“版面体检失败: No module named 'arcpy'”而判失败；而同一个检查在
    ArcGIS 解释器里是能跑的。现在有 code_runner 时改走它。
    """
    from pathlib import Path

    from gis_cli.recipes.store import RecipeLibrary

    class _Result:
        ok = True
        result = {"verdict": {"ok": True, "checks": []}, "summary": "版面体检通过"}
        error = None

    class _Runner:
        def __init__(self):
            self.code = ""

        def run(self, code, timeout=None, workspace=None):
            self.code = code
            return _Result()

    runner = _Runner()
    lib = RecipeLibrary(code_runner=runner)
    aprx = tmp_path / "map.aprx"
    aprx.write_text("x", encoding="utf-8")

    result = lib.run_assertions(
        [{"type": "map_layout_ok", "path": str(aprx), "image": str(tmp_path / "map.jpg")}]
    )[0]
    assert result.ok is True, result.detail
    assert "check_layout" in runner.code, "必须让 code_runner 去跑 check_layout"
    assert "arcpy" not in runner.code, "不要把 arcpy 直接写进模板（由 qc 内部 import）"


def test_shared_legend_bounds_assertion(tmp_path):
    """统一图例断言：多份 spec 的实测分级必须完全一致（4 期密度图对比的硬判据）。"""
    import json

    from gis_cli.recipes.store import RecipeLibrary

    specs = tmp_path / "制图"
    specs.mkdir()
    for index, bounds in enumerate(([0, 50, 100, 200], [0, 50, 100, 200]), start=1):
        (specs / f"图{index}.spec.json").write_text(
            json.dumps({"renderer": {"applied_bounds": bounds}}, ensure_ascii=False), encoding="utf-8"
        )
    lib = RecipeLibrary()
    good = lib.run_assertions(
        [{"type": "shared_legend_bounds", "path": str(specs), "value": {"min_files": 2}}]
    )[0]
    assert good.ok is True, good.detail

    # 混入一份不同分级 → 必须失败
    (specs / "图3.spec.json").write_text(
        json.dumps({"renderer": {"applied_bounds": [0, 10, 20, 30]}}, ensure_ascii=False), encoding="utf-8"
    )
    bad = lib.run_assertions(
        [{"type": "shared_legend_bounds", "path": str(specs), "value": {"min_files": 2}}]
    )[0]
    assert bad.ok is False and "共用 2 套分级" in bad.detail, bad.detail


def test_shared_legend_bounds_requires_enough_files(tmp_path):
    """图不够时不能静默放行（否则“统一图例”根本没被验证）。"""
    from gis_cli.recipes.store import RecipeLibrary

    empty = tmp_path / "空目录"
    empty.mkdir()
    result = RecipeLibrary().run_assertions(
        [{"type": "shared_legend_bounds", "path": str(empty), "value": {"min_files": 4}}]
    )[0]
    assert result.ok is False
    assert "少于要求的 4 份" in result.detail


def test_class_bounds_flow_into_renderer_spec():
    """class_bounds 应进入渲染规格并改成 Manual 分级（级数=上界个数）。"""
    from gis_cli.cartography.design import design_layout

    facts = {
        "layers": [
            {
                "path": "x/density",
                "name": "density",
                "geometry": "Raster",
                "feature_count": None,
                "fields": ["Value"],
                "crs_wkid": 32610,
                "extent": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
                "field_stats": {"Value": {"min": 0.0, "max": 400.0, "distinct": 100, "null_rate": 0.0}},
                "renderer_hint": "graduated",
            }
        ],
        "extent": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
        "crs_wkid": 32610,
    }
    spec = design_layout(
        facts,
        {"renderer": {"mode": "graduated", "field": "Value"}, "class_bounds": "0,50,100,200,400"},
    )
    renderer = spec["renderer"]
    assert renderer["explicit_bounds"] == [0.0, 50.0, 100.0, 200.0, 400.0]
    assert renderer["classification_method"] == "Manual"
    assert renderer["class_count"] == 4  # 5 个边界 → 4 级（边界语义）


def test_normalize_intent_keeps_custom_keys():
    """normalize_intent 不能丢自定义键——否则 class_bounds / category_names 静默失效。"""
    from gis_cli.cartography.design import normalize_intent

    normalized = normalize_intent(
        {"theme": " 老年人口 ", "class_bounds": "0,50,100", "category_names": "1:水域;2:林地"}
    )
    assert normalized["theme"] == "老年人口"
    assert normalized["class_bounds"] == "0,50,100"
    assert normalized["category_names"] == "1:水域;2:林地"


def test_class_bounds_edges_semantics():
    """分级边界语义：n 个边界 → n-1 级（避免首级区间倒置）。"""
    from gis_cli.cartography.design import design_layout

    facts = {
        "layers": [
            {
                "path": "x/raster",
                "name": "raster",
                "geometry": "Raster",
                "feature_count": None,
                "fields": ["Value"],
                "crs_wkid": 32610,
                "extent": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
                "field_stats": {"Value": {"min": 1201.0, "max": 3019.0, "distinct": 100, "null_rate": 0.0}},
                "renderer_hint": "graduated",
            }
        ],
        "extent": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
        "crs_wkid": 32610,
    }
    spec = design_layout(facts, {"renderer": {"mode": "graduated", "field": "Value"}, "class_bounds": "1200,1500,1800"})
    renderer = spec["renderer"]
    assert renderer["explicit_bounds"] == [1200.0, 1500.0, 1800.0]
    assert renderer["class_count"] == 2  # 3 个边界 → 2 级
    assert renderer["classification_method"] == "Manual"


def test_stamp_arrows_accepts_both_key_spellings(tmp_path):
    """回归：箭头坐标两种写法都要认，并在一张都用不了时报错（不假装成功）。

    实测（14 届 4a-2）：stamp_arrows 只认 ``from``/``to``，而任务提示词与
    create_map 工具描述给的是 ``x0/y0/x1/y1``。模型照提示词写，get("from")
    取不到就默认 [0, 0]，19 条箭头全画到原点，drawn 却报 19——“成功”了但图上
    什么都没有，模型只能反复改参数去猜。
    """
    from PIL import Image

    from gis_cli.cartography.image_overlay import stamp_arrows

    img_path = tmp_path / "map.jpg"
    Image.new("RGB", (400, 400), "white").save(img_path)
    extent = {"xmin": 0, "ymin": 0, "xmax": 1000, "ymax": 1000, "width": 1000, "height": 1000}
    box = (0.0, 0.0, 100.0, 100.0)

    for arrow in (
        {"from": [100, 100], "to": [500, 500], "label": "1982→1983"},
        {"x0": 100, "y0": 100, "x1": 500, "y1": 500, "label": "1982→1983"},
    ):
        result = stamp_arrows(
            str(img_path), [arrow], extent=extent, box_mm=box, page_height_mm=100.0, dpi=25.4
        )
        assert result.get("ok") is True, result
        assert result.get("drawn") == 1, result

    # 坐标完全缺失 → 必须报失败，而不是画到原点还报成功
    bad = stamp_arrows(
        str(img_path), [{"label": "x"}], extent=extent, box_mm=box, page_height_mm=100.0, dpi=25.4
    )
    assert bad.get("ok") is False, bad
    assert "坐标" in str(bad.get("detail")), bad


def test_layer_paths_made_absolute(monkeypatch, tmp_path):
    """回归：传给 ArcGIS 的图层路径必须是绝对路径。

    实测（14 届 4a-2）：addDataFromPath("input/study_area.shp") 传相对路径时，
    ArcGIS 不把它当本地文件，而当成 Web 服务去解析，直接报

        RuntimeError: AUTOMATIC 失败，可能需要提供更具体的 web_service_type。

    换成绝对路径就正常——而模型习惯写相对路径（基准文档就这么写的）。
    """
    from pathlib import Path

    from gis_cli.cartography import _absolute_layers
    from gis_cli.cartography.apply import _as_absolute

    monkeypatch.chdir(tmp_path)

    # create_map 入口：相对 → 绝对，绝对保持，空串丢弃，非字符串原样保留
    got = _absolute_layers(["input/a.shp", "", "  "])
    assert len(got) == 1, got
    assert Path(got[0]).is_absolute(), got
    assert got[0].endswith("a.shp"), got

    already = str(tmp_path / "b.shp")
    assert _absolute_layers([already])[0] == already

    class _Layer:
        pass

    obj = _Layer()
    assert _absolute_layers([obj])[0] is obj

    # render 调用点：同样的解析（两处都做，任一处漏掉都会炸）
    assert Path(_as_absolute("input/c.shp")).is_absolute()
    assert _as_absolute(already) == already
    assert _as_absolute("") == ""


def test_map_to_px_coordinate_mapping():
    """数据坐标 → 像素：含 y 翻转与 mm→px 换算（迁移图箭头定位的数学核心）。"""
    from gis_cli.cartography.image_overlay import map_to_px

    extent = {"xmin": 0, "ymin": 0, "xmax": 1000, "ymax": 2000, "width": 1000, "height": 2000}
    box = (10.0, 20.0, 100.0, 200.0)  # x, y(左下), w, h (mm)
    mm2px = 254 / 25.4
    corner, opposite, center = map_to_px(
        [(0, 0), (1000, 2000), (500, 1000)], extent, box, page_height_mm=297.0, dpi=254.0
    )
    assert abs(corner[0] - 10 * mm2px) < 0.01 and abs(corner[1] - (297 - 20) * mm2px) < 0.01
    assert abs(opposite[0] - 110 * mm2px) < 0.01 and abs(opposite[1] - (297 - 220) * mm2px) < 0.01
    assert abs(center[0] - 60 * mm2px) < 0.01 and abs(center[1] - (297 - 120) * mm2px) < 0.01


def test_map_to_px_uses_measured_extent():
    """换算必须支持实测范围（相机微调过的 extent），而不是只吃设计值。"""
    from gis_cli.cartography.image_overlay import map_to_px

    box = (0.0, 0.0, 100.0, 100.0)
    narrow = map_to_px([(5.0, 5.0)], {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}, box,
                       page_height_mm=100.0, dpi=25.4)
    wide = map_to_px([(5.0, 5.0)], {"xmin": -5, "ymin": -5, "xmax": 15, "ymax": 15}, box,
                     page_height_mm=100.0, dpi=25.4)
    # 同一个数据点，范围越大 → 越靠近图框中心
    assert narrow[0][0] == 50.0 and wide[0][0] == 50.0
    assert abs(narrow[0][1] - 50.0) < 0.01 and abs(wide[0][1] - 50.0) < 0.01


def test_raster_class_counts_between():
    """分类栅格按值计数区间断言（土地覆盖更新面积核对用）。"""
    from gis_cli.recipes.store import RecipeLibrary

    class _Outcome:
        ok = True
        error = ""

        def __init__(self, result):
            self.result = result

    class _Runner:
        def __init__(self, result):
            self.result = result

        def run(self, code, timeout=0):
            return _Outcome(self.result)

    lib = RecipeLibrary()
    lib.code_runner = _Runner({"counts": {"1": 143430, "7": 46240}})
    ok = lib.run_assertions([
        {"type": "raster_class_counts_between", "path": "gdb://lc", "scale": 0.0001,
         "value": {"1": [10, 20], "7": [4, 6]}}
    ])[0]
    assert ok.ok is True, ok.detail
    bad = lib.run_assertions([
        {"type": "raster_class_counts_between", "path": "gdb://lc", "scale": 0.0001,
         "value": {"1": [1, 5]}}
    ])[0]
    assert bad.ok is False and "不在" in bad.detail


def test_create_map_tool_exposes_full_intent_surface(tmp_path):
    """回归：create_map 工具要把渲染/分级/叠加/版式参数都暴露出来。

    实测（14 届 4a-1/4a-2）：模型为了知道怎么设栅格分级、色带、overlay，
    用 execute_code 读了 214 次我们自己的源码（cartography/*.py）。
    根因是这些参数只存在于"直接写代码"的路径里，工具面上没有。
    工具描述里也点名"这些参数就是全部接口，不要去读源码"。
    """
    from pathlib import Path

    from gis_cli.engine.tools import EngineContext, build_default_registry

    captured: dict = {}

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            captured["code"] = code

            class R:
                ok = True
                result = {"output": "out.jpg", "aprx": "out.aprx", "qc_ok": True, "qc_summary": ""}
                error = None
                stdout = ""
                stderr = ""

            return R()

    registry = build_default_registry()
    tool = [t for t in registry._tools.values() if t.name == "create_map"][0]
    props = set((tool.parameters or {}).get("properties") or {})
    for key in (
        "mode", "explicit_bounds", "class_bounds", "classification", "color_ramp",
        "category_names", "category_colors", "overlay", "overrides",
    ):
        assert key in props, f"create_map 缺少参数 {key}"

    ctx = EngineContext(workspace=Path("."), code_runner=_Runner())
    registry.execute(
        "create_map",
        {
            "layers": ["input/a.shp"],
            "output_path": "out.jpg",
            "mode": "raster_stretch",
            "explicit_bounds": "0,2,4,6,8",
            "color_ramp": "Purples",
            "overlay": {"arrows": [{"from": [1, 2], "to": [3, 4], "label": "a"}]},
            "overrides": {"scale_bar.element_width_mm": 55},
        },
        ctx,
    )
    code = captured["code"]
    # 这些键必须真的进到 create_map 的 intent 里（不是只在 schema 里摆着）
    for token in ("raster_stretch", "0,2,4,6,8", "Purples", "arrows", "element_width_mm"):
        assert token in code, f"{token} 没有传到 create_map"


def test_raster_range_falls_back_when_describe_min_missing():
    """回归：Describe 取不到 min 时要退到 raster_stats（同一产物不能两个结论）。

    实测（14 届 4a-1 回归）：同一个栅格，raster_not_constant 拿到 range=None
    而紧随其后的 raster_stat_between 拿到 max=6.012。前者走 Describe（当时
    一个 try 包住 MINIMUM+MAXIMUM，MIN 抛异常把 MAX 也置 None），后者走
    _raster_stats。同一条产物两个矛盾结论 = 误判成品为坏。
    """
    from gis_cli.recipes.store import RecipeLibrary

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            class R:
                ok = True
                error = None

            r = R()
            if "GetRasterProperties" in code:
                # Describe 路径：MINIMUM 失败 → 只有 max
                r.result = {"exists": True, "raster_min": None, "raster_max": 6.012}
            else:
                # RasterToNumPyArray 路径正常
                r.result = {"sum": 1.0, "min": 0.0, "max": 6.012, "mean": 0.5, "count": 10}
            return r

    lib = RecipeLibrary(code_runner=_Runner())
    assert lib._raster_range("x.gdb/kde") == (0.0, 6.012)
    result = lib.run_assertions([{"type": "raster_not_constant", "path": "x.gdb/kde"}])[0]
    assert result.ok is True, result.detail


def test_hygiene_sees_files_in_subdirectories():
    """交付图常放 output/成果图/ 下，只扫顶层会漏掉里面的过程残留。

    实测 14 届四a：顶层只有 2 个文件，成果图/ 里还有 14 个
    ``_m0.aprx`` / ``_q1.jpg`` / ``_r1.jpg``，卫生检查报“无问题、散文件=2”。
    """
    problems, _warnings, detail = hygiene_report(
        _inventory(["landcover"], files=["成果图/_m0.aprx", "成果图/密度_1982_1986.jpg"])
    )
    assert any("_m0.aprx" in p for p in problems), problems
    assert "散文件=2" in detail, detail
