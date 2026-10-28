# -*- coding: utf-8 -*-
"""成果包（P0-4）：单文件 HTML + PDF。

不需要 ArcGIS / 联网：内嵌、抽稀、人话摘要都在主进程里做，内核只在
"把 GDB 要素类读成几何" 这一步出现（用注入的 code_runner，测试里给桩）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from gis_cli.datacatalog.store import Catalog, DatasetRecord
from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.report import (
    HTML_NAME,
    MAX_HTML_BYTES,
    build_deliverables,
)
from gis_cli.report import html_pack, narrative, pdf_pack
from gis_cli.trace.recorder import RunRecorder


# ------------------------------------------------------------------ 夹具
def _geojson_layers() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                        [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]],
                    ],
                },
            },
            {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [5, 5]}},
        ],
    }


class _Req:
    def __init__(self, ident: str, text: str) -> None:
        self.id = ident
        self.text = text
        self.acceptance = ""


class _State:
    goal = "统计各街道老年人口并出图"
    final_summary = "按 12 个街道汇总了老年人口，并生成了分布图。"
    methodology = "先做空间连接再汇总\n按自然间断点分级出图"
    requirements = [_Req("r1", "按街道汇总老年人口")]
    status = "completed"

    def __init__(self, artifacts: list[str]) -> None:
        self.artifacts = artifacts


class _StubVerifier:
    def __init__(self, verdict: dict) -> None:
        self.verdict = verdict

    def verify(self, state):  # noqa: ANN001 - 与 Verifier 接口一致
        return self.verdict


def _workspace(tmp_path: Path) -> Path:
    out = tmp_path / "workspace" / "output"
    out.mkdir(parents=True, exist_ok=True)
    return tmp_path / "workspace"


def _write_image(path: Path, size=(3000, 1200), color=(200, 60, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _verdict(*paths: Path) -> dict:
    structural = []
    for path in paths:
        record = {"path": str(path), "exists": True}
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            record["image"] = {"ok": True}
        else:
            record["feature_count"] = 2
        structural.append(record)
    return {
        "pass": True,
        "summary": "验收通过",
        "structural": structural,
        "semantic": {"per_requirement": [{"id": "r1", "status": "PASS"}]},
    }


# ------------------------------------------------------------------ 几何转换
def test_esri_geometry_conversion_handles_hole_and_order():
    """Esri 的环不保证顺序：小环套在大环里就该是洞，而不是第二块面。"""
    inner = [[3, 3], [3, 7], [7, 7], [7, 3], [3, 3]]
    outer = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
    geometry = html_pack.esri_geometry_to_geojson({"rings": [inner, outer]})

    assert geometry["type"] == "Polygon"
    assert len(geometry["coordinates"]) == 2  # 外圈 + 一个洞
    assert geometry["coordinates"][0][0] == [0.0, 0.0]


def test_esri_geometry_conversion_disjoint_rings_become_multipolygon():
    a = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
    b = [[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]]
    geometry = html_pack.esri_geometry_to_geojson({"rings": [a, b]})

    assert geometry["type"] == "MultiPolygon"
    assert len(geometry["coordinates"]) == 2


def test_esri_geometry_conversion_point_line_and_z_drop():
    point = html_pack.esri_geometry_to_geojson({"x": 1, "y": 2})
    assert point == {"type": "Point", "coordinates": [1.0, 2.0]}
    line = html_pack.esri_geometry_to_geojson({"paths": [[[0, 0, 5], [1, 1], [2, 2]]]})
    assert line == {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]}
    assert html_pack.esri_geometry_to_geojson({"paths": [[[0, "坏"]]]}) is None


def test_layer_probe_only_uses_safe_kernel_api():
    """持久内核里 MakeFeatureLayer/SelectLayerByAttribute 会挂死，别碰。"""
    code = html_pack.layer_probe_code("output/结果.gdb/街道")
    assert "SearchCursor" in code and "SHAPE@JSON" in code
    assert "MakeFeatureLayer" not in code and "SelectLayerByAttribute" not in code


def test_load_layer_reads_geojson_file(tmp_path: Path):
    path = tmp_path / "图层.geojson"
    path.write_text(json.dumps(_geojson_layers(), ensure_ascii=False), encoding="utf-8")

    collection = html_pack.load_layer(path)

    assert collection is not None and len(collection["features"]) == 2


def test_load_layer_uses_injected_runner_for_gdb(tmp_path: Path):
    """GDB 要素类没有文件可读，几何只能问内核；内核返回 Esri JSON 字符串。"""
    class _Outcome:
        ok = True
        result = {
            "error": "",
            "geoms": [json.dumps({"rings": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]]})],
        }

    class _Runner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, code, **kwargs):  # noqa: ANN001
            self.calls += 1
            assert "SearchCursor" in code
            return _Outcome()

    runner = _Runner()
    collection = html_pack.load_layer(tmp_path / "output" / "结果.gdb" / "街道", code_runner=runner)

    assert runner.calls == 1
    assert collection is not None and len(collection["features"]) == 1
    assert collection["features"][0]["geometry"]["type"] == "Polygon"


def test_load_layer_resolves_relative_gdb_path_against_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """相对路径要变成绝对路径再进内核；GDB 内要素类靠库目录判“在不在地上”。"""
    gdb = tmp_path / "output" / "结果.gdb"
    gdb.mkdir(parents=True)
    seen: dict[str, str] = {}

    class _Outcome:
        ok = True
        result = {"error": "", "geoms": []}

    class _Runner:
        def run(self, code, **kwargs):  # noqa: ANN001
            seen["code"] = code
            return _Outcome()

    monkeypatch.chdir(tmp_path)
    html_pack.load_layer("output/结果.gdb/街道", code_runner=_Runner())

    # 探针把路径以 repr 形式写进代码（反斜杠会被转义）
    assert repr(str(gdb / "街道")) in seen["code"]


# ------------------------------------------------------------------ 抽稀与体积
def test_thin_features_respects_coordinate_budget():
    points = [[float(i), float(i % 7)] for i in range(50_000)]
    collection = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": points}}],
    }

    thinned = html_pack.thin_features(collection, max_coords=5_000)

    assert html_pack.count_points(thinned) <= 5_000 + 2
    assert html_pack.count_points(thinned) >= 2


def test_prepare_image_shrinks_to_1600_and_encodes_jpeg(tmp_path: Path):
    path = _write_image(tmp_path / "大图.png", size=(3000, 1200))

    mime, data = html_pack.prepare_image(path)

    assert mime == "image/jpeg"
    import io

    with Image.open(io.BytesIO(data)) as img:
        assert img.size == (1600, 640)


def test_prepare_image_keeps_png_when_alpha(tmp_path: Path):
    path = tmp_path / "透明.png"
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(path)

    mime, _data = html_pack.prepare_image(path)

    assert mime == "image/png"


def _write_noise_image(path: Path, size=(1600, 900)) -> Path:
    """随机噪声图：JPEG 压不动，用来逼体积上限真的生效。"""
    import random

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = random.Random(7).randbytes(size[0] * size[1] * 3)
    Image.frombytes("RGB", size, raw).save(path, format="PNG")
    return path


def test_build_deliverables_drops_images_to_stay_under_byte_limit(tmp_path: Path):
    ws = _workspace(tmp_path)
    images = [_write_noise_image(ws / "output" / f"图{i}.png") for i in range(3)]
    assert images[0].stat().st_size > 1_000_000  # 确认噪声图确实大

    result = build_deliverables(
        workspace=ws,
        state=_State([str(p) for p in images]),
        verdict=_verdict(*images),
        want_pdf=False,
        max_bytes=400_000,
    )

    assert result.html is not None
    assert result.html.stat().st_size <= 400_000
    assert result.images < len(images)
    assert any("未内嵌" in note for note in result.notes)


def _write_multiring_layer(path: Path, *, features: int = 200, rings: int = 40) -> Path:
    """多环面图层：一堆小环，抽稀压不下去，只能靠“少画一层”来压体积。"""
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [r * 1e-3, 0.0],
                            [r * 1e-3, 5e-4],
                            [r * 1e-3 + 5e-4, 5e-4],
                            [r * 1e-3 + 5e-4, 0.0],
                            [r * 1e-3, 0.0],
                        ]
                        for r in range(rings)
                    ],
                },
            }
            for _ in range(features)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_deliverables_drops_layers_to_stay_under_byte_limit(tmp_path: Path):
    """图丢光了还超限：继续丢图层，写清丢了哪层，层数不能虚报（DAIR-34）。"""
    ws = _workspace(tmp_path)
    images = [_write_noise_image(ws / "output" / f"图{i}.png") for i in range(2)]
    layers = [_write_multiring_layer(ws / "output" / f"多环面{i}.geojson") for i in range(4)]

    result = build_deliverables(
        workspace=ws,
        state=_State([str(p) for p in images + layers]),
        verdict=_verdict(*images, *layers),
        want_pdf=False,
        max_bytes=2_000_000,
    )

    html = result.html.read_text(encoding="utf-8")
    assert result.html.stat().st_size <= 2_000_000
    assert result.images == 0
    # 报的层数要和页面上真画出来的层一致（不能写“内联 4 个图层”却只画 2 个）
    assert 0 < result.layers < len(layers)
    assert html.count('"geojson":') == result.layers
    dropped = [note for note in result.notes if "图层" in note and "未内联" in note]
    assert len(dropped) == len(layers) - result.layers
    assert any("已按体积上限裁剪" in note for note in result.notes)
    # 量过的字节就是写出去的字节：Windows 上 write_text 默认会把 \n 翻成 \r\n
    assert b"\r\n" not in result.html.read_bytes()


# ------------------------------------------------------------------ HTML
def test_html_is_single_file_without_any_external_request(tmp_path: Path):
    ws = _workspace(tmp_path)
    image = _write_image(ws / "output" / "成果图.png")
    layer_path = ws / "output" / "街区.geojson"
    layer_path.write_text(json.dumps(_geojson_layers(), ensure_ascii=False), encoding="utf-8")

    result = build_deliverables(
        workspace=ws,
        state=_State([str(image), str(layer_path)]),
        verdict=_verdict(image, layer_path),
        want_pdf=False,
    )

    html = result.html.read_text(encoding="utf-8")
    assert result.html.name == HTML_NAME
    assert "http://" not in html and "https://" not in html
    assert "data:image/jpeg;base64," in html
    assert 'id="map"' in html and "evenodd" in html
    assert "window.__LAYERS__" in html
    assert narrative.DISCLAIMER in html
    assert result.layers == 1 and result.images == 1
    assert result.html.stat().st_size <= MAX_HTML_BYTES


def test_unreadable_layer_note_reaches_the_page(tmp_path: Path):
    """图层导不出几何时，页面上要写清楚，不能默默少画一层。"""
    ws = _workspace(tmp_path)
    layer = ws / "output" / "结果.gdb" / "街道"
    (ws / "output" / "结果.gdb").mkdir(parents=True, exist_ok=True)
    layer.write_text("", encoding="utf-8")  # 没有内核可问，几何取不到

    result = build_deliverables(
        workspace=ws,
        state=_State([str(layer)]),
        verdict={"pass": True},
        want_pdf=False,
    )

    html = result.html.read_text(encoding="utf-8")
    assert result.layers == 0
    assert "导不出几何" in html


def test_no_artifacts_means_no_report(tmp_path: Path):
    ws = _workspace(tmp_path)

    result = build_deliverables(workspace=ws, state=_State([]), verdict={"pass": True}, want_pdf=False)

    assert result.paths == []
    assert not (ws / "output" / HTML_NAME).exists()


# ------------------------------------------------------------------ PDF
def test_pdf_is_skipped_gracefully_without_matplotlib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _boom():
        raise ImportError("没有 matplotlib")

    monkeypatch.setattr(pdf_pack, "_import_pyplot", _boom)
    content = narrative.build_content(goal="任务", summary="结论")

    assert pdf_pack.write_pdf(content, out_path=tmp_path / "x.pdf") is None

    ws = _workspace(tmp_path)
    image = _write_image(ws / "output" / "成果图.png")
    result = build_deliverables(
        workspace=ws,
        state=_State([str(image)]),
        verdict=_verdict(image),
        want_pdf=True,
    )
    assert result.pdf is None
    assert any("PDF" in note for note in result.notes)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("matplotlib") is None,
    reason="本机没装 matplotlib（extras 可选依赖）",
)
def test_pdf_wrap_keeps_numbers_intact_and_loses_nothing():
    text = "整体上老年人口集中在中心城区，外围街道占比不足 10%，详见 output/result.csv。"
    lines = pdf_pack.wrap_text(text, limit=28)

    assert any("10%" in line for line in lines)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("matplotlib") is None,
    reason="本机没装 matplotlib（extras 可选依赖）",
)
def test_pdf_written_when_matplotlib_available(tmp_path: Path):
    content = narrative.build_content(goal="任务", summary="一句话结论", methodology="先汇总再制图")
    path = pdf_pack.write_pdf(content, out_path=tmp_path / "报告.pdf")

    assert path is not None and path.stat().st_size > 1000
    assert path.read_bytes()[:4] == b"%PDF"


# ------------------------------------------------------------------ 人话摘要
def test_plain_verifications_avoid_assertion_jargon(tmp_path: Path):
    ws = _workspace(tmp_path)
    image = _write_image(ws / "output" / "成果图.png")
    verdict = _verdict(image)

    lines = narrative.plain_verifications(verdict, [_Req("r1", "按街道汇总老年人口")])
    joined = "".join(lines)

    assert "不是空白图" in joined
    assert "已满足" in joined
    for jargon in ("断言", "feature_count", "PASS", "structural"):
        assert jargon not in joined


def test_key_numbers_use_measured_feature_counts(tmp_path: Path):
    numbers = narrative.key_numbers(
        [],
        [
            {"path": "a", "exists": True, "feature_count": 1200},
            {"path": "b", "exists": True, "feature_count": 48},
            {"path": "c", "exists": True, "kind": "RasterDataset", "raster_min": 0.0, "raster_max": 6.1},
        ],
    )
    by_label = {item.label: item for item in numbers}

    assert by_label["矢量图层"].value == "2 个"
    assert "1,248" in by_label["矢量图层"].note
    assert "0 ~ 6.1" in by_label["栅格成果"].note


def test_recipes_from_trace_pairs_calls_and_merges_retries(tmp_path: Path):
    trace = tmp_path / "run.jsonl"
    events = [
        {"seq": 1, "event": "tool_call", "payload": {"tool": "run_recipe", "args": {"recipe_id": "buffer_dissolve", "params": {"distance": 500}}}},
        {"seq": 2, "event": "tool_result", "payload": {"tool": "run_recipe", "ok": False}},
        {"seq": 3, "event": "tool_call", "payload": {"tool": "run_recipe", "args": {"recipe_id": "buffer_dissolve", "params": {"distance": 500}}}},
        {"seq": 4, "event": "tool_result", "payload": {"tool": "run_recipe", "ok": True}},
        {"seq": 5, "event": "tool_call", "payload": {"tool": "execute_code", "args": {"code": "print(1)"}}},
        "半行坏 JSON",
    ]
    trace.write_text("\n".join(json.dumps(e, ensure_ascii=False) if isinstance(e, dict) else e for e in events), encoding="utf-8")

    calls = narrative.recipes_from_trace(trace)

    assert len(calls) == 1
    assert calls[0].recipe_id == "buffer_dissolve"
    assert calls[0].count == 2 and calls[0].ok is True
    assert "buffer_dissolve" in calls[0].describe()


def test_source_rows_read_catalog_provenance(tmp_path: Path):
    ws = _workspace(tmp_path)
    catalog = Catalog(tmp_path / ".gis_agent" / "catalog.db")
    catalog.upsert(
        DatasetRecord(
            path=str(ws / "input" / "人口.csv"),
            kind="table",
            name="人口",
            source="csv_upload",
            license="用户自有",
            acquired_at="2026-10-01T08:00:00+00:00",
        )
    )
    catalog.upsert(DatasetRecord(path=str(ws / "output" / "成果.gdb" / "汇总"), kind="vector", name="汇总"))

    rows = narrative.source_rows(catalog, ws)

    assert [row.name for row in rows] == ["人口"]
    assert rows[0].source == "用户上传的表格/点位"
    assert rows[0].license == "用户自有"
    assert rows[0].acquired_at.startswith("2026-10-01")


# ------------------------------------------------------------------ 与 finish 挂钩
def test_loop_generates_report_after_successful_finish(tmp_path: Path):
    ws = _workspace(tmp_path)
    image = _write_image(ws / "output" / "成果图.png")

    class _LLM:
        class _Cfg:
            model = "mock"

        config = _Cfg()

        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, **kwargs):  # noqa: ANN001
            self.calls += 1
            args = {"summary": "完成", "methodology": "先汇总再出图", "artifacts": [str(image)]}
            return LLMResponse(
                tool_calls=[ToolCall(id="c1", name="finish", arguments=args, raw_arguments=json.dumps(args))],
                model="mock",
            )

    loop = AgentLoop(
        workspace=ws,
        llm_client=_LLM(),
        registry=build_default_registry(),
        context=EngineContext(workspace=ws),
        verifier=_StubVerifier(_verdict(image)),
        recorder=RunRecorder("run_report", ws / "traces"),
        config=LoopConfig(max_turns=3, plan_card=False),
        auto_extract_requirements=False,
    )

    state = loop.run("出图")

    assert state.status == "completed"
    html = ws / "output" / HTML_NAME
    assert html.exists()
    assert str(html) in state.artifacts


def test_loop_can_disable_report(tmp_path: Path):
    ws = _workspace(tmp_path)
    image = _write_image(ws / "output" / "成果图.png")

    class _LLM:
        class _Cfg:
            model = "mock"

        config = _Cfg()

        def chat(self, messages, **kwargs):  # noqa: ANN001
            args = {"summary": "完成", "artifacts": [str(image)]}
            return LLMResponse(
                tool_calls=[ToolCall(id="c1", name="finish", arguments=args, raw_arguments=json.dumps(args))],
                model="mock",
            )

    loop = AgentLoop(
        workspace=ws,
        llm_client=_LLM(),
        registry=build_default_registry(),
        context=EngineContext(workspace=ws),
        verifier=_StubVerifier(_verdict(image)),
        recorder=RunRecorder("run_report_off", ws / "traces"),
        config=LoopConfig(max_turns=3, plan_card=False, report_enabled=False),
        auto_extract_requirements=False,
    )

    state = loop.run("出图")

    assert state.status == "completed"
    assert not (ws / "output" / HTML_NAME).exists()
