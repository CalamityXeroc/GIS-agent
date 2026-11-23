# -*- coding: utf-8 -*-
"""溯源卡（P1-4）：数据 → 步骤 → 参数 → 验证，一页可审计。

不需要 ArcGIS / 联网：卡片只复述 catalog、trace JSONL 与 verdict 里已有的证据。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.report import PROVENANCE_NAME, build_deliverables, narrative, provenance


# ------------------------------------------------------------------ 夹具
class _Record:
    def __init__(
        self,
        path: str,
        name: str,
        source: str = "local",
        license: str = "未标注",
        acquired_at: str = "2026-10-01T09:00:00",
    ) -> None:
        self.path = path
        self.name = name
        self.source = source
        self.license = license
        self.acquired_at = acquired_at


class _Catalog:
    def __init__(self, records: list[_Record]) -> None:
        self._records = records

    def list_datasets(self, limit: int = 200) -> list[_Record]:
        return self._records[:limit]


class _BoomCatalog:
    def list_datasets(self, limit: int = 200):  # noqa: ANN201 - 故意炸
        raise RuntimeError("catalog 打不开")


class _Library:
    def get(self, recipe_id: str):
        return {
            "buffer": SimpleNamespace(name="缓冲区分析"),
            "zonal": SimpleNamespace(name="分区统计"),
        }.get(recipe_id)


class _Req:
    def __init__(self, ident: str, text: str) -> None:
        self.id = ident
        self.text = text
        self.acceptance = ""


class _State:
    run_id = "run-20261028"
    goal = "统计各街道老年人口并出图"
    final_summary = "按 12 个街道汇总了老年人口。"
    methodology = ""
    status = "completed"
    requirements = [_Req("r1", "按街道汇总老年人口")]

    def __init__(self, artifacts: list[str]) -> None:
        self.artifacts = artifacts


def _workspace(tmp_path: Path) -> Path:
    out = tmp_path / "workspace" / "output"
    out.mkdir(parents=True, exist_ok=True)
    return tmp_path / "workspace"


def _write_trace(tmp_path: Path, *, broken_tail: bool = False) -> Path:
    path = tmp_path / "trace" / "run-20261028.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "seq": 1,
            "ts": "2026-10-28T01:00:00Z",
            "event": "tool_call",
            "payload": {
                "tool": "run_recipe",
                "args": {"recipe_id": "buffer", "params": {"distance": 500, "units": "meters"}},
            },
        },
        {
            "seq": 2,
            "ts": "2026-10-28T01:00:01Z",
            "event": "tool_result",
            "payload": {"tool": "run_recipe", "ok": True},
        },
        {
            "seq": 3,
            "ts": "2026-10-28T01:00:02Z",
            "event": "tool_call",
            "payload": {
                "tool": "run_recipe",
                "args": {"recipe_id": "zonal", "params": {"field": "pop65", "statistic": "SUM"}},
            },
        },
        {
            "seq": 4,
            "ts": "2026-10-28T01:00:03Z",
            "event": "tool_result",
            "payload": {"tool": "run_recipe", "ok": False},
        },
    ]
    text = "\n".join(json.dumps(evt, ensure_ascii=False) for evt in events) + "\n"
    if broken_tail:
        # trace 是崩溃安全写的：末行可能是半行 JSON
        text += '{"seq": 5, "event": "tool_call", "payload": {"tool": "run_re'
    path.write_text(text, encoding="utf-8")
    return path


def _write_trace_cut_mid_char(tmp_path: Path, *, only_broken: bool = False) -> Path:
    """DAIR-46 实测形态：末行截在「≈」（U+2248）中间，是非法 UTF-8。

    原 trace 前 500 字节合法，末尾是 ``…\\xe2\\x89\\x88 564``。``only_broken=True``
    时整份 trace 只剩这条坏行。旧用例的 ``broken_tail`` 截在 ASCII 处，永远是
    合法 UTF-8，所以这条链一直没被测到。
    """
    path = tmp_path / "trace" / "run-20261028-cut.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b""
    if not only_broken:
        evt = {
            "seq": 1,
            "ts": "2026-10-28T01:00:00Z",
            "event": "tool_call",
            "payload": {
                "tool": "run_recipe",
                "args": {"recipe_id": "buffer", "params": {"distance": 500, "units": "meters"}},
            },
        }
        raw = (json.dumps(evt, ensure_ascii=False) + "\n").encode("utf-8")
    tail = '{"seq": 2, "event": "tool_result", "payload": {"tool": "run_recipe", "ok": true, "summary": "缓冲区面积 ≈'
    raw += tail.encode("utf-8")[:-1]  # 砍掉「≈」的最后一个字节：末两字节 e2 89
    path.write_bytes(raw)
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    return path


def _verdict(*paths: Path) -> dict:
    return {
        "pass": True,
        "summary": "验收通过",
        "structural": [
            {"path": str(path), "exists": True, "kind": "FeatureClass", "feature_count": 12}
            for path in paths
        ],
        "semantic": {"pass": True, "per_requirement": [{"id": "r1", "status": "PASS"}]},
        "missing": [],
        "problems": [],
    }


def _point_layer(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "Point", "coordinates": [1, 2]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------------------ 四段内容
def test_card_covers_four_sections_with_real_param_values(tmp_path: Path):
    ws = _workspace(tmp_path)
    result_csv = ws / "output" / "汇总.csv"
    result_csv.write_text("街道,pop65\n", encoding="utf-8")
    catalog = _Catalog(
        [
            _Record(
                str(ws / "data" / "街道.shp"),
                "街道边界",
                source="boundary",
                license="CC BY 4.0",
                acquired_at="2026-10-01T09:00:00",
            )
        ]
    )

    card = provenance.build_card(
        state=_State([str(result_csv)]),
        verdict=_verdict(result_csv),
        catalog=catalog,
        trace_path=_write_trace(tmp_path),
        recipes=_Library(),
        workspace=ws,
    )

    assert card.run_id == "run-20261028"
    assert card.goal == "统计各街道老年人口并出图"
    # 四段都非空（数据 / 步骤 / 参数 / 验证），产物也登记上了
    assert card.data and card.steps and card.params and card.verifications and card.artifacts
    assert "街道边界" in card.data[0] and "CC BY 4.0" in card.data[0]
    assert any("缓冲区分析" in line and "执行成功" in line for line in card.steps)
    assert any("分区统计" in line and "执行失败" in line for line in card.steps)
    # 参数摘要里是实际调用值，不是"有参数"三个字
    assert any("distance=500" in line for line in card.params)
    assert any("units=meters" in line for line in card.params)
    assert any("field=pop65" in line for line in card.params)
    assert any("通过" in line for line in card.verifications)
    assert any("已满足" in line for line in card.verifications)
    assert card.artifacts == ["output/汇总.csv"]
    assert card.missing == []

    html = provenance.render_card_html(card)
    assert "<h2>溯源卡</h2>" in html
    assert "distance=500" in html
    for title in ("数据", "步骤", "参数", "验证", "产物"):
        assert f">{title}</h3>" in html


# ------------------------------------------------------------------ 退化输入
def test_card_survives_missing_trace_and_broken_catalog(tmp_path: Path):
    ws = _workspace(tmp_path)

    card = provenance.build_card(
        state=_State([str(ws / "output" / "x.csv")]),
        verdict=None,
        catalog=_BoomCatalog(),
        trace_path=tmp_path / "nope.jsonl",
        recipes=None,
        workspace=ws,
    )

    html = provenance.render_card_html(card)
    # 缺的段落写"未记录"，而不是空着或编一句
    assert html.count(provenance.NOT_RECORDED) >= 3
    assert card.steps == [] and card.params == [] and card.data == [] and card.verifications == []
    assert any("数据" in item for item in card.missing)
    assert any("trace" in item for item in card.missing)
    assert any("验证" in item for item in card.missing)
    # 产物已经登记了，就不许在"未记录"里再提一句（这条能真红：产物为空时
    # 会多出"产物：本次没有登记产物"）
    assert not any("产物" in item for item in card.missing)


def test_step_gap_tells_missing_empty_and_directory_apart(tmp_path: Path):
    """trace 不存在 / 是空文件 / 是目录：三种原因各写各的，不混成一句。"""
    ws = _workspace(tmp_path)
    catalog = _Catalog([_Record(str(ws / "data" / "街道.shp"), "街道边界")])
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    empty = trace_dir / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    folder = trace_dir / "as-dir.jsonl"
    folder.mkdir()

    def step_gap(trace_path: Path) -> str:
        card = provenance.build_card(
            state=_State([str(ws / "output" / "x.csv")]),
            verdict=None,
            catalog=catalog,
            trace_path=trace_path,
            recipes=None,
            workspace=ws,
        )
        return next(item for item in card.missing if item.startswith("步骤："))

    assert step_gap(trace_dir / "nope.jsonl") == (
        "步骤：trace 文件不存在（nope.jsonl），执行过程未记录"
    )
    assert step_gap(empty) == "步骤：trace 为空（empty.jsonl），执行过程未记录"
    assert step_gap(folder) == (
        "步骤：trace 路径是目录而非文件（as-dir.jsonl），执行过程未记录"
    )


def test_card_generated_at_is_injectable_so_same_input_same_output(tmp_path: Path):
    ws = _workspace(tmp_path)
    kwargs = dict(
        state=_State([str(ws / "output" / "汇总.csv")]),
        verdict=None,
        catalog=_Catalog([]),
        trace_path=_write_trace(tmp_path),
        recipes=_Library(),
        workspace=ws,
    )

    first = provenance.build_card(**kwargs, generated_at="2026-10-28 01:00")
    second = provenance.build_card(**kwargs, generated_at="2026-10-28 01:00")

    assert first == second
    assert first.generated_at == "2026-10-28 01:00"
    assert provenance.render_card_html(first) == provenance.render_card_html(second)
    # 不传时仍按当下取（默认行为没变）
    assert provenance.build_card(**kwargs).generated_at


def test_card_tolerates_half_written_trace(tmp_path: Path):
    ws = _workspace(tmp_path)

    card = provenance.build_card(
        state=_State([]),
        verdict={"pass": True},
        catalog=_Catalog([]),
        trace_path=_write_trace(tmp_path, broken_tail=True),
        recipes=None,
        workspace=ws,
    )

    # 半行 JSON 被跳过，前面的调用照旧进卡；配方名退回 id
    assert any("buffer" in line for line in card.steps)
    assert any("distance=500" in line for line in card.params)
    assert provenance.render_card_html(card)


def test_card_without_any_input_still_renders():
    card = provenance.build_card()

    html = provenance.render_card_html(card)
    assert "<h2>溯源卡</h2>" in html
    assert html.count(provenance.NOT_RECORDED) >= 4
    assert card.missing


# ------------------------------------------------------------------ 单页 / 无外链
def test_card_has_no_external_links_and_stays_small(tmp_path: Path):
    ws = _workspace(tmp_path)
    records = [
        _Record(
            f"https://example.com/dl/{index}.shp",
            f"远程数据{index}",
            source="download",
            license="CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        )
        for index in range(20)
    ]
    artifacts = [str(ws / "output" / f"结果{index}.tif") for index in range(40)]

    card = provenance.build_card(
        state=_State(artifacts),
        verdict=_verdict(),
        catalog=_Catalog(records),
        trace_path=_write_trace(tmp_path),
        recipes=_Library(),
        workspace=ws,
    )

    html = provenance.render_card_html(card)
    page = provenance.render_card_page(card)
    assert "http://" not in html and "https://" not in html
    assert "http://" not in page and "https://" not in page
    assert len(html.encode("utf-8")) < provenance.EXPECTED_MAX_CARD_BYTES
    assert "<!doctype html>" in page and "<h2>溯源卡</h2>" in page
    # 每段封顶，超出的写"另有 N 项"
    assert len(card.data) == provenance.MAX_LINES
    assert len(card.artifacts) == provenance.MAX_LINES
    assert any("另有" in line for line in card.artifacts)


# ------------------------------------------------------------------ 成果包集成
def test_build_deliverables_writes_card_and_embeds_section(tmp_path: Path):
    ws = _workspace(tmp_path)
    layer = _point_layer(ws / "output" / "街区.geojson")

    result = build_deliverables(
        workspace=ws,
        state=_State([str(layer)]),
        verdict=_verdict(layer),
        catalog=_Catalog([_Record(str(ws / "data" / "街区.shp"), "街区", source="boundary")]),
        trace_path=_write_trace(tmp_path),
        recipes=_Library(),
        want_pdf=False,
    )

    html = result.html.read_text(encoding="utf-8")
    assert html.count("<h2>溯源卡</h2>") == 1
    assert "distance=500" in html
    assert result.provenance is not None
    assert result.provenance.name == PROVENANCE_NAME
    page = result.provenance.read_text(encoding="utf-8")
    assert "<h2>溯源卡</h2>" in page and "<!doctype html>" in page
    assert "http://" not in page and "https://" not in page
    assert str(result.provenance) in result.paths


def test_trace_cut_mid_chinese_char_still_writes_package_and_card(tmp_path: Path):
    """trace 截在汉字中间（非法 UTF-8）时，成果包与溯源卡都要照写。

    旧代码 UnicodeDecodeError 一路穿透到 ``engine/loop.py``，两个产物一个都不落盘。
    """
    ws = _workspace(tmp_path)
    layer = _point_layer(ws / "output" / "街区.geojson")

    result = build_deliverables(
        workspace=ws,
        state=_State([str(layer)]),
        verdict=_verdict(layer),
        catalog=_Catalog([_Record(str(ws / "data" / "街区.shp"), "街区")]),
        trace_path=_write_trace_cut_mid_char(tmp_path),
        recipes=_Library(),
        want_pdf=False,
    )

    # 两个产物都落盘（旧代码这里两个都是 None）
    assert result.html is not None and result.html.exists()
    assert result.provenance is not None and result.provenance.exists()
    # 坏行只丢自己：截断之前的调用照旧进卡，半行里的 tool_result 不算数
    page = result.provenance.read_text(encoding="utf-8")
    assert "distance=500" in page
    # 被截断的 tool_result 丢了，就写「未记录」，不许编一个「执行成功」
    assert f"缓冲区分析（buffer）：结果{provenance.NOT_RECORDED}" in page


def test_card_survives_trace_that_is_only_a_broken_line(tmp_path: Path):
    """整份 trace 只剩一条截在汉字中间的坏行：段落写「未记录」，产物照写。"""
    ws = _workspace(tmp_path)
    layer = _point_layer(ws / "output" / "街区.geojson")

    result = build_deliverables(
        workspace=ws,
        state=_State([str(layer)]),
        verdict=_verdict(layer),
        catalog=_Catalog([_Record(str(ws / "data" / "街区.shp"), "街区")]),
        trace_path=_write_trace_cut_mid_char(tmp_path, only_broken=True),
        recipes=_Library(),
        want_pdf=False,
    )

    assert result.html is not None and result.html.exists()
    page = result.provenance.read_text(encoding="utf-8")
    assert "步骤：trace 里没有配方调用，执行过程未记录" in page
    assert page.count(provenance.NOT_RECORDED) >= 2


def test_content_failure_degrades_instead_of_dropping_the_package(tmp_path: Path, monkeypatch):
    """正文生成炸了也只进 notes：丢掉 trace 重来一次，成果包照写。"""
    ws = _workspace(tmp_path)
    layer = _point_layer(ws / "output" / "街区.geojson")
    real = narrative.build_content

    def _boom(**kwargs):
        if kwargs.get("trace_path"):
            raise UnicodeDecodeError("utf-8", b"\xe2", 0, 1, "unexpected end of data")
        return real(**kwargs)

    monkeypatch.setattr(narrative, "build_content", _boom)

    result = build_deliverables(
        workspace=ws,
        state=_State([str(layer)]),
        verdict=_verdict(layer),
        trace_path=_write_trace_cut_mid_char(tmp_path),
        want_pdf=False,
    )

    assert result.html is not None and result.html.exists()
    assert any("降级" in note for note in result.notes)


def test_card_failure_does_not_block_delivery(tmp_path: Path, monkeypatch):
    ws = _workspace(tmp_path)
    layer = _point_layer(ws / "output" / "街区.geojson")

    def _boom(**_kwargs):
        raise RuntimeError("卡片炸了")

    monkeypatch.setattr(provenance, "build_card", _boom)

    result = build_deliverables(
        workspace=ws,
        state=_State([str(layer)]),
        verdict=_verdict(layer),
        want_pdf=False,
    )

    assert result.html is not None and result.html.exists()
    assert result.provenance is None
    assert any("溯源卡" in note for note in result.notes)
    assert "<h2>溯源卡</h2>" not in result.html.read_text(encoding="utf-8")
