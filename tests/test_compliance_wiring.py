# -*- coding: utf-8 -*-
"""合规闸门的接线：create_map 自检 → 边车 → verify_outputs / finish → 卡片与 UI。

这里不跑 ArcGIS：

- ``create_map`` 的自检逻辑抽在 ``_compliance_selfcheck`` 里，可以拿假 result 直接调；
- 验收侧只读 ``.compliance.json`` 边车（``Verifier.compliance_check``）；
- 工具层用假 code_runner 捕获生成的代码（不需要真出图）。

覆盖的契约：
1. 两处落点都要通（create_map 出图后自检 / finish 汇总进 verify_outputs）；
2. 阻断要真的挡住交付，且**不许**走"同一理由重复拒绝就降级放行"；
3. 责任边界文案出现在卡片、UI 与验收结论里。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gis_cli.cartography import _compliance_selfcheck
from gis_cli.compliance import NOTICE, MapContext, check_map, load_rules
from gis_cli.compliance.facts import LayerRef, catalog_rows_for, collect_facts, context_from_spec
from gis_cli.compliance.store import (
    SIDECAR_SUFFIX,
    aggregate,
    collect_reports,
    read_sidecar,
    report_from_sidecar,
    sidecar_path,
    write_sidecar,
)
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.engine.verifier import Verifier


# ------------------------------------------------------------------ 夹具
class _Record:
    def __init__(self, path: str, *, name: str = "", source: str = "", license: str = "") -> None:
        self.path = path
        self.name = name
        self.source = source
        self.license = license


class _Catalog:
    def __init__(self, records: list[_Record]) -> None:
        self._records = records

    def list_datasets(self, limit: int = 200) -> list[_Record]:
        return self._records[:limit]


def _spec(*, title: str = "示意图", layers: list[str] | None = None, **intent) -> dict:
    spec = {
        "title": {"text": title},
        "legend": {"title": "图例"},
        "renderer": {"layer_paths": list(layers or [])},
        "intent": dict(intent),
    }
    return spec


def _state(*artifacts: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifacts=[str(item) for item in artifacts],
        goal="出图",
        requirements=[],
        tasklist=[],
        final_summary="",
        methodology="",
        created_at="",
    )


def _write_map(tmp_path: Path, name: str = "地图.jpg") -> Path:
    """写一张**真能过图片非空白检查**的小图（验收结构体检会算方差）。"""
    import numpy as np
    from PIL import Image

    img = tmp_path / "output" / name
    img.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261104)
    pixels = (rng.random((120, 160)) * 255).astype("uint8")
    Image.fromarray(pixels).save(img, format="JPEG")
    return img


def _selfcheck(tmp_path: Path, *, sensitive: bool = True, catalog_rows=None, intent: dict | None = None) -> dict:
    layer = tmp_path / "input" / ("国界线.shp" if sensitive else "街道.shp")
    img = _write_map(tmp_path)
    result: dict = {"output": str(img), "spec": _spec(layers=[str(layer)])}
    _compliance_selfcheck(
        result,
        layers=[str(layer)],
        intent=dict(intent or {}),
        catalog_rows=catalog_rows if catalog_rows is not None else [],
    )
    return result


# --------------------------------------------------------- create_map 自检
def test_create_map_selfcheck_writes_report_and_sidecar(tmp_path: Path):
    """出图后自检：结论进 result["compliance"]，边车挨着图落盘。"""
    result = _selfcheck(
        tmp_path,
        catalog_rows=[{"path": str(tmp_path / "input" / "国界线.shp"), "source": "boundary"}],
    )

    compliance = result["compliance"]
    assert compliance["level"] == "block"
    assert compliance["blocked"] is True
    assert [item["rule_id"] for item in compliance["findings"]] == [
        "national_boundary_unapproved",
        "approval_missing",
    ]
    assert Path(result["compliance_sidecar"]).name == Path(result["output"]).stem + SIDECAR_SUFFIX
    assert sidecar_path(result["output"]).is_file()


def test_create_map_selfcheck_is_quiet_for_a_clean_public_map(tmp_path: Path):
    result = _selfcheck(
        tmp_path,
        sensitive=False,
        catalog_rows=[],
        intent={"approval_number": "GS(2023)1234号"},
    )
    # 断言里带上 findings：CI 上这条曾经在无 ArcGIS 的机器上返回 warn，
    # 只报 "warn != pass" 看不出是哪条规则命中（诊断信息就留在这里，别再删）。
    assert result["compliance"]["level"] == "pass", result["compliance"]
    assert result["compliance"]["findings"] == [], result["compliance"]


def test_create_map_selfcheck_never_breaks_rendering(tmp_path: Path, monkeypatch):
    """闸门自己坏掉只记 compliance_error，不许把已经出好的图带崩。"""
    import gis_cli.compliance as compliance_mod

    def boom(*args, **kwargs):
        raise RuntimeError("规则库炸了")

    monkeypatch.setattr(compliance_mod, "check_map", boom)
    result = _selfcheck(tmp_path)
    assert result["compliance_error"] == "规则库炸了"
    assert Path(result["output"]).is_file()


# --------------------------------------------------------- 上下文构造
def test_context_from_spec_collects_on_map_texts_and_declarations(tmp_path: Path):
    spec = _spec(
        title="北京市人口图",
        layers=[str(tmp_path / "input" / "国界线.shp")],
        approval_number="GS(2023)1234号",
        watermark="未送审",
        audience="internal",
    )
    spec["overlay"] = {"labels": [{"at": [1, 2], "text": "南海诸岛"}]}
    rows = [{"path": str(tmp_path / "input" / "国界线.shp"), "source": "boundary"}]

    context = context_from_spec(spec, catalog_rows=rows)

    assert "北京市人口图" in context.texts
    assert "南海诸岛" in context.texts           # 叠加标注是真画在图上的字
    assert context.approval_number == "GS(2023)1234号"
    assert context.watermark == "未送审"
    assert context.audience == "internal"
    assert context.catalog_available is True
    assert context.layers[0].registered is True
    assert context.layers[0].source == "boundary"

    facts = collect_facts(context)
    assert facts["approval"] == "valid"
    assert facts["watermark"] == "present"
    assert facts["sensitive_count"] == 1


def test_context_without_catalog_marks_sources_unverified(tmp_path: Path):
    context = context_from_spec(_spec(layers=[str(tmp_path / "input" / "省界.shp")]), catalog_rows=None)
    assert context.catalog_available is False
    assert context.layers[0].registered is False
    assert collect_facts(context)["boundary_source"] == "unknown"


def test_catalog_rows_for_filters_to_the_layers_drawn(tmp_path: Path):
    catalog = _Catalog(
        [
            _Record(str(tmp_path / "input" / "国界线.shp"), source="boundary", name="国界线"),
            _Record(str(tmp_path / "input" / "人口.shp"), source="csv_upload", name="人口"),
            _Record(str(tmp_path / "input" / "x.gdb" / "roads"), source="download", name="roads"),
        ]
    )
    rows = catalog_rows_for(catalog, [str(tmp_path / "input" / "国界线.shp")])
    assert len(rows) == 1
    assert rows[0]["source"] == "boundary"
    # 拿不到数据目录 → None（= 来源无法核验），不是空列表（= 目录里没有）
    assert catalog_rows_for(None, [str(tmp_path / "input" / "国界线.shp")]) is None


# --------------------------------------------------------- 边车与汇总
def test_sidecar_roundtrip_and_recheck_uses_current_rules(tmp_path: Path):
    """边车存的是上下文：规则改了/关了，验收立刻按新口径重算。"""
    img = _write_map(tmp_path)
    context = MapContext(
        texts=["国界线示意图"],
        layers=[LayerRef(name="国界线", path=str(tmp_path / "input" / "国界线.shp"), registered=False)],
        catalog_available=True,
        image=str(img),
    )
    report = check_map(context)
    write_sidecar(img, report, context=context)

    stored = read_sidecar(img)
    assert stored["version"] == 1 and stored["report"]["level"] == "block"

    again = report_from_sidecar(img)
    assert again is not None and again.level == "block"

    # 关掉阻断规则后重算：同一边车，结论变警告（证明它用的是上下文不是存档结论）
    softened = report_from_sidecar(img, enabled={"national_boundary_unapproved": False})
    assert softened is not None and softened.level == "warn"


def test_sidecar_falls_back_to_archived_report_when_context_is_broken(tmp_path: Path):
    img = _write_map(tmp_path)
    report = check_map(MapContext(texts=["人口图"], catalog_available=True))
    write_sidecar(img, report)
    payload = json.loads(sidecar_path(img).read_text(encoding="utf-8"))
    payload["context"] = "not-a-dict"
    sidecar_path(img).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    restored = report_from_sidecar(img)
    assert restored is not None
    assert restored.level == report.level
    assert [item.rule_id for item in restored.hits] == [item.rule_id for item in report.hits]


def test_missing_or_broken_sidecar_is_ignored(tmp_path: Path):
    img = _write_map(tmp_path)
    assert report_from_sidecar(img) is None
    sidecar_path(img).write_text("{半截", encoding="utf-8")
    assert read_sidecar(img) is None
    assert collect_reports([str(img)]) == []


def test_aggregate_dedupes_per_image_and_reports_block(tmp_path: Path):
    first = _write_map(tmp_path, "图1.jpg")
    second = _write_map(tmp_path, "图2.jpg")
    context = MapContext(
        texts=["国界线示意图"],
        layers=[LayerRef(name="国界线", path="input/国界线.shp", registered=True)],
        catalog_available=True,
    )
    report = check_map(context)
    for image in (first, second):
        write_sidecar(image, report, context=context)

    entries = collect_reports([str(first), str(second), str(first)])
    combined = aggregate(entries)

    assert combined["checked"] == 2
    assert combined["blocked"] is True
    assert combined["level"] == "block"
    assert combined["counts"]["block"] == 2          # 两张图各一条阻断
    assert combined["counts"]["info"] == 2
    assert "阻断 2 项" in combined["summary"]
    assert {item["image"] for item in combined["findings"]} == {"图1.jpg", "图2.jpg"}
    assert combined["notice"] == NOTICE
    assert "不做合规判定" in combined["notice"]


def test_aggregate_counts_one_map_once_when_the_aprx_is_registered_too(tmp_path: Path):
    """create_map 会把同名 .aprx 一起登记成产物，但它与 .jpg 共用一个边车：只许算一张图。"""
    img = _write_map(tmp_path, "地图.jpg")
    aprx = img.with_suffix(".aprx")
    aprx.write_bytes(b"PK\x03\x04 fake aprx")  # 合规闸门只读边车，不看 aprx 内容

    context = MapContext(
        texts=["国界线示意图"],
        layers=[LayerRef(name="国界线", path=str(tmp_path / "input" / "国界线.shp"), registered=False)],
        catalog_available=True,
        image=str(img),
    )
    write_sidecar(img, check_map(context), context=context)
    assert sidecar_path(aprx) == sidecar_path(img)  # 前提：两者落到同一个边车

    only_image = aggregate(collect_reports([str(img)]))
    with_aprx = aggregate(collect_reports([str(img), str(aprx)]))

    assert only_image["checked"] == 1
    assert with_aprx["checked"] == 1
    assert with_aprx["images"] == ["地图.jpg"]
    assert with_aprx["findings"] == only_image["findings"]
    assert with_aprx["counts"] == only_image["counts"]
    assert with_aprx["summary"] == only_image["summary"]

    # 真实验收路径：state.artifacts 里同时有 .jpg 与 .aprx，结论也只算一张图
    verdict = Verifier(workspace=tmp_path).verify(_state(img, aprx))
    compliance = verdict["compliance"]
    assert compliance["checked"] == 1
    assert compliance["images"] == ["地图.jpg"]
    assert compliance["counts"] == only_image["counts"]
    assert len(compliance["findings"]) == len(only_image["findings"])
    assert "（1 张图）" in compliance["summary"]


def test_aggregate_without_any_selfcheck_says_so_instead_of_passing():
    combined = aggregate([])
    assert combined["level"] == "none"
    assert combined["blocked"] is False
    assert combined["checked"] == 0
    assert "没有可核验的自检记录" in combined["summary"]


# --------------------------------------------------------- 验收（verify_outputs / finish）
def test_verifier_blocks_delivery_and_asks_for_the_missing_evidence(tmp_path: Path):
    result = _selfcheck(tmp_path)
    img = Path(result["output"])

    verdict = Verifier(workspace=tmp_path).verify(_state(img))

    compliance = verdict["compliance"]
    assert compliance["checked"] == 1
    assert compliance["blocked"] is True
    assert verdict["pass"] is False
    assert verdict["hard_block"] is True
    assert "合规闸门" in verdict["summary"]
    assert "审图号" in verdict["repair_hint"] or "水印" in verdict["repair_hint"]


def test_verifier_only_warns_for_unregistered_boundary_source(tmp_path: Path):
    img = _write_map(tmp_path)
    context = MapContext(
        texts=["省界线示意图", "GS(2023)1234号"],
        layers=[LayerRef(name="省界", path=str(tmp_path / "input" / "省界.shp"), registered=False)],
        catalog_available=True,
        image=str(img),
    )
    write_sidecar(img, check_map(context), context=context)

    verdict = Verifier(workspace=tmp_path).verify(_state(img))

    assert verdict["compliance"]["level"] == "warn"
    assert verdict["compliance"]["blocked"] is False
    assert verdict["hard_block"] is False
    # 警告不拦交付，但必须出现在验收结论里
    assert "合规闸门" in verdict["summary"] or verdict["pass"] is True


def test_verifier_without_sidecar_keeps_old_behaviour(tmp_path: Path):
    img = _write_map(tmp_path)
    verdict = Verifier(workspace=tmp_path).verify(_state(img))
    assert verdict["compliance"]["checked"] == 0
    assert verdict["compliance"]["findings"] == []
    assert verdict["hard_block"] is False
    assert verdict["pass"] is True


def test_hard_block_is_not_degraded_by_repeated_rejections(tmp_path: Path):
    """合规阻断不走"同一理由连拒三次就降级放行"（普通拒绝才有那条路）。"""
    from gis_cli.engine.llm_client import LLMResponse, ToolCall
    from gis_cli.engine.loop import AgentLoop, LoopConfig
    from gis_cli.trace.recorder import RunRecorder

    class ScriptedLLM:
        class _Cfg:
            model = "mock"

        config = _Cfg()

        def __init__(self, count: int) -> None:
            self.left = count

        def chat(self, messages, **kwargs):
            self.left -= 1
            return LLMResponse(
                tool_calls=[ToolCall(id=f"c{self.left}", name="finish", arguments={"summary": "done"}, raw_arguments="{}")],
                model="mock",
            )

    class BlockedVerifier:
        def verify(self, state):
            return {
                "pass": False,
                "summary": "验收未通过：合规闸门阻断",
                "missing": ["合规：国界图缺审图号"],
                "problems": [],
                "hard_block": True,
            }

    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=ScriptedLLM(4),
        registry=build_default_registry(),
        context=EngineContext(workspace=tmp_path),
        verifier=BlockedVerifier(),
        recorder=RunRecorder("run_hard_block", tmp_path / "traces"),
        config=LoopConfig(max_turns=8),
        auto_extract_requirements=False,
    )
    state = loop.run("测试任务")

    assert state.status == "failed"
    assert "多次验收未通过" in state.error
    verdicts = [
        json.loads(line)["payload"]
        for line in (tmp_path / "traces" / "run_hard_block.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("event") == "verification"
    ]
    assert verdicts and all(record.get("degraded") is not True for record in verdicts)


# --------------------------------------------------------- 工具层接线
class _CaptureRunner:
    """不执行生成的代码，只把 payload 回给工具（用来验证工具层的处理）。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.codes: list[str] = []

    def run(self, code, timeout=None, workspace=None):
        self.codes.append(code)

        class R:
            ok = True
            error = None
            stdout = ""
            stderr = ""

        R.result = self.payload
        return R


def _create_map_tool(tmp_path: Path, runner: _CaptureRunner, args: dict | None = None):
    registry = build_default_registry()
    ctx = EngineContext(workspace=tmp_path, code_runner=runner)
    payload = {
        "output": str(tmp_path / "output" / "map.jpg"),
        "aprx": str(tmp_path / "output" / "map.aprx"),
        "qc_ok": True,
        "qc_summary": "版面体检全部通过",
        "spec": _spec(),
    }
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output" / "map.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 2048)
    runner.payload = payload
    return registry.execute(
        "create_map",
        {"layers": [str(tmp_path / "input" / "国界线.shp")], "output_path": str(tmp_path / "output" / "map.jpg"), **(args or {})},
        ctx,
    )


def test_create_map_tool_passes_compliance_declarations_and_catalog(tmp_path: Path):
    registry = build_default_registry()
    tool = registry.get("create_map")
    assert tool is not None
    assert "approval_number" in tool.parameters["properties"]
    assert "watermark" in tool.parameters["properties"]
    assert "audience" in tool.parameters["properties"]

    runner = _CaptureRunner({})
    ctx = EngineContext(workspace=tmp_path, code_runner=runner, catalog=_Catalog([]))
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    registry.execute(
        "create_map",
        {
            "layers": [str(tmp_path / "input" / "国界线.shp")],
            "output_path": str(tmp_path / "output" / "map.jpg"),
            "approval_number": "GS(2023)1234号",
            "watermark": "未送审",
            "audience": "internal",
        },
        ctx,
    )
    code = runner.codes[-1]
    assert "catalog_rows=" in code          # 数据目录随出图一起传进内核
    assert "approval_number" in code and "GS(2023)1234号" in code
    assert "watermark" in code and "未送审" in code
    assert "audience" in code and "internal" in code


def test_create_map_tool_blocks_on_compliance_but_keeps_artifacts(tmp_path: Path):
    runner = _CaptureRunner({})
    registry = build_default_registry()
    ctx = EngineContext(workspace=tmp_path, code_runner=runner, catalog=_Catalog([]))
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output" / "map.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 2048)
    runner.payload = {
        "output": str(tmp_path / "output" / "map.jpg"),
        "aprx": str(tmp_path / "output" / "map.aprx"),
        "qc_ok": True,
        "spec": _spec(),
        "compliance": {
            "level": "block",
            "blocked": True,
            "findings": [
                {
                    "rule_id": "national_boundary_unapproved",
                    "title": "国界与敏感要素（未审图）",
                    "level": "block",
                    "message": "图面含 1 个国界/敏感要素图层（国界线），未见审图号",
                    "advice": "取得审图号并随图标注。",
                }
            ],
            "summary": "合规闸门：阻断 1 项",
        },
    }
    observation = registry.execute(
        "create_map",
        {"layers": [str(tmp_path / "input" / "国界线.shp")], "output_path": str(tmp_path / "output" / "map.jpg")},
        ctx,
    )
    assert observation.ok is False
    assert observation.error["type"] == "ComplianceBlocked"
    assert observation.artifacts                 # 图已经出了，产物照常登记（别让模型重复出图）
    assert "approval_number" in observation.hint and "watermark" in observation.hint


def test_create_map_tool_only_warns_for_non_blocking_findings(tmp_path: Path):
    runner = _CaptureRunner({})
    registry = build_default_registry()
    ctx = EngineContext(workspace=tmp_path, code_runner=runner, catalog=_Catalog([]))
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output" / "map.jpg").write_bytes(b"\xff\xd8\xff" + b"0" * 2048)
    runner.payload = {
        "output": str(tmp_path / "output" / "map.jpg"),
        "aprx": str(tmp_path / "output" / "map.aprx"),
        "qc_ok": True,
        "spec": _spec(),
        "compliance": {
            "level": "warn",
            "blocked": False,
            "findings": [
                {
                    "rule_id": "boundary_source_unregistered",
                    "title": "边界数据来源未登记",
                    "level": "warn",
                    "message": "1 个边界类图层里有 1 个来源未登记（国界线）",
                    "advice": "用 ingest 登记来源与许可。",
                }
            ],
            "summary": "合规闸门：警告 1 项",
        },
    }
    observation = registry.execute(
        "create_map",
        {"layers": [str(tmp_path / "input" / "国界线.shp")], "output_path": str(tmp_path / "output" / "map.jpg")},
        ctx,
    )
    assert observation.ok is True
    assert "合规提醒" in observation.summary
    assert "来源未登记" in observation.hint


# --------------------------------------------------------- 卡片与 UI 文案
def test_provenance_card_shows_compliance_block_and_notice(tmp_path: Path):
    from gis_cli.report import provenance

    verdict = {
        "pass": False,
        "summary": "验收未通过",
        "compliance": {
            "level": "block",
            "blocked": True,
            "checked": 1,
            "findings": [
                {
                    "rule_id": "national_boundary_unapproved",
                    "title": "国界与敏感要素（未审图）",
                    "level": "block",
                    "message": "图面含 1 个国界/敏感要素图层（国界线），未见审图号",
                    "image": "国界图.jpg",
                }
            ],
            "summary": "合规闸门：阻断 1 项（1 张图）",
        },
    }
    card = provenance.build_card(verdict=verdict, workspace=tmp_path)

    assert any("阻断" in line for line in card.compliance)
    html = provenance.render_card_html(card)
    assert "合规闸门" in html
    assert "不做合规判定" in html              # 责任边界与卡片同一节呈现
    assert "国界图" in html


def test_card_says_not_recorded_for_runs_without_a_compliance_verdict(tmp_path: Path):
    from gis_cli.report import provenance

    card = provenance.build_card(verdict={"pass": True}, workspace=tmp_path)
    assert card.compliance and "未记录" in card.compliance[0]
    assert not any("合规" in item for item in card.missing)   # 不往 missing 里刷屏


def test_ui_verification_event_shows_compliance_findings_and_notice():
    from gis_cli.ui.events import to_unit

    unit = to_unit(
        "verification",
        {
            "pass": False,
            "summary": "验收未通过：合规闸门阻断 1 项",
            "compliance": {
                "level": "block",
                "blocked": True,
                "findings": [
                    {
                        "rule_id": "national_boundary_unapproved",
                        "title": "国界与敏感要素（未审图）",
                        "level": "block",
                        "message": "图面含 1 个国界/敏感要素图层（国界线），未见审图号",
                    }
                ],
                "notice": NOTICE,
            },
        },
    )
    assert unit is not None
    assert unit.ok is False
    assert any("合规阻断" in line for line in unit.lines)
    assert any("不做合规判定" in line for line in unit.lines)


def test_default_rule_library_is_the_one_shipped_with_the_package():
    """规则库必须随包分发（``gis_cli/compliance/rules.yaml``），不能被 cwd 影响。"""
    rules = load_rules()
    assert rules.version == 1 and len(rules.rules) == 4


def test_aggregate_surfaces_selfcheck_errors_instead_of_silence(tmp_path: Path):
    """规则库/边车坏了要说出来：``checked`` 为 1 但结论是异常，不是“没问题”。"""
    from gis_cli.compliance import ComplianceReport

    img = _write_map(tmp_path)
    write_sidecar(img, ComplianceReport(error="规则库炸了"))
    combined = aggregate(collect_reports([str(img)]))

    assert combined["checked"] == 1
    assert combined["problems"] and "规则库炸了" in combined["problems"][0]
    assert "自检异常" in combined["summary"]
