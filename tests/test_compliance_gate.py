# -*- coding: utf-8 -*-
"""地图合规闸门（P1-2 / W3）：规则库、事实判定、20 个合成样本的漏报/误报口径。

验收口径照 P1 实施方案 §4：**20 个合成样本**（构造已知问题，不依赖真实图件）、
**0 漏报**、**误报 ≤10%**、**单图 ≤30 s**。样本覆盖每条规则的正例（该报）、
反例（不该报）与边界情形（审图号格式不对、国界图层来源已登记、图层名带"敏感"
但不是国界……）—— 误报率只能靠反例量出来。

这个文件只跑纯 Python 代码，不需要 ArcGIS。
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from gis_cli.compliance import NOTICE, ComplianceReport, MapContext, check_map, load_rules
from gis_cli.compliance.facts import LayerRef, collect_facts
from gis_cli.compliance.rules import RuleError, parse_ruleset, worst_level

R1 = "national_boundary_unapproved"
R2 = "boundary_source_unregistered"
R3 = "watermark_missing"
R4 = "approval_missing"

SRC = Path(__file__).resolve().parents[1] / "src" / "gis_cli"
COMPLIANCE_DIR = SRC / "compliance"


# ------------------------------------------------------------------ 夹具
def _layer(
    name: str,
    *,
    source: str = "",
    license: str = "unknown",
    registered: bool = True,
) -> LayerRef:
    return LayerRef(name=name, path=f"input/{name}.shp", source=source, license=license, registered=registered)


def _ctx(
    *,
    layers: tuple[LayerRef, ...] = (),
    texts: tuple[str, ...] = (),
    audience: str = "public",
    approval_number: str = "",
    watermark: str = "",
    catalog: bool = True,
) -> MapContext:
    return MapContext(
        texts=list(texts),
        layers=list(layers),
        audience=audience,
        approval_number=approval_number,
        watermark=watermark,
        catalog_available=catalog,
    )


def _fired(report: ComplianceReport) -> set[str]:
    return {item.rule_id for item in report.hits}


#: 20 个合成样本：``(样本名, 上下文, 期望命中的规则)``。
#:
#: 期望集合就是"人工构造时就知道该报什么"——漏报 = 期望有、实际没有；
#: 误报 = 实际有、期望没有。所以每条样本都必须写清期望，包括"什么都不该报"。
SAMPLES: list[tuple[str, MapContext, set[str]]] = [
    (
        "国界图·无审图号无水印",
        _ctx(layers=(_layer("国界线", source="boundary", license="天地图条款"),)),
        {R1, R4},
    ),
    (
        "国界图·审图号有效·来源已登记",
        _ctx(
            layers=(_layer("国界线", source="boundary", license="天地图条款"),),
            texts=("全国国界线示意图", "审图号：GS(2023)1234号"),
        ),
        set(),
    ),
    (
        "国界图·已标未送审水印",
        _ctx(
            layers=(_layer("国界线", source="boundary", license="天地图条款"),),
            texts=("未送审 · 不得对外发布",),
        ),
        set(),
    ),
    (
        "国界图·审图号格式不对",
        _ctx(
            layers=(_layer("国界线", source="boundary", license="天地图条款"),),
            texts=("审图号：GS2023-1234",),
        ),
        {R1, R4},
    ),
    (
        "军事设施图·无审图号",
        _ctx(layers=(_layer("军事设施", source="boundary", license="用户自有"),)),
        {R1, R4},
    ),
    (
        "省界图·来源未登记",
        _ctx(layers=(_layer("省界", registered=False),)),
        {R2, R4},
    ),
    (
        "省界图·来源已登记",
        _ctx(layers=(_layer("省界", source="boundary", license="天地图条款"),)),
        {R4},
    ),
    (
        "省界图·只登记成本机文件",
        _ctx(layers=(_layer("省界", source="local", registered=True),)),
        {R2, R4},
    ),
    (
        "省界图·没拿到数据目录",
        _ctx(layers=(_layer("省界", registered=False),), catalog=False),
        {R4},
    ),
    (
        "内部使用图·缺水印",
        _ctx(texts=("社区适老化评估图",), audience="internal"),
        {R3, R4},
    ),
    (
        "内部使用图·有水印",
        _ctx(texts=("社区适老化评估图", "内部使用"), audience="internal"),
        set(),
    ),
    (
        "普通人口图·无审图号",
        _ctx(layers=(_layer("街道", registered=True),), texts=("街道人口分布图",)),
        {R4},
    ),
    (
        "普通人口图·审图号有效",
        _ctx(texts=("街道人口分布图", "GS(2021)0088号")),
        set(),
    ),
    (
        "国界图·审图号有效·来源未登记",
        _ctx(
            layers=(_layer("国界线", registered=False),),
            texts=("GS(2020)1234号",),
        ),
        {R2},
    ),
    (
        "国界图·审图号有效·来源只登记成本机",
        _ctx(
            layers=(_layer("国界线", source="local", registered=True),),
            texts=("GS(2020)1234号",),
        ),
        {R2},
    ),
    (
        "国界图·审图号写在叠加标注里",
        _ctx(
            layers=(_layer("国界线", source="boundary", license="天地图条款"),),
            texts=("国界线示意图", "GS（2019）第1822号"),
        ),
        set(),
    ),
    (
        "国界图·水印写在叠加标注里",
        _ctx(
            layers=(_layer("国界线", source="boundary", license="天地图条款"),),
            texts=("国界线示意图", "未送审 内部使用"),
        ),
        set(),
    ),
    (
        "生态敏感区图·不是国界",
        _ctx(layers=(_layer("生态敏感区", source="local", registered=True),), texts=("生态敏感区分布图",)),
        {R4},
    ),
    (
        "南海子公园绿地·不是南海诸岛",
        _ctx(layers=(_layer("南海子公园绿地", source="local", registered=True),), texts=("绿地公平性",)),
        {R4},
    ),
    (
        "国界+省界混图·省界未登记",
        _ctx(
            layers=(
                _layer("国界线", source="boundary", license="天地图条款"),
                _layer("省界", registered=False),
            )
        ),
        {R1, R2, R4},
    ),
]


# ------------------------------------------------------------------ 验收口径
def test_twenty_synthetic_samples_have_zero_misses_and_zero_false_positives():
    """20 个合成样本：0 漏报、误报 0%（验收上限 10%）、单图 ≤30 s。"""
    assert len(SAMPLES) == 20
    missed: list[str] = []
    false_positive: list[str] = []
    slowest = (0.0, "")
    for name, context, expected in SAMPLES:
        started = time.perf_counter()
        report = check_map(context)
        elapsed = time.perf_counter() - started
        slowest = max(slowest, (elapsed, name))
        fired = _fired(report)
        if expected - fired:
            missed.append(f"{name}: 该报没报 {sorted(expected - fired)}")
        if fired - expected:
            false_positive.append(f"{name}: 不该报却报了 {sorted(fired - expected)}")

    assert not missed, "漏报（验收要求 0）：" + "；".join(missed)
    false_positive_rate = len(false_positive) / len(SAMPLES)
    assert false_positive_rate <= 0.10, "误报率超 10%：" + "；".join(false_positive)
    assert not false_positive, "误报（本口径要求 0）：" + "；".join(false_positive)
    assert slowest[0] < 30.0, f"单图自检超过 30 s：{slowest[0]:.3f}s（{slowest[1]}）"
    # 正常规模下应该是毫秒级：这条只是防止以后把 IO / 模型调用塞进判定路径
    assert slowest[0] < 1.0


@pytest.mark.parametrize("name,context,expected", SAMPLES, ids=[item[0] for item in SAMPLES])
def test_each_sample_fires_exactly_the_expected_rules(name, context, expected):
    """逐条样本可单独看：命中集合与期望集合完全一致（便于定位是哪条规则漂了）。"""
    report = check_map(context)
    assert _fired(report) == expected
    # 每条规则都有独立结论：命中/通过/关闭/错误，四选一，不能没有
    assert {item.status for item in report.results} <= {"hit", "pass", "off", "error"}
    assert len(report.results) == len(load_rules().rules)
    if expected:
        assert report.level == worst_level(
            item.level for item in report.results if item.rule_id in expected
        )
    else:
        assert report.level == "pass"


# ------------------------------------------------------------------ 规则库
def test_default_library_declares_the_four_rules_from_the_plan():
    rules = load_rules()
    assert rules.ids == [R1, R2, R3, R4]
    levels = {rule.id: rule.level for rule in rules.rules}
    assert levels == {R1: "block", R2: "warn", R3: "warn", R4: "info"}
    # 责任边界文案只有一份：rules.yaml 与代码常量必须一致
    assert rules.notice == NOTICE
    assert "不做合规判定" in rules.notice


def test_rule_library_is_extensible_without_touching_code():
    """在 YAML 里加一条规则就能生效（事实复用现有键，不改 Python）。"""
    payload = {
        "version": 1,
        "notice": NOTICE,
        "levels": {"block": "阻断", "warn": "警告", "info": "提示"},
        "rules": [
            {
                "id": "too_many_layers",
                "title": "图层过多",
                "level": "info",
                "when": {"all": [{"fact": "layer_count", "op": "gt", "value": 2}]},
                "message": "本图叠了 {layer_count} 个图层，图例可能放不下",
                "advice": "合并同类图层或拆图。",
            }
        ],
    }
    ruleset = parse_ruleset(payload)
    report = check_map(
        _ctx(layers=(_layer("街道"), _layer("社区"), _layer("绿地"))),
        rules=ruleset,
    )
    assert [item.rule_id for item in report.hits] == ["too_many_layers"]
    assert "3 个图层" in report.hits[0].message


def test_each_rule_can_be_switched_off_individually():
    """单条开关：关掉的规则结论是 off（不是 pass），其它规则照常给结论。"""
    context = _ctx(layers=(_layer("国界线", registered=False),))
    report = check_map(context, enabled={R1: False})
    status = {item.rule_id: item.status for item in report.results}
    assert status[R1] == "off"
    assert status[R2] == "hit" and status[R4] == "hit"
    assert report.level == "warn"  # 阻断被关掉，剩下最重的是警告

    all_off = check_map(context, enabled={R1: False, R2: False, R3: False, R4: False})
    assert all_off.level == "pass"
    assert all(item.status == "off" for item in all_off.results)
    assert "未发现问题" in all_off.summary


@pytest.mark.parametrize(
    "payload,expectation",
    [
        ({"version": 2, "notice": "x", "rules": []}, "version"),
        ({"version": 1, "notice": "", "rules": []}, "notice"),
        ({"version": 1, "notice": "x", "rules": []}, "rules"),
        (
            {"version": 1, "notice": "x", "rules": [{"id": "Bad-Id", "title": "t", "level": "warn",
                                                    "when": {"fact": "layer_count", "op": "gt", "value": 0},
                                                    "message": "m"}]},
            "id",
        ),
        (
            {"version": 1, "notice": "x", "rules": [{"id": "unknown_level", "title": "t", "level": "error",
                                                     "when": {"fact": "layer_count", "op": "gt", "value": 0},
                                                     "message": "m"}]},
            "level",
        ),
        (
            {"version": 1, "notice": "x", "rules": [{"id": "bad_fact", "title": "t", "level": "warn",
                                                     "when": {"fact": "nope", "op": "gt", "value": 0},
                                                     "message": "m"}]},
            "nope",
        ),
        (
            {"version": 1, "notice": "x", "rules": [{"id": "bad_op", "title": "t", "level": "warn",
                                                     "when": {"fact": "layer_count", "op": "between", "value": 0},
                                                     "message": "m"}]},
            "between",
        ),
        (
            {"version": 1, "notice": "x", "rules": [{"id": "bad_text", "title": "t", "level": "warn",
                                                     "when": {"fact": "layer_count", "op": "gt", "value": 0},
                                                     "message": "引用了 {nope_fact}"}]},
            "nope_fact",
        ),
        (
            {"version": 1, "notice": "x", "rules": [
                {"id": "dup", "title": "t", "level": "warn",
                 "when": {"fact": "layer_count", "op": "gt", "value": 0}, "message": "m"},
                {"id": "dup", "title": "t2", "level": "warn",
                 "when": {"fact": "layer_count", "op": "gt", "value": 0}, "message": "m"},
            ]},
            "重复",
        ),
    ],
)
def test_broken_rule_library_fails_loudly_at_load_time(payload, expectation):
    """规则写错必须在装载时报错：拼错 fact 会让规则恒假，静默通过比误报更糟。"""
    with pytest.raises(RuleError) as err:
        parse_ruleset(payload)
    assert expectation in str(err.value)


def test_missing_rule_file_is_reported_not_swallowed(tmp_path: Path):
    report = check_map(_ctx(), rules_path=str(tmp_path / "nope.yaml"))
    assert report.error
    assert "读不到规则库" in report.error
    assert "未完成" in report.summary


# ------------------------------------------------------------------ 事实判定
@pytest.mark.parametrize(
    "layers,expected",
    [
        ((), (0, 0)),
        ((_layer("国界线"),), (1, 1)),
        ((_layer("省界"),), (0, 1)),
        ((_layer("海岸线"),), (0, 1)),
        ((_layer("国家级自然保护区"),), (0, 0)),   # "国家级"不是国界
        ((_layer("军事设施"),), (1, 1)),
        ((_layer("national_boundary"),), (1, 1)),
        ((_layer("study_area"),), (0, 0)),
    ],
)
def test_sensitive_and_boundary_layer_detection(layers, expected):
    facts = collect_facts(_ctx(layers=layers))
    assert (facts["sensitive_count"], facts["boundary_count"]) == expected


@pytest.mark.parametrize(
    "texts,expected",
    [
        ((), ("absent", "未见审图号")),
        (("审图号：GS(2023)1234号",), ("valid", "审图号 GS(2023)1234号")),
        (("GS（2019）第1822号",), ("valid", "审图号 GS（2019）第1822号")),
        (("审图号：GS2023-1234",), ("malformed", None)),
        (("审图号：2023-1234",), ("malformed", None)),
        (("地图审图号待补",), ("malformed", None)),
    ],
)
def test_approval_number_is_read_from_map_texts(texts, expected):
    facts = collect_facts(_ctx(texts=texts))
    assert facts["approval"] == expected[0]
    if expected[1]:
        assert facts["approval_note"] == expected[1]
    else:
        assert "格式无法识别" in facts["approval_note"]


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "absent"),
        ({"texts": ("未送审",)}, "present"),
        ({"texts": ("内部资料 · 禁止外传",)}, "present"),
        ({"watermark": "内部使用"}, "present"),
        ({"texts": ("已送审",)}, "absent"),   # "已送审"不是水印
        ({"texts": ("Not For Publication",)}, "present"),
    ],
)
def test_watermark_detection(kwargs, expected):
    assert collect_facts(_ctx(**kwargs))["watermark"] == expected


def test_catalog_unavailable_never_accuses_missing_source():
    """拿不到数据目录时，来源状态是 unknown：不许把"没核验"说成"未登记"。"""
    facts = collect_facts(_ctx(layers=(_layer("省界", registered=False),), catalog=False))
    assert facts["boundary_source"] == "unknown"
    assert facts["unregistered_count"] == 0
    report = check_map(_ctx(layers=(_layer("省界", registered=False),), catalog=False))
    assert R2 not in _fired(report)


def test_audience_normalization_only_internal_is_special():
    from gis_cli.compliance.facts import normalize_audience

    assert normalize_audience("internal") == "internal"
    assert normalize_audience("内部") == "internal"
    assert normalize_audience("public") == "public"
    assert normalize_audience("") == "public"      # 漏标按对外处理（更保守）
    assert normalize_audience(None) == "public"


# ------------------------------------------------------------------ 纯函数
def test_compliance_modules_never_import_arcpy_or_gis_stack():
    """闸门是纯 Python 判定：不碰 arcpy / GDAL / 模型，才能在没有 ArcGIS 的机器上跑。"""
    forbidden = {"arcpy", "numpy", "PIL", "rasterio", "geopandas", "requests", "openai"}
    for path in sorted(COMPLIANCE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not (set(names) & forbidden), f"{path.name} 不该 import {sorted(set(names) & forbidden)}"


def test_check_map_writes_nothing_to_disk(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = check_map(_ctx(layers=(_layer("国界线"),)))
    assert report.level in {"block", "warn", "info", "pass"}
    assert list(tmp_path.iterdir()) == []


def test_report_serializes_and_restores_for_the_verifier():
    """``to_dict`` / ``from_dict`` 是边车与验收之间的契约，必须能原样往返。"""
    report = check_map(_ctx(layers=(_layer("国界线", registered=False),)))
    restored = ComplianceReport.from_dict(report.to_dict())
    assert restored.level == report.level == "block"
    assert [item.rule_id for item in restored.hits] == [item.rule_id for item in report.hits]
    assert restored.facts["boundary_source"] == report.facts["boundary_source"]
    # 存档里必须带责任边界文案，验收/UI 直接拿来显示
    assert "不做合规判定" in report.to_dict()["notice"]


# ------------------------------------------------------------------ 规则库落点
def test_packaged_rule_library_sits_inside_the_package():
    """内置规则库必须跟包走（pyproject 里声明了 package-data）。"""
    from gis_cli.compliance import rules as rules_mod

    assert rules_mod.PACKAGED_RULES == COMPLIANCE_DIR / "rules.yaml"
    assert rules_mod.PACKAGED_RULES.is_file()
    assert rules_mod.OVERRIDE_RULES.name == "compliance_rules.yaml"
    assert rules_mod.OVERRIDE_RULES.parent.name == "config"


def test_config_override_wins_over_the_packaged_library(tmp_path: Path, monkeypatch):
    """``config/compliance_rules.yaml`` 存在时用它（用户改规则不改源码）。"""
    from gis_cli.compliance import rules as rules_mod

    override = tmp_path / "compliance_rules.yaml"
    override.write_text(
        "version: 1\n"
        f"notice: \"{NOTICE}\"\n"
        "rules:\n"
        "  - id: demo_only\n"
        "    title: 演示规则\n"
        "    level: info\n"
        "    when:\n"
        "      all:\n"
        "        - fact: layer_count\n"
        "          op: gt\n"
        "          value: 0\n"
        "    message: \"本次叠了 {layer_count} 个图层\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rules_mod, "OVERRIDE_RULES", override)

    loaded = rules_mod.load_rules()
    assert loaded.ids == ["demo_only"]
    report = check_map(_ctx(layers=(_layer("街道"),)), rules=loaded)
    assert [item.rule_id for item in report.hits] == ["demo_only"]


def test_collect_facts_tolerates_an_empty_context():
    """没有上下文 = 没有任何证据：事实全零，而不是抛异常。"""
    facts = collect_facts(None)  # type: ignore[arg-type]
    assert facts["layer_count"] == 0
    assert facts["boundary_source"] == "unknown"
    # 空上下文：只剩“审图号缺失”的提示（提示级不拦交付）
    report = check_map(None)  # type: ignore[arg-type]
    assert report.level == "info" and not report.blocked


def test_sensitive_terms_in_the_title_are_enough():
    """真实图件的图层名常常是 boundary.shp 这类泛名，国界线索全在图名上。"""
    report = check_map(_ctx(texts=("某国国界线示意图",), layers=(_layer("study_area"),)))
    assert R1 in _fired(report)
    assert "图名/注记提到" in report.hits[0].message
    # 已取得审图号就不阻断（文字线索与图层线索走同一条件）
    report = check_map(_ctx(texts=("某国国界线示意图", "GS(2023)1234号")))
    assert R1 not in _fired(report)
