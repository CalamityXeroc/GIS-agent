# -*- coding: utf-8 -*-
"""成果溯源卡（P1-4）：数据 → 步骤 → 参数 → 验证，一页可审计。

硬约束（P1 实施方案 §2）：

- **只复用现有采集**：数据来自 ``narrative.source_rows()``，步骤与参数来自
  ``narrative.recipes_from_trace()``，验证来自调用方传入的 ``verdict``，
  产物来自 ``state.artifacts``。这里不新增埋点、不调模型、不碰 arcpy。
- :func:`build_card` 是纯函数：只读入参（生成时间由 ``generated_at`` 注入，
  不传才取当下），不联网、不写盘。
- :func:`render_card_html` 返回**内联样式、零外链**的 ``<section>``：嵌进成果包
  就是一节，套上 :func:`render_card_page` 就是能离线打开的独立页面。
- 拿不到证据的段落写"未记录"，原因收进 ``card.missing``——绝不补一句
  "效果良好"。
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import narrative
from ..compliance import NOTICE as COMPLIANCE_NOTICE

logger = logging.getLogger(__name__)

#: 独立卡片文件名（成果包里写 ``output/溯源卡.html``）。
CARD_NAME = "溯源卡.html"
#: 卡面时间的写法（默认值与调用点共用，避免两处格式漂移）。
CARD_TIME_FORMAT = "%Y-%m-%d %H:%M"
#: 单卡体积的**预期值**（验收条件的观测口径，不是代码里的硬闸门：超了只由
#: 测试报警，代码不会截断或拒绝渲染）。
EXPECTED_MAX_CARD_BYTES = 200 * 1024
#: 每段最多几行（总计划风险项：溯源卡不能变成流水账）。
MAX_LINES = 8
#: 一行里单个值的字数上限。
TEXT_LIMIT = 160
#: 缺证据时的固定写法。
NOT_RECORDED = "未记录"

#: 卡片里不允许出现"看起来能点"的外链：命中后把冒号转成实体（渲染出来仍是
#: 原样文字，但字符串里不再有 ``http://`` / ``https://``）。
_URL_SCHEME = re.compile(r"https?://", re.IGNORECASE)

_SECTION_STYLE = (
    "margin:24px 0;padding:18px 20px;border:1px solid #dbe2ea;border-radius:4px;"
    "background:#fbfcfd;color:#1b2430;"
    "font:14px/1.7 -apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif"
)
_META_STYLE = "margin:0 0 4px;color:#5c6a7a;font-size:13px"
_H3_STYLE = "margin:16px 0 6px;font-size:15px;line-height:1.4;color:#155e9b"
_LIST_STYLE = "margin:0;padding-left:20px"
_EMPTY_STYLE = "margin:0;color:#5c6a7a;font-size:13px"
_FOOT_STYLE = "margin:18px 0 0;color:#5c6a7a;font-size:12px"


@dataclass
class ProvenanceCard:
    """一次运行的溯源卡（每段都是已经格式化好的短句，渲染层只排版）。"""

    run_id: str = ""
    generated_at: str = ""
    goal: str = ""
    data: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    verifications: list[str] = field(default_factory=list)
    #: 合规闸门结论（只复述 create_map / 验收已经算好的分级结论，这里不做判断）
    compliance: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ 小工具
def _text(value: Any, limit: int = TEXT_LIMIT) -> str:
    """压成一行并截断（报告里的长句一律先过这里）。"""
    flat = " ".join(str(value if value is not None else "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _value_text(value: Any) -> str:
    """参数值写给人看：容器走 JSON，标量原样。"""
    if isinstance(value, (dict, list, tuple)):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:  # pragma: no cover - 极端值，退回 str
            value = str(value)
    return _text(value, 80)


def _clip(lines: list[str], unit: str = "项") -> list[str]:
    """每段封顶 ``MAX_LINES`` 行，超出的写一句"另有 N 项"（不静默截断）。"""
    if len(lines) <= MAX_LINES:
        return lines
    rest = len(lines) - (MAX_LINES - 1)
    return lines[: MAX_LINES - 1] + [f"…另有 {rest} {unit}未列出"]


def _esc(value: Any) -> str:
    text = html.escape(str(value if value is not None else ""), quote=True)
    return _URL_SCHEME.sub(lambda m: m.group(0).replace(":", "&#58;"), text)


# ------------------------------------------------------------------ 四段取材
def _data_lines(catalog: Any, workspace: Any) -> tuple[list[str], str]:
    """数据段：catalog 登记的名称/来源/许可/采集时间（复用 ``source_rows``）。"""
    if catalog is None:
        return [], "数据：没有数据目录（catalog），来源与许可无从登记"
    try:
        rows = narrative.source_rows(catalog, workspace)
    except Exception as exc:  # 老库/锁/实现异常都不该拖垮卡片
        logger.debug("溯源卡读 catalog 失败: %s", exc)
        return [], "数据：读数据目录失败，来源与许可未记录"
    if not rows:
        return [], "数据：数据目录里没有本次用到的数据登记"
    lines = [
        _text(f"{row.name} · {row.source} · 许可 {row.license} · 取得 {row.acquired_at} · {row.path}")
        for row in rows
    ]
    return _clip(lines), ""


def _recipe_calls(trace_path: Any, recipes: Any) -> list[narrative.RecipeCall]:
    """trace 里的配方调用（配方名从配方库补，库不在就退回 id）。"""
    try:
        calls = narrative.recipes_from_trace(trace_path)
    except Exception as exc:  # pragma: no cover - recipes_from_trace 已吞掉坏行
        logger.debug("溯源卡读 trace 失败: %s", exc)
        return []
    for call in calls:
        if recipes is not None:
            call.title = narrative.recipe_title(recipes, call.recipe_id)
    return calls


def _step_gap(trace_path: Any, calls: Sequence[narrative.RecipeCall]) -> str:
    if calls:
        return ""
    if not trace_path:
        return "步骤：本次没有 trace 文件，执行过程未记录"
    path = Path(str(trace_path))
    name = _text(path.name, 60)
    if not path.exists():
        return f"步骤：trace 文件不存在（{name}），执行过程未记录"
    if path.is_dir():
        return f"步骤：trace 路径是目录而非文件（{name}），执行过程未记录"
    try:
        empty = path.stat().st_size == 0
    except OSError:  # pragma: no cover - 权限/占用
        empty = False
    if empty:
        return f"步骤：trace 为空（{name}），执行过程未记录"
    return "步骤：trace 里没有配方调用，执行过程未记录"


def _step_lines(calls: Iterable[narrative.RecipeCall]) -> list[str]:
    lines = []
    for call in calls:
        status = {True: "执行成功", False: "执行失败", None: "结果未记录"}[call.ok]
        times = f"，共 {call.count} 次" if call.count > 1 else ""
        lines.append(_text(f"{call.title}（{call.recipe_id}）：{status}{times}"))
    return _clip(lines)


def _param_lines(calls: Iterable[narrative.RecipeCall]) -> list[str]:
    """参数段：一行一个实际调用值（不写默认值，不写没传的参数）。"""
    lines = [
        _text(f"{call.title} · {key}={_value_text(value)}")
        for call in calls
        for key, value in call.params.items()
    ]
    return _clip(lines)


def _param_gap(calls: Sequence[narrative.RecipeCall]) -> str:
    if not calls:
        return ""  # 步骤段已经说明"没有 trace"，不必重复
    if not any(call.params for call in calls):
        return "参数：trace 里的配方调用没有入参记录"
    return ""


def _verification_lines(
    verdict: dict[str, Any] | None,
    requirements: Iterable[Any],
) -> tuple[list[str], str]:
    """验证段：验收结论 + 断言明细（复用 ``plain_verifications`` 的人话口径）。"""
    if not isinstance(verdict, dict) or not verdict:
        return [], "验证：本次没有验收结论（verdict），断言明细未记录"
    lines: list[str] = []
    if "pass" in verdict:
        lines.append("验收结论：" + ("通过" if verdict.get("pass") else "未通过"))
    if verdict.get("degraded"):
        lines.append("降级放行：同一问题重复出现且成果未变化，需人工复核")
    try:
        lines.extend(narrative.plain_verifications(verdict, requirements))
    except Exception as exc:
        logger.debug("溯源卡读 verdict 失败: %s", exc)
        return _clip(lines), "验证：验收结论解析失败，断言明细未记录"
    if not lines:
        return [], "验证：没有可复述的断言明细"
    return _clip(lines), ""


def _compliance_lines(data: Any) -> list[str]:
    """合规闸门段：只复述验收结论里的 ``verdict["compliance"]``（不重新判定）。

    总计划要求"责任边界与溯源卡同一节呈现"，所以这一段紧跟在"验证"后，
    并在渲染时带上闸门自己的免责声明（见 :data:`COMPLIANCE_NOTICE`）。

    这里**不往** ``card.missing`` 里塞东西：这一节自己会写清楚"没有自检记录"，
    而 missing 是"某段证据缺了"的清单，不该被每个非制图任务刷屏。
    """
    if not isinstance(data, Mapping):
        return ["未记录（本次验收没有合规自检结论：不是 create_map 出的图，或旧版本出图）"]
    if data.get("error"):
        return ["闸门未完成：" + _text(data.get("error"), 120)]
    findings = [item for item in (data.get("findings") or []) if isinstance(item, Mapping)]
    checked = int(data.get("checked") or 0)
    lines: list[str] = []
    level = str(data.get("level") or "none")
    label = {"block": "阻断", "warn": "警告", "info": "提示"}.get(level, "未发现阻断/警告")
    lines.append(_text(f"总体结论：{label}（检查 {checked} 张图的合规自检记录）"))
    for item in findings:
        level_name = {"block": "阻断", "warn": "警告", "info": "提示"}.get(str(item.get("level")), "提示")
        where = f"［{item.get('image')}］" if item.get("image") else ""
        lines.append(_text(f"{level_name} · {item.get('title') or item.get('rule_id')}：{item.get('message')}{where}"))
    if not findings and checked:
        lines.append("未发现需要提醒的问题")
    return _clip(lines)


def _artifact_lines(artifacts: Iterable[Any], workspace: Any) -> tuple[list[str], str]:
    paths = [str(item) for item in (artifacts or []) if str(item).strip()]
    if not paths:
        return [], "产物：本次没有登记产物"
    lines = [_text(narrative.relative_path(path, workspace)) for path in paths]
    return _clip(lines), ""


# ------------------------------------------------------------------ 组装
def build_card(
    *,
    state: Any = None,
    verdict: dict[str, Any] | None = None,
    catalog: Any = None,
    trace_path: str | Path | None = None,
    recipes: Any = None,
    workspace: str | Path | None = None,
    generated_at: str | None = None,
) -> ProvenanceCard:
    """把一次运行的材料整理成溯源卡（纯函数：只读入参，不写盘）。

    ``generated_at`` 是入参：传了就照用（同输入同输出，便于比对/加指纹），
    不传才取当下时间。

    任何一段拿不到证据都不抛异常：那一段留空（渲染成"未记录"），原因进
    ``card.missing``。
    """
    card = ProvenanceCard(
        run_id=_text(getattr(state, "run_id", "") or "", 80),
        generated_at=(
            generated_at
            if generated_at is not None
            else datetime.now().strftime(CARD_TIME_FORMAT)
        ),
        goal=_text(getattr(state, "goal", "") or "", 200),
    )

    calls = _recipe_calls(trace_path, recipes)
    data, data_gap = _data_lines(catalog, workspace)
    steps = _step_lines(calls)
    params = _param_lines(calls)
    verifications, verify_gap = _verification_lines(
        verdict, getattr(state, "requirements", []) or []
    )
    compliance = _compliance_lines(verdict.get("compliance") if isinstance(verdict, Mapping) else None)
    artifacts, artifact_gap = _artifact_lines(getattr(state, "artifacts", []) or [], workspace)

    card.data = data
    card.steps = steps
    card.params = params
    card.verifications = verifications
    card.compliance = compliance
    card.artifacts = artifacts
    for gap in (data_gap, _step_gap(trace_path, calls), _param_gap(calls), verify_gap, artifact_gap):
        if gap:
            card.missing.append(gap)
    return card


# ------------------------------------------------------------------ 渲染
def _meta_line(card: ProvenanceCard) -> str:
    bits = [
        f"运行 {_esc(card.run_id or NOT_RECORDED)}",
        f"生成 {_esc(card.generated_at or NOT_RECORDED)}",
    ]
    if card.goal:
        bits.append(f"目标：{_esc(card.goal)}")
    return "　·　".join(bits)


def _block(title: str, lines: Sequence[str]) -> str:
    head = f'<h3 style="{_H3_STYLE}">{_esc(title)}</h3>'
    if not lines:
        return head + f'<p style="{_EMPTY_STYLE}">{NOT_RECORDED}</p>'
    items = "".join(f"<li>{_esc(line)}</li>" for line in lines)
    return head + f'<ul style="{_LIST_STYLE}">{items}</ul>'


def _missing_block(lines: Sequence[str]) -> str:
    if not lines:
        return ""
    items = "".join(f"<li>{_esc(line)}</li>" for line in lines)
    return f'<h3 style="{_H3_STYLE}">{NOT_RECORDED}</h3><ul style="{_LIST_STYLE}">{items}</ul>'


def render_card_html(card: ProvenanceCard) -> str:
    """内联样式、零外链的 ``<section>``（可嵌成果包，也可独立成页）。"""
    return "\n".join(
        part
        for part in [
            f'<section style="{_SECTION_STYLE}">',
            "<h2>溯源卡</h2>",
            f'<p style="{_META_STYLE}">{_meta_line(card)}</p>',
            _block("数据", card.data),
            _block("步骤", card.steps),
            _block("参数", card.params),
            _block("验证", card.verifications),
            _block("合规闸门", card.compliance),
            f'<p style="{_EMPTY_STYLE}">{_esc(COMPLIANCE_NOTICE)}</p>',
            _block("产物", card.artifacts),
            _missing_block(card.missing),
            f'<p style="{_FOOT_STYLE}">'
            "本卡只复述本次运行的登记证据（数据目录、trace、验收结论），不构成合规审查结论。"
            "</p>",
            "</section>",
        ]
        if part
    )


_PAGE_CSS = """
body{margin:0;background:#eef1f5;color:#1b2430}
main{max-width:900px;margin:0 auto;padding:32px 24px 48px;background:#fff}
main section{margin:0}
h2{font-size:20px;margin:0 0 10px;padding-left:10px;border-left:5px solid #155e9b}
@media (max-width:640px){main{padding:20px 14px 32px}}
@media print{body{background:#fff}main{max-width:none;padding:0}}
"""


def render_card_page(card: ProvenanceCard) -> str:
    """独立成页：同一份 ``<section>`` 套一层最小外壳（仍然零外链）。"""
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>溯源卡 · {_esc(card.run_id or '本次运行')}</title>",
            f"<style>{_PAGE_CSS}</style>",
            "</head>",
            "<body>",
            f"<main>{render_card_html(card)}</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


__all__ = [
    "CARD_NAME",
    "CARD_TIME_FORMAT",
    "EXPECTED_MAX_CARD_BYTES",
    "MAX_LINES",
    "NOT_RECORDED",
    "ProvenanceCard",
    "build_card",
    "render_card_html",
    "render_card_page",
]
