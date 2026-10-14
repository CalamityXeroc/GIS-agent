# -*- coding: utf-8 -*-
"""渲染层：把引擎事件画成"像主流 agent CLI"的转录。

消息角色（一屏最多三色，避免花）::

    你 ▸ 用户输入                     亮白 + 加粗
    ✻ 思考 · 1.2s                     浅灰（模型真实推理，默认展开）
    │ 推理正文（左侧竖线引导，悬挂缩进）
    ⏺ 工具 ×N                         青
    └ 一行摘要（长内容折叠，Ctrl+O 展开）
    ● Agent                           亮白（最终答复，独立成块）
    · 系统/状态                       浅灰

关键约定：
- **工具调用合并**：连续同一工具合并为 ``⏺ tool ×N``，只输出**一个** ``└`` 摘要行；
- **答案不埋在工具行里**：``finish`` 的正文单独渲染成 ``● Agent`` 块；
- 长文本（修复根因、工具原始输出）一律折叠，正文只留一行，避免刷屏。
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Iterable

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .events import RenderUnit, StatusTracker, cleanup_error, format_duration, format_tokens, unescape_text
from .theme import glyph

#: 状态 → 角色色
_STATUS_ROLE = {
    "completed": "gis.ok",
    "failed": "gis.err",
    "max_turns": "gis.warn",
    "deadline_exceeded": "gis.warn",
    "llm_unavailable": "gis.err",
    "runner_error": "gis.err",
    "stopped": "gis.warn",
}

#: 正文折行宽度（留出缩进与边框）
_BODY_WIDTH = 88
#: 思考块最多显示的行数（默认展开，但极端长的推理不至于淹没屏幕）
_THOUGHT_MAX_LINES = 40


from rich.cells import cell_len


def disp_width(text: str) -> int:
    """显示宽度（中文按 2 列）——按字符数算宽度是之前中文行溢出的根因。"""
    return cell_len(str(text or ""))


def _cut_by_width(line: str, width: int) -> tuple[str, str]:
    """按显示宽度切一刀，返回 (头部, 尾部)。"""
    used = 0
    for index, char in enumerate(line):
        char_width = disp_width(char)
        if used + char_width > width:
            return line[:index], line[index:]
        used += char_width
    return line, ""


def wrap_lines(text: str, width: int = _BODY_WIDTH) -> list[str]:
    """按显示宽度折行：优先在空格断；不劈单词；不留 ``3.`` 这种孤行。"""
    out: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        while disp_width(line) > width:
            head, tail = _cut_by_width(line, width)
            space = head.rfind(" ")
            # 尽量在空格处断开；且别让尾部只剩 "3." 这种短碎片
            if space > 0 and disp_width(tail.strip()) >= 6:
                head, tail = head[:space], line[space + 1:]
            elif space > 0 and disp_width(head[:space]) >= width // 2:
                head, tail = head[:space], line[space + 1:]
            out.append(head.rstrip())
            line = tail.lstrip()
        if line:
            out.append(line)
    return out


def _wrap(text: str, width: int = _BODY_WIDTH, indent: str = "") -> list[str]:
    """折行 + 统一缩进（结果卡片等沿用）。"""
    return [indent + line if line else "" for line in wrap_lines(text, width)]


def prefixed_block(
    lines: list[str],
    *,
    cont_prefix: str,
    last_prefix: str,
    cont_style: str,
    text_style: str,
) -> Text:
    """给折行后的正文加行首引导：中间行 │，最后一行 └。"""
    out = Text()
    total = len(lines)
    for index, line in enumerate(lines):
        is_last = index == total - 1
        out.append(last_prefix if is_last else cont_prefix, style=cont_style)
        out.append(line, style=text_style)
        if not is_last:
            out.append("\n")
    return out


def _short(text: str, limit: int = 88) -> str:
    """按显示宽度截断（中文场景下按字符数截断会导致行溢出）。"""
    text = " ".join(str(text or "").split())
    if disp_width(text) <= limit:
        return text
    head, _tail = _cut_by_width(text, max(1, limit - 1))
    return head + "…"


class ToolOutputStore:
    """折叠的详细输出（``Ctrl+O`` 展开时取用）。只保留最近 ``limit`` 条。"""

    def __init__(self, limit: int = 40) -> None:
        self._items: deque[tuple[str, str]] = deque(maxlen=limit)

    def add(self, title: str, body: str) -> None:
        if body:
            self._items.append((title, body))

    def latest(self) -> tuple[str, str] | None:
        return self._items[-1] if self._items else None

    def all(self) -> list[tuple[str, str]]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


# ------------------------------------------------------------------ 单块渲染


def user_block(text: str) -> Text:
    """用户输入：``你 ▸ …``（亮白加粗）。"""
    lines = wrap_lines(text, _BODY_WIDTH - 2) or [""]
    out = Text()
    out.append(f"{glyph('user')} ", style="gis.user.tag")
    for index, line in enumerate(lines):
        if index:
            out.append("\n  ")
        out.append(line, style="gis.user")
    return out


def thought_block(unit: RenderUnit, *, elapsed: float | None = None) -> Text:
    """思考块：标题一行 + 竖线引导的正文（默认展开，浅灰）。"""
    out = Text()
    bar = glyph("bar")
    out.append(f"{glyph('thought')} ", style="gis.thought.tag")
    out.append("思考", style="gis.thought.tag")
    if elapsed is not None and elapsed > 0.05:
        out.append(f" · {format_duration(elapsed)}", style="gis.thought.tag")
    if not unit.meta.get("has_reasoning", True):
        out.append("（模型自述）", style="gis.thought.tag")
    out.append("\n")

    body = (unit.text or "").strip()
    if not body:
        actions = unit.meta.get("actions") or []
        body = "调用：" + ", ".join(actions) if actions else "（本轮无额外说明）"
    # 折行宽度必须按控制台实际宽度来（否则 rich 会二次折行，续行会跑到最左边没有 │ 引导）
    width = int(unit.meta.get("_width") or 0) or _BODY_WIDTH
    lines: list[str] = []
    for paragraph in body.splitlines():
        lines.extend(wrap_lines(paragraph, max(24, width - 2)))
    lines = [line for line in lines if line != ""] or ["（无）"]
    if len(lines) > _THOUGHT_MAX_LINES:
        hidden = len(lines) - _THOUGHT_MAX_LINES
        lines = lines[:_THOUGHT_MAX_LINES] + [f"…（另有 {hidden} 行）"]
    block = prefixed_block(
        lines,
        cont_prefix=f"{bar} ",
        last_prefix=f"{glyph('cont')} ",
        cont_style="gis.thought.bar",
        text_style="gis.thought",
    )
    block.no_wrap = True
    block.overflow = "fold"
    out.append(block)
    return out


def tool_call_line(
    tool: str,
    *,
    args: str = "",
    count: int = 1,
    elapsed: float | None = None,
    result: str = "",
    ok: bool = True,
    folded: bool = False,
    budget: int = 0,
) -> Text:
    """工具行：``⏺ tool ×N  参数 · 结果摘要  耗时``。

    结果**内联**在同一行（主流 CLI 的做法），避免每个工具都拖一行 ``└``；
    只有失败时才由调用方另起一行显示细节。
    """
    # 宽度预算：结果摘要比参数重要，先砍参数、再砍摘要，保证一行放得下
    if budget:
        overhead = 4 + len(tool) + (len(f" ×{count}") if count > 1 else 0)
        room = budget - overhead
        if result:
            result_budget = max(24, int(room * 0.55))
            result = _short(result, result_budget)
            room -= disp_width(result) + 3
        if args and room < 24:
            args = _short(args, max(12, room))
        elif args and disp_width(args) > room:
            args = _short(args, max(12, room))

    out = Text()
    out.append(f"{glyph('step')} ", style="gis.tool")
    out.append(tool, style="gis.tool_name")
    if count > 1:
        out.append(f" ×{count}", style="gis.num")
    if args:
        out.append("  " + args, style="gis.tool.summary")
    if result:
        if not ok:
            out.append(f"  {glyph('bad')} ", style="gis.err")
            out.append(result, style="gis.err")
        else:
            out.append(f"  {glyph('bullet')} ", style="gis.system")
            out.append(result, style="gis.tool.summary")
    if folded:
        out.append("  Ctrl+O 明细", style="gis.system")
    if elapsed is not None and elapsed > 0.05:
        out.append(f"  {format_duration(elapsed)}", style="gis.system")
    return out


def tool_result_line(summary: str, *, ok: bool = True, folded: bool = False) -> Text:
    """工具结果行（每个工具组**只出现一次** ``└``）。"""
    out = Text()
    out.append(f"{glyph('cont')} ", style="gis.thought.bar")
    out.append(f"{glyph('ok')} " if ok else f"{glyph('bad')} ", style="gis.ok" if ok else "gis.err")
    out.append(summary, style="gis.tool.summary" if ok else "gis.err")
    if folded:
        out.append("   ⌃O 明细", style="gis.system")
    return out


def answer_block(text: str, *, width: int = 0) -> Text:
    """最终答复：``● Agent`` 独立成块（亮白）。"""
    width = width or _BODY_WIDTH
    out = Text()
    out.append(f"{glyph('answer')} ", style="gis.answer.tag")
    out.append("Agent", style="gis.answer.tag")
    out.append("\n")
    body = str(text or "").strip() or "（无说明）"
    # 折行宽度跟控制台一致，避免 rich 二次折行（续行会顶到最左边）
    lines: list[str] = []
    for paragraph in body.splitlines():
        lines.extend(wrap_lines(paragraph, max(24, width - 4)) or [""])
    for index, line in enumerate(lines):
        out.append("  " + line, style="gis.answer")
        if index != len(lines) - 1:
            out.append("\n")
    out.no_wrap = True
    out.overflow = "fold"
    return out


def system_line(text: str, *, role: str = "gis.system", mark: str = "") -> Text:
    """系统/状态行：``· 文本``（浅灰）。"""
    out = Text()
    out.append(f"{mark or glyph('dot')} ", style=role)
    out.append(text, style=role)
    return out


def turn_rule(turn: int, *, width: int = _BODY_WIDTH) -> Text:
    """轮次分隔线（浅灰）。"""
    label = f"{glyph('dot')} 第 {turn} 轮 "
    tail = max(0, width - len(label) - 1)
    out = Text()
    out.append(label, style="gis.system")
    out.append(glyph("rule") * tail, style="gis.thought.bar")
    return out


# ------------------------------------------------------------------ 转录视图


class TranscriptView:
    """把事件流渲染成转录，并**合并连续工具调用**。

    用法::

        view = TranscriptView(console, tracker, store)
        view.handle("llm_result", payload)   # 引擎事件直接喂进来
        view.flush()                          # 一轮结束时清空挂起的工具组

    这类"先挂起、遇到非工具事件再输出"的做法，是把 ``catalog_query ×12``
    收成一行、避免满屏 ``└`` 的关键。
    """

    def __init__(self, console: Any, tracker: StatusTracker, store: ToolOutputStore | None = None) -> None:
        self.console = console
        self.tracker = tracker
        # 注意：ToolOutputStore 定义了 __len__，空实例是 falsy，
        # 所以这里必须显式判 None（`store or ...` 会把调用方传入的空 store 换掉）。
        self.store = store if store is not None else ToolOutputStore()
        self._pending_tool: str = ""
        self._pending_args: str = ""
        self._pending_count: int = 0
        self._pending_elapsed: float = 0.0
        self._pending_result: str = ""
        self._pending_result_ok: bool = True
        self._pending_folded: bool = False
        self._last_turn: int = 0

    # -- 工具组缓冲 ----------------------------------------------------

    def _flush_tools(self) -> None:
        if not self._pending_count:
            return
        ok = self._pending_result_ok
        summary = self._pending_result if ok else ""
        budget = max(40, int(getattr(self.console, "width", 0) or _BODY_WIDTH) - 2)
        self.console.print(
            tool_call_line(
                self._pending_tool,
                args=self._pending_args,
                count=self._pending_count,
                elapsed=self._pending_elapsed or None,
                result=summary,
                ok=ok,
                folded=self._pending_folded,
                budget=budget,
            )
        )
        if not ok and self._pending_result:
            # 失败才另起一行：用 └ 收口，明确指出问题
            self.console.print(tool_result_line(self._pending_result, ok=False))
        self._pending_tool = ""
        self._pending_args = ""
        self._pending_count = 0
        self._pending_elapsed = 0.0
        self._pending_result = ""
        self._pending_result_ok = True
        self._pending_folded = False

    def _buffer_tool_call(self, tool: str, args: str) -> None:
        if self._pending_count and tool != self._pending_tool:
            self._flush_tools()
        self._pending_tool = tool
        self._pending_count += 1
        if args and not self._pending_args:
            self._pending_args = args

    def _buffer_tool_result(self, summary: str, ok: bool, folded: bool) -> None:
        # 只保留该组最后一条结果作摘要（前面的同工具结果没信息量）
        self._pending_result = summary
        self._pending_result_ok = ok
        self._pending_folded = folded or self._pending_folded
        if folded:
            self.store.add(self._pending_tool or "tool", summary)

    # -- 事件入口 ------------------------------------------------------

    def handle(self, event: str, payload: dict | None = None) -> None:
        """处理一个引擎事件（与 ``on_event`` 同签名）。"""
        from . import events as ui_events

        data = dict(payload or {})
        unit = ui_events.to_unit(event, data)
        if unit is None:
            return
        self.render(unit, event=event, elapsed=self.tracker.action_elapsed)

    def render(self, unit: RenderUnit, *, event: str = "", elapsed: float | None = None) -> None:
        kind = unit.kind
        # 让折行按真实控制台宽度算（否则续行会失去 │ 引导）
        unit.meta["_width"] = max(40, int(getattr(self.console, "width", 0) or _BODY_WIDTH) - 2)
        if kind == "step" and unit.text == "finish":
            # finish 的正文由 ● Agent 块呈现，这里不再重复一行工具调用
            return

        if kind == "step":
            self._buffer_tool_call(unit.text, (unit.lines or [""])[0])
            return
        if kind == "result":
            self._buffer_tool_result(unit.text, bool(unit.ok), bool(unit.lines))
            return

        # 任何非工具事件都先冲掉挂起的工具组，保证顺序正确
        self._flush_tools()

        if kind == "banner":
            # 用户刚打过这句话，不再重复回显（避免"你 ▸ …"紧跟着"任务 …"）
            return
        if kind == "turn":
            match = re.search(r"\d+", unit.text or "")
            turn = int(match.group(0)) if match else 0
            if turn and turn != self._last_turn:
                self._last_turn = turn
                width = max(40, int(getattr(self.console, "width", 0) or _BODY_WIDTH) - 2)
                self.console.print(turn_rule(turn, width=width))
            return
        if kind == "thought":
            self.console.print(thought_block(unit, elapsed=elapsed))
            return
        if kind == "answer":
            self.console.print(answer_block(unit.text, width=int(unit.meta.get("_width") or 0)))
            return
        if kind == "note":
            self.console.print(system_line(_short(unit.text, _BODY_WIDTH)))
            return
        if kind == "verify":
            passed = bool(unit.ok)
            detail = unit.text or ""
            for prefix in ("验收通过", "验收未通过"):
                if detail.startswith(prefix):
                    detail = detail[len(prefix):].lstrip("：: ")
                    break
            mark = glyph("ok") if passed else glyph("bad")
            detail = detail.strip().strip("（）()")
            suffix = f" · {_short(detail, 60)}" if detail else ""
            self.console.print(
                system_line(
                    f"验收{'通过' if passed else '未通过'}{suffix}",
                    role="gis.ok" if passed else "gis.err",
                    mark=mark,
                )
            )
            return
        if kind in ("warn", "error"):
            role = "gis.warn" if kind == "warn" else "gis.err"
            mark = glyph("warn") if kind == "warn" else glyph("bad")
            width = max(30, int(getattr(self.console, "width", 0) or _BODY_WIDTH) - 6)
            head = unit.text or ""
            detail = (unit.lines[0] if unit.lines else "") or ""
            tail = "  Ctrl+O 明细" if (detail and len(detail) > 40) else ""
            if detail:
                # 明细进 Ctrl+O，行内只留一小段；预算里要算上尾标与显示宽度
                room = max(16, width - disp_width(head) - disp_width(tail) - 4)
                head = f"{head} · {_short(detail, room)}"
            line_text = _short(head + tail, width)
            self.console.print(system_line(line_text, role=role, mark=mark))
            if detail and len(detail) > 40:
                self.store.add(unit.text or "明细", detail)
            return
        if kind == "final":
            self._flush_tools()
            return
        self.console.print(system_line(_short(unit.text, _BODY_WIDTH)))

    def flush(self) -> None:
        """一轮结束/退出前调用，确保挂起的工具组被打印。"""
        self._flush_tools()


# ------------------------------------------------------------------ 兼容接口


def render_unit(unit: RenderUnit, *, elapsed: float | None = None) -> RenderableType:
    """单条渲染（无状态版，供测试与简单场景使用）。"""
    if unit.kind == "banner":
        out = Text()
        out.append("任务  ", style="gis.status.key")
        out.append(unit.text, style="gis.agent")
        return out
    if unit.kind == "step":
        return tool_call_line(unit.text, args=(unit.lines or [""])[0], elapsed=elapsed)
    if unit.kind == "result":
        return tool_result_line(unit.text, ok=bool(unit.ok), folded=bool(unit.lines))
    if unit.kind == "thought":
        return thought_block(unit, elapsed=elapsed)
    if unit.kind == "answer":
        return answer_block(unit.text)
    if unit.kind == "verify":
        return system_line(unit.text, role="gis.ok" if unit.ok else "gis.err", mark=glyph("ok") if unit.ok else glyph("bad"))
    if unit.kind in ("warn", "error"):
        role = "gis.warn" if unit.kind == "warn" else "gis.err"
        return system_line(unit.text, role=role, mark=glyph("warn") if unit.kind == "warn" else glyph("bad"))
    if unit.kind == "turn":
        match = re.search(r"\d+", unit.text or "")
        return turn_rule(int(match.group(0)) if match else 0)
    if unit.kind == "note":
        return system_line(_short(unit.text, _BODY_WIDTH))
    return Text(unit.text, style="gis.agent")


# ------------------------------------------------------------------ 复合渲染


def banner(
    *,
    model: str,
    api_base: str = "",
    workspace: str = "",
    catalog: str = "",
    recipes: int = 0,
    session_id: str = "",
    kernel: str = "",
    version: str = "",
) -> Panel:
    """启动横幅：模型 / 工作区 / 数据目录 / 会话（对齐 pi 的 startup header）。"""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="gis.status.key", justify="right")
    grid.add_column(style="gis.status")
    if model:
        provider = _host_of(api_base)
        grid.add_row("模型", f"[gis.num]{model}[/]" + (f" · {provider}" if provider else ""))
    if workspace:
        grid.add_row("工作区", f"[gis.path]{workspace}[/]")
    if catalog:
        grid.add_row("数据目录", catalog)
    if recipes:
        grid.add_row("配方", f"{recipes} 个")
    if kernel:
        grid.add_row("内核", kernel)
    if session_id:
        grid.add_row("会话", session_id)
    title = "[gis.panel.title]GIS Agent[/]" + (f" [gis.dim]v{version}[/]" if version else "")
    return Panel(grid, title=title, title_align="left", border_style="gis.panel.border", padding=(0, 1))


def _host_of(url: str) -> str:
    if not url:
        return ""
    text = str(url)
    for prefix in ("https://", "http://"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.split("/")[0]


def result_card(state: dict[str, Any]) -> Panel:
    """结束卡片：任务清单 + 产出 + 方法论 + 状态。"""
    status = str(state.get("status") or "")
    role = _STATUS_ROLE.get(status, "gis.status")
    body: list[RenderableType] = []

    tasklist = str(state.get("tasklist_digest") or "").strip()
    if tasklist:
        body.append(Text.from_markup("[gis.status.key]任务清单[/]"))
        for line in wrap_lines(tasklist, 84):
            body.append(Text("  " + line if line else "", style="gis.agent"))
        body.append(Text(""))

    artifacts: Iterable[str] = state.get("artifacts") or []
    artifacts = list(artifacts)
    body.append(Text.from_markup("[gis.status.key]产出[/]"))
    if artifacts:
        for path in artifacts:
            line = Text()
            line.append(f"  {glyph('bullet')} ", style="gis.dim")
            line.append(str(path), style="gis.path")
            body.append(line)
    else:
        body.append(Text("  （无）", style="gis.dim"))

    methodology = unescape_text(str(state.get("methodology") or "")).strip()
    if methodology:
        body.append(Text(""))
        body.append(Text.from_markup("[gis.status.key]方法论[/]"))
        for line in wrap_lines(methodology, 84):
            body.append(Text("  " + line if line else "", style="gis.agent"))

    summary = cleanup_error(unescape_text(state.get("final_summary") or state.get("error") or ""), 320).strip()
    if summary:
        body.append(Text(""))
        for line in wrap_lines(summary, 84):
            body.append(Text("  " + line if line else "", style="gis.agent"))

    turn = state.get("turn")
    meta = Text()
    meta.append(f"状态 {status or '未知'}", style=role)
    if turn is not None:
        meta.append(f" · 轮次 {turn}", style="gis.dim")
    body.append(Text(""))
    body.append(meta)

    return Panel(Group(*body), title="结果", title_align="left", border_style="gis.panel.border", padding=(0, 1))


def footer_text(tracker: StatusTracker, *, width: int = 0) -> Text:
    """状态行：模型 · 状态 · 轮次 · 用时 · token · 工具 · 当前动作。"""
    text = Text()
    pieces: list[tuple[str, str]] = []
    if tracker.model:
        pieces.append((tracker.model, "gis.accent"))
    # 与 events.StatusTracker.footer 保持同一套中文映射：过去只列了 4 个，
    # stopped（用户中断）等状态会直接显示英文。
    status_label = {
        "running": "运行中",
        "awaiting_user": "等待输入",
        "completed": "已完成",
        "idle": "空闲",
        "failed": "失败",
        "stopped": "已停止",
        "max_turns": "达轮次上限",
        "deadline_exceeded": "超时停止",
        "llm_unavailable": "模型不可用",
        "runner_error": "运行出错",
    }.get(tracker.status, tracker.status or "空闲")
    pieces.append((status_label, _STATUS_ROLE.get(tracker.status, "gis.status")))
    pieces.append((f"第 {tracker.turn} 轮" if tracker.turn else "未开始", "gis.dim"))
    pieces.append((format_duration(tracker.elapsed), "gis.dim"))
    pieces.append((f"↑{format_tokens(tracker.tokens_in)} ↓{format_tokens(tracker.tokens_out)}", "gis.dim"))
    if tracker.tool_count:
        pieces.append((f"工具 {tracker.tool_count}", "gis.dim"))
    if tracker.repair_count:
        pieces.append((f"修复 {tracker.repair_count}", "gis.warn"))
    if tracker.current_action:
        pieces.append((f"→ {tracker.current_action}", "gis.tool"))

    plain_len = 0
    for index, (text_value, role) in enumerate(pieces):
        if index:
            text.append(f" {glyph('dot')} ", style="gis.dim")
            plain_len += 3
        if width and plain_len + len(text_value) > width - 2:
            text.append("…", style="gis.dim")
            break
        text.append(text_value, style=role)
        plain_len += len(text_value)
    return text


def hint_line(*, busy: bool = False) -> Text:
    """输入框下边框里的快捷键提示。"""
    text = Text()
    if busy:
        text.append("Esc", style="gis.num")
        text.append(" 中断 · ", style="gis.dim")
        text.append("Enter", style="gis.num")
        text.append(" 排队纠偏 · ", style="gis.dim")
    else:
        text.append("Enter", style="gis.num")
        text.append(" 发送 · ", style="gis.dim")
        text.append("Shift+Enter", style="gis.num")
        text.append(" 换行 · ", style="gis.dim")
    text.append("/", style="gis.num")
    text.append(" 命令 · ", style="gis.dim")
    text.append("@", style="gis.num")
    text.append(" 文件 · ", style="gis.dim")
    text.append("Ctrl+O", style="gis.num")
    text.append(" 明细 · ", style="gis.dim")
    text.append("Ctrl+C", style="gis.num")
    text.append(" 退出", style="gis.dim")
    return text


def hint_text(*, busy: bool = False) -> str:
    """快捷键提示的纯文本版（给 prompt_toolkit 底栏用）。"""
    if busy:
        head = "Esc 中断 · Enter 排队纠偏 · "
    else:
        head = "Enter 发送 · Shift+Enter 换行 · "
    return head + "/ 命令 · @ 文件 · Ctrl+O 明细 · Ctrl+C 退出"


__all__ = [
    "ToolOutputStore",
    "TranscriptView",
    "answer_block",
    "banner",
    "footer_text",
    "hint_line",
    "hint_text",
    "render_unit",
    "result_card",
    "system_line",
    "thought_block",
    "tool_call_line",
    "tool_result_line",
    "turn_rule",
    "user_block",
]
