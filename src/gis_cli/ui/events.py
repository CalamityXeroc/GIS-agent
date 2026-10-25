# -*- coding: utf-8 -*-
"""引擎事件 → 渲染单元（纯逻辑，不碰 TTY，便于单测）。

引擎侧 ``on_event(event, payload)`` 会发这些事件（见 ``engine/loop.py``）::

    run_start / requirements_extracted / turn / llm_result / tool_call / tool_result
    repair_attempt / verification / llm_error / deadline_exceeded / run_end

本模块只做两件事：
1. :func:`to_unit` 把事件翻译成 :class:`RenderUnit`（调用方决定怎么画）；
2. :class:`StatusTracker` 累积状态行的数据（轮次/模型/token/耗时/当前动作）。

这样"UI 长什么样"与"引擎发了什么"解耦：TUI、纯文本、JSON 三种输出可共用同一层。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- 渲染单元

#: 单元类别 → 调用方的默认呈现方式
#: ``step`` 工具动作、``result`` 工具结果、``note`` 提示、``warn`` 警告、
#: ``error`` 错误、``verify`` 验收、``turn`` 轮次、``banner`` 启动横幅、``final`` 结束卡片
UNIT_KINDS = (
    "banner",
    "turn",
    "step",
    "result",
    "note",
    "warn",
    "error",
    "verify",
    "final",
    "final_line",
    "thought",
    "answer",
)


@dataclass
class RenderUnit:
    """一条可渲染的记录。``text`` 是主行，``lines`` 是附加行（可折叠）。"""

    kind: str
    text: str
    lines: list[str] = field(default_factory=list)
    ok: bool | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "lines": list(self.lines), "ok": self.ok, "meta": dict(self.meta)}


# ------------------------------------------------------------------ 小工具


def _short(value: Any, limit: int = 72) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def unescape_text(text: Any) -> str:
    """把正文里的字面量转义还原成真实字符。

    实测坑：模型经 JSON 传输的答复常带字面的反斜杠 n（两个字符），直接显示会看到
    ``CSV。\\n\\n数据事实…`` 这种"假换行"，读起来像乱码。
    """
    raw = "" if text is None else str(text)
    backslash = chr(92)
    if f"{backslash}n" in raw or f"{backslash}t" in raw or f"{backslash}r" in raw:
        raw = raw.replace(f"{backslash}r{backslash}n", "\n")
        raw = raw.replace(f"{backslash}n", "\n").replace(f"{backslash}t", "\t")
    return raw


def cleanup_error(text: Any, limit: int = 160) -> str:
    """把模型/网关报错压成人类可读一行。

    实测：余额不足、限流这类错误带一整段 JSON（含 traceId/retryAfterSeconds），
    原样打出来会刷满屏幕且看不出重点。这里抽掉噪声字段、优先取 message。
    """
    import re as _re

    raw = "" if text is None else str(text)
    raw = _re.sub(r'"traceId"\s*:\s*"[^"]*"', "", raw)
    raw = _re.sub(r'"retryAfterSeconds"\s*:\s*\d+', "", raw)
    raw = _re.sub(r"\s+", " ", raw).strip(" ,;")
    hit = _re.search(r'"message"\s*:\s*"([^"]+)"', raw)
    if hit:
        code = _re.search(r'"code"\s*:\s*"([^"]+)"', raw)
        raw = (f"{code.group(1)}：" if code else "") + hit.group(1)
    return _short(raw, limit)


def _wrap(text: str, width: int = 96) -> list[str]:
    """按宽度换行（中英文混排按字符数近似），保留原段落分隔。"""
    out: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.rstrip()
        if not line:
            out.append("")
            continue
        while len(line) > width:
            out.append(line[:width])
            line = line[width:]
        out.append(line)
    return out


def describe_args(tool: str, args: dict[str, Any] | None, limit: int = 96) -> str:
    """把工具参数压成一行摘要（跳过 code 全文，只留首行）。"""
    if not isinstance(args, dict) or not args:
        return ""
    parts: list[str] = []
    for key, value in args.items():
        if key == "code":
            first = next((ln.strip() for ln in str(value).splitlines() if ln.strip()), "")
            parts.append(f"code={_short(first, 40)}")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}={len(value)}项")
        elif isinstance(value, dict):
            parts.append(f"{key}={len(value)}键")
        else:
            parts.append(f"{key}={_short(value, 28)}")
        if sum(len(p) for p in parts) > limit:
            parts.append("…")
            break
    return " ".join(parts)


def format_duration(seconds: float) -> str:
    """``6m12s`` / ``1h02m`` 这类紧凑写法。"""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    hours = int(seconds // 3600)
    return f"{hours}h{int((seconds % 3600) // 60):02d}m"


def format_tokens(count: int) -> str:
    """``38.2k`` / ``1.4M`` —— 状态行要短。"""
    count = int(count or 0)
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


# ------------------------------------------------------------------ 事件翻译


def to_unit(event: str, payload: dict[str, Any] | None = None) -> RenderUnit | None:
    """把引擎事件翻译成渲染单元；返回 ``None`` 表示该事件不进 transcript。"""
    data = dict(payload or {})

    if event == "run_start":
        return RenderUnit("banner", str(data.get("goal", "")), meta={"run_id": str(data.get("run_id", ""))})

    if event == "llm_delta":
        # 流式增量：一次一个词，不适合当“一条记录”走 transcript；
        # 由渲染层用实时块消费（见 TranscriptView.live_delta）。
        return None

    if event == "requirements_extracted":
        count = int(data.get("count") or 0)
        if count <= 0:
            return None
        return RenderUnit("note", f"抽取验收需求 {count} 条", lines=[str(data.get("requirements", ""))])

    if event == "turn":
        turn = int(data.get("turn") or 0)
        if turn <= 1:
            return None
        return RenderUnit("turn", f"第 {turn} 轮", lines=[str(data.get("tasklist", ""))])

    if event == "llm_result":
        actions = [a for a in (data.get("actions") or []) if isinstance(a, dict)]
        reasoning = ""
        said = ""
        for action in actions:
            reasoning = reasoning or str(action.get("reasoning") or "")
            said = said or str(action.get("content") or "")
        # 真实思考：优先 reasoning（推理模型），否则用模型自述内容
        thought = (reasoning or said).strip()
        names = [str(a.get("tool") or a.get("kind") or "?") for a in actions]
        return RenderUnit(
            "thought",
            thought,
            meta={
                "model": _short(data.get("model") or "", 32),
                "actions": names,
                "usage": data.get("usage") or {},
                "has_reasoning": bool(reasoning.strip()),
            },
        )

    if event == "tool_call":
        tool = str(data.get("tool") or "?")
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        return RenderUnit("step", tool, lines=[describe_args(tool, args)], meta={"tool": tool})

    if event == "tool_result":
        tool = str(data.get("tool") or "?")
        ok = bool(data.get("ok", True))
        raw = " ".join(str(data.get("summary", "")).split())
        # finish 的返回其实是"最终答复"，交给 answer 单元渲染，避免埋进工具行
        if tool == "finish":
            body = unescape_text(raw)
            for prefix in ("验收通过。", "验收通过：", "验收通过"):
                if body.startswith(prefix):
                    body = body[len(prefix):].lstrip("：: ")
                    break
            return RenderUnit("answer", body, ok=ok, meta={"tool": tool, "raw": raw})
        # 工具结果只留一行；超长（修复说明常 300+ 字）进折叠区，正文不再铺开
        return RenderUnit("result", _short(raw, 88), lines=[raw] if len(raw) > 88 else [], ok=ok, meta={"tool": tool})

    if event == "repair_attempt":
        attempt = data.get("attempt")
        reason = " ".join(str(data.get("error") or data.get("reason") or "").split())
        return RenderUnit(
            "warn",
            f"自动修复第 {attempt} 次",
            lines=[reason] if reason else [],
            ok=False,
            meta={"short": _short(reason, 70)},
        )

    if event == "verification":
        passed = bool(data.get("pass"))
        return RenderUnit("verify", _short(data.get("summary", ""), 200), ok=passed, meta={"verdict": data})

    if event == "llm_error":
        return RenderUnit("error", "模型调用失败", lines=[cleanup_error(data.get("error", ""), 200)])

    if event == "deadline_exceeded":
        return RenderUnit("warn", "达到运行时限，正在收尾", lines=[_short(data.get("error", ""), 160)])

    if event == "context_compacted":
        return RenderUnit("note", "上下文已压缩", meta={"summarized": data.get("summarized_messages")})

    if event == "vision_images":
        files = data.get("files") or []
        return RenderUnit("note", f"附上成图自检（{len(files)} 张）", lines=[_short(f, 120) for f in files[:3]])

    # ---------------------------------------------------------- 子代理
    # 子代理跑几分钟时父会话是阻塞的：这几个事件是唯一的“还在动”信号，
    # 不转发界面会静默到用户以为卡死。
    if event == "subagent_start":
        sid = str(data.get("subtask_id") or "?")
        task = _short(data.get("task") or "", 70)
        return RenderUnit(
            "note",
            f"子代理 {sid} 开始：{task}",
            lines=[f"预算 {data.get('max_turns', '?')} 轮 / {data.get('timeout_s', '?')}s"],
            meta={"subtask_id": sid},
        )

    if event == "subagent_tool":
        sid = str(data.get("subtask_id") or "?")
        tool = str(data.get("tool") or "?")
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        return RenderUnit("step", f"[{sid}] {tool}", lines=[describe_args(tool, args)], meta={"tool": tool, "subtask_id": sid})

    if event == "subagent_warn":
        sid = str(data.get("subtask_id") or "?")
        tool = str(data.get("tool") or "?")
        return RenderUnit(
            "warn",
            f"[{sid}] {tool} 失败",
            lines=[_short(data.get("summary") or "", 160)],
            ok=False,
            meta={"tool": tool, "subtask_id": sid},
        )

    if event == "subagent_end":
        sid = str(data.get("subtask_id") or "?")
        status = str(data.get("status") or "?")
        ok = status == "completed"
        label = {
            "completed": "完成",
            "max_turns": "撞轮次上限",
            "deadline_exceeded": "超时",
            "llm_unavailable": "模型不可用",
            "failed": "失败",
            "stopped": "被中断",
        }.get(status, status)
        lines = [unescape_text(data.get("summary") or "")] if data.get("summary") else []
        leaks = data.get("namespace_leaks") or []
        if leaks:
            lines.append("未加前缀的内核变量：" + "、".join(str(x) for x in leaks[:5]))
        return RenderUnit(
            "verify" if ok else "warn",
            f"子代理 {sid} {label}（{data.get('turns', '?')} 轮 / {data.get('elapsed_s', '?')}s，产物 {data.get('artifacts', 0)} 个）",
            lines=lines,
            ok=ok,
            meta={"subtask_id": sid, "status": status},
        )

    if event == "run_end":
        return _run_end_unit(data)

    return None


def _tasklist_lines(state: dict[str, Any]) -> list[str]:
    """任务清单文本：优先用引擎给的 digest，其次从 tasklist 数组自行拼。"""
    digest = str(state.get("tasklist_digest") or "").strip()
    if digest:
        return digest.splitlines()
    items = state.get("tasklist") or []
    if not isinstance(items, list):
        return []
    mark = {"done": "x", "completed": "x", "failed": "!", "skipped": "-", "in_progress": ">"}
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "pending").lower()
        box = mark.get(status, " ")
        title = _short(item.get("title") or item.get("id") or "", 90)
        lines.append(f"- [{box}] {title}")
    return lines


def _run_end_unit(state: dict[str, Any]) -> RenderUnit:
    status = str(state.get("status") or "")
    artifacts = [str(p) for p in (state.get("artifacts") or [])]
    lines: list[str] = []
    lines.extend(_tasklist_lines(state))
    if artifacts:
        lines.append("")
        lines.append("产出：")
        lines.extend(f"  {p}" for p in artifacts)
    methodology = str(state.get("methodology") or "").strip()
    if methodology:
        lines.append("")
        lines.append("方法论：")
        lines.extend(methodology.splitlines())
    summary = str(state.get("final_summary") or state.get("error") or "").strip()
    ok = status == "completed"
    text = {"completed": "任务完成", "max_turns": "达到轮次上限", "deadline_exceeded": "达到时限"}.get(status, status or "结束")
    if summary:
        lines.append("")
        lines.extend(_wrap(cleanup_error(summary, 400), 96))
    return RenderUnit("final", text, lines=lines, ok=ok, meta={"status": status, "artifacts": artifacts})


# ------------------------------------------------------------------ 状态累积


@dataclass
class StatusTracker:
    """状态行数据源：轮次、模型、token、耗时、当前动作、运行状态。

    线程说明：``apply`` 由工作线程调用，``footer`` 由渲染线程调用；
    只做属性写入/读取，整数自增在 CPython 下原子，够用（无需锁）。
    """

    model: str = ""
    provider: str = ""
    workspace: str = ""
    session_id: str = ""
    turn: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    status: str = "idle"
    current_action: str = ""
    action_started: float = 0.0
    action_elapsed: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float = 0.0
    verify_pass: bool | None = None
    tool_count: int = 0
    repair_count: int = 0
    #: 流式实时字数（等待期 footer 显示"思考中 · 12s · 340 字"）
    stream_chars: int = 0
    #: 首个增量到达的时刻（算首字延迟 TTFB）
    first_delta_at: float = 0.0
    #: 排队提示（运行中用户又输入了消息时显示，让用户知道消息没丢）
    queued_hint: str = ""

    # -- 更新 ---------------------------------------------------------

    def apply(self, event: str, payload: dict[str, Any] | None = None) -> None:
        data = dict(payload or {})
        now = time.monotonic()
        if event == "run_start":
            self.status = "running"
            self.started_at = now
            self.current_action = "准备"
            self.action_started = now
        elif event == "turn":
            self.turn = int(data.get("turn") or self.turn)
        elif event == "llm_result":
            model = str(data.get("model") or "")
            if model:
                self.model = model
            # 用量在这里累加：任何走 apply 的路径都不会漏（曾因只在某个调用点加而显示 ↑0 ↓0）
            self.add_usage(data.get("usage") if isinstance(data.get("usage"), dict) else None)
            self.stream_chars = 0
        elif event == "llm_delta":
            text = str(data.get("text") or "")
            if text:
                if not self.first_delta_at:
                    self.first_delta_at = now
                self.stream_chars += len(text)
        elif event == "queued":
            # 运行中用户又提交了消息：立刻给可见反馈（消息没丢，本轮结束后继续）
            count = int(data.get("count") or 0)
            self.queued_hint = f"已排队 {count} 条，结束后合并继续" if count > 0 else ""
        elif event == "tool_call":
            self.tool_count += 1
            self.current_action = str(data.get("tool") or "")
            self.action_started = now
        elif event == "tool_result":
            if self.action_started:
                self.action_elapsed = now - self.action_started
            self.current_action = ""
        elif event == "repair_attempt":
            self.repair_count += 1
        elif event == "subagent_turn":
            # 子代理的轮次不能覆盖父的 turn（那是两个预算）
            self.current_action = f"子代理 {data.get('subtask_id')} · 第 {data.get('turn')} 轮"
        elif event == "subagent_tool":
            self.current_action = f"子代理 {data.get('subtask_id')} · {data.get('tool')}"
        elif event == "subagent_end":
            self.current_action = ""
        elif event == "verification":
            self.verify_pass = bool(data.get("pass"))
        elif event == "run_end":
            self.status = str(data.get("status") or "stopped")
            self.finished_at = now
            self.current_action = ""

    def add_usage(self, usage: dict[str, Any] | None) -> None:
        """累加 token 用量（``llm_result`` 之外的路径也可调用）。"""
        if not isinstance(usage, dict):
            return
        self.tokens_in += int(usage.get("prompt_tokens") or 0)
        self.tokens_out += int(usage.get("completion_tokens") or 0)

    # -- 读取 ---------------------------------------------------------

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.monotonic()
        return max(0.0, end - self.started_at)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def footer(self, *, width: int = 0, show_action: bool = True) -> str:
        """一行状态文本（不含样式；样式由渲染层按角色上色）。"""
        bits: list[str] = []
        if self.model:
            bits.append(self.model)
        status_text = {
            "idle": "空闲",
            "running": "运行中",
            "awaiting_user": "等待输入",
            "completed": "已完成",
            "failed": "失败",
            # 引擎 run_end 给的是 stopped（用户中断）与这几个；以前直接显示英文
            "stopped": "已停止",
            "max_turns": "达轮次上限",
            "deadline_exceeded": "超时停止",
            "llm_unavailable": "模型不可用",
            "runner_error": "运行出错",
        }.get(self.status, self.status or "空闲")
        bits.append(status_text)
        idle = self.status in ("", "idle") and not self.turn
        if not idle:
            bits.append(f"第 {self.turn} 轮" if self.turn else "未开始")
            bits.append(format_duration(self.elapsed))
            bits.append(f"↑{format_tokens(self.tokens_in)} ↓{format_tokens(self.tokens_out)}")
            if self.tool_count:
                bits.append(f"工具 {self.tool_count}")
            if self.repair_count:
                bits.append(f"修复 {self.repair_count}")
            if show_action and self.current_action:
                bits.append(f"→ {self.current_action}")
            elif show_action and self.stream_chars:
                # 等待期活体提示：模型正在流式输出（避免盯着空屏不知死没死）
                # 这里不引 theme.glyph：events 是底层模块，theme 反向依赖它会成环。
                bits.append(f"思考中 · {self.stream_chars} 字")
        elif self.queued_hint:
            # 空闲但有排队中的后续任务：让用户知道它会被接着处理
            bits.append(self.queued_hint)
        if self.queued_hint and not idle:
            # 运行中排队也要可见（用户实测：静默排队让人以为消息丢了）
            bits.append(self.queued_hint)
        text = " · ".join(bits)
        if width and len(text) > width:
            text = text[: max(0, width - 1)] + "…"
        return text


__all__ = [
    "RenderUnit",
    "StatusTracker",
    "UNIT_KINDS",
    "describe_args",
    "format_duration",
    "format_tokens",
    "to_unit",
]
