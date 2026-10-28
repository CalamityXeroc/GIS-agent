# -*- coding: utf-8 -*-
"""交互式对话入口（新引擎）。

体验对齐 pi / Claude Code：启动横幅 → 滚动转录（提问/工具动作/结果/验收）
→ 底部输入框（多行、历史、``/`` 命令、``@`` 文件）→ 状态行。

要点：
- 引擎跑在**工作线程**里，主线程始终可以输入：运行中输入=纠偏，``Esc``=中断；
- 每一轮结束后不停机，下一条输入直接接着做（引擎每轮都是独立任务，靠会话前言续上下文）；
- 转录落盘 ``workspace/.gis_agent/sessions/<id>.jsonl``，支持 ``--continue`` 与 ``/resume``。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from rich.text import Text

from ..ui import events as ui_events
from ..ui import logging_setup, plan_card as ui_plan_card, prompt as ui_prompt, render as ui_render, session_store, theme as ui_theme, wiring
from ..ui.session_runner import EngineSession


class InteractiveChat:
    """一次交互会话。"""

    def __init__(
        self,
        *,
        workspace: Optional[Path] = None,
        config: Optional[Path] = None,
        theme_mode: str = "system",
        no_color: bool = False,
        color: bool = False,
        max_turns: int = 0,
        refresh_catalog: bool = False,
        resume_id: str = "",
        continue_last: bool = False,
        prompt_input: Any = None,
        prompt_output: Any = None,
    ) -> None:
        self.workspace = Path(workspace or Path.cwd())
        self.config = config
        self.theme_mode = theme_mode
        self.no_color = no_color
        #: 可注入的 prompt_toolkit 输入/输出（无头测试用；生产留 None）
        self._prompt_input = prompt_input
        self._prompt_output = prompt_output
        # color=None → 自动判定（默认不上色）；True → 强制上色
        self.color: bool | None = True if color else None
        self.max_turns = max_turns
        #: 启动时是否同步扫全数据目录。默认 False：启动不阻塞，
        #: 改由首个任务开始时后台扫描（见 engine/runner.py 的 prewarmer）。
        #: 传 True（``--refresh-catalog``）时恢复“启动就扫全”的旧行为。
        self.force_refresh_catalog = bool(refresh_catalog)
        self.console = ui_theme.make_console(theme_mode, no_color=no_color, color=self.color)
        self.sessions = session_store.SessionStore(self.workspace)
        self.session_id = resume_id or (self.sessions.latest_id() if continue_last else "") or self.sessions.new_id()
        self.pending_context = ""
        if resume_id or continue_last:
            self.pending_context = self.sessions.context_prefix(self.session_id)
        self.bundle: Any = None
        self.session: EngineSession | None = None
        self.reader: ui_prompt.PromptInput | None = None
        #: 计划卡闸门：引擎跑在工作线程，跨线程 ``input()`` 会抢输入框的按键，
        #: 所以闸门只投递请求，按键由主线程（prompt_toolkit）读。
        self.plan_gate: ui_plan_card.ChatPlanGate | None = None
        #: 已按了 ``e``，下一行是“怎么改”的那一句
        self._plan_asking_revision = False
        self.store = ui_render.ToolOutputStore()
        #: 转录视图（合并连续工具调用、区分思考/答复/系统）
        self.view = ui_render.TranscriptView(self.console, ui_events.StatusTracker(workspace=str(self.workspace)), self.store)
        self.last_trace = ""

    # ------------------------------------------------------------ 引擎

    def _build(self) -> int:
        from ..engine.runner import build_engine

        logging_setup.setup_engine_logging(self.workspace)
        helper = ui_events
        tracker = helper.StatusTracker(workspace=str(self.workspace))

        renderer_holder: dict[str, Any] = {}

        def _on_event(name: str, payload: dict) -> None:
            render = renderer_holder.get("render")
            if render is not None:
                render(name, payload)
            if name == "tool_result":
                summary = str((payload or {}).get("summary", ""))
                self.sessions.append(
                    self.session_id, "tool", summary[:400], tool=(payload or {}).get("tool"), ok=(payload or {}).get("ok")
                )
            if name == "run_end":
                self._record_finish(payload)

        # 计划卡闸门：只有真能收键（真终端）时才停下来等确认；
        # 管道/重定向下照旧只打印（行为与之前一致，只是文案改成“未确认”）。
        # 判定必须在 ``status(Live)`` **之外**：Live 期间 ``sys.stdout`` 是 rich 的
        # ``FileProxy``，``isatty()`` 恒为 False（DAIR-37：真终端也不暂停）。
        interactive_ok = ui_prompt.interactive_terminal()
        with self.console.status("[gis.dim]启动引擎（加载配方/配置）…[/]"):
            self.plan_gate = ui_plan_card.ChatPlanGate(interactive=interactive_ok)
            try:
                self.bundle = build_engine(
                    workspace=self.workspace,
                    config_path=self.config,
                    # 启动**不扫数据目录**：它改由首个任务开始时后台扫描，
                    # 与首轮模型调用重叠（实测目录扫描曾占启动耗时 158s）。
                    # 想启动就扫全的用 ``--refresh-catalog``。
                    refresh_catalog=bool(self.force_refresh_catalog),
                    on_event=_on_event,
                    plan_gate=self.plan_gate,
                )
            except Exception as exc:
                self.console.print(ui_render.render_unit(ui_events.RenderUnit("error", "引擎启动失败", lines=[str(exc)[:200]])))
                return 1

        if self.max_turns:
            self.bundle.loop.config.max_turns = self.max_turns

        self.session = EngineSession(
            self.bundle,
            renderer=lambda name, payload: self._render_event(name, payload),
            on_finish=self._on_finish,
        )
        # 视图与状态行共用同一个 tracker，保证 footer/框内状态一致
        self.view.tracker = self.session.tracker
        renderer_holder["render"] = self.session.drain_events
        self.last_trace = str(getattr(getattr(self.bundle.loop, "recorder", None), "jsonl_path", ""))
        return 0

    def _render_event(self, name: str, payload: dict) -> None:
        if self.session is None:
            return
        if name == "run_end":
            self.view.flush()
            return
        # 流式增量：边到边画（不走 to_unit——那是“一条事件一块”的粒度）
        if name == "llm_delta":
            kind = str((payload or {}).get("kind") or "content")
            self.view.live_delta(kind, str((payload or {}).get("text") or ""))
            return
        if name in ("tool_call", "tool_result"):
            # 工具开始/结束意味着模型这一轮的输出已经完了，收掉实时块
            self.view.end_live()
        # 转录视图负责分组与样式：连续同工具合并、长文本折叠、思考/答复分块
        self.view.handle(name, payload)

    def _record_finish(self, payload: dict) -> None:
        status = str((payload or {}).get("status") or "")
        summary = str((payload or {}).get("final_summary") or (payload or {}).get("error") or "")
        self.sessions.append(self.session_id, "assistant", summary[:1200], status=status)

    def _on_finish(self, state: Any) -> None:
        """工作线程里的收尾渲染（patch_stdout 会保证不冲乱输入框）。

        若同时有排队的补充说明，本轮结束后**自动合并继续**（同一任务的后续），
        不需要用户重新输入。
        """
        if state is None:
            return
        self.view.flush()
        self.console.print()
        wiring.render_finish(self.console, self.session.tracker, state, trace_path=self.last_trace, show_trace=False)
        self._maybe_suggest_distill(state)
        # 排队的消息 = 同一任务的补充说明 → 合并成一条新任务自动接着做
        merged = self.session.merged_task("") if self.session else ""
        if merged.strip():
            self.console.print(
                Text(f"（接着处理排队的补充说明，{merged.count(chr(10)) + 1} 行）", style="gis.dim")
            )
            self._pending_followup = merged

    def _maybe_suggest_distill(self, state: Any) -> None:
        """成功后提示/自动把这次运行沉淀成配方（engine.distill_after_success）。

        ``suggest``（默认）只打一行提示，不调模型；``auto`` 后台蒸馏；
        ``off`` 完全不做。默认选 suggest：网关抖动时自动蒸馏会多一次
        LLM 往返，不该偷偷花掉用户的时间。
        """
        if str(getattr(state, "status", "")) != "completed" or self.bundle is None:
            return
        try:
            mode = str(getattr(self.bundle.loop.config, "distill_after_success", "suggest") or "suggest").lower()
        except Exception:
            mode = "suggest"
        if mode == "off":
            return
        if mode == "auto":
            import threading

            def _worker() -> None:
                try:
                    result = self.bundle.distill()
                    text = (
                        f"已自动沉淀配方：{result['recipe']['id']}（门禁{'通过' if result.get('gate', {}).get('ok') else '未过'}）"
                        if result.get("recipe")
                        else f"自动沉淀跳过：{result.get('reason')}"
                    )
                except Exception as exc:  # pragma: no cover - 后台任务不能影响主流程
                    text = f"自动沉淀失败：{exc}"
                self.console.print(Text(text, style="gis.dim"))

            threading.Thread(target=_worker, name="gis-agent-distill", daemon=True).start()
            return
        self.console.print(
            Text("（本次运行可沉淀为可复用配方：输入 /distill 试试，/distill adopt 直接入库）", style="gis.dim")
        )

    def _emit_queued(self, count: int) -> None:
        """把“已排队”同时推进状态行（事件流）与转录。"""
        try:
            self.session.tracker.apply("queued", {"count": count})
        except Exception:  # pragma: no cover - 状态行只是提示，不影响主流程
            pass
        try:
            self.view.flush()
        except Exception:
            pass

    def _take_followup(self) -> str:
        """取走待自动执行的后续任务（主循环在下一轮开始时调用）。"""
        text = getattr(self, "_pending_followup", "")
        self._pending_followup = ""
        return text

    # ------------------------------------------------------------ 命令

    def _command_registry(self) -> Any:
        """建命令表：名字/说明取自 ui.commands.COMMANDS，这里只挂处理器。

        **每次调用重建**，以便捕获最新的会话/模型状态。
        """
        from ..ui.commands import COMMANDS, CommandRegistry

        return CommandRegistry(COMMANDS).with_handlers(
            {
                "help": self._cmd_help,
                "model": self._cmd_model,
                "theme": self._cmd_theme,
                "color": self._cmd_color,
                "workspace": self._cmd_workspace,
                "status": self._cmd_status,
                "tools": self._cmd_tools,
                "skills": self._cmd_skills,
                "recipes": self._cmd_recipes,
                "distill": self._cmd_distill,
                "trace": self._cmd_trace,
                "session": self._cmd_session,
                "resume": self._cmd_resume,
                "clear": self._cmd_clear,
                "export": self._cmd_export,
                "exit": self._cmd_exit,
            },
            arg_providers={
                # 惰性：只在按下补全那一刻才查（网关探测有几秒，不能每次打字都跑）
                "model": self._model_arg_choices,
                "resume": self._resume_choices,
            },
        )

    def _model_arg_choices(self) -> list[tuple[str, str]]:
        """``/model`` 的参数候选：当前可用的模型名（惰性调用）。

        **补全时不去问网关**：探测有几秒超时，会卡住打字。所以只读上次
        ``/model`` 列过的缓存；没缓存过就先给配置里能看到的名字
        （用户按 Tab 得到的是立即可用的，而不是卡 4 秒）。
        """
        cached = getattr(self, "_model_items", None)
        if cached:
            return list(cached)
        from ..engine import model_catalog as catalog

        config_path = catalog.resolve_config_path(self.workspace, self.config)
        data = catalog.load_config_data(config_path)
        return [(name, "配置里列出") for name in catalog.models_from_config_data(data)]

    def _refresh_model_items(self) -> None:
        """刷新模型候选项缓存（由 ``/model`` 无参数时触发；含在线清单）。"""
        from ..engine import model_catalog as catalog

        config_path = catalog.resolve_config_path(self.workspace, self.config)
        data = catalog.load_config_data(config_path)
        items = catalog.available_models(
            config_data=data,
            api_base=str(data.get("api_base") or ""),
            api_key=str(data.get("api_key") or ""),
        )
        self._model_items = [(item.name, item.note or item.source) for item in items]

    def _current_model(self) -> str:
        if self.session is not None and self.session.tracker.model:
            return self.session.tracker.model
        if self.bundle is not None:
            return str(getattr(self.bundle.llm.config, "model", "") or "")
        return ""

    def _handle_command(self, line: str) -> bool:
        """查表分发；返回 False 表示要退出。

        额外支持 ``/skill:名字 [参数]``：直接把某个技能的全文注入对话，
        让 agent 照着做（与 pi 的 ``/skill:name`` 一致）。自然语言相关时
        模型会自己调 run_skill，这个命令是“手动指定”的入口。
        """
        parts = line.strip().split(maxsplit=1)
        word = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        if word.lower().startswith("/skill:"):
            return self._run_skill_command(word.split(":", 1)[1], arg)
        spec = self._command_registry().find(word)
        if spec is None or spec.handler is None:
            self.console.print(Text(f"未知命令 {word}，输入 /help 查看可用命令", style="gis.warn"))
            return True
        return bool(spec.handler(arg))

    def _run_skill_command(self, name: str, arg: str) -> bool:
        """加载技能并把正文作为任务发给引擎（技能跟随自然语言一起工作）。"""
        from ..skills.catalog import discover_skills, read_skill_text

        target = None
        try:
            for item in discover_skills(self.workspace):
                if item.name.lower() == name.strip().lower():
                    target = item
                    break
        except Exception as exc:
            self.console.print(Text(f"技能查找失败：{exc}", style="gis.warn"))
            return True
        if target is None:
            self.console.print(Text(f"没有找到技能 {name}（用 /skills 看全部）", style="gis.warn"))
            return True
        body = read_skill_text(target)
        task = (
            f"请严格按下面的技能说明执行任务。\n\n"
            f"## 技能：{target.name}\n{body}\n\n"
            f"技能目录（脚本相对它引用）：{target.root}\n"
        )
        if arg:
            task += f"\n用户要求：{arg}\n"
        self.console.print(Text(f"已加载技能 {target.name}", style="gis.ok"))
        self._dispatch(task, echo=False)
        return True

    # ------------------------------------------------------------ 各命令实现

    def _cmd_help(self, arg: str) -> bool:  # noqa: ARG002 - 统一签名
        self.console.print(Text(self._help_text(), style="gis.agent"))
        return True

    def _help_text(self) -> str:
        from ..ui.commands import KEY_HINT_LINES

        return self._command_registry().help_text(keys=KEY_HINT_LINES)

    def _cmd_exit(self, arg: str) -> bool:  # noqa: ARG002
        return False

    def _cmd_clear(self, arg: str) -> bool:  # noqa: ARG002
        self.console.clear()
        if self.bundle is not None:
            wiring.banner_for(self.console, self.bundle, version=_version())
        return True

    def _cmd_status(self, arg: str) -> bool:  # noqa: ARG002
        self._print_status()
        return True

    def _cmd_theme(self, arg: str) -> bool:
        if arg in ("system", "dark", "light"):
            self.theme_mode = arg
            self.console = ui_theme.make_console(arg, no_color=self.no_color, color=self.color)
            self.console.print(Text(f"配色已切到 {arg}", style="gis.ok"))
        else:
            self.console.print(Text("当前配色 " + self.theme_mode + "；用法：/theme system|dark|light", style="gis.dim"))
        return True

    def _cmd_color(self, arg: str) -> bool:
        if arg in ("on", "off"):
            self.color = arg == "on"
            self.console = ui_theme.make_console(self.theme_mode, no_color=self.no_color, color=self.color)
            self.console.print(Text(f"颜色已{'开启' if self.color else '关闭'}", style="gis.ok"))
        else:
            shown = "开" if self.color else "关"
            self.console.print(Text(f"颜色当前{shown}；用法：/color on|off", style="gis.dim"))
        return True

    def _cmd_workspace(self, arg: str) -> bool:
        self.console.print(Text(f"工作区 {self.workspace}", style="gis.path"))
        if arg:
            self.console.print(
                Text("切换工作区需要重启会话（引擎的数据目录与内核都绑在当前工作区）", style="gis.dim")
            )
        return True

    def _cmd_tools(self, arg: str) -> bool:  # noqa: ARG002
        self._print_inventory("tools")
        return True

    def _cmd_skills(self, arg: str) -> bool:  # noqa: ARG002
        self._print_inventory("skills")
        return True

    def _cmd_recipes(self, arg: str) -> bool:  # noqa: ARG002
        self._print_inventory("recipes")
        return True

    def _cmd_distill(self, arg: str) -> bool:
        """把上一次成功运行沉淀成配方：/distill [list|replay [id] [adopt]|adopt [id]|reject <id>]。"""
        if self.bundle is None:
            self.console.print(Text("（引擎未就绪）", style="gis.dim"))
            return True
        from ..recipes.lifecycle import RecipeLifecycle

        lifecycle = RecipeLifecycle(self.bundle.recipes, self.bundle.workspace,
                                    code_runner=self.bundle.code_runner)
        sub = (arg or "").strip().split()
        action = sub[0].lower() if sub else ""

        if action == "list":
            self._print_candidates(lifecycle)
            return True
        if action == "reject" and len(sub) > 1:
            ok = lifecycle.reject(sub[1])
            self.console.print(Text(f"{'已删除候选' if ok else '没有这个候选'}: {sub[1]}",
                                    style="gis.ok" if ok else "gis.dim"))
            return True

        rid = sub[1] if len(sub) > 1 else ""
        # 带 id 时只动**已有候选**：先验后采必须是同一个配方。不带 id 的
        # replay/adopt 会重新蒸馏出一个新候选，“验的不是采的那个”（实测踩到）。
        if action == "replay" and rid:
            res = lifecycle.replay(rid)
            lines = [f"回放 {rid}: {'通过' if res.ok else '未通过'}"]
            for check in res.checks:
                if not check["ok"]:
                    lines.append(f"  [FAIL] {check['name']}: {check['detail'][:80]}")
            for warn in res.warnings:
                lines.append(f"  [警告] {warn[:80]}")
            if res.ok and len(sub) > 2 and sub[2].lower() == "adopt":
                path, note = lifecycle.adopt(rid)
                lines.append(f"已采纳: {path}（{note}）" if path else f"采纳失败：{note}")
            elif not res.ok and len(sub) > 2:
                lines.append("回放未通过，不予采纳")
            self.console.print(Text("\n".join(lines), style="gis.ok" if res.ok else "gis.warn"))
            return True
        if action == "adopt" and rid:
            path, note = lifecycle.adopt(rid)
            self.console.print(Text(
                f"已采纳: {path}（{note}；下次 build_engine 自动加载）" if path else f"采纳失败：{note}",
                style="gis.ok" if path else "gis.warn",
            ))
            return True

        self.console.print(Text("正在蒸馏上一次成功运行…", style="gis.dim"))
        result = self.bundle.distill(replay=action == "replay", adopt=action == "adopt")
        if not result.get("recipe"):
            self.console.print(Text(f"未能蒸馏：{result.get('reason')}", style="gis.warn"))
            return True

        gate = result.get("gate") or {}
        lines = [f"配方: {result['recipe']['id']} — {result['recipe']['name']}",
                 f"参数: {len(result['recipe'].get('params') or {})} 个 | "
                 f"门禁: {'通过' if gate.get('ok') else '未通过'}"]
        for check in gate.get("checks") or []:
            if not check["ok"]:
                lines.append(f"  [FAIL] {check['name']}: {check['detail'][:80]}")
        for warn in gate.get("warnings") or []:
            lines.append(f"  [警告] {warn[:80]}")
        replay_result = result.get("replay")
        if replay_result:
            lines.append(f"回放: {'通过' if replay_result.get('ok') else '未通过'}")
            for check in replay_result.get("checks") or []:
                if not check["ok"]:
                    lines.append(f"  [FAIL] {check['name']}: {check['detail'][:80]}")
        if result.get("adopted"):
            lines.append(f"已采纳: {result['adopted']}")
        elif gate.get("ok"):
            lines.append("下一步：/distill replay 回放验证，/distill adopt 采纳入库")
        self.console.print(Text("\n".join(lines), style="gis.agent"))
        return True

    def _print_candidates(self, lifecycle: Any) -> None:
        rows = lifecycle.list_candidates()
        if not rows:
            self.console.print(Text("（暂无候选配方；跑完一个任务后 /distill 试试）", style="gis.dim"))
            return
        lines = [f"  {r.get('id')} — {r.get('name')} [{'门禁通过' if r.get('gate_ok') else '门禁未过'}]"
                 for r in rows]
        self.console.print(Text("候选配方：\n" + "\n".join(lines[:40]), style="gis.agent"))

    def _cmd_session(self, arg: str) -> bool:  # noqa: ARG002
        self._print_session()
        return True

    def _cmd_export(self, arg: str) -> bool:  # noqa: ARG002
        path = self.sessions.export(self.session_id)
        style = "gis.ok" if path else "gis.dim"
        self.console.print(Text(f"已导出 {path}" if path else "（无内容可导出）", style=style))
        return True

    def _cmd_trace(self, arg: str) -> bool:
        if not self.last_trace:
            self.console.print(Text("（暂无追踪文件）", style="gis.dim"))
            return True
        self.console.print(Text(self.last_trace, style="gis.path"))
        if arg.strip().lower() == "full":
            self.console.print(Text(_tail_of(self.last_trace, lines=80), style="gis.code"))
        else:
            self.console.print(Text(_tail_of(self.last_trace, lines=8), style="gis.dim"))
            self.console.print(Text("（/trace full 看完整末尾 80 行）", style="gis.dim"))
        return True

    def _cmd_model(self, arg: str) -> bool:
        """查看或切换模型；切换要同时在运行时与配置文件里生效。"""
        from ..engine import model_catalog as catalog

        config_path = catalog.resolve_config_path(self.workspace, self.config)
        data = catalog.load_config_data(config_path)
        current = self._current_model() or str(data.get("model") or "")

        if not arg:
            self._refresh_model_items()
            items = catalog.available_models(
                config_data=data,
                api_base=str(data.get("api_base") or ""),
                api_key=str(data.get("api_key") or ""),
            )
            self.console.print(Text("可用模型：", style="gis.status.key"))
            self.console.print(Text(catalog.format_model_list(items, current), style="gis.agent"))
            self.console.print(Text("用法：/model <名字>（也可输入 /model 后按 Tab 从菜单选）", style="gis.dim"))
            return True

        name = arg.strip()
        if self.session is not None and self.session.busy:
            self.console.print(Text("任务运行中，等这轮结束再切模型", style="gis.warn"))
            return True
        if self.bundle is None:
            self.console.print(Text("引擎未就绪，无法切换", style="gis.warn"))
            return True
        catalog.apply_model(self.bundle.llm.config, name)
        ok, note = catalog.persist_model(config_path, name)
        style = "gis.ok" if ok else "gis.warn"
        self.console.print(Text(f"模型已切到 {name}（本会话立即生效）", style="gis.ok"))
        self.console.print(Text(note if ok else f"注意：{note}（仅本会话生效）", style=style))
        return True

    def _resume_choices(self) -> list[tuple[str, str]]:
        """``/resume`` 的候选：最近会话（菜单里能↑↓选）。"""
        try:
            items = self.sessions.list_sessions(limit=8)
        except Exception:
            return []
        return [(item.session_id, item.title) for item in items]

    def _cmd_resume(self, arg: str) -> bool:
        self._resume(arg)
        return True

    def _print_status(self) -> None:
        lines = [f"会话 {self.session_id}", f"工作区 {self.workspace}"]
        if self.session is not None:
            lines.append(f"模型 {self.session.tracker.model or '-'}")
            lines.append(f"状态 {self.session.tracker.status} · 轮次 {self.session.tracker.turn}")
        info = ui_theme.theme_diagnostics()
        lines.append(f"配色 {info['mode']}（GIS_AGENT_THEME={info['env_theme'] or '-'}）")
        try:
            from ..ui import console_env

            lines.append(console_env.describe())
        except Exception:
            pass
        if self.bundle is not None:
            meta = wiring.engine_meta(self.bundle)
            lines.append(f"数据目录 {meta['catalog']} · 配方 {meta['recipes']} 个")
            lines.append(f"日志 {self.workspace / '.gis_agent' / 'logs' / 'engine.log'}")
        self.console.print(Text("\n".join(lines), style="gis.agent"))

    def _print_inventory(self, kind: str) -> None:
        """列出工具 / 技能 / 配方（带一行说明；以前工具只给名字）。"""
        if kind == "recipes":
            if self.bundle is None:
                return
            try:
                items = [f"{r.id} — {r.name}" for r in self.bundle.recipes.all()]
            except Exception as exc:
                items = [f"（读取失败：{exc}）"]
            text = "\n".join(f"  {item}" for item in items[:80]) or "  （无）"
            # 候选配方也要露一眼：否则用户不知道有东西等着采纳
            try:
                from ..recipes.lifecycle import RecipeLifecycle

                pending = RecipeLifecycle(self.bundle.recipes, self.bundle.workspace).list_candidates()
            except Exception:
                pending = []
            if pending:
                text += "\n\n候选（未采纳，/distill list 查看）：\n" + "\n".join(
                    f"  {r.get('id')} — {r.get('name')}" for r in pending[:10]
                )
            self.console.print(Text(f"配方：\n{text}", style="gis.agent"))
            return
        if kind == "tools":
            if self.bundle is None:
                return
            text = self._tool_lines()
            self.console.print(Text(f"工具：\n{text}", style="gis.agent"))
            return
        if kind == "skills":
            self.console.print(Text(self._skill_lines(), style="gis.agent"))
            return

    def _tool_lines(self) -> str:
        """工具清单：名字 + 描述 + 参数名（描述以前被丢掉了）。"""
        try:
            registry = self.bundle.loop.registry
            tools = list(getattr(registry, "_tools", {}).values())
        except Exception as exc:
            return f"  （读取失败：{exc}）"
        if not tools:
            return "  （无）"
        lines: list[str] = []
        for tool in tools:
            params = list(((tool.parameters or {}).get("properties") or {}).keys())
            arg_text = f"（{', '.join(params)}）" if params else ""
            desc = " ".join(str(tool.description or "").split())[:70]
            lines.append(f"  {tool.name}{arg_text} — {desc}")
        return "\n".join(lines)

    def _skill_lines(self, *, limit: int = 40) -> str:
        """技能清单：扫 SKILL.md（与引擎注入 system prompt 的是同一套发现逻辑）。"""
        try:
            from ..skills.catalog import discover_skills

            found = discover_skills(self.workspace)
        except Exception as exc:
            return f"技能：\n  （读取失败：{exc}）"
        if not found:
            return "技能：\n  （未找到 SKILL.md；放到 workspace/skills/<名字>/SKILL.md 即可）"
        shown = found[:limit]
        lines = [
            f"技能 {len(found)} 个（可用 /skill:名字 手动加载；自然语言相关时 agent 会自己调 run_skill）："
        ]
        width = max(len(item.name) for item in shown)
        for item in shown:
            desc = " ".join(str(item.description or "").split())[:56]
            lines.append(f"  {item.name:<{width}}  {desc}")
        if len(found) > len(shown):
            lines.append(f"  …（还有 {len(found) - len(shown)} 个，候选太多时先列这些）")
        return "\n".join(lines)

    def _print_session(self) -> None:
        records = self.sessions.load(self.session_id)
        text = session_store.summarize_records(records) or "（本会话还没有记录）"
        self.console.print(Text(f"会话 {self.session_id}（{len(records)} 条）\n{text}", style="gis.agent"))

    def _resume(self, arg: str) -> None:
        if arg:
            target = arg
        else:
            items = self.sessions.list_sessions(limit=8)
            if not items:
                self.console.print(Text("（没有可续的会话）", style="gis.dim"))
                return
            self.console.print(Text("最近会话：", style="gis.status.key"))
            for index, item in enumerate(items, 1):
                self.console.print(Text(f"  {index}. {item.session_id}  {item.title}", style="gis.agent"))
            self.console.print(Text("用法：/resume <会话号或 id>", style="gis.dim"))
            return
        if arg.isdigit():
            items = self.sessions.list_sessions(limit=8)
            index = int(arg) - 1
            if 0 <= index < len(items):
                target = items[index].session_id
            else:
                self.console.print(Text("序号超出范围", style="gis.warn"))
                return
        self.session_id = target
        self.pending_context = self.sessions.context_prefix(target)
        self.console.print(Text(f"已切到会话 {target}，下一条任务会带上它的上下文", style="gis.ok"))

    # ------------------------------------------------------------ 主循环

    def run(self, initial_task: str = "") -> int:
        code = self._build()
        if code != 0:
            return code

        wiring.banner_for(self.console, self.bundle, version=_version())
        self.console.print(Text(f"会话 {self.session_id} · 直接输入任务开始（/help 查看命令）", style="gis.dim"))
        notice = ui_prompt.degrade_notice()
        if notice:
            self.console.print(Text(f"! {notice}", style="gis.warn"))
        if self.pending_context:
            self.console.print(Text("（已带上上一轮会话的上下文）", style="gis.dim"))
        self.console.print()

        self.reader = ui_prompt.PromptInput(
            self.workspace,
            mode=self.theme_mode,
            no_color=self.no_color,
            on_toggle_output=self._toggle_output,
            on_interrupt=self._interrupt,
            input=self._prompt_input,
            output=self._prompt_output,
            registry=self._command_registry(),
            # 流式实时预览（输入框上方）：模型正在生成的思考/回复。
            # 终端不支持 ANSI 时，rich Live 会满屏乱码，这里改由
            # prompt_toolkit 重绘（走 Win32 API，conhost 里也正确）。
            stream_provider=lambda: self.view.live_preview(),
        )

        pending_initial = initial_task.strip()
        try:
            while True:
                # 上一轮结束后若有排队的补充说明 → 直接接着做，不用用户再输入
                # （回显由 _dispatch 统一负责，这里不再重复打印）
                followup = self._take_followup()
                if followup and not (self.session and self.session.busy):
                    self._dispatch(followup)
                    continue
                if pending_initial:
                    text = pending_initial
                    pending_initial = ""
                else:
                    busy = bool(self.session and self.session.busy)
                    try:
                        # 标记由输入框自己画（"│ › "），这里不要再传 "›"。
                        # 状态传**方法本身**（不是快照）：输入框每帧重算，
                        # 运行中轮次/token 才会实时跳（否则是进框前那一下的快照）。
                        text = self.reader.read_line_with_stdout_patch(
                            "", busy=busy, status=self._status_text
                        )
                    except KeyboardInterrupt:
                        self.console.print(Text("（按 Ctrl+D 或输入 /exit 退出）", style="gis.dim"))
                        continue
                if text is None:
                    break
                text = text.strip()
                if text.startswith("/"):
                    if not self._handle_command(text):
                        break
                    continue
                # 计划卡正在等人确认：下一行就是答复（s / e / q / 一句改动）。
                # 放在命令处理之后：``/exit`` 之类随时可用（退出会先把闸门判 cancel）。
                if self._plan_step(text):
                    continue
                if not text:
                    continue
                self._dispatch(text)
        finally:
            self._shutdown()
        return 0

    def _status_text(self) -> str:
        """框下状态行的文本（模型 · 状态 · 轮次 · 用时 · token · 工具）。"""
        tracker = self.session.tracker if self.session else self.view.tracker
        text = ui_render.footer_text(tracker, width=0).plain.strip()
        request = self._plan_request()
        if request is not None:
            # 状态行已经写着“待确认计划”；这里只补按键（太长会被终端截掉）
            hint = "怎么改（一句话）？" if self._plan_asking_revision else "s 开始 / e 修改 / q 取消"
            text = f"{text} · {hint}" if text else hint
        return text

    # ------------------------------------------------------------ 计划确认

    def _plan_request(self) -> ui_plan_card.PlanRequest | None:
        """当前待答复的确认请求（没有则 None）。"""
        if self.plan_gate is None:
            return None
        request = self.plan_gate.pending
        return request if request is not None and not request.answered else None

    def _plan_step(self, text: str) -> bool:
        """把这一行输入当“计划确认答复”处理；返回 True = 已消费掉这行输入。

        动作词与 ``ui.plan_card.parse_decision`` 同一套（loop 入口也用那一份）。
        ``e`` 会再问一句“怎么改”，所以分两行。
        """
        request = self._plan_request()
        if request is None:
            self._plan_asking_revision = False
            return False
        if self._plan_asking_revision:
            self._plan_asking_revision = False
            revision = (text or "").strip()
            if not revision:
                self.console.print(Text("（没听到改动内容：再按 e 说一句，或按 s 直接开始）", style="gis.dim"))
                return True
            self._resolve_plan(request, "revise", revision)
            return True
        decision = ui_plan_card.parse_decision(text)
        if decision is None:  # e：先问一句怎么改
            self._plan_asking_revision = True
            self.console.print(
                Text("（怎么改？一句话说完按回车；直接按 s 开始）", style="gis.dim")
            )
            return True
        self._resolve_plan(request, decision.action, decision.text)
        return True

    def _resolve_plan(self, request: ui_plan_card.PlanRequest, action: str, text: str = "") -> None:
        """答复闸门，并在转录里留下“确认了什么”（事后能查）。"""
        if not request.resolve(action, text):
            return
        label = {
            "start": "开始执行",
            "revise": f"按修改重抽计划 —— {text}",
            "cancel": "取消，这次任务不开始",
        }.get(action, action)
        style = "gis.ok" if action == "start" else ("gis.warn" if action == "cancel" else "gis.agent")
        self.console.print(Text(f"（计划确认：{label}）", style=style))
        self.sessions.append(self.session_id, "user", f"[计划确认] {label}")

    def _dispatch(self, text: str, *, echo: bool = False) -> None:
        """派发用户输入。

        **总是回显**：输入框提交后会被擦掉（``erase_when_done``），不回显的话
        滚动转录里就看不到“我刚才说了什么”，也无法确认提交成功（实测用户因此
        以为没提交而重复输入）。
        """
        assert self.session is not None
        self.console.print(ui_render.user_block(text))
        self.console.print()
        self.sessions.append(self.session_id, "user", text)
        if self.session.busy:
            # 运行中：排队（不注入当前任务）。合并成同一个任务的补充说明，
            # 本轮结束后自动接着做 —— 要换任务请开新对话。
            self.session.guide(text)
            count = self.session.queued_count
            # 事件走一遍：状态行（footer）与转录都能看到“已排队”，
            # 用户就不会以为消息丢了而重复输入。
            self._emit_queued(count)
            self.console.print(Text(f"（已排队 {count} 条，本轮结束后合并继续）", style="gis.dim"))
            return
        task = text
        if self.pending_context:
            task = f"{self.pending_context}\n\n本次任务：{text}"
            self.pending_context = ""
        started = self.session.start(task)
        if not started:
            self.session.guide(text)
            self.console.print(Text(f"（{self.session.queue_hint()}）", style="gis.dim"))

    # ------------------------------------------------------------ 快捷键

    def _toggle_output(self) -> None:
        latest = self.store.latest()
        if latest is None:
            return
        title, body = latest
        self.console.print(Text(f"── {title} 最近一次输出 ──", style="gis.status.key"))
        self.console.print(Text(body, style="gis.code"))

    def _interrupt(self) -> bool:
        # 计划卡正等着确认：Esc 等同按 q（引擎线程停在闸门上，不取消就一直等）
        if self.plan_gate is not None and self.plan_gate.cancel_pending():
            self.console.print(Text("（已取消计划确认，这次任务不开始）", style="gis.warn"))
            return True
        if self.session is None or not self.session.busy:
            return False
        ok = self.session.interrupt()
        if ok:
            self.console.print(Text("（已请求中断，等待当前步骤收尾…）", style="gis.warn"))
        return ok

    # ------------------------------------------------------------ 收尾

    def _shutdown(self) -> None:
        # 退出时还挂在计划闸门上：先判成 cancel，否则线程会一直等下去
        if self.plan_gate is not None:
            self.plan_gate.cancel_pending()
        if self.session is not None and self.session.busy:
            self.session.interrupt()
            self.session.wait(timeout=20)
        if self.bundle is not None:
            try:
                self.bundle.close()
            except Exception:
                pass
        self.console.print(Text(f"会话已保存：{self.sessions.path_for(self.session_id)}", style="gis.dim"))


def _version() -> str:
    from ..ui.theme import engine_version

    return engine_version()


def _tail_of(path: str, *, lines: int = 8) -> str:
    """读文件末尾几行（读不到就返回提示，不让 /trace 报错）。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.readlines()
    except Exception as exc:
        return f"（读取失败：{exc}）"
    return "".join(content[-max(1, lines) :]).rstrip() or "（文件是空的）"


__all__ = ["InteractiveChat"]
