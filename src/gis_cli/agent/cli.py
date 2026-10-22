"""Agent CLI 命令行接口。

提供与 GIS Agent 交互的命令行界面。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("缺少必要的依赖包。请运行: pip install typer rich")
    sys.exit(1)

# 旧引擎（plan-then-execute）已整体迁入 ``gis_cli.legacy`` 并从打包中排除。
# 这里只剩新引擎命令：chat / loop / reload-skills / benchmark-models。
#
# 历史：旧引擎曾在模块顶层 import，把 pydantic/langchain 整条链拉进来
# （实测 0.75s，占 import 总耗时 1.12s 的 67%）。现在那些命令已删除，
# 新引擎启动只剩自己的依赖。

# 控制台自适应：把代码页尽量切到 UTF-8，并让 Python 流与控制台实际编码一致。
# 踩过的坑：无条件强制 stdout=UTF-8 时，GBK 控制台会把 UTF-8 字节按 GBK 解码 → 整屏乱码。
try:
    from ..ui.console_env import configure_stdio as _configure_stdio

    _CONSOLE_INFO = _configure_stdio()
except Exception:  # pragma: no cover - 极端环境兜底
    _CONSOLE_INFO = {}
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


# 创建 CLI 应用
app = typer.Typer(
    name="gis-agent",
    help="GIS Agent - AI 驱动的 GIS 工作流助手",
    add_completion=False
)

console = Console()








@app.command()
def chat(
    message: str = typer.Argument(None, help="首条任务（可省略，进入后再输入）"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="llm_config.json 路径"),
    theme: str = typer.Option("system", "--theme", help="配色：system（跟随终端）/ dark / light"),
    no_color: bool = typer.Option(False, "--no-color", help="纯文本模式：不输出颜色（默认在老终端就是纯文本）"),
    color: bool = typer.Option(False, "--color", help="强制上色（确认终端支持 ANSI 时用）"),
    max_turns: int = typer.Option(0, "--max-turns", help="单任务最大轮次（0 表示用配置默认值）"),
    refresh_catalog: bool = typer.Option(False, "--refresh-catalog/--no-refresh-catalog", help="启动时同步扫全数据目录（默认不扫：改由首个任务后台扫描，启动更快）"),
    continue_last: bool = typer.Option(False, "--continue/--no-continue", help="接续上一次会话"),
    resume: str = typer.Option("", "--resume", help="接续指定会话 id"),
):
    """交互式对话（新引擎）：工具动作/结果实时呈现，运行中可纠偏，Esc 中断。"""
    from .interactive import InteractiveChat

    chat_session = InteractiveChat(
        workspace=workspace,
        config=config,
        theme_mode=theme,
        no_color=no_color,
        color=color,
        max_turns=max_turns,
        refresh_catalog=refresh_catalog,
        resume_id=resume,
        continue_last=continue_last,
    )
    raise typer.Exit(code=chat_session.run(message or ""))






def _load_benchmark_tasks(tasks_file: Path | None) -> list[str] | None:
    if tasks_file is None:
        return None
    if not tasks_file.exists():
        raise FileNotFoundError(f"tasks 文件不存在: {tasks_file}")

    text = tasks_file.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if tasks_file.suffix.lower() == ".json":
        import json

        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(x).strip() for x in payload if str(x).strip()]
        raise ValueError("JSON tasks 文件必须是字符串数组")

    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line and not line.startswith("#")]








@app.command("resume-run")
def resume_run_cmd(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="llm_config.json 路径"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只看有没有可续的断点，不真的跑"),
):
    """从断点续跑上一次中断的任务（网关抖动 / 撞时限后不用从头再来）。

    读 ``.gis_agent/runs/<run_id>.json`` + 同名 ``.messages.json``：
    模型上下文、需求清单、产物都在，接着往下跑。
    """
    from ..engine.runner import build_engine

    ws = (workspace or Path.cwd()).resolve()
    bundle = build_engine(workspace=ws, config_path=config)
    try:
        found = bundle.loop.find_checkpoint()
        if found is None:
            console.print(
                Panel.fit(
                    "没有可续跑的断点。\n\n"
                    "只有这些状态能续：llm_unavailable / deadline_exceeded / max_turns / stopped / failed，"
                    "且需要 .gis_agent/runs/ 下有配套的 .messages.json（旧版本运行没存消息历史）。",
                    title="无可续断点",
                )
            )
            raise typer.Exit(code=1)
        state_path, _messages_path = found
        console.print(Panel.fit(f"断点: {state_path}", title="发现可续运行"))
        if dry_run:
            raise typer.Exit(code=0)
        state = bundle.loop.resume_from_checkpoint()
        console.print(
            Panel.fit(
                f"status={state.status}\nturns={state.turn}\n"
                f"artifacts={len(state.artifacts)}\n\n{state.final_summary or state.error}",
                title="续跑结束",
            )
        )
        raise typer.Exit(code=0 if state.status == "completed" else 1)
    finally:
        bundle.close()


@app.command("distill")
def distill_cmd(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="llm_config.json 路径"),
    run_id: str = typer.Option("", "--run", help="指定 run_id（默认取最近一次 completed）"),
    adopt: bool = typer.Option(False, "--adopt", help="采纳进 workspace/recipes/（下次自动加载）"),
    replay: bool = typer.Option(False, "--replay", help="回放验证：用默认参数在隔离目录重跑一遍"),
    verify: str = typer.Option("", "--verify", help="只回放已有候选（不重新蒸馏）；加 --adopt 采纳该候选"),
    list_only: bool = typer.Option(False, "--list", help="只看已有候选，不蒸馏"),
    reject: str = typer.Option("", "--reject", help="删除指定候选"),
):
    """把一次成功运行沉淀成可复用配方（候选 → 门禁 → 可选回放 → 可选采纳）。

    只有 ``completed`` 的运行才值得沉淀；蒸馏出的配方先进候选目录，
    默认不自动进配方库。
    """
    from ..engine.runner import build_engine
    from ..recipes.lifecycle import RecipeLifecycle

    ws = (workspace or Path.cwd()).resolve()
    bundle = build_engine(workspace=ws, config_path=config, refresh_catalog=False)
    try:
        lifecycle = RecipeLifecycle(bundle.recipes, ws, code_runner=bundle.code_runner)

        if list_only:
            rows = lifecycle.list_candidates()
            if not rows:
                console.print(Panel.fit("还没有候选配方。跑完一个任务后执行 `gis-agent distill`。", title="候选配方"))
                raise typer.Exit(code=0)
            table = Table(title="候选配方", show_lines=False)
            for col in ("id", "名称", "门禁", "来源"):
                table.add_column(col)
            for row in rows:
                table.add_row(
                    str(row.get("id", "")),
                    str(row.get("name", ""))[:28],
                    "通过" if row.get("gate_ok") else "未通过",
                    str(row.get("provenance", ""))[:24],
                )
            console.print(table)
            raise typer.Exit(code=0)

        if reject:
            ok = lifecycle.reject(reject)
            console.print(f"{'已删除候选' if ok else '没有这个候选'}: {reject}")
            raise typer.Exit(code=0 if ok else 1)

        if verify:
            console.print(f"回放验证候选 {verify}…")
            res = lifecycle.replay(verify)
            lines = [f"回放: {'通过' if res.ok else '未通过'}"]
            for check in res.checks:
                lines.append(f"  [{'OK ' if check.ok else 'FAIL'}] {check.name}: {check.detail[:90]}")
            for warn in res.warnings:
                lines.append(f"  [警告] {warn[:100]}")
            # `--verify X --adopt`：采纳**刚回放过的那一个**。不加这个组合，
            # 只能先 --verify 再 --adopt，而 --adopt 会重新蒸馏出一个新候选，
            # 等于“验的不是采的那个”（实测踩到：回放通过的没被采纳）。
            if adopt:
                if not res.ok:
                    lines.append("回放未通过，不予采纳")
                    console.print(Panel.fit("\n".join(lines), title="回放验证"))
                    raise typer.Exit(code=1)
                adopted_path, note = lifecycle.adopt(verify)
                if adopted_path is None:
                    lines.append(f"采纳失败：{note}")
                else:
                    lines.append(f"已采纳: {adopted_path}（{note}；下次 build_engine 自动加载）")
            console.print(Panel.fit("\n".join(lines), title="回放验证"))
            raise typer.Exit(code=0 if res.ok else 1)

        console.print(f"正在从 {run_id or '最近一次 completed 运行'} 蒸馏配方…")
        result = bundle.distill(run_id=run_id or None, replay=replay, adopt=adopt)

        if not result.get("ok") and not result.get("recipe"):
            console.print(Panel.fit(str(result.get("reason") or "蒸馏失败"), title="未能蒸馏"))
            raise typer.Exit(code=1)

        gate = result.get("gate") or {}
        lines = [f"配方: {result['recipe']['id']} — {result['recipe']['name']}",
                 f"参数: {len(result['recipe'].get('params') or {})} 个",
                 f"门禁: {'通过' if gate.get('ok') else '未通过'}"]
        for check in gate.get("checks") or []:
            lines.append(f"  [{'OK ' if check['ok'] else 'FAIL'}] {check['name']}: {check['detail'][:90]}")
        for warn in gate.get("warnings") or []:
            lines.append(f"  [警告] {warn[:100]}")
        for note in result.get("sanitized") or []:
            lines.append(f"  [净化] {note[:100]}")

        replay_result = result.get("replay")
        if replay_result:
            lines.append(f"回放: {'通过' if replay_result.get('ok') else '未通过'}")
            for check in replay_result.get("checks") or []:
                lines.append(f"  [{'OK ' if check['ok'] else 'FAIL'}] {check['name']}: {check['detail'][:90]}")

        lines.append(f"候选文件: {result.get('candidate')}")
        if result.get("adopted"):
            lines.append(f"已采纳: {result['adopted']}（下次 build_engine 自动加载）")
        elif gate.get("ok"):
            lines.append("下一步：`gis-agent distill --replay` 回放验证，`--adopt` 采纳入库")
        console.print(Panel.fit("\n".join(lines), title="配方蒸馏"))
        raise typer.Exit(code=0 if result.get("ok") else 1)
    finally:
        bundle.close()


@app.command("reload-skills")
def reload_skills_cmd(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录")
):
    """热重载 workspace/skills 下的 SKILL.md 技能。"""
    agent = create_agent(workspace=workspace, dry_run=True, llm_enabled=False)
    count = agent.reload_custom_skills()
    console.print(Panel.fit(
        f"[bold green]技能重载完成[/bold green]\n\n已加载自定义技能数量: [bold]{count}[/bold]",
        title="SKILL.md Hot Reload"
    ))


















@app.command("benchmark-models")
def benchmark_models(
    models: str = typer.Option(
        "gpt-4o-mini,claude-3-5-sonnet-20241022,Qwen/Qwen3.5-32B",
        "--models",
        help="逗号分隔模型列表",
    ),
    repeats: int = typer.Option(1, "--repeats", "-r", help="每个任务重复次数"),
    tasks_file: Optional[Path] = typer.Option(None, "--tasks-file", help="任务列表文件（txt/json）"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
    max_turns: int = typer.Option(30, "--max-turns", help="单任务最大轮次"),
):
    """跨模型基准：跑**真实引擎循环**，测任务是否真的做完。

    成功判据是**验收通过**，不是"产出了非空计划"——"计划对"与"做得对"
    是两回事，真正会翻车的是多轮工具调用、代码自修复、验收收敛。
    """
    from .evaluation import EngineModelBenchmark

    workspace_root = workspace or Path.cwd()
    selected_models = [m.strip() for m in models.split(",") if m.strip()]
    if not selected_models:
        console.print("[bold red]未提供有效模型列表[/bold red]")
        raise typer.Exit(code=1)

    tasks = _load_benchmark_tasks(tasks_file)
    out_dir = workspace_root / "workspace" / "output" / "evaluation"

    cfg_path = workspace_root / "config" / "llm_config.json"
    report = EngineModelBenchmark(
        output_dir=out_dir,
        models=selected_models,
        tasks=tasks,
        repeats=max(1, repeats),
        config_path=str(cfg_path) if cfg_path.exists() else None,
        max_turns=max_turns,
    ).run()

    lines = [
        "[bold green]跨模型评测完成[/bold green]",
        "",
        f"[bold]Models:[/bold] {', '.join(selected_models)}",
        f"[bold]Tasks:[/bold] {len(report.tasks)}",
        f"[bold]Repeats:[/bold] {report.repeats}",
        f"[bold]Success Rate Gap:[/bold] {report.success_rate_gap:.2%}",
        f"[bold]Max Parse Error Rate:[/bold] {report.max_parse_error_rate:.2%}",
    ]
    # 主指标：验收通过率（"完成"不等于"过验收"）
    for metric in report.model_metrics:
        lines.append(
            f"  {metric.model}: 验收 {metric.verify_pass_rate:.0%} · "
            f"完成 {metric.completed_rate:.0%} · "
            f"平均 {metric.avg_turns:.1f} 轮 · {metric.avg_tokens:.0f} tokens"
        )
    lines += [
        "",
        f"[bold]JSON 报告:[/bold] {report.report_path}",
        f"[bold]Markdown 报告:[/bold] {str(Path(report.report_path).with_suffix('.md'))}",
    ]
    console.print(Panel.fit("\n".join(lines), title=title))






def _engine_version() -> str:
    """版本号（与交互界面一致，见 ui.theme.engine_version）。"""
    try:
        from ..ui.theme import engine_version

        return engine_version()
    except Exception:
        return ""


def _run_engine_task(
    task: str,
    workspace: Optional[Path],
    config: Optional[Path],
    refresh_catalog: bool = True,
    max_turns: int = 0,
    theme_mode: str = "system",
    no_color: bool = False,
    color: bool = False,
) -> int:
    """Run one task with the new observe-decide-act engine.

    渲染统一走 :mod:`gis_cli.ui`：启动横幅 → 工具动作/结果（可折叠）→ 验收 → 结果卡片 + 状态行。
    """
    from ..engine.runner import build_engine
    from ..ui import events as ui_events
    from ..ui import render as ui_render
    from ..ui import theme as ui_theme

    ui_console = ui_theme.make_console(theme_mode, no_color=no_color, color=True if color else None)
    ws = workspace or Path.cwd()
    # 引擎内部日志写文件，不再直接刷到终端（实测：网关报错会把整段 JSON 打屏）
    ui_logging = __import__("gis_cli.ui.logging_setup", fromlist=["setup_engine_logging"])
    ui_logging.setup_engine_logging(ws)
    tracker = ui_events.StatusTracker(workspace=str(ws))
    store = ui_render.ToolOutputStore()
    interrupt = {"requested": False}

    def _on_event(name: str, payload: dict) -> None:
        tracker.apply(name, payload)
        if name == "tool_result":
            summary = str((payload or {}).get("summary", ""))
            if len(summary) > 200:
                store.add(str((payload or {}).get("tool", "")), summary)
        unit = ui_events.to_unit(name, payload)
        if unit is None:
            return
        if unit.kind == "final":
            return  # 由结果卡片统一呈现
        elapsed = tracker.action_elapsed if name == "tool_result" else None
        ui_console.print(ui_render.render_unit(unit, elapsed=elapsed))

    with ui_console.status("[gis.dim]启动引擎（加载数据目录/内核）…[/]"):
        try:
            bundle = build_engine(
                workspace=ws,
                config_path=config,
                refresh_catalog=refresh_catalog,
                on_event=_on_event,
            )
        except Exception as exc:
            ui_console.print(Panel.fit(str(exc), title="引擎启动失败", border_style="red"))
            return 1

    if max_turns:
        bundle.loop.config.max_turns = max_turns

    counts = {}
    try:
        counts = bundle.catalog.counts() or {}
    except Exception:
        counts = {}
    catalog_text = "、".join(f"{kind} {n}" for kind, n in counts.items()) or "空"
    try:
        recipes_count = len(bundle.recipes.all())
    except Exception:
        recipes_count = 0

    tracker.model = getattr(bundle.llm.config, "model", "") or ""
    tracker.provider = getattr(bundle.llm.config, "api_base", "") or ""
    tracker.status = "running"

    ui_console.print(
        ui_render.banner(
            model=tracker.model,
            api_base=tracker.provider,
            workspace=str(ws),
            catalog=catalog_text,
            recipes=recipes_count,
            session_id=str(getattr(bundle.loop.state, "run_id", "") or ""),
            kernel="ready" if getattr(bundle, "code_runner", None) is not None else "-",
            version=_engine_version(),
        )
    )

    try:
        state = bundle.loop.run(task)
    except KeyboardInterrupt:
        ui_console.print("[gis.warn]已中断（Ctrl+C）[/]")
        try:
            bundle.close()
        except Exception:
            pass
        return 130

    payload = state.to_dict() if hasattr(state, "to_dict") else dict(state)
    # 状态行兜底：即便 run_end 没发出（异常/中断路径），也以最终 state 为准
    tracker.turn = int(getattr(state, "turn", tracker.turn) or tracker.turn)
    tracker.status = str(getattr(state, "status", tracker.status) or tracker.status)
    tracker.current_action = ""
    if not tracker.finished_at:
        import time as _time

        tracker.finished_at = _time.monotonic()
    ui_console.print()
    ui_console.print(ui_render.result_card(payload))

    trace_path = getattr(getattr(bundle.loop, "recorder", None), "jsonl_path", "")
    if trace_path:
        ui_console.print(Text(f"追踪 {trace_path}", style="gis.dim"))
    if state.status == "awaiting_user":
        ui_console.print(Text("引擎在等待你的回答：补充信息后再次运行，或在对话模式里直接回复。", style="gis.warn"))
    ui_console.print(ui_render.footer_text(tracker, width=ui_console.width))

    bundle.close()
    return 0 if state.status == "completed" else 1


@app.command()
def loop(
    task: str = typer.Argument(..., help="任务描述（自然语言）"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="llm_config.json 路径"),
    refresh_catalog: bool = typer.Option(True, "--refresh-catalog/--no-refresh-catalog", help="执行前增量刷新数据目录"),
    max_turns: int = typer.Option(0, "--max-turns", help="最大轮次（0 表示用配置默认值）"),
    theme: str = typer.Option("system", "--theme", help="配色：system（跟随终端）/ dark / light"),
    no_color: bool = typer.Option(False, "--no-color", help="关闭颜色（老终端或重定向到文件时用）"),
):
    """使用新引擎（观察-决策-执行循环 + 代码自修复）执行 GIS 任务。"""
    raise typer.Exit(
        code=_run_engine_task(task, workspace, config, refresh_catalog, max_turns, theme, no_color)
    )


def main():
    """主入口。"""
    app()


if __name__ == "__main__":
    main()
