"""Agent CLI 命令行接口。

提供与 GIS Agent 交互的命令行界面。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("缺少必要的依赖包。请运行: pip install typer rich")
    sys.exit(1)

from .agent import GISAgent, AgentConfig
from .evaluation import AgentEvaluationLoop, MultiModelBenchmark
from .executor import ExecutionMode
from .llm import create_llm_client, LLMConfig
from .model_adaptation import BAMLBridge


# 创建 CLI 应用
app = typer.Typer(
    name="gis-agent",
    help="GIS Agent - AI 驱动的 GIS 工作流助手",
    add_completion=False
)

console = Console()


def read_user_input() -> str:
    """读取用户输入，支持粘贴多行文本。
    
    方案：使用标准 input() 但收集所有粘贴的行。
    - 短命令直接按 Enter 发送
    - 多行文本粘贴后，输入空行或 '//' 结束
    - 或者使用 Ctrl+Z (Windows) / Ctrl+D (Unix) 结束
    """
    import select
    import sys
    
    lines = []
    
    # 第一行
    try:
        first_line = input()
    except EOFError:
        return ""
    
    # 如果是短命令或单行，直接返回
    # 检查是否有更多数据在缓冲区（表示粘贴了多行）
    if sys.stdin.isatty():
        # 交互模式，检查是否是简单命令
        stripped = first_line.strip().lower()
        # 常见的短命令直接返回
        if stripped in ("", "exit", "quit", "q", "退出", "执行", "确认", 
                        "取消", "1", "2", "3", "是", "否", "yes", "no",
                        "y", "n", "帮助", "help", "状态", "status"):
            return first_line
        
        # 检查是否以数字开头（如 "1"、"2." 等选择）
        if stripped and stripped[0].isdigit():
            return first_line
    
    lines.append(first_line)
    
    # 继续读取后续行
    consecutive_empty = 0
    
    while True:
        try:
            line = input()
            
            # 结束标记
            if line.strip() == "//":
                break
            
            if line == "":
                consecutive_empty += 1
                if consecutive_empty >= 1:  # 一个空行就结束
                    break
                lines.append("")
            else:
                consecutive_empty = 0
                lines.append(line)
                
        except EOFError:
            break
    
    return "\n".join(lines).strip()


def create_agent(
    workspace: Optional[Path] = None,
    dry_run: bool = True,
    llm_enabled: bool = True,
    baml_precheck: bool = True,
    baml_strict: bool = False,
    baml_require: Optional[list[str]] = None,
    expert_mode: bool = True,
) -> GISAgent:
    """Create and configure the GIS Agent."""
    workspace_root = workspace or Path.cwd()
    safe_session = re.sub(r"[^A-Za-z0-9_-]", "_", str(workspace_root.resolve()))
    config = AgentConfig(
        session_id=f"ws_{safe_session}",
        workspace_path=workspace_root,
        memory_path=workspace_root / ".gis_agent_memory",
        state_path=workspace_root / ".gis_agent_state" / "workflow_state.json",
        default_mode=ExecutionMode.DRY_RUN if dry_run else ExecutionMode.EXECUTE,
        llm_enabled=llm_enabled,
        expert_mode=expert_mode,
    )
    
    llm_client = None
    llm_config: LLMConfig | None = None
    llm_init_error: str | None = None
    if llm_enabled:
        try:
            cfg_path = workspace_root / "config" / "llm_config.json"
            if cfg_path.exists():
                llm_config = LLMConfig.from_file(str(cfg_path))
            else:
                llm_config = LLMConfig.from_env()
            if llm_config.api_key:
                llm_client = create_llm_client(llm_config)
            else:
                llm_init_error = "未检测到 API Key，LLM 已禁用。"
        except Exception as exc:
            llm_init_error = f"LLM 初始化失败：{exc}"

    if llm_enabled and baml_precheck and llm_config is not None:
        _run_baml_preflight(
            llm_config=llm_config,
            strict_override=baml_strict,
            required_override=baml_require,
        )
    
    agent = GISAgent(config=config, llm_client=llm_client)
    if llm_init_error:
        agent.memory.set_context("llm_init_error", llm_init_error)
    else:
        agent.memory.set_context("llm_init_error", "")
    return agent


def _run_baml_preflight(
    llm_config: LLMConfig,
    strict_override: bool = False,
    required_override: Optional[list[str]] = None,
) -> None:
    if not bool(getattr(llm_config, "enable_baml_standardizer", True)):
        return
    if not bool(getattr(llm_config, "enable_baml_preflight", True)):
        return

    function_map = getattr(llm_config, "baml_functions", {})
    if not isinstance(function_map, dict):
        function_map = {}

    bridge = BAMLBridge(
        enabled=True,
        function_map=function_map,
        allow_builtin_fallback=bool(getattr(llm_config, "enable_baml_builtin_fallback", True)),
    )
    required = required_override or getattr(
        llm_config,
        "baml_required_capabilities",
        ["intent", "task_refine", "planning", "recovery"],
    )
    validation = bridge.validate_required_capabilities(required)

    strict = bool(getattr(llm_config, "baml_preflight_strict", False)) or strict_override
    if validation.get("ok"):
        return

    missing = validation.get("missing", []) if isinstance(validation.get("missing"), list) else []
    message = (
        "BAML 预检未通过：缺少必需能力 " + ", ".join(missing) +
        "。可运行 gis-agent baml-check --strict 诊断，或在 llm_config.json 中调整 baml_functions 映射。"
    )

    if strict:
        raise RuntimeError(message)

    console.print(Panel.fit(message, title="BAML 预检警告", border_style="yellow"))


@app.command()
def chat(
    message: str = typer.Argument(None, help="发送给 Agent 的初始消息"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
    execute: bool = typer.Option(False, "--execute", "-x", help="执行模式（默认为预览模式）"),
    no_llm: bool = typer.Option(False, "--no-llm", help="禁用 LLM 集成"),
    baml_precheck: bool = typer.Option(True, "--baml-precheck/--no-baml-precheck", help="启动前执行 BAML 预检"),
    baml_strict: bool = typer.Option(False, "--baml-strict", help="BAML 预检严格模式（失败即阻断）"),
    baml_require: str = typer.Option("", "--baml-require", help="BAML 必需能力列表（逗号分隔）"),
):
    """启动与 GIS Agent 的交互式对话。"""
    require_caps = [item.strip() for item in baml_require.split(",") if item.strip()] or None
    try:
        agent = create_agent(
            workspace=workspace,
            dry_run=not execute,
            llm_enabled=not no_llm,
            baml_precheck=baml_precheck,
            baml_strict=baml_strict,
            baml_require=require_caps,
        )
    except RuntimeError as exc:
        console.print(Panel.fit(str(exc), title="启动失败", border_style="red"))
        raise typer.Exit(code=1)
    
    console.print(Panel.fit(
        "[bold blue]GIS Agent[/bold blue] - AI 驱动的 GIS 工作流助手\n"
        "输入 GIS 任务或问题，按两次回车发送。输入 'exit' 退出。\n"
        "[dim]支持粘贴多行文本，完成后按两次回车确认。[/dim]",
        title="欢迎"
    ))
    
    # 处理初始消息（如果有）
    if message:
        _process_message(agent, message)
    
    # 交互循环
    while True:
        try:
            console.print("\n[bold green]你:[/bold green] ", end="")
            user_input = read_user_input()
            
            if user_input.lower() in ("exit", "quit", "q", "退出"):
                console.print("[dim]再见！[/dim]")
                break
            
            if not user_input.strip():
                continue
            
            _process_message(agent, user_input)
            
        except KeyboardInterrupt:
            console.print("\n[dim]已中断。再见！[/dim]")
            break


def _process_message(agent: GISAgent, message: str) -> None:
    """处理消息并显示响应。"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        progress.add_task("思考中...", total=None)
        response = agent.chat(message)
    
    # 显示响应
    console.print("\n[bold blue]Agent:[/bold blue]")
    
    # 尝试渲染为 Markdown
    try:
        md = Markdown(response.content)
        console.print(md)
    except Exception:
        console.print(response.content)
    
    # 显示建议（如果有）
    if response.suggestions:
        console.print("\n[dim]建议操作:[/dim]")
        for i, suggestion in enumerate(response.suggestions, 1):
            console.print(f"  {i}. {suggestion}")


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


@app.command()
def run(
    task: str = typer.Argument(..., help="任务描述"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="预览模式或执行模式"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录"),
    baml_precheck: bool = typer.Option(True, "--baml-precheck/--no-baml-precheck", help="启动前执行 BAML 预检"),
    baml_strict: bool = typer.Option(False, "--baml-strict", help="BAML 预检严格模式（失败即阻断）"),
    baml_require: str = typer.Option("", "--baml-require", help="BAML 必需能力列表（逗号分隔）"),
):
    """直接执行 GIS 任务。"""
    require_caps = [item.strip() for item in baml_require.split(",") if item.strip()] or None
    try:
        agent = create_agent(
            workspace=workspace,
            dry_run=dry_run,
            llm_enabled=True,
            baml_precheck=baml_precheck,
            baml_strict=baml_strict,
            baml_require=require_caps,
        )
    except RuntimeError as exc:
        console.print(Panel.fit(str(exc), title="启动失败", border_style="red"))
        raise typer.Exit(code=1)
    
    console.print(f"[bold]任务:[/bold] {task}")
    console.print(f"[bold]模式:[/bold] {'预览' if dry_run else '执行'}")
    console.print()
    
    # 创建计划
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task_id = progress.add_task("正在创建计划...", total=None)
        plan = agent.create_plan(task)
        progress.update(task_id, description="计划已创建")
    
    # 显示计划
    console.print(Panel.fit(
        "\n".join([
            f"[bold]{plan.goal}[/bold]",
            "",
            *[f"{i}. [{s.tool}] {s.description}" for i, s in enumerate(plan.steps, 1)]
        ]),
        title="执行计划"
    ))
    
    # 执行
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task_id = progress.add_task("正在执行...", total=len(plan.steps))
        
        def on_progress(msg: str, pct: float):
            progress.update(task_id, description=msg, completed=int(pct / 100 * len(plan.steps)))
        
        agent.executor.on_progress = on_progress
        result = agent.execute_plan()
    
    # 显示结果
    if result.success:
        outputs_lines = [f"  - {o}" for o in result.outputs] if result.outputs else ["  （无）"]
        console.print(Panel.fit(
            "\n".join([
                "[bold green]成功！[/bold green]",
                "",
                "[bold]输出:[/bold]",
                *outputs_lines
            ]),
            title="结果"
        ))
    else:
        console.print(Panel.fit(
            f"[bold red]失败[/bold red]\n\n{result.error}",
            title="结果"
        ))


@app.command()
def tools():
    """列出可用工具。"""
    from ..core import ToolRegistry
    
    # 工具中文描述
    tool_desc_cn = {
        "scan_layers": "扫描目录中的 GIS 图层",
        "merge_layers": "合并多个 GIS 图层",
        "project_layers": "投影坐标系转换",
        "export_map": "导出地图为 PDF/PNG",
        "quality_check": "数据质量检查"
    }
    
    category_cn = {
        "data": "数据处理",
        "analysis": "空间分析",
        "visualization": "可视化",
        "utility": "工具"
    }
    
    registry = ToolRegistry.instance()
    tools_list = registry.list_tools()
    
    console.print(Panel.fit("[bold]可用工具[/bold]"))
    
    for tool in tools_list:
        desc = tool_desc_cn.get(tool.name, tool.description[:100])
        cat = category_cn.get(tool.category.value, tool.category.value)
        console.print(f"\n[bold cyan]{tool.name}[/bold cyan]")
        console.print(f"  {desc}")
        console.print(f"  [dim]分类: {cat}[/dim]")


@app.command()
def skills():
    """列出可用技能。"""
    from ..skills import SkillRegistry, get_skill_loader
    
    # 技能中文描述
    skill_desc_cn = {
        "thematic_map": "专题图制作 - 扫描数据合并投影导出地图",
        "data_integration": "数据集成 - 扫描验证转换合并数据",
        "quality_assurance": "质量保证 - 扫描数据检查质量生成报告"
    }
    
    registry = SkillRegistry.instance()
    skills_list = registry.list_skills()
    
    console.print(Panel.fit("[bold]可用技能[/bold]"))
    
    for skill in skills_list:
        desc = skill_desc_cn.get(skill.name, skill.description)
        console.print(f"\n[bold cyan]{skill.name}[/bold cyan]")
        console.print(f"  {desc}")
        if skill.steps:
            console.print(f"  [dim]步骤: {'  '.join(skill.steps)}[/dim]")

    # 自定义 SKILL.md 技能
    loader = get_skill_loader()
    custom_skills = loader.list_skills()
    if custom_skills:
        console.print("\n[bold]自定义技能 (SKILL.md)[/bold]")
        for skill in custom_skills:
            mode = "可执行" if getattr(skill, "is_executable", False) else "文档参考"
            console.print(f"\n[bold magenta]{skill.name}[/bold magenta]")
            console.print(f"  {skill.description}")
            console.print(f"  [dim]类型: {mode}[/dim]")
            if skill.triggers:
                console.print(f"  [dim]触发词: {', '.join(skill.triggers[:6])}[/dim]")


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


@app.command("exec-status")
def exec_status(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录")
):
    """显示执行适配器状态（legacy/langchain/shadow）。"""
    agent = create_agent(workspace=workspace, dry_run=True, llm_enabled=False)
    summary = agent.get_session_summary()
    adapter = summary.get("execution_adapter") or {}
    mode = adapter.get("mode", "legacy")
    shadow_compare = adapter.get("shadow_compare", {})
    lines = [
        f"[bold]执行后端模式:[/bold] {mode}",
        f"[bold]配置路径:[/bold] {adapter.get('config_path', 'config/execution_adapter_config.json')}",
    ]
    if shadow_compare:
        lines.extend([
            "",
            "[bold]Shadow 对照:[/bold]",
            f"- legacy_success: {shadow_compare.get('legacy_success')}",
            f"- langchain_success: {shadow_compare.get('langchain_success')}",
            f"- legacy_outputs: {shadow_compare.get('legacy_outputs')}",
            f"- langchain_outputs: {shadow_compare.get('langchain_outputs')}",
        ])
    metrics = adapter.get("metrics", {})
    if metrics:
        lines.extend([
            "",
            "[bold]运行指标:[/bold]",
            f"- runs: {metrics.get('runs', 0)}",
            f"- legacy_runs: {metrics.get('legacy_runs', 0)}",
            f"- shadow_runs: {metrics.get('shadow_runs', 0)}",
            f"- langchain_attempts: {metrics.get('langchain_attempts', 0)}",
            f"- langchain_successes: {metrics.get('langchain_successes', 0)}",
            f"- langchain_fallbacks: {metrics.get('langchain_fallbacks', 0)}",
            f"- last_backend_used: {adapter.get('last_backend_used', 'legacy')}",
        ])
    if adapter.get("last_error"):
        lines.extend(["", f"[bold]最近错误:[/bold] {adapter.get('last_error')}"])
    if adapter.get("history_count") is not None:
        lines.extend([
            "",
            "[bold]历史事件:[/bold]",
            f"- history_count: {adapter.get('history_count', 0)}",
            f"- history_path: {adapter.get('history_path', 'config/execution_adapter_history.json')}",
        ])
    guard = adapter.get("guard_recommendation", {})
    if guard:
        lines.extend([
            "",
            "[bold]Guard 建议:[/bold]",
            f"- can_recommend_switch: {guard.get('can_recommend_switch')}",
            f"- target_mode: {guard.get('target_mode')}",
            f"- fallback_rate: {guard.get('fallback_rate')}",
            f"- attempts: {guard.get('attempts')}",
            f"- fallbacks: {guard.get('fallbacks')}",
        ])
    console.print(Panel.fit("\n".join(lines), title="Execution Adapter Status"))


@app.command("exec-history")
def exec_history(
    limit: int = typer.Option(10, "--limit", "-n", help="显示最近 N 条"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
):
    """显示执行适配器历史事件。"""
    agent = create_agent(workspace=workspace, dry_run=True, llm_enabled=False)
    if not hasattr(agent.executor, "load_history"):
        console.print("[bold red]当前执行器不支持历史查询[/bold red]")
        raise typer.Exit(code=1)
    history = agent.executor.load_history()
    if not history:
        console.print("[dim]暂无执行历史[/dim]")
        return
    last_items = history[-max(1, limit):]
    lines = []
    for idx, item in enumerate(last_items, 1):
        lines.append(
            f"{idx}. {item.get('timestamp')} | mode={item.get('mode')} | "
            f"backend={item.get('backend_used')} | success={item.get('success')} | "
            f"fallback={item.get('fallback')}"
        )
    console.print(Panel.fit("\n".join(lines), title="Execution Adapter History"))


@app.command("exec-reload")
def exec_reload(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录")
):
    """热加载执行适配器配置。"""
    agent = create_agent(workspace=workspace, dry_run=True, llm_enabled=False)
    payload = {}
    if hasattr(agent.executor, "reload_config"):
        payload = agent.executor.reload_config()
    console.print(Panel.fit(
        "\n".join([
            "[bold green]执行适配器已重载[/bold green]",
            "",
            f"[bold]mode:[/bold] {payload.get('mode', 'legacy')}",
            f"[bold]config:[/bold] {payload.get('config_path', 'config/execution_adapter_config.json')}",
        ]),
        title="Execution Adapter Reload"
    ))


@app.command("exec-mode")
def exec_mode(
    mode: str = typer.Argument(..., help="legacy | langchain | shadow"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
):
    """切换执行适配器模式并持久化到配置文件。"""
    agent = create_agent(workspace=workspace, dry_run=True, llm_enabled=False)
    if not hasattr(agent.executor, "set_mode"):
        console.print("[bold red]当前执行器不支持动态模式切换[/bold red]")
        raise typer.Exit(code=1)
    try:
        payload = agent.executor.set_mode(mode, persist=True)
    except Exception as exc:
        console.print(f"[bold red]切换失败:[/bold red] {exc}")
        raise typer.Exit(code=1)
    console.print(Panel.fit(
        "\n".join([
            "[bold green]执行模式已更新[/bold green]",
            "",
            f"[bold]mode:[/bold] {payload.get('mode', mode)}",
            f"[bold]config:[/bold] {payload.get('config_path', 'config/execution_adapter_config.json')}",
        ]),
        title="Execution Adapter Mode"
    ))


@app.command("exec-metrics-reset")
def exec_metrics_reset(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
):
    """重置执行适配器运行指标。"""
    agent = create_agent(workspace=workspace, dry_run=True, llm_enabled=False)
    if not hasattr(agent.executor, "reset_metrics"):
        console.print("[bold red]当前执行器不支持指标重置[/bold red]")
        raise typer.Exit(code=1)
    payload = agent.executor.reset_metrics(persist=True)
    console.print(Panel.fit(
        "\n".join([
            "[bold green]执行指标已重置[/bold green]",
            "",
            f"[bold]metrics_path:[/bold] {payload.get('metrics_path', 'config/execution_adapter_metrics.json')}",
            f"[bold]runs:[/bold] {(payload.get('metrics') or {}).get('runs', 0)}",
        ]),
        title="Execution Adapter Metrics Reset"
    ))


@app.command("exec-guard-apply")
def exec_guard_apply(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
):
    """按 Guard 策略自动应用推荐模式。"""
    agent = create_agent(workspace=workspace, dry_run=True, llm_enabled=False)
    if not hasattr(agent.executor, "apply_guard_recommendation"):
        console.print("[bold red]当前执行器不支持 Guard 自动切换[/bold red]")
        raise typer.Exit(code=1)
    payload = agent.executor.apply_guard_recommendation(persist=True)
    guard = payload.get("guard_recommendation", {})
    console.print(Panel.fit(
        "\n".join([
            "[bold green]Guard 推荐已应用[/bold green]",
            "",
            f"[bold]mode:[/bold] {payload.get('mode', 'legacy')}",
            f"[bold]target_mode:[/bold] {guard.get('target_mode', 'shadow')}",
            f"[bold]reason:[/bold] {guard.get('reason', '')}",
        ]),
        title="Execution Guard Apply"
    ))


@app.command("recover")
def recover(
    failed_tool: str = typer.Option("merge_layers", "--failed-tool", help="失败工具名"),
    failed_error: str = typer.Option("参数验证失败", "--failed-error", help="失败错误信息"),
):
    """演示恢复策略模板（用于快速验证恢复链路）。"""
    from .planner import AgentPlanner, Plan, PlanStep

    planner = AgentPlanner(llm_client=None)
    failed_step = PlanStep(
        id="step_failed_1",
        tool=failed_tool,
        description=f"失败步骤: {failed_tool}",
        input={},
        error=failed_error
    )
    demo_plan = Plan(id="demo_recovery", goal="demo", steps=[failed_step])
    recovery = planner.suggest_recovery(demo_plan, failed_step)

    if not recovery:
        console.print("[bold red]未生成恢复计划[/bold red]")
        return

    strategy = recovery.metadata.get("recovery_strategy", "unknown")
    lines = [f"[bold]策略[/bold]: {strategy}", "", "[bold]恢复步骤[/bold]:"]
    for i, step in enumerate(recovery.steps, 1):
        lines.append(f"{i}. [{step.tool}] {step.description}")
    console.print(Panel.fit("\n".join(lines), title="Recovery Preview"))


@app.command("eval-loop")
def eval_loop(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
    limit: int = typer.Option(30, "--limit", "-n", help="分析最近 N 个会话"),
):
    """运行第三阶段评测闭环并生成报告。"""
    workspace_root = workspace or Path.cwd()
    memory_dir = workspace_root / ".gis_agent_memory"
    evaluation_dir = workspace_root / "workspace" / "output" / "evaluation"
    exec_history = workspace_root / "config" / "execution_adapter_history.json"

    loop = AgentEvaluationLoop(
        memory_dir=memory_dir,
        history_dir=evaluation_dir,
        execution_history_path=exec_history,
    )
    report = loop.run(session_limit=max(1, limit))
    metrics = report.metrics

    lines = [
        "[bold green]评测完成[/bold green]",
        "",
        f"[bold]Sessions:[/bold] {metrics.sessions_analyzed}",
        f"[bold]Context Hit Rate:[/bold] {metrics.context_hit_rate:.2%} ({metrics.context_hits}/{metrics.context_checks})",
        f"[bold]Plan Continuity Rate:[/bold] {metrics.plan_continuity_rate:.2%} ({metrics.continuity_hits}/{metrics.continuity_checks})",
        f"[bold]Mis-Trigger Rate:[/bold] {metrics.mis_trigger_rate:.2%} ({metrics.mis_trigger_count}/{metrics.mis_trigger_checks})",
        f"[bold]Clip Hijack Events:[/bold] {metrics.clip_hijack_events}",
        "",
        f"[bold]JSON 报告:[/bold] {report.report_path}",
        f"[bold]趋势文件:[/bold] {str((evaluation_dir / 'metrics_history.json'))}",
    ]
    if report.recommendations:
        lines.extend(["", "[bold]改进建议:[/bold]"])
        lines.extend([f"- {tip}" for tip in report.recommendations])

    console.print(Panel.fit("\n".join(lines), title="Evaluation Loop"))


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
):
    """运行跨模型基准评测并输出成功率差、解析错误率和时延指标。"""
    workspace_root = workspace or Path.cwd()
    selected_models = [m.strip() for m in models.split(",") if m.strip()]
    if not selected_models:
        console.print("[bold red]未提供有效模型列表[/bold red]")
        raise typer.Exit(code=1)

    tasks = _load_benchmark_tasks(tasks_file)

    cfg_path = workspace_root / "config" / "llm_config.json"
    base_config = None
    try:
        if cfg_path.exists():
            base_config = LLMConfig.from_file(str(cfg_path))
        else:
            base_config = LLMConfig.from_env()
    except Exception:
        base_config = LLMConfig.from_env()

    benchmark = MultiModelBenchmark(
        output_dir=workspace_root / "workspace" / "output" / "evaluation",
        base_config=base_config,
    )
    report = benchmark.run(models=selected_models, tasks=tasks, repeats=max(1, repeats))

    lines = [
        "[bold green]跨模型评测完成[/bold green]",
        "",
        f"[bold]Models:[/bold] {', '.join(selected_models)}",
        f"[bold]Tasks:[/bold] {len(report.tasks)}",
        f"[bold]Repeats:[/bold] {report.repeats}",
        f"[bold]Success Rate Gap:[/bold] {report.success_rate_gap:.2%}",
        f"[bold]Max Parse Error Rate:[/bold] {report.max_parse_error_rate:.2%}",
        "",
        f"[bold]JSON 报告:[/bold] {report.report_path}",
        f"[bold]Markdown 报告:[/bold] {str(Path(report.report_path).with_suffix('.md'))}",
    ]
    console.print(Panel.fit("\n".join(lines), title="Model Benchmark"))


@app.command("baml-check")
def baml_check(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="工作空间目录"),
    strict: bool = typer.Option(False, "--strict", help="严格模式：缺少必需能力时返回非零退出码"),
    require: str = typer.Option("intent,task_refine,planning,recovery", "--require", help="严格模式下必需能力（逗号分隔）"),
):
    """检查 BAML client 可用性与函数映射绑定状态。"""
    workspace_root = workspace or Path.cwd()
    cfg_path = workspace_root / "config" / "llm_config.json"
    try:
        if cfg_path.exists():
            cfg = LLMConfig.from_file(str(cfg_path))
        else:
            cfg = LLMConfig.from_env()
    except Exception:
        cfg = LLMConfig.from_env()

    bridge = BAMLBridge(
        enabled=bool(getattr(cfg, "enable_baml_standardizer", True)),
        function_map=getattr(cfg, "baml_functions", {}) if isinstance(getattr(cfg, "baml_functions", {}), dict) else {},
        allow_builtin_fallback=bool(getattr(cfg, "enable_baml_builtin_fallback", True)),
    )
    diag = bridge.diagnostics()

    lines = [
        "[bold green]BAML 自检完成[/bold green]",
        "",
        f"[bold]enabled:[/bold] {diag.get('enabled')}",
        f"[bold]client_available:[/bold] {diag.get('client_available')}",
    ]

    functions = diag.get("functions", {}) if isinstance(diag.get("functions", {}), dict) else {}
    for capability in ["intent", "task_refine", "planning", "recovery"]:
        entries = functions.get(capability, []) if isinstance(functions.get(capability, []), list) else []
        if not entries:
            continue
        lines.append("")
        lines.append(f"[bold]{capability}[/bold]")
        for entry in entries:
            name = entry.get("name")
            exists = bool(entry.get("exists"))
            callable_flag = bool(entry.get("callable"))
            status = "OK" if (exists and callable_flag) else "MISSING"
            lines.append(f"- {name}: {status}")

    if not bool(diag.get("client_available")):
        lines.extend(
            [
                "",
                "[bold yellow]提示:[/bold yellow] 当前未检测到 baml_client，将自动回退到非 BAML 路径。",
            ]
        )

    required_caps = [item.strip().lower() for item in str(require).split(",") if item.strip()]
    validation = bridge.validate_required_capabilities(required_caps)
    if strict:
        lines.extend([
            "",
            f"[bold]strict required:[/bold] {', '.join(validation.get('required', []))}",
            f"[bold]strict ok:[/bold] {validation.get('ok')}",
        ])
        missing = validation.get("missing", []) if isinstance(validation.get("missing"), list) else []
        if missing:
            lines.append(f"[bold red]missing:[/bold red] {', '.join(missing)}")

    console.print(Panel.fit("\n".join(lines), title="BAML Check"))

    if strict and not bool(validation.get("ok")):
        raise typer.Exit(code=1)


@app.command()
def status(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="会话 ID")
):
    """显示 Agent 状态。"""
    workspace_root = Path.cwd()
    llm_config = None
    try:
        cfg_path = workspace_root / "config" / "llm_config.json"
        if cfg_path.exists():
            llm_config = LLMConfig.from_file(str(cfg_path))
        else:
            llm_config = LLMConfig.from_env()
    except Exception:
        llm_config = None

    agent = create_agent()
    
    if session_id:
        agent.load_session(session_id)
    
    summary = agent.get_session_summary()

    baml_lines = []
    if llm_config is not None:
        function_map = getattr(llm_config, "baml_functions", {}) if isinstance(getattr(llm_config, "baml_functions", {}), dict) else {}
        bridge = BAMLBridge(
            enabled=bool(getattr(llm_config, "enable_baml_standardizer", True)),
            function_map=function_map,
            allow_builtin_fallback=bool(getattr(llm_config, "enable_baml_builtin_fallback", True)),
        )
        diag = bridge.diagnostics()
        capability_ready = diag.get("capability_ready", {}) if isinstance(diag.get("capability_ready", {}), dict) else {}
        baml_lines = [
            "",
            f"[bold]BAML 启用:[/bold] {diag.get('enabled')}",
            f"[bold]BAML client 可用:[/bold] {diag.get('client_available')}",
            f"[bold]intent ready:[/bold] {capability_ready.get('intent')}",
            f"[bold]task_refine ready:[/bold] {capability_ready.get('task_refine')}",
            f"[bold]planning ready:[/bold] {capability_ready.get('planning')}",
            f"[bold]recovery ready:[/bold] {capability_ready.get('recovery')}",
        ]
    
    console.print(Panel.fit(
        "\n".join([
            f"[bold]会话:[/bold] {summary['session_id']}",
            f"[bold]对话轮次:[/bold] {summary['memory']['total_turns']}",
            f"[bold]工具调用:[/bold] {summary['memory']['tool_calls']}",
            "",
            f"[bold]当前计划:[/bold] {summary['current_plan']['goal'] if summary['current_plan'] else '无'}",
            f"[bold]最近结果:[/bold] {'成功' if summary['last_result'] and summary['last_result']['success'] else '无'}",
            f"[bold]执行后端:[/bold] {(summary.get('execution_adapter') or {}).get('mode', 'legacy')}",
            *baml_lines,
        ]),
        title="Agent 状态"
    ))


def main():
    """主入口。"""
    app()


if __name__ == "__main__":
    main()
