from __future__ import annotations

import json

import typer

from .orchestrator import TaskOrchestrator
from .safety import requires_confirmation

app = typer.Typer(help="General-purpose GIS task solving CLI")


def _record_to_dict(record: object) -> dict:
    from dataclasses import asdict

    return asdict(record)


@app.command("task-create")
def task_create(prompt: str) -> None:
    orch = TaskOrchestrator()
    record = orch.create_task(prompt)

    typer.echo(f"task_id: {record.task_id}")
    typer.echo(f"intent: {record.plan.intent if record.plan else 'unknown'}")
    typer.echo(f"risk: {record.risk.value}")
    typer.echo(f"runner_profile: {record.metadata.get('runner_profile', 'integration')}")
    nodes = record.metadata.get("workflow_nodes", [])
    if nodes:
        typer.echo(f"workflow_nodes: {len(nodes)}")
    if requires_confirmation(record.risk):
        typer.echo("warning: high-risk intent detected, confirmation required before destructive run")


@app.command("task-create-doc")
def task_create_doc(
    file_path: str = typer.Option("", "--file", help="Path to a task document text file"),
    text: str = typer.Option("", "--text", help="Inline task document content"),
) -> None:
    if not file_path and not text:
        raise typer.BadParameter("Provide either --file or --text.")

    orch = TaskOrchestrator()
    if file_path:
        record = orch.create_task_from_document_file(file_path)
    else:
        record = orch.create_task_from_document(text, source_name="inline")

    typer.echo(f"task_id: {record.task_id}")
    typer.echo(f"intent: {record.plan.intent if record.plan else 'unknown'}")
    typer.echo(f"planner_mode: {record.plan.planner_mode if record.plan else 'rule'}")
    typer.echo(f"risk: {record.risk.value}")
    typer.echo(f"runner_profile: {record.metadata.get('runner_profile', 'integration')}")
    nodes = record.metadata.get("workflow_nodes", [])
    if nodes:
        typer.echo("workflow_nodes:")
        for node in nodes[:10]:
            typer.echo(f"- {node}")
    capabilities = record.metadata.get("required_capabilities", [])
    if capabilities:
        typer.echo("required_capabilities:")
        for cap in capabilities:
            typer.echo(f"- {cap}")

    unresolved = record.metadata.get("unresolved_scoring_items", [])
    if unresolved:
        typer.echo("unresolved_scoring_items:")
        for item in unresolved[:5]:
            typer.echo(f"- {item}")
    if record.plan and record.plan.clarifying_questions:
        typer.echo("clarifying_questions:")
        for q in record.plan.clarifying_questions:
            typer.echo(f"- {q}")


@app.command("task-plan")
def task_plan(task_id: str) -> None:
    orch = TaskOrchestrator()
    record = orch.plan_task(task_id)
    typer.echo(json.dumps(_record_to_dict(record), ensure_ascii=False, indent=2))


@app.command("task-run")
def task_run(
    task_id: str,
    dry_run: bool = typer.Option(True, "--dry-run/--exec"),
    nodes: str = typer.Option(
        "",
        "--nodes",
        help="Comma-separated workflow nodes for partial rerun, e.g. check_geometry,repair_geometry",
    ),
) -> None:
    orch = TaskOrchestrator()
    record = orch.get_task(task_id)
    if requires_confirmation(record.risk) and not dry_run:
        ok = typer.confirm("High-risk task. Continue execution?")
        if not ok:
            raise typer.Exit(code=1)

    override_nodes = [x.strip() for x in nodes.split(",") if x.strip()] if nodes else None
    result = orch.run_task(task_id, dry_run=dry_run, override_nodes=override_nodes)
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


@app.command("task-show")
def task_show(task_id: str) -> None:
    orch = TaskOrchestrator()
    record = orch.get_task(task_id)
    typer.echo(json.dumps(_record_to_dict(record), ensure_ascii=False, indent=2))


@app.command("task-summary")
def task_summary(task_id: str) -> None:
    orch = TaskOrchestrator()
    summary = orch.get_task_summary(task_id)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command("task-suggest-rerun")
def task_suggest_rerun(
    task_id: str,
    include_skipped: bool = typer.Option(True, "--include-skipped/--no-include-skipped"),
) -> None:
    orch = TaskOrchestrator()
    nodes = orch.suggest_rerun_nodes(task_id, include_skipped=include_skipped)
    typer.echo(json.dumps(nodes, ensure_ascii=False, indent=2))


@app.command("task-rerun-failed")
def task_rerun_failed(
    task_id: str,
    dry_run: bool = typer.Option(True, "--dry-run/--exec"),
    include_skipped: bool = typer.Option(True, "--include-skipped/--no-include-skipped"),
    auto_fix: bool = typer.Option(True, "--auto-fix/--no-auto-fix"),
) -> None:
    orch = TaskOrchestrator()
    result = orch.rerun_failed_nodes(
        task_id,
        dry_run=dry_run,
        include_skipped=include_skipped,
        auto_apply_strategy=auto_fix,
    )
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


@app.command("task-suggest-fix")
def task_suggest_fix(
    task_id: str,
    include_skipped: bool = typer.Option(True, "--include-skipped/--no-include-skipped"),
) -> None:
    orch = TaskOrchestrator()
    plan = orch.suggest_recovery_plan(task_id, include_skipped=include_skipped)
    typer.echo(json.dumps(plan, ensure_ascii=False, indent=2))


@app.command("task-list")
def task_list(
    skip: int = typer.Option(0, "--skip", help="Number of tasks to skip"),
    limit: int = typer.Option(100, "--limit", help="Maximum number of tasks to return"),
    status: str = typer.Option("", "--status", help="Filter by task status"),
    profile: str = typer.Option("", "--profile", help="Filter by recovery profile"),
) -> None:
    """List tasks with optional pagination and filtering."""
    orch = TaskOrchestrator()
    all_task_ids = orch.list_tasks()
    
    # Filter by status
    if status.strip():
        filtered_ids = []
        for task_id in all_task_ids:
            try:
                record = orch.get_task(task_id)
                if record.status.value == status.strip():
                    filtered_ids.append(task_id)
            except Exception:
                pass
        all_task_ids = filtered_ids
    
    # Filter by profile
    if profile.strip():
        filtered_ids = []
        for task_id in all_task_ids:
            try:
                record = orch.get_task(task_id)
                recovery_profile = orch._resolve_recovery_profile(record)
                if recovery_profile == profile.strip().lower():
                    filtered_ids.append(task_id)
            except Exception:
                pass
        all_task_ids = filtered_ids
    
    # Apply pagination
    limit = min(limit, 1000) if limit > 0 else 100
    start = max(0, skip)
    end = start + limit
    paginated_ids = all_task_ids[start:end]
    
    result = {
        "total": len(all_task_ids),
        "skip": skip,
        "limit": limit,
        "returned": len(paginated_ids),
        "task_ids": paginated_ids,
    }
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("task-search")
def task_search(
    query: str = typer.Option("", "--query", "-q", help="Search term to match against task prompts"),
    skip: int = typer.Option(0, "--skip", help="Number of tasks to skip"),
    limit: int = typer.Option(100, "--limit", help="Maximum number of tasks to return"),
) -> None:
    """Search tasks by prompt text."""
    orch = TaskOrchestrator()
    all_task_ids = orch.list_tasks()
    matched_ids = []
    
    if query.strip():
        query_lower = query.lower()
        for task_id in all_task_ids:
            try:
                record = orch.get_task(task_id)
                if query_lower in record.prompt.lower():
                    matched_ids.append(task_id)
            except Exception:
                pass
    else:
        matched_ids = all_task_ids
    
    # Apply pagination
    limit = min(limit, 1000) if limit > 0 else 100
    start = max(0, skip)
    end = start + limit
    paginated_ids = matched_ids[start:end]
    
    result = {
        "query": query,
        "total": len(matched_ids),
        "skip": skip,
        "limit": limit,
        "returned": len(paginated_ids),
        "task_ids": paginated_ids,
    }
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("agent-run")
def agent_run(
    prompt: str,
    dry_run: bool = typer.Option(True, "--dry-run/--exec"),
    max_recovery_rounds: int = typer.Option(2, "--max-recovery-rounds"),
    include_skipped: bool = typer.Option(True, "--include-skipped/--no-include-skipped"),
    auto_fix: bool = typer.Option(True, "--auto-fix/--no-auto-fix"),
) -> None:
    orch = TaskOrchestrator()
    record = orch.create_task(prompt)
    if requires_confirmation(record.risk) and not dry_run:
        ok = typer.confirm("High-risk task. Continue execution?")
        if not ok:
            raise typer.Exit(code=1)
    result = orch.run_agent_for_task(
        task_id=record.task_id,
        dry_run=dry_run,
        max_recovery_rounds=max_recovery_rounds,
        include_skipped=include_skipped,
        auto_apply_strategy=auto_fix,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("agent-run-task")
def agent_run_task(
    task_id: str,
    dry_run: bool = typer.Option(True, "--dry-run/--exec"),
    max_recovery_rounds: int = typer.Option(2, "--max-recovery-rounds"),
    include_skipped: bool = typer.Option(True, "--include-skipped/--no-include-skipped"),
    auto_fix: bool = typer.Option(True, "--auto-fix/--no-auto-fix"),
) -> None:
    orch = TaskOrchestrator()
    record = orch.get_task(task_id)
    if requires_confirmation(record.risk) and not dry_run:
        ok = typer.confirm("High-risk task. Continue execution?")
        if not ok:
            raise typer.Exit(code=1)
    result = orch.run_agent_for_task(
        task_id=task_id,
        dry_run=dry_run,
        max_recovery_rounds=max_recovery_rounds,
        include_skipped=include_skipped,
        auto_apply_strategy=auto_fix,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("recovery-show")
def recovery_show(
    profile: str = typer.Option("", "--profile", help="Recovery strategy profile, e.g. mapping"),
) -> None:
    orch = TaskOrchestrator()
    target_profile = profile.strip() or None
    typer.echo(json.dumps(orch.get_recovery_strategy_info(target_profile), ensure_ascii=False, indent=2))


@app.command("recovery-reload")
def recovery_reload() -> None:
    orch = TaskOrchestrator()
    info = orch.reload_recovery_rules()
    typer.echo(json.dumps(info, ensure_ascii=False, indent=2))


@app.command("rag-status")
def rag_status() -> None:
    """Show LLM+RAG planner runtime status."""
    orch = TaskOrchestrator()
    typer.echo(json.dumps(orch.get_planner_rag_status(), ensure_ascii=False, indent=2))


@app.command("rag-reload")
def rag_reload() -> None:
    """Reload LLM+RAG planner config."""
    orch = TaskOrchestrator()
    typer.echo(json.dumps(orch.reload_planner_rag_config(), ensure_ascii=False, indent=2))


@app.command("rag-enable")
def rag_enable(
    enabled: bool = typer.Option(True, "--enabled/--disabled", help="Enable or disable LLM+RAG planner"),
) -> None:
    """Quickly toggle LLM+RAG planner mode in config file."""
    from pathlib import Path

    cfg_path = Path("config") / "llm_rag_config.json"
    payload = {}
    if cfg_path.exists():
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    if not isinstance(payload, dict):
        payload = {}

    payload["enabled"] = bool(enabled)
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    orch = TaskOrchestrator()
    typer.echo(json.dumps(orch.reload_planner_rag_config(), ensure_ascii=False, indent=2))


@app.command("export-stats")
def export_stats(
    format: str = typer.Option("json", "--format", help="Output format: json or csv"),
    output_file: str = typer.Option("", "--output", help="Save output to file (optional)"),
    task_ids: str = typer.Option("", "--task-ids", help="Comma-separated task IDs (exports all if not specified)"),
) -> None:
    """Export task execution statistics in JSON or CSV format."""
    orch = TaskOrchestrator()
    task_id_list = [x.strip() for x in task_ids.split(",") if x.strip()] if task_ids else None
    
    try:
        result = orch.export_tasks_stats(format=format, task_ids=task_id_list)
        
        if output_file:
            from pathlib import Path
            Path(output_file).write_text(result, encoding="utf-8")
            typer.echo(f"✓ 导出成功: {output_file}")
        else:
            typer.echo(result)
    except ValueError as e:
        typer.echo(f"❌ 错误: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("task-delete")
def task_delete(
    task_id: str,
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Delete a task record."""
    orch = TaskOrchestrator()
    
    if not force:
        ok = typer.confirm(f"确定删除任务 {task_id}?")
        if not ok:
            raise typer.Exit(code=1)
    
    result = orch.delete_task(task_id, force=force)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("task-archive")
def task_archive(task_id: str) -> None:
    """Archive a task (move to archive directory)."""
    orch = TaskOrchestrator()
    result = orch.archive_task(task_id)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("task-restore")
def task_restore(task_id: str) -> None:
    """Restore a task from archive."""
    orch = TaskOrchestrator()
    result = orch.restore_task(task_id)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("task-archive-list")
def task_archive_list() -> None:
    """List archived tasks."""
    orch = TaskOrchestrator()
    archived = orch.get_archive_list()
    typer.echo(json.dumps({"count": len(archived), "task_ids": archived}, ensure_ascii=False, indent=2))


@app.command("task-cleanup")
def task_cleanup(
    max_age_days: int = typer.Option(30, "--max-age-days", help="Delete tasks older than this many days"),
    status: str = typer.Option("COMPLETED,FAILED", "--status", help="Comma-separated status list to clean"),
    dry_run: bool = typer.Option(True, "--dry-run/--exec", help="Preview before deleting"),
) -> None:
    """Clean up old or completed tasks."""
    orch = TaskOrchestrator()
    status_list = [s.strip() for s in status.split(",") if s.strip()]
    
    result = orch.cleanup_tasks(
        max_age_days=max_age_days,
        status_filter=status_list,
        dry_run=dry_run
    )
    
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("strategy-stats")
def strategy_stats() -> None:
    """Get strategy efficiency statistics."""
    orch = TaskOrchestrator()
    stats = orch.get_strategy_statistics()
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command("strategy-report")
def strategy_report(
    format: str = typer.Option("json", "--format", help="Output format: json or html"),
    output_file: str = typer.Option("", "--output", help="Save report to file (optional)"),
) -> None:
    """Generate strategy effectiveness report."""
    orch = TaskOrchestrator()
    
    try:
        result = orch.generate_strategy_report(format=format)
        
        if output_file:
            from pathlib import Path
            Path(output_file).write_text(result, encoding="utf-8")
            typer.echo(f"✓ 报告生成成功: {output_file}")
        else:
            typer.echo(result)
    except ValueError as e:
        typer.echo(f"❌ 错误: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
