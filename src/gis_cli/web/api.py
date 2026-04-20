from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..orchestrator import TaskOrchestrator

app = FastAPI(title="GIS Task CLI API", version="0.1.0")
orch = TaskOrchestrator()


class CreateTaskRequest(BaseModel):
    prompt: str


class CreateDocTaskRequest(BaseModel):
    text: str
    source_name: str = "inline"


class CreateDocFileTaskRequest(BaseModel):
    file_path: str


class RunTaskRequest(BaseModel):
    dry_run: bool = True
    nodes: list[str] | None = None


class RerunFailedRequest(BaseModel):
    dry_run: bool = True
    include_skipped: bool = True
    auto_fix: bool = True


class AgentRunRequest(BaseModel):
    prompt: str
    dry_run: bool = True
    max_recovery_rounds: int = 2
    include_skipped: bool = True
    auto_fix: bool = True


class AgentRunTaskRequest(BaseModel):
    dry_run: bool = True
    max_recovery_rounds: int = 2
    include_skipped: bool = True
    auto_fix: bool = True


class CleanupRequest(BaseModel):
    max_age_days: int = 30
    status_filter: list[str] = ["COMPLETED", "FAILED"]
    dry_run: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tasks")
def create_task(req: CreateTaskRequest) -> dict:
    record = orch.create_task(req.prompt)
    return asdict(record)


@app.post("/tasks/from-document")
def create_task_from_document(req: CreateDocTaskRequest) -> dict:
    record = orch.create_task_from_document(req.text, source_name=req.source_name)
    return asdict(record)


@app.post("/tasks/from-document-file")
def create_task_from_document_file(req: CreateDocFileTaskRequest) -> dict:
    try:
        record = orch.create_task_from_document_file(req.file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(record)


@app.get("/tasks")
def list_tasks(
    skip: int = 0,
    limit: int = 100,
    status: str = "",
    profile: str = "",
) -> dict:
    """
    List tasks with optional pagination and filtering.
    
    Query parameters:
    - skip: Number of tasks to skip (default: 0)
    - limit: Maximum number of tasks to return (default: 100, max: 1000)
    - status: Filter by status (DRAFT, RUNNING, COMPLETED, FAILED, etc.)
    - profile: Filter by recovery profile (mapping, integration, quality, default)
    """
    all_task_ids = orch.list_tasks()
    
    # Filter by status if provided
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
    
    # Filter by recovery profile if provided
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
    
    return {
        "total": len(all_task_ids),
        "skip": skip,
        "limit": limit,
        "returned": len(paginated_ids),
        "task_ids": paginated_ids,
    }


@app.get("/tasks/search")
def search_tasks(
    query: str = "",
    skip: int = 0,
    limit: int = 100,
) -> dict:
    """
    Search tasks by prompt text and metadata.
    
    Query parameters:
    - query: Search term to match against task prompts
    - skip: Number of tasks to skip (default: 0)
    - limit: Maximum number of tasks to return (default: 100)
    """
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
    
    return {
        "query": query,
        "total": len(matched_ids),
        "skip": skip,
        "limit": limit,
        "returned": len(paginated_ids),
        "task_ids": paginated_ids,
    }



@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    try:
        record = orch.get_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(record)


@app.get("/tasks/{task_id}/summary")
def get_task_summary(task_id: str) -> dict:
    try:
        summary = orch.get_task_summary(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return summary


@app.post("/tasks/{task_id}/plan")
def plan_task(task_id: str) -> dict:
    try:
        record = orch.plan_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(record)


@app.post("/tasks/{task_id}/run")
def run_task(task_id: str, req: RunTaskRequest) -> dict:
    try:
        result = orch.run_task(task_id, dry_run=req.dry_run, override_nodes=req.nodes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(result)


@app.get("/tasks/{task_id}/suggest-rerun")
def suggest_rerun(task_id: str, include_skipped: bool = True) -> dict:
    try:
        nodes = orch.suggest_rerun_nodes(task_id, include_skipped=include_skipped)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"task_id": task_id, "nodes": nodes}


@app.get("/tasks/{task_id}/suggest-fix")
def suggest_fix(task_id: str, include_skipped: bool = True) -> dict:
    try:
        plan = orch.suggest_recovery_plan(task_id, include_skipped=include_skipped)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return plan


@app.post("/tasks/{task_id}/rerun-failed")
def rerun_failed(task_id: str, req: RerunFailedRequest) -> dict:
    try:
        result = orch.rerun_failed_nodes(
            task_id,
            dry_run=req.dry_run,
            include_skipped=req.include_skipped,
            auto_apply_strategy=req.auto_fix,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(result)


@app.post("/agent/run")
def agent_run(req: AgentRunRequest) -> dict:
    return orch.run_agent(
        prompt=req.prompt,
        dry_run=req.dry_run,
        max_recovery_rounds=req.max_recovery_rounds,
        include_skipped=req.include_skipped,
        auto_apply_strategy=req.auto_fix,
    )


@app.post("/agent/run/{task_id}")
def agent_run_task(task_id: str, req: AgentRunTaskRequest) -> dict:
    try:
        return orch.run_agent_for_task(
            task_id=task_id,
            dry_run=req.dry_run,
            max_recovery_rounds=req.max_recovery_rounds,
            include_skipped=req.include_skipped,
            auto_apply_strategy=req.auto_fix,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/agent/recovery-strategy")
def get_recovery_strategy(profile: str = "") -> dict:
    target_profile = profile.strip() or None
    return orch.get_recovery_strategy_info(target_profile)


@app.post("/agent/recovery-strategy/reload")
def reload_recovery_strategy() -> dict:
    return orch.reload_recovery_rules()


@app.get("/planner/rag/status")
def get_planner_rag_status() -> dict:
    """Get LLM+RAG planner runtime status."""
    return orch.get_planner_rag_status()


@app.post("/planner/rag/reload")
def reload_planner_rag() -> dict:
    """Reload LLM+RAG planner config."""
    return orch.reload_planner_rag_config()


@app.get("/tasks/stats/export")
def export_tasks_stats(
    format: str = "json",
    task_ids: str = "",
) -> dict:
    """
    Export task execution statistics in JSON or CSV format.
    
    Query parameters:
    - format: 'json' or 'csv' (default: 'json')
    - task_ids: comma-separated task IDs (exports all if not specified)
    """
    try:
        task_id_list = [x.strip() for x in task_ids.split(",") if x.strip()] if task_ids else None
        result = orch.export_tasks_stats(format=format, task_ids=task_id_list)
        
        if format == "csv":
            return {
                "format": "csv",
                "data": result,
                "count": len([line for line in result.split("\n") if line.strip() and not line.startswith("task_id")])
            }
        else:
            import json as json_lib
            data = json_lib.loads(result)
            return {
                "format": "json",
                "data": data,
                "count": len(data)
            }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/tasks/{task_id}/delete")
def delete_task(task_id: str) -> dict:
    """Delete a task record."""
    result = orch.delete_task(task_id, force=True)
    if result.get("success"):
        return result
    else:
        raise HTTPException(status_code=404, detail=result.get("error", "Unknown error")) from None


@app.post("/tasks/{task_id}/archive")
def archive_task(task_id: str) -> dict:
    """Archive a task (move to archive directory)."""
    result = orch.archive_task(task_id)
    if result.get("success"):
        return result
    else:
        raise HTTPException(status_code=404, detail=result.get("error", "Unknown error")) from None


@app.post("/tasks/{task_id}/restore")
def restore_task(task_id: str) -> dict:
    """Restore a task from archive."""
    result = orch.restore_task(task_id)
    if result.get("success"):
        return result
    else:
        raise HTTPException(status_code=404, detail=result.get("error", "Unknown error")) from None


@app.get("/tasks/archive/list")
def archive_list() -> dict:
    """List archived tasks."""
    archived = orch.get_archive_list()
    return {
        "count": len(archived),
        "task_ids": archived,
    }


@app.post("/tasks/cleanup")
def cleanup_tasks(req: CleanupRequest) -> dict:
    """Clean up old or completed tasks."""
    result = orch.cleanup_tasks(
        max_age_days=req.max_age_days,
        status_filter=req.status_filter,
        dry_run=req.dry_run,
    )
    return result


@app.get("/strategy/stats")
def get_strategy_stats() -> dict:
    """Get strategy efficiency statistics."""
    return orch.get_strategy_statistics()


@app.get("/strategy/report")
def get_strategy_report(
    format: str = "json",
) -> dict | str:
    """
    Generate strategy effectiveness report.
    
    Query parameters:
    - format: 'json' or 'html' (default: 'json')
    """
    try:
        result = orch.generate_strategy_report(format=format)
        
        if format == "html":
            # Return HTML as string wrapped in JSON
            return {"format": "html", "content": result}
        else:
            import json as json_lib
            data = json_lib.loads(result)
            return {"format": "json", "data": data}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def run() -> None:
    import uvicorn

    uvicorn.run("gis_cli.web.api:app", host="127.0.0.1", port=8010, reload=False)
