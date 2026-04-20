from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .catalog import IntentCatalog
from .document_parser import DocumentTaskParser
from .models import ExecutionResult, RiskLevel, TaskRecord, TaskStatus, utc_now_iso
from .planner_adapter import PlannerAdapter
from .requirement_mapper import RequirementMapper
from .runner.arcpy_runner import ArcPyRunner
from .safety import evaluate_risk
from .storage import TaskStore
from .workflow_assembler import WorkflowAssembler
from .recovery_state_machine import RecoveryInput, RecoveryStateMachine


class TaskOrchestrator:
    DEFAULT_RECOVERY_RULES: dict = {
        "reason_to_prep_nodes": {
            "missing_input": ["scan_inputs"],
            "merge_failed": ["scan_inputs", "merge_by_layer"],
            "merge_or_dissolve_failed": ["merge_by_layer"],
            "arcpy_unavailable": [],
            "implementation_pending": [],
            "runner_exception": ["scan_inputs"],
        },
        "implementation_fallback_nodes": {
            "export_map_layout": ["build_quality_report"],
        },
        "reason_recommendations": {
            "arcpy_unavailable": "当前环境缺少 ArcPy，请切换到 ArcGIS Pro Python 环境后重试。",
            "implementation_pending": "节点 {node} 尚未实现，建议先跳过或替换为可用节点。",
        },
        "rule_flags": {
            "replace_implementation_pending_with_fallback": True,
            "skip_implementation_pending": True,
            "force_dry_run_when_arcpy_unavailable": True,
        },
        "rule_priorities": {
            "replace_implementation_pending_with_fallback": 100,
            "skip_implementation_pending": 50,
            "force_dry_run_when_arcpy_unavailable": 100,
        },
    }

    def __init__(self, recovery_config_path: Path | None = None) -> None:
        self.catalog = IntentCatalog()
        self.planner = PlannerAdapter(self.catalog)
        self.document_parser = DocumentTaskParser()
        self.requirement_mapper = RequirementMapper()
        self.workflow_assembler = WorkflowAssembler()
        self.store = TaskStore()
        self.runner = ArcPyRunner()
        self.recovery_config_path = recovery_config_path or Path("config") / "recovery_strategies.json"
        self._recovery_config_mtime: float | None = None
        self.recovery_strategy_meta: dict = {}
        self.recovery_config_payload: dict = {}
        self.recovery_rules = self._load_recovery_rules()
        self.recovery_state_machine = RecoveryStateMachine()

    def create_task(self, prompt: str) -> TaskRecord:
        plan = self.planner.build_plan(prompt)
        matched_intent = next(i for i in self.catalog.all() if i.name == plan.intent)
        risk = evaluate_risk(prompt, RiskLevel(matched_intent.risk))

        now = utc_now_iso()
        record = TaskRecord(
            task_id=uuid4().hex[:10],
            prompt=prompt,
            status=TaskStatus.DRAFT,
            risk=risk,
            created_at=now,
            updated_at=now,
            plan=plan,
            metadata={"catalog_version": self.catalog.version},
        )
        assembled = self.workflow_assembler.assemble(plan.intent, required_capabilities=[])
        record.metadata["runner_profile"] = assembled.runner_profile
        record.metadata["workflow_nodes"] = assembled.workflow_nodes
        record.metadata["unresolved_capabilities"] = assembled.unresolved_capabilities
        self.store.save(record)
        return record

    def create_task_from_document(self, document_text: str, source_name: str = "inline") -> TaskRecord:
        parsed = self.document_parser.parse_text(document_text)
        mapped = self.requirement_mapper.map_document_requirements(
            constraints=parsed.constraints,
            scoring_items=parsed.scoring_items,
            extracted_actions=parsed.extracted_actions,
        )

        hint_suffix = "；".join(mapped.required_capabilities[:6])
        normalized_prompt = parsed.normalized_prompt
        if hint_suffix:
            normalized_prompt = f"{normalized_prompt}；能力提示:{hint_suffix}"

        record = self.create_task(normalized_prompt)
        record.metadata["document_source"] = source_name
        record.metadata["document_title"] = parsed.title
        record.metadata["document_constraints"] = parsed.constraints
        record.metadata["document_scoring_items"] = parsed.scoring_items
        record.metadata["document_extracted_actions"] = parsed.extracted_actions
        record.metadata["required_capabilities"] = mapped.required_capabilities
        record.metadata["suggested_intents"] = mapped.suggested_intents
        record.metadata["unresolved_scoring_items"] = mapped.unresolved_items
        assembled = self.workflow_assembler.assemble(
            record.plan.intent if record.plan else "data_integration",
            required_capabilities=mapped.required_capabilities,
        )
        record.metadata["runner_profile"] = assembled.runner_profile
        record.metadata["workflow_nodes"] = assembled.workflow_nodes
        record.metadata["unresolved_capabilities"] = assembled.unresolved_capabilities
        record.updated_at = utc_now_iso()
        self.store.save(record)
        return record

    def create_task_from_document_file(self, file_path: str) -> TaskRecord:
        parsed = self.document_parser.parse_file(file_path)
        return self.create_task_from_document(parsed.normalized_prompt, source_name=file_path)

    def plan_task(self, task_id: str) -> TaskRecord:
        record = self.store.load(task_id)
        record.plan = self.planner.build_plan(record.prompt)
        required_capabilities = record.metadata.get("required_capabilities", [])
        assembled = self.workflow_assembler.assemble(
            record.plan.intent,
            required_capabilities=required_capabilities,
        )
        record.metadata["runner_profile"] = assembled.runner_profile
        record.metadata["workflow_nodes"] = assembled.workflow_nodes
        record.metadata["unresolved_capabilities"] = assembled.unresolved_capabilities
        record.status = TaskStatus.READY
        record.updated_at = utc_now_iso()
        self.store.save(record)
        return record

    def run_task(
        self,
        task_id: str,
        dry_run: bool = False,
        override_nodes: list[str] | None = None,
    ) -> ExecutionResult:
        record = self.store.load(task_id)
        if override_nodes:
            record.metadata["workflow_nodes"] = override_nodes
            record.metadata["workflow_override_active"] = True
        else:
            record.metadata["workflow_override_active"] = False
        record.status = TaskStatus.RUNNING
        record.updated_at = utc_now_iso()
        self.store.save(record)

        try:
            result = self.runner.run(record, dry_run=dry_run)
        except Exception as exc:
            result = ExecutionResult(
                task_id=task_id,
                success=False,
                dry_run=dry_run,
                message="Task execution failed in runner.",
                outputs=[],
                details={
                    "error_code": "runner_exception",
                    "error_message": str(exc),
                    "workflow_nodes": record.metadata.get("workflow_nodes", []),
                },
            )

        record.metadata["last_execution"] = {
            "success": result.success,
            "dry_run": result.dry_run,
            "outputs": result.outputs,
            "details": result.details,
        }
        self._record_execution_summary(record, result, stage="run_task")

        record.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        record.updated_at = utc_now_iso()
        self.store.save(record)
        return result

    def suggest_rerun_nodes(self, task_id: str, include_skipped: bool = True) -> list[str]:
        record = self.store.load(task_id)
        last_execution = record.metadata.get("last_execution", {})
        details = last_execution.get("details", {})
        node_execution = details.get("node_execution", {})

        if not isinstance(node_execution, dict):
            return []

        retry_status = {"failed", "pending"}
        if include_skipped:
            retry_status.add("skipped")

        suggested = [
            node for node, status in node_execution.items() if str(status).lower() in retry_status
        ]
        return suggested

    def suggest_recovery_plan(self, task_id: str, include_skipped: bool = True) -> dict:
        self._ensure_recovery_rules_fresh()
        record = self.store.load(task_id)
        last_execution = record.metadata.get("last_execution", {})
        details = last_execution.get("details", {})
        node_execution = details.get("node_execution", {})
        node_reasons = details.get("node_reasons", {})

        rerun_nodes = self.suggest_rerun_nodes(task_id, include_skipped=include_skipped)
        recommendations: list[str] = []
        recovery_profile = self._resolve_recovery_profile(record)
        effective_rules = self._build_effective_recovery_rules(recovery_profile)
        reason_to_prep_nodes = effective_rules.get("reason_to_prep_nodes", {})
        reason_recommendations = effective_rules.get("reason_recommendations", {})

        prep_nodes: list[str] = []
        for node in list(rerun_nodes):
            reason = str(node_reasons.get(node, "")).lower()
            if reason in reason_to_prep_nodes:
                for pn in reason_to_prep_nodes[reason]:
                    if pn not in prep_nodes:
                        prep_nodes.append(pn)

            if reason in reason_recommendations:
                template = str(reason_recommendations[reason])
                msg = template.replace("{node}", node)
                if msg not in recommendations:
                    recommendations.append(msg)

        final_nodes: list[str] = []
        for node in prep_nodes + rerun_nodes:
            if node not in final_nodes:
                final_nodes.append(node)

        if not recommendations and final_nodes:
            recommendations.append("建议先执行准备节点，再重跑失败节点。")

        return {
            "task_id": task_id,
            "runner_profile": record.metadata.get("runner_profile", "integration"),
            "recovery_profile": recovery_profile,
            "recovery_strategy": self.get_recovery_strategy_info(recovery_profile),
            "nodes": final_nodes,
            "base_rerun_nodes": rerun_nodes,
            "recommendations": recommendations,
            "node_execution": node_execution if isinstance(node_execution, dict) else {},
            "node_reasons": node_reasons if isinstance(node_reasons, dict) else {},
        }

    def rerun_failed_nodes(
        self,
        task_id: str,
        dry_run: bool = True,
        include_skipped: bool = True,
        auto_apply_strategy: bool = True,
    ) -> ExecutionResult:
        self._ensure_recovery_rules_fresh()
        recovery = self.suggest_recovery_plan(task_id, include_skipped=include_skipped)
        recovery_profile = str(recovery.get("recovery_profile", "default"))
        effective_rules = self._build_effective_recovery_rules(recovery_profile)
        strategy = self._build_recovery_strategy(
            recovery,
            auto_apply=auto_apply_strategy,
            effective_rules=effective_rules,
        )
        nodes = strategy.get("effective_nodes", [])
        effective_dry_run = dry_run or bool(strategy.get("force_dry_run", False))
        if not nodes:
            record = self.store.load(task_id)
            message = "No failed/pending nodes to rerun."
            if strategy.get("applied"):
                message = "No eligible nodes to rerun after applying recovery strategy."
            result = ExecutionResult(
                task_id=task_id,
                success=True,
                dry_run=effective_dry_run,
                message=message,
                outputs=[],
                details={
                    "runner_profile": record.metadata.get("runner_profile", "integration"),
                    "workflow_nodes": [],
                    "node_execution": {},
                    "effective_dry_run": effective_dry_run,
                    "recovery_profile": recovery_profile,
                    "recovery_strategy": self.get_recovery_strategy_info(recovery_profile),
                    "recommendations": recovery.get("recommendations", []),
                    "recovery_plan": recovery,
                    "strategy_applied": strategy.get("applied", False),
                    "strategy": strategy,
                },
            )
            self._record_execution_summary(record, result, stage="rerun_failed_nodes")
            record.updated_at = utc_now_iso()
            self.store.save(record)
            return result
        result = self.run_task(task_id, dry_run=effective_dry_run, override_nodes=nodes)
        result.details["recommendations"] = recovery.get("recommendations", [])
        result.details["recovery_plan"] = recovery
        result.details["strategy_applied"] = strategy.get("applied", False)
        result.details["strategy"] = strategy
        result.details["effective_dry_run"] = effective_dry_run
        result.details["recovery_profile"] = recovery_profile
        result.details["recovery_strategy"] = self.get_recovery_strategy_info(recovery_profile)
        return result

    def _build_recovery_strategy(self, recovery: dict, auto_apply: bool, effective_rules: dict) -> dict:
        nodes = list(recovery.get("nodes", []))
        base_rerun_nodes = set(recovery.get("base_rerun_nodes", []))
        node_reasons = recovery.get("node_reasons", {})
        if not isinstance(node_reasons, dict):
            node_reasons = {}
        strategy = self.recovery_state_machine.run(
            RecoveryInput(
                nodes=nodes,
                base_rerun_nodes=base_rerun_nodes,
                node_reasons={k: str(v) for k, v in node_reasons.items()},
                auto_apply=auto_apply,
                effective_rules=effective_rules,
            )
        )

        recommendations = recovery.get("recommendations", [])
        if not isinstance(recommendations, list):
            recommendations = []
        skipped_nodes = strategy.get("skipped_nodes", [])
        replaced_nodes = strategy.get("replaced_nodes", [])
        if skipped_nodes:
            auto_hint = "已自动跳过未实现节点，可在节点实现后重新纳入重跑。"
            if auto_hint not in recommendations:
                recommendations.append(auto_hint)
            recovery["recommendations"] = recommendations

        if replaced_nodes:
            replacement_hint = "检测到未实现节点，已自动替换为可执行降级节点。"
            if replacement_hint not in recommendations:
                recommendations.append(replacement_hint)
            recovery["recommendations"] = recommendations

        if strategy.get("force_dry_run"):
            dry_run_hint = "检测到 ArcPy 不可用，已自动切换为 dry-run 以避免执行失败。"
            if dry_run_hint not in recommendations:
                recommendations.append(dry_run_hint)
            recovery["recommendations"] = recommendations

        return strategy

    def _load_recovery_rules(self, force: bool = False) -> dict:
        if not force and self.recovery_rules if hasattr(self, "recovery_rules") else False:
            return self.recovery_rules

        merged = dict(self.DEFAULT_RECOVERY_RULES)
        for key in (
            "reason_to_prep_nodes",
            "implementation_fallback_nodes",
            "reason_recommendations",
            "rule_flags",
            "rule_priorities",
        ):
            merged[key] = dict(self.DEFAULT_RECOVERY_RULES.get(key, {}))

        if not self.recovery_config_path.exists():
            self.recovery_config_payload = {}
            self._recovery_config_mtime = None
            self.recovery_strategy_meta = {
                "source": str(self.recovery_config_path),
                "loaded_from_file": False,
                "version": self._compute_rules_version(merged),
                "mtime": None,
            }
            return merged

        try:
            raw_text = self.recovery_config_path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except Exception:
            self.recovery_config_payload = {}
            self._recovery_config_mtime = None
            self.recovery_strategy_meta = {
                "source": str(self.recovery_config_path),
                "loaded_from_file": False,
                "version": self._compute_rules_version(merged),
                "mtime": None,
                "error": "invalid_recovery_config",
            }
            return merged

        if not isinstance(payload, dict):
            self.recovery_config_payload = {}
            self._recovery_config_mtime = None
            self.recovery_strategy_meta = {
                "source": str(self.recovery_config_path),
                "loaded_from_file": False,
                "version": self._compute_rules_version(merged),
                "mtime": None,
                "error": "invalid_recovery_config_schema",
            }
            return merged

        for key in (
            "reason_to_prep_nodes",
            "implementation_fallback_nodes",
            "reason_recommendations",
            "rule_flags",
            "rule_priorities",
        ):
            value = payload.get(key)
            if isinstance(value, dict):
                merged[key].update(value)
        self.recovery_config_payload = payload

        try:
            self._recovery_config_mtime = self.recovery_config_path.stat().st_mtime
        except Exception:
            self._recovery_config_mtime = None
        self.recovery_strategy_meta = {
            "source": str(self.recovery_config_path),
            "loaded_from_file": True,
            "version": self._compute_rules_version(merged),
            "mtime": self._recovery_config_mtime,
            "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
        }
        return merged

    def _compute_rules_version(self, rules: dict) -> str:
        normalized = json.dumps(rules, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def get_task(self, task_id: str) -> TaskRecord:
        return self.store.load(task_id)

    def list_tasks(self) -> list[str]:
        return self.store.list_task_ids()

    def get_task_summary(self, task_id: str) -> dict:
        record = self.store.load(task_id)
        history = record.metadata.get("execution_history", [])
        if not isinstance(history, list):
            history = []
        last_execution = record.metadata.get("last_execution", {})
        if not history and isinstance(last_execution, dict) and last_execution:
            history = [
                {
                    "success": bool(last_execution.get("success", False)),
                    "dry_run": bool(last_execution.get("dry_run", True)),
                }
            ]
        total_runs = len(history)
        success_runs = len([x for x in history if isinstance(x, dict) and x.get("success") is True])
        failure_runs = len([x for x in history if isinstance(x, dict) and x.get("success") is False])
        dry_runs = len([x for x in history if isinstance(x, dict) and x.get("dry_run") is True])
        exec_runs = len([x for x in history if isinstance(x, dict) and x.get("dry_run") is False])
        return {
            "task_id": task_id,
            "status": record.status.value,
            "runner_profile": record.metadata.get("runner_profile", "integration"),
            "total_runs": total_runs,
            "success_runs": success_runs,
            "failure_runs": failure_runs,
            "dry_runs": dry_runs,
            "exec_runs": exec_runs,
            "last_success": bool(last_execution.get("success", False)),
            "last_dry_run": bool(last_execution.get("dry_run", True)),
            "last_outputs_count": len(last_execution.get("outputs", []))
            if isinstance(last_execution.get("outputs", []), list)
            else 0,
            "recovery_profile": self._resolve_recovery_profile(record),
            "recovery_strategy": self.get_recovery_strategy_info(self._resolve_recovery_profile(record)),
        }

    def _record_execution_summary(self, record: TaskRecord, result: ExecutionResult, stage: str) -> None:
        history = record.metadata.get("execution_history", [])
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "at": utc_now_iso(),
                "stage": stage,
                "success": result.success,
                "dry_run": result.dry_run,
                "message": result.message,
                "workflow_nodes": result.details.get("workflow_nodes", []),
                "strategy_applied": result.details.get("strategy_applied", False),
                "error_code": result.details.get("error_code"),
            }
        )
        record.metadata["execution_history"] = history[-30:]

    def run_agent(
        self,
        prompt: str,
        dry_run: bool = True,
        max_recovery_rounds: int = 2,
        include_skipped: bool = True,
        auto_apply_strategy: bool = True,
    ) -> dict:
        record = self.create_task(prompt)
        return self.run_agent_for_task(
            task_id=record.task_id,
            dry_run=dry_run,
            max_recovery_rounds=max_recovery_rounds,
            include_skipped=include_skipped,
            auto_apply_strategy=auto_apply_strategy,
        )

    def run_agent_for_task(
        self,
        task_id: str,
        dry_run: bool = True,
        max_recovery_rounds: int = 2,
        include_skipped: bool = True,
        auto_apply_strategy: bool = True,
    ) -> dict:
        self._ensure_recovery_rules_fresh()
        trace: list[dict] = []
        task_record = self.get_task(task_id)
        recovery_profile = self._resolve_recovery_profile(task_record)

        initial = self.run_task(task_id, dry_run=dry_run)
        trace.append(
            {
                "stage": "initial_run",
                "success": initial.success,
                "message": initial.message,
                "dry_run": initial.dry_run,
                "workflow_nodes": initial.details.get("workflow_nodes", []),
                "error_code": initial.details.get("error_code"),
                "recovery_profile": recovery_profile,
                "recovery_strategy": self.get_recovery_strategy_info(recovery_profile),
            }
        )
        final_result = initial

        for idx in range(max(0, max_recovery_rounds)):
            if final_result.success:
                break

            recovery_result = self.rerun_failed_nodes(
                task_id,
                dry_run=dry_run,
                include_skipped=include_skipped,
                auto_apply_strategy=auto_apply_strategy,
            )
            final_result = recovery_result
            trace.append(
                {
                    "stage": f"recovery_round_{idx + 1}",
                    "success": recovery_result.success,
                    "message": recovery_result.message,
                    "dry_run": recovery_result.dry_run,
                    "workflow_nodes": recovery_result.details.get("workflow_nodes", []),
                    "strategy_applied": recovery_result.details.get("strategy_applied", False),
                    "strategy_rules": recovery_result.details.get("strategy", {}).get("applied_rules", []),
                    "recovery_profile": recovery_profile,
                    "recovery_strategy": self.get_recovery_strategy_info(recovery_profile),
                }
            )

            terminal_messages = {
                "No failed/pending nodes to rerun.",
                "No eligible nodes to rerun after applying recovery strategy.",
            }
            if recovery_result.message in terminal_messages:
                break

        task_record = self.get_task(task_id)
        return {
            "task_id": task_id,
            "status": task_record.status.value,
            "success": final_result.success,
            "dry_run": dry_run,
            "attempts": len(trace),
            "recovery_profile": recovery_profile,
            "recovery_strategy": self.get_recovery_strategy_info(recovery_profile),
            "trace": trace,
            "final_result": asdict(final_result),
            "task": asdict(task_record),
        }

    def get_recovery_strategy_info(self, profile: str | None = None) -> dict:
        active_profile = profile or "default"
        info = dict(self.recovery_strategy_meta)
        info["active_profile"] = active_profile
        info["available_profiles"] = sorted(self._available_recovery_profiles())
        effective_rules = self._build_effective_recovery_rules(active_profile)
        info["effective_version"] = self._compute_rules_version(effective_rules)
        return info

    def reload_recovery_rules(self) -> dict:
        self.recovery_rules = self._load_recovery_rules(force=True)
        return self.get_recovery_strategy_info()

    def get_planner_rag_status(self) -> dict:
        return self.planner.get_llm_rag_status()

    def reload_planner_rag_config(self) -> dict:
        return self.planner.reload_llm_rag_config()

    def _ensure_recovery_rules_fresh(self) -> None:
        if not self.recovery_config_path.exists():
            return
        try:
            current_mtime = self.recovery_config_path.stat().st_mtime
        except Exception:
            return
        if self._recovery_config_mtime is None or current_mtime > self._recovery_config_mtime:
            self.recovery_rules = self._load_recovery_rules(force=True)

    def _resolve_recovery_profile(self, record: TaskRecord) -> str:
        runner_profile = str(record.metadata.get("runner_profile", "")).strip().lower()
        if runner_profile and runner_profile in self._available_recovery_profiles():
            return runner_profile
        return "default"

    def _available_recovery_profiles(self) -> set[str]:
        profiles = {"default"}
        payload_profiles = self.recovery_config_payload.get("profiles", {})
        if isinstance(payload_profiles, dict):
            for key in payload_profiles:
                profiles.add(str(key))
        return profiles

    def _build_effective_recovery_rules(self, profile: str) -> dict:
        effective = {
            "reason_to_prep_nodes": dict(self.recovery_rules.get("reason_to_prep_nodes", {})),
            "implementation_fallback_nodes": dict(
                self.recovery_rules.get("implementation_fallback_nodes", {})
            ),
            "reason_recommendations": dict(self.recovery_rules.get("reason_recommendations", {})),
            "rule_flags": dict(self.recovery_rules.get("rule_flags", {})),
            "rule_priorities": dict(self.recovery_rules.get("rule_priorities", {})),
        }
        payload_profiles = self.recovery_config_payload.get("profiles", {})
        if not isinstance(payload_profiles, dict):
            return effective
        profile_payload = payload_profiles.get(profile)
        if not isinstance(profile_payload, dict):
            return effective
        for key in (
            "reason_to_prep_nodes",
            "implementation_fallback_nodes",
            "reason_recommendations",
            "rule_flags",
            "rule_priorities",
        ):
            value = profile_payload.get(key)
            if isinstance(value, dict):
                effective[key].update(value)
        return effective

    def export_tasks_stats(self, format: str = "json", task_ids: list[str] | None = None) -> str:
        """
        导出任务执行统计信息为JSON或CSV格式。
        
        Args:
            format: 输出格式，支持 'json' 或 'csv'
            task_ids: 要导出的任务ID列表，不指定则导出所有任务
            
        Returns:
            格式化的统计数据字符串
        """
        if format not in ("json", "csv"):
            raise ValueError(f"不支持的格式: {format}。支持的格式: json, csv")
        
        all_task_ids = task_ids or self.list_tasks()
        summaries = []
        
        for task_id in all_task_ids:
            try:
                summary = self.get_task_summary(task_id)
                summaries.append(summary)
            except Exception as e:
                summaries.append({
                    "task_id": task_id,
                    "error": str(e)
                })
        
        if format == "json":
            return json.dumps(summaries, ensure_ascii=False, indent=2)
        else:  # csv
            if not summaries:
                return ""
            
            import csv
            import io
            
            output = io.StringIO()
            fieldnames = [
                "task_id", "status", "runner_profile", "recovery_profile",
                "total_runs", "success_runs", "failure_runs", "dry_runs", "exec_runs",
                "last_success", "last_dry_run", "last_outputs_count"
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for summary in summaries:
                if "error" not in summary:
                    writer.writerow(summary)
                else:
                    writer.writerow({
                        "task_id": summary["task_id"],
                        "status": f"ERROR: {summary['error']}"
                    })
            
            return output.getvalue()

    def delete_task(self, task_id: str, force: bool = False) -> dict:
        """
        Delete a task record.
        
        Args:
            task_id: Task ID to delete
            force: Skip confirmation for destructive operation
            
        Returns:
            Status dict with deletion details
        """
        try:
            record = self.store.load(task_id)
        except FileNotFoundError:
            return {"success": False, "error": f"Task {task_id} not found"}
        
        task_file = self.store._task_file(task_id)
        try:
            task_file.unlink()
            return {
                "success": True,
                "task_id": task_id,
                "status": record.status.value,
                "message": f"删除任务 {task_id}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def archive_task(self, task_id: str, archive_dir: Path | None = None) -> dict:
        """
        Archive a task (move to archive directory).
        
        Args:
            task_id: Task ID to archive
            archive_dir: Directory to archive to (default: .gis-cli/archive)
            
        Returns:
            Status dict with archival details
        """
        if archive_dir is None:
            archive_dir = Path(".gis-cli") / "archive"
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            record = self.store.load(task_id)
        except FileNotFoundError:
            return {"success": False, "error": f"Task {task_id} not found"}
        
        task_file = self.store._task_file(task_id)
        archive_file = archive_dir / f"{task_id}.json"
        
        try:
            task_file.rename(archive_file)
            return {
                "success": True,
                "task_id": task_id,
                "status": record.status.value,
                "archive_path": str(archive_file),
                "message": f"存档任务 {task_id} 到 {archive_file}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup_tasks(
        self,
        max_age_days: int = 30,
        status_filter: list[str] | None = None,
        dry_run: bool = True,
    ) -> dict:
        """
        Clean up old or completed tasks.
        
        Args:
            max_age_days: Delete tasks older than this many days
            status_filter: Only clean tasks with these statuses (default: COMPLETED, FAILED)
            dry_run: If True, only report what would be deleted
            
        Returns:
            Status dict with cleanup results
        """
        from datetime import datetime, timedelta, timezone
        
        if status_filter is None:
            status_filter = ["COMPLETED", "FAILED"]
        
        status_filter = [s.upper() for s in status_filter]
        
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        all_task_ids = self.list_tasks()
        
        to_delete = []
        for task_id in all_task_ids:
            try:
                record = self.store.load(task_id)
                
                # Check status filter
                if record.status.value.upper() not in status_filter:
                    continue
                
                # Check age
                if record.created_at < cutoff_date:
                    to_delete.append({
                        "task_id": task_id,
                        "status": record.status.value,
                        "created_at": record.created_at,
                    })
            except Exception:
                pass
        
        if dry_run:
            return {
                "dry_run": True,
                "would_delete": len(to_delete),
                "tasks": to_delete,
            }
        else:
            deleted_count = 0
            deleted_ids = []
            for item in to_delete:
                result = self.delete_task(item["task_id"], force=True)
                if result.get("success"):
                    deleted_count += 1
                    deleted_ids.append(item["task_id"])
            
            return {
                "deleted_count": deleted_count,
                "deleted_ids": deleted_ids,
                "message": f"删除了 {deleted_count} 个任务"
            }

    def get_archive_list(self, archive_dir: Path | None = None) -> list[str]:
        """Get list of archived tasks."""
        if archive_dir is None:
            archive_dir = Path(".gis-cli") / "archive"
        
        if not archive_dir.exists():
            return []
        
        return [p.stem for p in archive_dir.glob("*.json")]

    def restore_task(self, task_id: str, archive_dir: Path | None = None) -> dict:
        """
        Restore a task from archive.
        
        Args:
            task_id: Task ID to restore
            archive_dir: Directory to restore from (default: .gis-cli/archive)
            
        Returns:
            Status dict with restoration details
        """
        if archive_dir is None:
            archive_dir = Path(".gis-cli") / "archive"
        
        archive_file = archive_dir / f"{task_id}.json"
        
        if not archive_file.exists():
            return {"success": False, "error": f"Archive {task_id} not found"}
        
        task_file = self.store._task_file(task_id)
        
        try:
            archive_file.rename(task_file)
            return {
                "success": True,
                "task_id": task_id,
                "message": f"恢复任务 {task_id} 从存档"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_strategy_statistics(self) -> dict:
        """
        Get statistics on recovery strategy usage and effectiveness.
        
        Returns:
            Dict with strategy statistics
        """
        all_task_ids = self.list_tasks()
        
        strategy_stats = {
            "total_tasks": len(all_task_ids),
            "tasks_with_recovery": 0,
            "recovery_success_rate": 0.0,
            "profile_distribution": {},
            "recovery_rounds_stats": {
                "total_recovery_attempts": 0,
                "successful_recovery": 0,
                "failed_recovery": 0,
            },
            "task_status_distribution": {},
        }
        
        tasks_with_recovery = 0
        total_with_recovery_success = 0
        recovery_round_count = 0
        
        for task_id in all_task_ids:
            try:
                record = self.store.load(task_id)
                history = record.metadata.get("execution_history", [])
                
                if len(history) > 1:
                    tasks_with_recovery += 1
                    recovery_round_count += len(history) - 1
                    
                    first_success = history[0].get("success", False) if history else False
                    last_success = history[-1].get("success", False) if history else False
                    
                    if last_success and not first_success:
                        total_with_recovery_success += 1
                
                # Track profile distribution
                profile = self._resolve_recovery_profile(record)
                strategy_stats["profile_distribution"][profile] = (
                    strategy_stats["profile_distribution"].get(profile, 0) + 1
                )
                
                # Track status distribution
                status = record.status.value
                strategy_stats["task_status_distribution"][status] = (
                    strategy_stats["task_status_distribution"].get(status, 0) + 1
                )
            except Exception:
                pass
        
        strategy_stats["tasks_with_recovery"] = tasks_with_recovery
        if tasks_with_recovery > 0:
            strategy_stats["recovery_success_rate"] = 100.0 * total_with_recovery_success / tasks_with_recovery
        
        strategy_stats["recovery_rounds_stats"]["total_recovery_attempts"] = recovery_round_count
        strategy_stats["recovery_rounds_stats"]["successful_recovery"] = total_with_recovery_success
        strategy_stats["recovery_rounds_stats"]["failed_recovery"] = (
            tasks_with_recovery - total_with_recovery_success
        )
        
        return strategy_stats

    def generate_strategy_report(self, format: str = "json") -> str:
        """
        Generate a strategy effectiveness report.
        
        Args:
            format: 'json' or 'html'
            
        Returns:
            Formatted report string
        """
        if format not in ("json", "html"):
            raise ValueError(f"不支持的格式: {format}。支持的格式: json, html")
        
        stats = self.get_strategy_statistics()
        
        if format == "json":
            return json.dumps(stats, ensure_ascii=False, indent=2)
        else:  # html
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GIS Task Recovery Strategy Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .stat-section {{ margin-top: 30px; padding: 10px; border: 1px solid #ddd; background: #f9f9f9; }}
        .value {{ font-weight: bold; color: #2196F3; }}
    </style>
</head>
<body>
    <h1>GIS Task Recovery Strategy Report</h1>
    <div class="stat-section">
        <h2>Summary Statistics</h2>
        <p>Total Tasks: <span class="value">{stats['total_tasks']}</span></p>
        <p>Tasks with Recovery: <span class="value">{stats['tasks_with_recovery']}</span></p>
        <p>Recovery Success Rate: <span class="value">{stats['recovery_success_rate']:.1f}%</span></p>
    </div>
    
    <div class="stat-section">
        <h2>Recovery Rounds Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Count</th>
            </tr>
            <tr>
                <td>Total Recovery Attempts</td>
                <td>{stats['recovery_rounds_stats']['total_recovery_attempts']}</td>
            </tr>
            <tr>
                <td>Successful Recovery</td>
                <td>{stats['recovery_rounds_stats']['successful_recovery']}</td>
            </tr>
            <tr>
                <td>Failed Recovery</td>
                <td>{stats['recovery_rounds_stats']['failed_recovery']}</td>
            </tr>
        </table>
    </div>
    
    <div class="stat-section">
        <h2>Profile Distribution</h2>
        <table>
            <tr>
                <th>Profile</th>
                <th>Task Count</th>
            </tr>
"""
            for profile, count in sorted(stats['profile_distribution'].items()):
                html += f"            <tr><td>{profile}</td><td>{count}</td></tr>\n"
            
            html += """        </table>
    </div>
    
    <div class="stat-section">
        <h2>Task Status Distribution</h2>
        <table>
            <tr>
                <th>Status</th>
                <th>Task Count</th>
            </tr>
"""
            for status, count in sorted(stats['task_status_distribution'].items()):
                html += f"            <tr><td>{status}</td><td>{count}</td></tr>\n"
            
            html += """        </table>
    </div>
</body>
</html>"""
            
            return html
