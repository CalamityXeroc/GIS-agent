# -*- coding: utf-8 -*-
"""The engine loop: bounded observe -> decide -> act with code self-repair.

The loop keeps an explicit, user-visible task list, but unlike the legacy
plan-then-execute engine the model sees every observation and can change
course. GIS work is done through ``execute_code``; failures are repaired with
the real traceback and the real data schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..trace.recorder import RunRecorder, new_run_id
from .repair import CodeRepairer
from .state import Observation, Requirement, TaskState
from .tool_codec import Action, AutoCodec, Protocol

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Runtime knobs for the agent loop."""

    max_turns: int = 60
    max_repairs: int = 4
    exec_timeout: float = 600.0
    tool_protocol: str = "auto"
    context_soft_limit_tokens: int = 120_000
    keep_recent_messages: int = 24
    max_finish_rejections: int = 3
    verbose: bool = False
    max_diagnosis_turns: int = 3
    escalate_after_failures: int = 2


_DIAGNOSIS_TOOLS = {"catalog_query", "read_document", "list_recipes"}

_ALLOWED_ROOT = {"input", "output", ".gis_agent", "config", "logs", "README.md"}


class AgentLoop:
    """Bounded agent loop with tool calling and code repair."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        llm_client: Any,
        registry: Any,
        context: Any,
        code_runner: Any = None,
        catalog: Any = None,
        verifier: Any = None,
        recipes: Any = None,
        recorder: RunRecorder | None = None,
        config: LoopConfig | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        auto_extract_requirements: bool = True,
        use_error_memory: bool = True,
        use_project_log: bool = True,
    ):
        self.workspace = Path(workspace)
        self.llm = llm_client
        self.registry = registry
        self.context = context
        self.code_runner = code_runner
        self.catalog = catalog
        self.verifier = verifier
        self.recipes = recipes
        self.recorder = recorder
        self.config = config or LoopConfig()
        self.on_event = on_event
        self.auto_extract_requirements = auto_extract_requirements
        self.use_error_memory = use_error_memory
        self.use_project_log = use_project_log
        self._hints_seen: list[str] = []
        self._design_notes: list[str] = []
        self._hygiene_warned: set[str] = set()
        self._finish_summary: str = ""
        self.codec = AutoCodec(llm_client, protocol=Protocol(self.config.tool_protocol))
        self.repairer = (
            CodeRepairer(
                llm_client=llm_client,
                code_runner=code_runner,
                max_attempts=self.config.max_repairs,
            )
            if code_runner is not None
            else None
        )
        self.state: TaskState | None = None
        self._messages: list[dict[str, Any]] = []
        self._finish_rejections = 0
        self._nudges = 0
        self._diagnosis_streak = 0
        self._consecutive_failures = 0

    # ------------------------------------------------------------------ public
    def run(
        self,
        goal: str,
        *,
        requirements: list[Requirement | dict[str, Any]] | None = None,
        doc_paths: list[str] | None = None,
    ) -> TaskState:
        """Run one task to completion (or to an ``ask_user`` pause)."""
        run_id = new_run_id()
        self.recorder = self.recorder or RunRecorder(run_id, self.workspace / ".gis_agent" / "traces")
        self.state = TaskState(run_id=run_id, goal=goal, workspace=str(self.workspace))
        self.context.state = self.state
        for item in requirements or []:
            if isinstance(item, Requirement):
                self.state.requirements.append(item)
            elif isinstance(item, dict):
                self.state.requirements.append(Requirement.from_dict(item))

        # Auto-extract requirements when none were provided (doc -> checklist).
        if not self.state.requirements and self.auto_extract_requirements:
            extracted = self._extract_requirements(goal, doc_paths)
            if extracted:
                self.state.requirements.extend(extracted)
                self.recorder.record(
                    "requirements_extracted",
                    {"count": len(extracted), "items": [r.to_dict() for r in extracted]},
                )
                self._emit(
                    "requirements_extracted",
                    {"count": len(extracted), "requirements": self.state.requirements_digest()},
                )

        self._emit("run_start", {"goal": goal, "run_id": run_id})
        self.recorder.record("run_start", {"goal": goal, "workspace": str(self.workspace)})

        if self.code_runner is not None:
            try:
                self.code_runner.reset()
            except Exception as exc:
                logger.warning("code runner reset failed: %s", exc)

        self._messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._initial_user_message(goal, doc_paths)},
        ]

        while self.state.turn < self.config.max_turns and self.state.status == "running":
            self.state.turn += 1
            self.recorder.record("turn", {"turn": self.state.turn})
            self._emit("turn", {"turn": self.state.turn, "tasklist": self.state.tasklist_digest()})

            self._compact_if_needed()
            task_type = (
                "agent_strong"
                if self._consecutive_failures >= max(1, self.config.escalate_after_failures)
                else "agent"
            )
            try:
                response, actions = self.codec.call(
                    self._messages,
                    self.registry.schemas(),
                    task_type=task_type,
                    temperature=0.1,
                )
            except Exception as exc:
                # Gateway/network outage: stop gracefully instead of crashing.
                self.state.status = "llm_unavailable"
                self.state.error = f"LLM 不可用: {exc}"
                self.recorder.record("llm_error", {"error": str(exc)[:500]})
                self._emit("llm_error", {"error": str(exc)[:300]})
                break
            self.recorder.record(
                "llm_result",
                {
                    "model": response.model,
                    "task_type": task_type,
                    "usage": response.usage,
                    "tool_calls": [a.to_dict() for a in actions],
                    "content": response.content[:2000],
                },
            )
            self._emit("llm_result", {"model": response.model, "actions": [a.to_dict() for a in actions]})

            if not actions:
                self._append_assistant(response.content, [])
                self._append_user("请继续：使用工具执行下一步，或调用 finish 交付。")
                continue

            self._append_assistant(response.content, response.tool_calls, actions)

            stop = False
            turn_observations: list[Observation] = []
            for action in actions:
                if action.parse_error:
                    observation = Observation(
                        ok=False,
                        summary=f"工具参数解析失败: {action.parse_error}",
                        error={"type": "ArgumentParseError", "message": action.parse_error},
                        hint="请重新输出合法的 JSON 参数。",
                    )
                    turn_observations.append(observation)
                    self._append_observation(observation, action)
                    continue
                observation, stop = self._handle_action(action)
                turn_observations.append(observation)
                self._append_observation(observation, action)
                if stop:
                    break

            # Speed guardrails: exploration budget + failure escalation.
            self._update_guards(actions, turn_observations)
            if stop:
                break

        if self.state.status == "running":
            self.state.status = "max_turns" if self.state.turn >= self.config.max_turns else "stopped"
            self.state.error = "达到最大轮次仍未完成" if self.state.status == "max_turns" else ""
        self._finalize()
        return self.state

    def resume(self, user_message: str) -> TaskState:
        """Resume after an ``ask_user`` pause."""
        if self.state is None:
            raise RuntimeError("No task to resume")
        self.state.status = "running"
        self._append_user(user_message)
        return self._continue()

    # ------------------------------------------------------- speed guardrails
    def _update_guards(self, actions: list[Action], observations: list[Observation]) -> None:
        """Exploration budget nudge + failure escalation counters."""
        if not actions:
            return
        if all(a.kind == "tool" and a.tool in _DIAGNOSIS_TOOLS for a in actions):
            self._diagnosis_streak += 1
            if self._diagnosis_streak > self.config.max_diagnosis_turns:
                self._append_user(
                    f"[系统提示] 数据侦查已连续 {self._diagnosis_streak} 轮，超过预算 "
                    f"{self.config.max_diagnosis_turns} 轮。请基于已有信息立即开始执行"
                    "（update_tasklist / execute_code），不要再做只读侦查。"
                )
        else:
            self._diagnosis_streak = 0

        if observations and any(o.ok for o in observations):
            self._consecutive_failures = 0
        elif observations:
            self._consecutive_failures += 1

    # ------------------------------------------------------------------ actions
    def _handle_action(self, action: Action) -> tuple[Observation, bool]:
        """Execute one action; returns (observation, should_stop)."""
        self.recorder.record("tool_call", {"tool": action.tool, "args": self._safe_args(action.args)})
        self._emit("tool_call", {"tool": action.tool, "args": self._safe_args(action.args)})

        if action.tool == "finish":
            return self._handle_finish(action)
        if action.tool == "ask_user":
            observation = self.registry.execute("ask_user", action.args, self.context)
            self.state.status = "awaiting_user"
            self.recorder.record("tool_result", {"tool": action.tool, "observation": observation.to_dict()})
            return observation, True

        if action.tool == "execute_code":
            observation = self._run_code_with_repair(action)
        else:
            observation = self.registry.execute(action.tool, action.args, self.context)
            observation = self._enrich_failure(observation)

        if observation.artifacts:
            for path in observation.artifacts:
                self.state.add_artifact(path)
        self._check_workspace_hygiene(observation)
        self._collect_design_note(observation)
        self.recorder.record(
            "tool_result",
            {
                "tool": action.tool,
                "ok": observation.ok,
                "summary": observation.summary,
                "artifacts": observation.artifacts,
            },
        )
        self._emit("tool_result", {"tool": action.tool, "ok": observation.ok, "summary": observation.summary})
        return observation, False

    def _check_workspace_hygiene(self, observation: Observation) -> None:
        """工作区根目录出现意外项时提醒（Windows 反斜杠转义常造出假目录）。"""
        try:
            strays = sorted(
                p.name
                for p in self.workspace.iterdir()
                if p.name not in _ALLOWED_ROOT and not p.name.startswith(".")
            )
        except Exception:
            return
        fresh = [name for name in strays if name not in self._hygiene_warned]
        if not fresh:
            return
        self._hygiene_warned.update(fresh)
        note = (
            "工作区根目录出现预期外的项："
            + ", ".join(fresh[:5])
            + "。产出必须写在 output/ 下；这通常是 Windows 路径反斜杠转义错误"
            "（如 `\"...\\data\\...\"`）导致，请核对路径（用原始字符串或正斜杠）并清理误建目录。"
        )
        observation.hint = (observation.hint + "\n" + note) if observation.hint else note
        self.recorder and self.recorder.record("workspace_hygiene", {"strays": fresh[:8]})

    def _collect_design_note(self, observation: Observation) -> None:
        """把制图配方回传的"设计说明"记下来，写入项目日志（下次同主题出图可沿用）。"""
        data = observation.data if isinstance(observation.data, dict) else {}
        candidates = [data, data.get("result"), data.get("design"), data.get("spec")]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            note = candidate.get("design_note") or candidate.get("summary_note")
            if note and str(note) not in self._design_notes:
                self._design_notes.append(str(note)[:400])
                self.recorder and self.recorder.record("map_design", {"design_note": str(note)[:400]})
                return

    def _run_code_with_repair(self, action: Action) -> Observation:
        observation = self.registry.execute("execute_code", action.args, self.context)
        observation = self._enrich_failure(observation)
        if observation.ok or self.repairer is None:
            return observation

        catalog_digest = self._focused_catalog_digest(str(action.args.get("code", "")))
        recipes_digest = self._recipes_digest(str(action.args.get("description", "")))

        def _on_attempt(record: Any) -> None:
            self.recorder.record("repair_attempt", record.to_dict())
            self._emit("repair_attempt", record.to_dict())

        repaired, fixed_code, attempts = self.repairer.run(
            intent=str(action.args.get("description", "") or self.state.goal),
            code=str(action.args.get("code", "") or ""),
            observation=observation,
            catalog_digest=catalog_digest,
            recipes_digest=recipes_digest,
            timeout=float(action.args.get("timeout_seconds", self.config.exec_timeout) or self.config.exec_timeout),
            workspace=str(self.workspace),
            on_attempt=_on_attempt,
        )
        if repaired.ok:
            for path in repaired.artifacts:
                self.state.add_artifact(path)
            return Observation(
                ok=True,
                summary=f"{repaired.summary}（自动修复 {len(attempts)} 次）",
                data=repaired.data,
                artifacts=repaired.artifacts,
                images=repaired.images,
            )
        return Observation(
            ok=False,
            summary=f"{repaired.summary}；自动修复 {len(attempts)} 次仍未成功",
            data=repaired.data,
            error=repaired.error,
            hint=repaired.hint,
        )

    def _handle_finish(self, action: Action) -> tuple[Observation, bool]:
        observation = self.registry.execute("finish", action.args, self.context)
        self._finish_summary = str(observation.summary or "")
        verdict = self.verifier.verify(self.state) if self.verifier is not None else {"pass": True, "summary": "未配置验证器"}
        self.recorder.record("verification", verdict)
        self._emit("verification", verdict)

        if verdict.get("pass"):
            self.state.status = "completed"
            return (
                Observation(
                    ok=True,
                    summary=f"验收通过。{observation.summary}",
                    data=verdict,
                ),
                True,
            )

        self._finish_rejections += 1
        if self._finish_rejections > self.config.max_finish_rejections:
            self.state.status = "failed"
            self.state.error = "多次验收未通过：" + str(verdict.get("summary"))
            return (Observation(ok=False, summary=f"验收多次未通过，终止：{verdict.get('summary')}", data=verdict), True)

        feedback = Observation(
            ok=False,
            summary=f"验收未通过（第 {self._finish_rejections} 次）：{verdict.get('summary')}",
            data=verdict,
            hint=str(verdict.get("repair_hint", "") or ""),
        )
        return feedback, False

    # ------------------------------------------------------------ requirements
    def _extract_requirements(self, goal: str, doc_paths: list[str] | None) -> list[Requirement]:
        """Best-effort requirement extraction from the goal and documents."""
        try:
            from ..requirements.extractor import RequirementExtractor

            extractor = RequirementExtractor(self.llm)
            return extractor.extract(
                goal,
                doc_paths=doc_paths,
                catalog_digest=self._catalog_digest(),
            )
        except Exception as exc:
            logger.warning("requirement extraction error: %s", exc)
            return []

    # ------------------------------------------------------------- messages
    # ------------------------------------------------------------ error memory
    def _enrich_failure(self, observation: Observation) -> Observation:
        """Append distilled fix suggestions to a failing observation's hint."""
        if observation.ok or not self.use_error_memory:
            return observation
        try:
            from ..runtime.error_memory import enrich_hint

            error = observation.error or {}
            text = " ".join(
                [
                    str(observation.summary or ""),
                    str(error.get("type", "")),
                    str(error.get("message", "")),
                    str(error.get("traceback", ""))[-2000:],
                    str(error.get("stderr", ""))[-1500:],
                ]
            )
            enriched = enrich_hint(observation.hint or "", text)
            if enriched and enriched != (observation.hint or ""):
                observation.hint = enriched
                for line in enriched.splitlines():
                    if line.startswith("[错误记忆/") and line not in self._hints_seen:
                        self._hints_seen.append(line)
                self.recorder and self.recorder.record(
                    "error_memory", {"hint": observation.hint[:600]}
                )
        except Exception as exc:  # pragma: no cover - 记忆模块不能影响主流程
            logger.warning("error memory failed: %s", exc)
        return observation

    def _system_prompt(self) -> str:
        output_dir = self.workspace / "output"
        return (
            "你是一个资深 GIS 智能体，目标是把用户的自然语言需求转化为可复现的 GIS 方法论并完成实际操作。\n\n"
            "## 工作环境\n"
            f"- 工作区: {self.workspace}\n"
            f"- 输入数据目录: {self.workspace / 'input'}\n"
            f"- 输出目录: {output_dir}（所有产出必须写到这里或其后代路径）\n"
            "- 执行环境: ArcGIS Pro 3.6（arcpy 已在持久内核中就绪）\n\n"
            "## 工作原则\n"
            f"0. **输出位置（硬性）**：所有产出（.gdb/.shp/.tif/地图/表）必须写入 {output_dir} 下，"
            "例如 CreateFileGDB 的第一参数用输出目录，不要直接写到工作区根目录。\n"
            "1. **数据先行**：写任何代码前先用 catalog_query 确认真实字段名、几何类型、坐标系；禁止臆造字段名。\n"
            "2. **可见计划**：开始任务先用 update_tasklist 建立 3-8 条任务清单，之后每完成一步更新状态。\n"
            "3. **代码即行动**：具体 GIS 操作通过 execute_code 完成。代码失败时系统会自动修复；你应关注错误是否属于需求理解偏差。\n"
            "4. **投影正确**：距离/面积/密度分析必须先投影到合适的投影坐标系（全国用 Albers，局部用高斯-克吕格/UTM）；仅展示可用 Web Mercator。\n"
            "5. **制图完整**：专题图必须包含图名、图例、比例尺、指北针，并确保地图框已绑定数据图层。\n"
            "6. **自检交付**：交付前调用 verify_outputs；确认产物满足需求后再调用 finish。\n"
            "7. **少问多做**：能从数据目录/文档推断的信息不要问用户；只在关键信息确实缺失时用 ask_user。\n"
            "8. **中文输出**：与用户交流、任务清单、方法论说明全部用中文。\n\n"
            "## 方法论纪律（常驻，违反会导致返工）\n"
            "- **先看后算**：任何计算前先确认真实字段名/几何类型/坐标系/要素数；名字一律从 catalog_query 或"
            "`arcpy.ListFields` 拿，禁止凭常识猜（中文名会被 DBF 截断）。\n"
            "- **坐标系最贵**：距离/面积/密度/缓冲必须先投影到投影坐标系并使用同一坐标系；写代码前打印"
            "`Describe(路径).spatialReference.factoryCode` 核对，错一次全盘重做。\n"
            "- **缺失值不编造**：空值/null/NoData 必须如实报告或按方法说明处理（补 0 要写清楚），"
            "不得静默当成 0、也不得直接丟弃；空值数量写进结论。\n"
            "- **单位与量纲**：面积 m²/km²、人口 人/万人、比例 0-1/0-100 要统一并写明；换算系数写进说明。\n"
            "- **数字要守恒**：交付前用独立途径核对总量（如分摊前后合计、分区合计 vs 全体合计），"
            "差异超 1% 必须查清或如实说明。\n"
            "- **方法可复现**：结论里写清公式与口径（数据源、筛选条件、统计范围），让第三者能重跑。\n"
            "- **坑要记下**：踩到的工具/API/数据坑写进 finish 说明，系统会记入项目日志供后续任务使用。\n\n"
            f"## 数据目录摘要\n{self._catalog_digest()}\n\n"
            f"## 需求清单\n{self.state.requirements_digest() or '（由你从用户需求中提炼，并用 update_tasklist 呈现）'}\n\n"
            f"## 操作目录（可用配方，共 {len(self.recipes.all()) if self.recipes is not None else 0} 个；"
            "优先用配方，配方不适用再用 execute_code）\n"
            f"{self._recipe_catalog()}"
        )

    def _initial_user_message(self, goal: str, doc_paths: list[str] | None) -> str:
        text = ""
        if self.use_project_log:
            try:
                from .project_log import digest as _log_digest

                recent = _log_digest(self.workspace)
                if recent:
                    text += recent + "\n\n"
            except Exception as exc:  # pragma: no cover
                logger.warning("project log digest failed: %s", exc)
        text += f"任务目标：\n{goal}\n"
        if doc_paths:
            text += "\n相关文档（可先用 read_document 读取）：\n"
            text += "\n".join(f"- {p}" for p in doc_paths)
        text += "\n请开始：先建立任务清单，再逐步执行。"
        return text

    def _recipe_catalog(self) -> str:
        """紧凑操作目录（不再反复 list_recipes 就能看到全部可用操作）。"""
        if self.recipes is None:
            return "（未配置配方库）"
        try:
            if hasattr(self.recipes, "catalog_digest"):
                return self.recipes.catalog_digest()
            return self._recipes_digest("")
        except Exception as exc:
            logger.warning("recipe catalog failed: %s", exc)
            return self._recipes_digest("")

    def _append_assistant(
        self,
        content: str,
        tool_calls: list[Any] | None = None,
        actions: list[Action] | None = None,
    ) -> None:
        protocol = self.codec.protocol_for(self.llm.config.model)
        if protocol == Protocol.NATIVE and tool_calls:
            message: dict[str, Any] = {"role": "assistant", "content": content or ""}
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                }
                for call in tool_calls
            ]
            self._messages.append(message)
        else:
            self._messages.append({"role": "assistant", "content": content or ""})

    def _append_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def _append_observation(self, observation: Observation, action: Action) -> None:
        protocol = self.codec.protocol_for(self.llm.config.model)
        payload = observation.to_message()
        if protocol == Protocol.NATIVE and action.tool_call_id:
            self._messages.append(
                {"role": "tool", "tool_call_id": action.tool_call_id, "content": payload}
            )
        else:
            self._messages.append(
                {"role": "user", "content": f"工具 {action.tool} 的返回：\n{payload}"}
            )

    # ------------------------------------------------------------ compaction
    def _estimate_tokens(self) -> int:
        total = 0
        for message in self._messages:
            content = message.get("content") or ""
            total += len(str(content)) // 2
        return total

    def _compact_if_needed(self) -> None:
        if self._estimate_tokens() < self.config.context_soft_limit_tokens:
            return
        keep = self.config.keep_recent_messages
        if len(self._messages) <= keep + 2:
            return
        head = self._messages[:2]
        middle = self._messages[2:-keep]
        tail = self._messages[-keep:]
        summary = self._summarize(middle)
        self._messages = head + [
            {"role": "user", "content": f"[进展摘要（早期轮次已压缩）]\n{summary}"}
        ] + tail
        self.recorder.record("context_compacted", {"summarized_messages": len(middle)})

    def _summarize(self, messages: list[dict[str, Any]]) -> str:
        joined = []
        for message in messages[-40:]:
            role = message.get("role")
            content = str(message.get("content") or "")[:1500]
            joined.append(f"{role}: {content}")
        prompt = (
            "请把下面的 GIS 任务执行记录压缩成事实摘要，保留：数据路径、字段名、坐标系、"
            "关键决策与参数、已产出的文件、失败原因与教训。不要省略具体名称与路径。\n\n"
            + "\n".join(joined)[:30000]
        )
        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}], task_type="compact", max_tokens=3000
            )
            return response.content.strip() or "（摘要为空）"
        except Exception as exc:
            logger.warning("compaction failed: %s", exc)
            return "（摘要生成失败，保留最近消息）"

    # ---------------------------------------------------------------- helpers
    def _catalog_digest(self) -> str:
        if self.catalog is None:
            return "（未配置数据目录）"
        try:
            from ..datacatalog.summary import build_digest

            return build_digest(self.catalog, max_datasets=30, max_fields=24, with_samples=False)
        except Exception as exc:
            logger.warning("catalog digest failed: %s", exc)
            return "（数据目录读取失败）"

    def _focused_catalog_digest(self, code: str) -> str:
        """Catalog digest limited to datasets referenced by the failing code."""
        if self.catalog is None:
            return ""
        try:
            import re

            from ..datacatalog.summary import build_focused_digest

            tokens = set(re.findall(r"[A-Za-z_][\w.\-]{2,}", code))
            tokens |= set(re.findall(r"[\u4e00-\u9fff]{2,}", code))
            digest = build_focused_digest(
                self.catalog, sorted(tokens), max_datasets=6, max_fields=12
            )
            if digest:
                return digest
        except Exception as exc:
            logger.warning("focused digest failed: %s", exc)
        return self._catalog_digest()[:3500]

    def _recipes_digest(self, query: str) -> str:
        if self.recipes is None:
            return "（未配置配方库）"
        try:
            items = self.recipes.search(query)
            if not items:
                return "（暂无匹配配方）"
            lines = []
            for item in items[:8]:
                params = ", ".join(item.get("params", {}).keys()) if isinstance(item.get("params"), dict) else ""
                lines.append(f"- {item.get('id')}: {item.get('name')} — {item.get('description', '')[:120]}（参数: {params}）")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("recipes digest failed: %s", exc)
            return "（配方库读取失败）"

    def _safe_args(self, args: dict[str, Any]) -> dict[str, Any]:
        safe = {}
        for key, value in (args or {}).items():
            if key == "code" and isinstance(value, str):
                safe["code"] = value[:400] + ("..." if len(value) > 400 else "")
            else:
                safe[key] = value
        return safe

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(event, payload)
            except Exception:
                pass

    def _finalize(self) -> None:
        if self.state is None:
            return
        try:
            self.state.save(str(self.workspace / ".gis_agent" / "runs" / f"{self.state.run_id}.json"))
        except Exception as exc:
            logger.warning("failed to save run state: %s", exc)
        if self.recorder is not None:
            self.recorder.finish(
                self.state.status,
                {
                    "goal": self.state.goal,
                    "artifacts": self.state.artifacts,
                    "turns": self.state.turn,
                },
            )
        if self.use_project_log:
            self._write_project_log()
        self._emit("run_end", self.state.to_dict())

    def _write_project_log(self) -> None:
        """把本次运行的事实结论追加到项目日志（确定性，不调用模型）。"""
        try:
            from .project_log import record_run

            record_run(
                self.workspace,
                goal=self.state.goal,
                status=self.state.status,
                turns=self.state.turn,
                artifacts=self.state.artifacts,
                summary=self._finish_summary or self.state.error or "",
                hints=self._hints_seen,
                design_notes=self._design_notes,
                run_id=self.state.run_id,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("project log write failed: %s", exc)

    def _continue(self) -> TaskState:
        """Continue an interrupted run with existing messages."""
        if self.state is None:
            raise RuntimeError("No task to continue")
        while self.state.turn < self.config.max_turns and self.state.status == "running":
            self.state.turn += 1
            if self.recorder is not None:
                self.recorder.record("turn", {"turn": self.state.turn, "resumed": True})
            self._compact_if_needed()
            try:
                response, actions = self.codec.call(
                    self._messages,
                    self.registry.schemas(),
                    task_type="agent",
                    temperature=0.1,
                )
            except Exception as exc:
                self.state.status = "llm_unavailable"
                self.state.error = f"LLM 不可用: {exc}"
                if self.recorder is not None:
                    self.recorder.record("llm_error", {"error": str(exc)[:500]})
                break
            self._append_assistant(response.content, response.tool_calls, actions)
            stop = False
            for action in actions:
                observation, stop = self._handle_action(action)
                self._append_observation(observation, action)
                if stop:
                    break
            if stop:
                break
        if self.state.status == "running":
            self.state.status = "max_turns" if self.state.turn >= self.config.max_turns else "stopped"
        self._finalize()
        return self.state
