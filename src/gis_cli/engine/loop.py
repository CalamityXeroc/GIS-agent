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
import time
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
    #: 图片块按固定成本估算 token（不能按 base64 字符串长度算）
    image_token_cost: int = 1_500
    #: 压缩滞回：超过软上限后要多出一截才压（避免在阈值上反复抖动）
    compact_hysteresis: float = 1.2
    #: 待压缩的消息至少这么多条才值一次 LLM 往返
    compact_min_messages: int = 8
    max_finish_rejections: int = 3
    verbose: bool = False
    max_diagnosis_turns: int = 3
    escalate_after_failures: int = 2
    #: 一轮里最多并发几个**只读**工具（catalog_query / read_file …）。
    #: 1 = 完全串行（旧行为）。只并连续的只读动作，写操作与 execute_code 仍串行。
    max_parallel_readonly: int = 3
    # 整跑截止时间：网关半死不活时，单轮 LLM 调用可能反复重试几十分钟，
    # 若无总截止，一次 run 会整夜空转（实测 11.7h/22 轮）。到时后置 deadline_exceeded。
    run_deadline_seconds: float = 7200.0


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
        catalog_prewarmer: Any = None,
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
        #: 数据目录后台预热器（启动不扫时用；``start_prewarm`` 与首轮 LLM 重叠）
        self.catalog_prewarmer = catalog_prewarmer
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
        self._diagnosis_streak = 0
        self._consecutive_failures = 0
        #: ``_run_actions`` 中“本动作导致应停止”的标记（ask_user / finish）
        self._stopped_on: Action | None = None

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
        self._deadline = None
        self.context.state = self.state
        for item in requirements or []:
            if isinstance(item, Requirement):
                self.state.requirements.append(item)
            elif isinstance(item, dict):
                self.state.requirements.append(Requirement.from_dict(item))

        # Auto-extract requirements when none were provided (doc -> checklist).
        if not self.state.requirements and self.auto_extract_requirements:
            # 先给可见反馈再调模型：这一步是一次完整的 LLM 往返（数秒到二十几秒），
            # 静默时用户会以为提交没成功而重复输入（实测反馈）。
            self._emit("note", {"text": "正在分析需求…"})
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

        # 数据目录后台扫描：与首轮模型调用重叠（启动时不扫，界面才能立刻可用）。
        # 先启动再拼 system prompt：摘要里能如实写上“正在扫描”，
        # 真正的数据由 catalog_query 工具在需要时等扫描完再给。
        self._start_catalog_prewarm()
        self._messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._initial_user_message(goal, doc_paths)},
        ]
        self._deadline = (
            time.monotonic() + self.config.run_deadline_seconds
            if self.config.run_deadline_seconds and self.config.run_deadline_seconds > 0
            else None
        )

        while self.state.turn < self.config.max_turns and self.state.status == "running":
            if self._deadline is not None and time.monotonic() > self._deadline:
                self.state.status = "deadline_exceeded"
                self.state.error = (
                    f"整跑截止时间已到（run_deadline_seconds={self.config.run_deadline_seconds:g}s，"
                    f"已完成 {self.state.turn} 轮）。通常是模型网关长时间不可用导致单轮反复重试；"
                    "可排查网关后用 resume 从 trace 续跑。"
                )
                self.recorder.record("deadline_exceeded", {"turns": self.state.turn, "error": self.state.error})
                self._emit("deadline_exceeded", {"turns": self.state.turn, "error": self.state.error})
                break
            self.state.turn += 1
            self.recorder.record("turn", {"turn": self.state.turn})
            self._emit("turn", {"turn": self.state.turn, "tasklist": self.state.tasklist_digest()})

            self._compact_if_needed()
            task_type = (
                "agent_strong"
                if self._consecutive_failures >= max(1, self.config.escalate_after_failures)
                else "agent"
            )
            self._sanitize_messages()
            try:
                response, actions = self.codec.call(
                    self._messages,
                    self.registry.schemas(),
                    task_type=task_type,
                    temperature=0.1,
                    on_delta=self._on_llm_delta,
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
            self._emit(
                "llm_result",
                {
                    "model": response.model,
                    "actions": [a.to_dict() for a in actions],
                    # UI 状态行要显示 token 用量：这里必须带上（trace 里早就有，事件里之前漏了）
                    "usage": getattr(response, "usage", None) or {},
                },
            )

            if not actions:
                self._append_assistant(response.content, [])
                self._append_user("请继续：使用工具执行下一步，或调用 finish 交付。")
                continue

            self._append_assistant(response.content, response.tool_calls, actions)

            stop = False
            turn_observations: list[Observation] = []
            for action, observation in self._run_actions(actions):
                if observation is not None:
                    turn_observations.append(observation)
                    self._append_observation(observation, action)
                if self._stopped_on is action:
                    stop = True
                    break

            self._append_turn_images(turn_observations)

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
    def _run_actions(
        self, actions: list[Action]
    ) -> list[tuple[Action, Observation | None]]:
        """执行本轮所有动作，**按原顺序**返回 ``(action, observation)``。

        并行的理由：模型一轮里经常同时发多个只读查询（如一次查三个数据集），
        串行时每个都要等一次完整的“读文件/查目录”。多个只读工具之间没有
        依赖，并发能省下真实等待。

        只并**连续的**一段只读动作（不跨越写操作），因为写操作可能依赖前面
        只读动作的结果，且顺序不能动。

        顺序保证：结果按 ``actions`` 的下标回填，而 ``_append_observation`` 在
        调用方按同一顺序做 —— 否则 tool_call_id 与结果会错配，模型会看到
        “参数是 A 的结果却是 B”。
        """
        results: list[tuple[Action, Observation | None]] = []
        self._stopped_on: Action | None = None
        index = 0
        total = len(actions)
        while index < total:
            action = actions[index]
            # 参数解析失败：不执行，直接给一条失败观察
            if action.parse_error:
                results.append(
                    (
                        action,
                        Observation(
                            ok=False,
                            summary=f"工具参数解析失败: {action.parse_error}",
                            error={"type": "ArgumentParseError", "message": action.parse_error},
                            hint="请重新输出合法的 JSON 参数。",
                        ),
                    )
                )
                index += 1
                continue

            # 收集一段连续的只读动作
            batch: list[Action] = []
            while (
                index + len(batch) < total
                and not actions[index + len(batch)].parse_error
                and self._is_parallelizable(actions[index + len(batch)])
            ):
                batch.append(actions[index + len(batch)])

            if len(batch) > 1:
                results.extend(self._run_readonly_batch(batch))
                index += len(batch)
                continue

            # 单个动作（或写操作）：串行
            observation, should_stop = self._handle_action(action)
            results.append((action, observation))
            index += 1
            if should_stop:
                self._stopped_on = action
                break
        return results

    def _is_parallelizable(self, action: Action) -> bool:
        """这个动作能否与其他只读动作并发。"""
        if action.kind != "tool":
            return False
        checker = getattr(self.registry, "is_readonly", None)
        return bool(checker and checker(action.tool))

    def _run_readonly_batch(
        self, batch: list[Action]
    ) -> list[tuple[Action, Observation | None]]:
        """并发执行一组只读动作，结果按原顺序返回。"""
        if len(batch) <= 1:
            observation, _ = self._handle_action(batch[0])
            return [(batch[0], observation)]

        ordered: list[Observation | None] = [None] * len(batch)
        workers = max(1, min(int(self.config.max_parallel_readonly), len(batch)))

        def _work(position: int, action: Action) -> None:
            try:
                observation, _ = self._handle_action(action)
            except Exception as exc:  # pragma: no cover - 防御
                logger.exception("parallel tool %s crashed", action.tool)
                observation = Observation(
                    ok=False,
                    summary=f"工具 {action.tool} 内部异常: {exc}",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            ordered[position] = observation

        if workers == 1:
            for position, action in enumerate(batch):
                _work(position, action)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gis-agent-ro") as pool:
                for future in [pool.submit(_work, i, a) for i, a in enumerate(batch)]:
                    future.result()
        return [(action, ordered[i]) for i, action in enumerate(batch)]

    def _handle_action(self, action: Action) -> tuple[Observation, bool]:
        """Execute one action; returns (observation, should_stop)."""
        # recorder 是可选的（未传时 `run()` 才惰性建；单测也可能不建）。
        # 这里曾直接调 self.recorder.record(...) —— 没有 recorder 时直接崩。
        if self.recorder is not None:
            self.recorder.record("tool_call", {"tool": action.tool, "args": self._safe_args(action.args)})
        self._emit("tool_call", {"tool": action.tool, "args": self._safe_args(action.args)})

        if action.tool == "finish":
            return self._handle_finish(action)
        if action.tool == "ask_user":
            observation = self.registry.execute("ask_user", action.args, self.context)
            self.state.status = "awaiting_user"
            if self.recorder is not None:
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
        if self.recorder is not None:
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
        # finish 是提前返回的：这里显式补一条 tool_result 事件，界面才能把答复
        # 渲染成独立的 "● Agent" 块（否则答复只存在于 final_summary 里，用户看不到）。
        self.recorder.record(
            "tool_result",
            {"tool": "finish", "ok": observation.ok, "summary": observation.summary, "artifacts": observation.artifacts},
        )
        self._emit("tool_result", {"tool": "finish", "ok": observation.ok, "summary": observation.summary})
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
        """Append distilled fix suggestions to a failing observation's hint.

        也处理**成功但带 stderr 告警**的观察：matplotlib 缺字形（图里中文变方框）
        只写 stderr，若只在失败时匹配，这类问题永远看不见。
        """
        if not self.use_error_memory:
            return observation
        data = observation.data if isinstance(observation.data, dict) else {}
        stderr_text = str(data.get("stderr") or "")
        if observation.ok and not stderr_text:
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
                    stderr_text[-1500:],
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
            "5. **制图完整**：专题图必须包含图名、图例、比例尺、指北针，并确保地图框已绑定数据图层。"
            "出图优先用 `create_map`（一次性传图层 + 意图，自动做版面与体检并回写可编辑 .aprx）；"
            "多张图要共用一套图例时，每张都传**同一个** explicit_bounds。\n"
            "6. **自检交付**：交付前调用 verify_outputs；确认产物满足需求后再调用 finish。\n"
            "7. **少问多做**：能从数据目录/文档推断的信息不要问用户；只在关键信息确实缺失时用 ask_user。\n"
            "8. **中文输出**：与用户交流、任务清单、方法论说明全部用中文。\n\n"
            "## 关于工具边界（省时间用）\n"
            "- 每个工具的 description 就是它的全部参数说明。**不要去读 gis_cli 的源码**来猜开关——"
            "实测有任务为此白跑了 200+ 次调用。工具里没写的参数，就是没有。\n"
            "- 需要“看效果再调”时：catalog_query 看数据字段/坐标系；view_image 看自己出的图；"
            "check_map_project 看 .aprx 里的渲染器与版面要素；这些都比读源码直接。\n"
            "- 制图相关的一律走 `create_map` / `graduated_colors_map` / `category_map` 这类入口，"
            "不要在 execute_code 里自己拼 arcpy.mp 代码（易错且不会自动体检）。\n\n"
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
            "- **按规范命名**：分类编码的类别名一律用任务给定的代码表（如土地覆盖 1=水域 2=林地 5=耕地 7=建筑区 8=裸地 11=牧场），"
            "报告与图例都不能只给编码或自创名称。\n"
            "- **交付前清点**：过程库只留必要中间件——同一个名称前缀出现多个版本（如 dem_mosaic_wgs84 / dem_mosaic_utm / dem_mosaic）"
            "是重做残留，必须删；`output/` 里不留 .pkl/.tmp/.bak 过程文件（可先移入过程库）。"
            "清理用 `cleanup_gdb`（先 dry_run 预览再执行），并保证结果库里的成果层一个不少。\n"
            "- **坑要记下**：踩到的工具/API/数据坑写进 finish 说明，系统会记入项目日志供后续任务使用。\n\n"
            f"## 数据目录摘要\n{self._catalog_digest()}\n\n"
            f"## 需求清单\n{self.state.requirements_digest() or '（由你从用户需求中提炼，并用 update_tasklist 呈现）'}\n\n"
            f"## 操作目录（可用配方，共 {len(self.recipes.all()) if self.recipes is not None else 0} 个；"
            "优先用配方，配方不适用再用 execute_code）\n"
            f"{self._recipe_catalog()}"
            + self._skills_section()
        )

    def _skills_section(self) -> str:
        """技能的 system prompt 段（只放名字+描述；真用时才读全文）。

        这是“自然语言触发技能”的关键：描述常驻上下文，模型才能判断
        “用户说要写 Word 文档”→ 该用 docx 技能 → 调 run_skill 读全文。
        只放描述是刻意的（见 pi/Claude 的 progressive disclosure），
        111 个技能如果全文常驻会直接爆上下文。
        """
        try:
            from ..skills.catalog import discover_skills, skills_prompt_section

            found = discover_skills(self.workspace)
            if not found:
                return ""
            logger.info("已发现 %s 个技能", len(found))
            return "\n\n" + skills_prompt_section(found)
        except Exception as exc:  # pragma: no cover - 技能扫描失败不影响主流程
            logger.warning("技能发现失败: %s", exc)
            return ""

    def _initial_user_message(self, goal: str, doc_paths: list[str] | None) -> str:
        text = ""
        if self.use_project_log:
            try:
                from .project_log import relevant_digest as _log_digest

                # 按**相关性**挑历史结论（而不只是最近几条）：
                # 同一个 workspace 里可能已跑过几十个任务，只取最近会把
                # “上次同类任务踩过的坑”挤出窗口（详见 relevant_digest）。
                recent = _log_digest(self.workspace, goal)
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

    def _sanitize_messages(self) -> None:
        """保证消息序列合法：role=tool 必须紧跟带 tool_calls 的 assistant 消息。

        严格端点（实测 DeepSeek）会直接 400：
        ``Messages with role 'tool' must be a response to a preceding message with 'tool_calls'``。
        上下文压缩、协议切换、附图插入都可能破坏这个顺序，这里统一兜底修正：
        孤立的 tool 消息降级成 user 文本（内容完整保留，不丢信息）。
        """
        fixed: list[dict[str, Any]] = []
        pending: set[str] = set()
        for message in self._messages:
            role = str(message.get("role", ""))
            if role == "assistant":
                pending = {
                    str(call.get("id"))
                    for call in (message.get("tool_calls") or [])
                    if isinstance(call, dict) and call.get("id")
                }
                fixed.append(message)
                continue
            if role == "tool":
                tool_id = str(message.get("tool_call_id") or "")
                if tool_id and tool_id in pending:
                    pending.discard(tool_id)
                    fixed.append(message)
                else:
                    fixed.append({"role": "user", "content": str(message.get("content") or "")})
                continue
            if role == "user":
                pending.clear()
            fixed.append(message)
        if len(fixed) != len(self._messages):
            logger.debug("消息序列已修正：%d -> %d", len(self._messages), len(fixed))
        self._messages = fixed

    def _append_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def inject_user_message(self, text: str) -> bool:
        """运行中注入一条用户消息（交互界面的"纠偏/补充"）。

        由 UI 线程调用，工作线程在下一轮开始时自然读到；返回是否已注入。
        运行已结束（或文本为空）时返回 ``False``，由调用方决定改成排队或新开轮次。
        """
        content = (text or "").strip()
        if not content:
            return False
        if self.state.status not in ("running", "awaiting_user"):
            return False
        self._append_user(content)
        try:
            self.recorder.record("user_injection", {"text": content[:2000]})
        except Exception:
            pass
        self._emit("user_injection", {"text": content[:400]})
        return True

    def request_stop(self, reason: str = "用户中断") -> bool:
        """请求尽快停止：把状态从 running 改掉，工作线程在下一个检查点退出。

        引擎每轮开头都会判断 ``status == "running"``，所以不需要信号/异步机制。
        """
        if self.state.status != "running":
            return False
        self.state.status = "stopped"
        self.state.error = reason
        try:
            self.recorder.record("stop_requested", {"reason": reason})
        except Exception:
            pass
        self._emit("stop_requested", {"reason": reason})
        return True

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

    def _vision_enabled(self) -> bool:
        """配置里开了识图、且客户端声明支持时才附图。"""
        try:
            if not bool(getattr(self.llm.config, "vision", False)):
                return False
            # 上游拒过图（如 400 "replace the image"）就不要再附：每轮白碰一次
            # 400 不仅浪费时间，还会把当轮对话搞脏。
            if getattr(self.llm, "vision_rejected", False):
                return False
            return True
        except Exception:  # pragma: no cover - 配置对象异常不影响主流程
            return False

    def _append_turn_images(self, observations: list[Observation]) -> None:
        """模型能识图时，把观察到的图片附进对话，让它自己核对图面。

        这正是过去“中文变方框的图一路通过验收”的补救：模型能看图，
        就能在图面上发现标题/图例不可读、地图被裁、图幅空白等问题。
        """
        if not self._vision_enabled():
            return
        try:
            from ..runtime.image_input import as_data_url, is_image_path

            candidates: list[str] = []
            for observation in observations:
                candidates += [str(item) for item in (observation.images or [])]
                candidates += [str(p) for p in (observation.artifacts or []) if is_image_path(p)]
            seen: set[str] = set()
            candidates = [c for c in candidates if not (c in seen or seen.add(c))]
            parts = []
            for item in candidates[:2]:
                url = as_data_url(item)
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            if not parts:
                return
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "上面工具的图片产出（请核对图面：图名/图例/比例尺是否齐全且可读，"
                                "中文有没有变成方框或乱码，图幅是否被裁切/过大面积空白）。"
                            ),
                        },
                        *parts,
                    ],
                }
            )
            if self.recorder is not None:
                self.recorder.record("vision_images", {"count": len(parts), "files": candidates[:2]})
        except Exception as exc:  # pragma: no cover - 附图失败不影响主流程
            logger.debug("attach observation images failed: %s", exc)

    # ------------------------------------------------------------ compaction
    def _estimate_tokens(self) -> int:
        """粗略估算对话 token 数（用于决定何时压缩）。

        关键：**图片块必须按固定成本算**，不能把 base64 当文本。
        实测坑：一张 400KB 的图 base64 后有 53 万字符，旧公式
        ``len(str(content))//2`` 算出 26.7 万 token（软上限才 12 万）——
        于是每附一张图就必定触发压缩，而压缩本身又是一次完整 LLM 调用。
        实测一个 32 轮的 run 里压了 16 次，后半段每轮白多一次往返。
        视觉 token 由模型侧计，这里只估"有没有图"，量级对就够了。
        """
        total = 0
        image_cost = max(1, int(self.config.image_token_cost))
        for message in self._messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        total += len(str(part)) // 2
                        continue
                    ptype = part.get("type")
                    if ptype == "image_url":
                        total += image_cost
                    elif ptype == "text":
                        total += len(str(part.get("text") or "")) // 2
                    else:
                        total += len(str(part)) // 2
                continue
            total += len(str(content or "")) // 2
        return total

    def _compact_if_needed(self) -> None:
        """上下文超限时压缩早期消息（带滞回与最少条数门槛）。

        滞回的必要性：旧逻辑"超一点就压"，而压缩后 tail 里可能仍超限，
        于是每轮都压一次（实测连压 16 轮，每轮多一次 LLM 往返）。
        现在要求超出软上限的一截（``compact_hysteresis``）才动手，
        且待压缩内容至少有 ``compact_min_messages`` 条——不够就不值一次调用。
        """
        estimated = self._estimate_tokens()
        threshold = self.config.context_soft_limit_tokens
        if threshold <= 0 or estimated < threshold * max(1.0, self.config.compact_hysteresis):
            return
        keep = self.config.keep_recent_messages
        if len(self._messages) <= keep + 2:
            return
        head = self._messages[:2]
        middle = self._messages[2:-keep]
        tail = self._messages[-keep:]
        if len(middle) < max(1, self.config.compact_min_messages):
            # 没多少可压的：压了也降不下来，白花一次调用
            return
        summary = self._summarize(middle)
        self._messages = head + [
            {"role": "user", "content": f"[进展摘要（早期轮次已压缩）]\n{summary}"}
        ] + tail
        if self.recorder is not None:
            self.recorder.record(
                "context_compacted",
                {"summarized_messages": len(middle), "estimated_tokens": estimated},
            )

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
        scanning = bool(getattr(self.catalog_prewarmer, "busy", False))
        try:
            from ..datacatalog.summary import build_digest

            digest = build_digest(self.catalog, max_datasets=30, max_fields=24, with_samples=False)
            if scanning:
                # 启动不再阻塞扫描：首次运行时目录可能还是空的。
                # 这里必须讲清楚，否则模型会以为“没有数据”而直接放弃。
                digest += (
                    "\n（注：数据目录正在后台扫描中。若上面为空或不全，"
                    "直接调用 catalog_query 即可——它会等扫描完成再返回真实字段。）"
                )
            return digest
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

    def _start_catalog_prewarm(self) -> None:
        """启动后台预热：数据目录扫描 + 内核启动（幂等；无预热器时只起内核）。

        两者都是"用户已经发了任务、但还没用到"的资源：
        - 目录扫描：与首轮 LLM 调用重叠；
        - 内核启动（实测 11.6~17.6s，大头是 import arcpy）：与两者重叠，
          这样第一个 execute_code 到达时内核已经就绪。
        """
        if self.catalog_prewarmer is not None:
            try:
                if self.catalog_prewarmer.start():
                    self._emit("catalog_scan", {"state": "started"})
                    self.recorder and self.recorder.record("catalog_scan", {"state": "started"})
            except Exception as exc:  # pragma: no cover - 扫描失败不影响任务
                logger.warning("catalog prewarm start failed: %s", exc)
        if self.code_runner is not None and hasattr(self.code_runner, "prewarm"):
            try:
                self.code_runner.prewarm()
            except Exception as exc:  # pragma: no cover - 预热失败不影响任务
                logger.debug("kernel prewarm failed: %s", exc)

    def _on_llm_delta(self, kind: str, text: str) -> None:
        """模型流式增量 → 界面事件（``llm_delta``）。

        只转发，不做任何决策；界面不在时（基准/单测）开销就是一次函数调用。
        """
        self._emit("llm_delta", {"kind": kind, "text": text})

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
                    # token 用量与缓存命中率：以前只在日志后缀里一闪而过，
                    # 跑完就无法回答“这次花了多少 / 缓存命中率如何”。
                    "usage": self._usage_summary(),
                },
            )
        if self.use_project_log:
            self._write_project_log()
        self._emit("run_end", self.state.to_dict())

    def _usage_summary(self) -> dict[str, Any]:
        """本次 run 的 token 用量汇总（客户端不支持时返回空结构）。"""
        try:
            getter = getattr(self.llm, "usage_summary", None)
            if callable(getter):
                return getter()
        except Exception as exc:  # pragma: no cover - 统计不能影响收尾
            logger.debug("usage summary failed: %s", exc)
        return {}

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
            self._sanitize_messages()
            try:
                response, actions = self.codec.call(
                    self._messages,
                    self.registry.schemas(),
                    task_type="agent",
                    temperature=0.1,
                    on_delta=self._on_llm_delta,
                )
            except Exception as exc:
                self.state.status = "llm_unavailable"
                self.state.error = f"LLM 不可用: {exc}"
                if self.recorder is not None:
                    self.recorder.record("llm_error", {"error": str(exc)[:500]})
                break
            self._append_assistant(response.content, response.tool_calls, actions)
            stop = False
            # 每轮都新建：续跑轮的工具观察也要附图/进对话。
            # 踩过的坑：这里曾经直接引用 ``turn_observations``，而它只在 ``run()``
            # 的作用域里定义 —— 任何走 resume 的路径都 NameError，ask_user
            # （模型“关键信息缺失时才问”的合法出口）用户答完必崩。
            turn_observations: list[Observation] = []
            for action, observation in self._run_actions(actions):
                if observation is not None:
                    turn_observations.append(observation)
                    self._append_observation(observation, action)
                if self._stopped_on is action:
                    stop = True
                    break
            self._append_turn_images(turn_observations)
            if stop:
                break
        if self.state.status == "running":
            self.state.status = "max_turns" if self.state.turn >= self.config.max_turns else "stopped"
        self._finalize()
        return self.state
