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
import os
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
    #: 单次失败允许的总修复时长（墙钟秒）。修不好就该回主循环换思路。
    max_repair_seconds: float = 600.0
    #: 成功后要不要把这次运行沉淀成配方：
    #: ``suggest``（默认，只提示不调模型）/ ``auto``（后台蒸馏）/ ``off``。
    #: 默认不自动：网关抖动时多一次 LLM 往返会把成功路径拖慢。
    distill_after_success: str = "suggest"
    exec_timeout: float = 600.0
    #: system prompt 里最多列多少个技能（每个约 120 字符）。子代理只给少数几个：
    #: 111 个技能全列是 13k 字符，而子任务只需要其中一两个。
    skills_max: int = 40
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
    #: 挂在 system prompt 最前面的身份说明。子代理用它声明「你是子代理、
    #: 内核共享、ask_user 不可用」；主代理留空。
    role_preamble: str = ""
    verbose: bool = False
    max_diagnosis_turns: int = 3
    escalate_after_failures: int = 2
    #: 一轮里最多并发几个**只读**工具（catalog_query / read_file …）。
    #: 1 = 完全串行（旧行为）。只并连续的只读动作，写操作与 execute_code 仍串行。
    max_parallel_readonly: int = 3
    # 整跑截止时间：网关半死不活时，单轮 LLM 调用可能反复重试几十分钟，
    # 若无总截止，一次 run 会整夜空转（实测 11.7h/22 轮）。到时后置 deadline_exceeded。
    run_deadline_seconds: float = 7200.0
    #: 执行前打印 5 行计划卡（P0-3）。非交互时只打印不拦截；闸门由调用方注入。
    plan_card: bool = True
    #: 计划卡上的合规标记。只写提示，不做判定（判定在 P1）。
    plan_compliance: str = "未送审，仅内部使用（不做合规判定）"
    #: 「e 修改」最多改几轮：改到第 3 次还在改，说明该重新描述任务，不是改卡。
    plan_max_revisions: int = 3

    #: 验收通过后自动生成成果包（单文件 HTML + PDF，P0-4）。
    #: 关掉只影响成果包，不影响验收与产物登记。
    report_enabled: bool = True


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
        should_stop: Callable[[], bool] | None = None,
        plan_gate: Any = None,
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
        #: 外部停止条件（子代理用它继承父的「用户中断」）。每轮开头检查一次。
        self.should_stop = should_stop
        #: 计划卡确认闸门（``ui.plan_card.PlanGate`` 之类，只要实现 ``ask(card)``）。
        #: ``None`` = 只打印计划卡、不等人（子代理与无头调用走这条路）。
        self.plan_gate = plan_gate
        #: ``run()`` 的持久化开关：子代理只留 trace，不写 runs/ 与项目日志。
        self._persist_state = True
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
                max_repair_seconds=self.config.max_repair_seconds,
            )
            if code_runner is not None
            else None
        )
        self.state: TaskState | None = None
        self._messages: list[dict[str, Any]] = []
        self._finish_rejections = 0
        #: 上一次被拒时的（缺失/问题签名, 产物清单）：两者都没变就说明“再拒也没用”
        self._last_rejection: tuple[tuple, tuple] | None = None
        #: 已发过的收尾提醒阈值（0.8 / 0.95），避免每轮重复提醒
        self._finish_nudges: set[float] = set()
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
        reset_runner: bool = True,
        persist_state: bool = True,
    ) -> TaskState:
        """Run one task to completion (or to an ``ask_user`` pause).

        ``reset_runner=False``：不清内核。子代理必须传 False —— 父代理在内核里
        攒的变量（df、投影对象）是它下一步要用的，清了就等于把父的活重做。
        ``persist_state=False``：不写 ``.gis_agent/runs/`` 与项目日志。子运行若
        落成正式 run，蒸馏与断点续跑会把它当成「最近一次成功运行」挑走。
        """
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

        self._persist_state = persist_state
        self._emit("run_start", {"goal": goal, "run_id": run_id})
        self.recorder.record("run_start", {"goal": goal, "workspace": str(self.workspace)})

        if reset_runner and self.code_runner is not None:
            try:
                self.code_runner.reset()
            except Exception as exc:
                logger.warning("code runner reset failed: %s", exc)

        # 数据目录后台扫描：与首轮模型调用重叠（启动时不扫，界面才能立刻可用）。
        # 先启动再拼 system prompt：摘要里能如实写上“正在扫描”，
        # 真正的数据由 catalog_query 工具在需要时等扫描完再给。
        self._start_catalog_prewarm(kernel=False)

        # 执行前先给人看一张 5 行计划卡（P0-3）：非交互只打印，TTY 等人确认。
        # 放在目录扫描之后：“用什么”才能列到刚扫出来的数据集；
        # 内核先不预热——用户按 q 取消时不该白起一次内核（实测 12~17s）。
        if not self._confirm_plan_or_continue(goal, doc_paths):
            self.state.status = "cancelled"
            self.state.error = "计划未确认（用户取消）"
            self.recorder.record("plan_cancelled", {"reason": self.state.error})
            self._emit("plan_cancelled", {"reason": self.state.error})
            self._finalize()
            return self.state

        self._start_catalog_prewarm()  # 计划已确认：现在才预热内核（扫描幂等，不会重跑）

        self._messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._initial_user_message(goal, doc_paths)},
        ]
        # messages 在**计划确认之后**才拼：`e` 改的那一句已并回 state.requirements，
        # system prompt 会照 requirements_digest() 的口径带上去。留一条 trace，
        # 事后能核实“模型当时看到的到底是哪一版需求”。
        self.recorder.record(
            "system_prompt",
            {
                "chars": len(self._messages[0]["content"]),
                "requirements": self.state.requirements_digest(),
            },
        )
        self._deadline = (
            time.monotonic() + self.config.run_deadline_seconds
            if self.config.run_deadline_seconds and self.config.run_deadline_seconds > 0
            else None
        )

        while self.state.turn < self.config.max_turns and self.state.status == "running":
            if self.should_stop is not None and self.should_stop():
                # 父任务被中断（用户 Ctrl-C）：子代理不能继续跑——它会一直占着
                # 内核与 LLM 配额，而父已经在等下一次输入了。
                self.state.status = "stopped"
                self.state.error = "父任务已停止"
                self._emit("stop_requested", {"reason": self.state.error})
                break
            if self._deadline is not None and time.monotonic() > self._deadline:
                self.state.status = "deadline_exceeded"
                self.state.error = (
                    f"整跑截止时间已到（run_deadline_seconds={self.config.run_deadline_seconds:g}s，"
                    f"已完成 {self.state.turn} 轮）。通常是模型网关长时间不可用导致单轮反复重试；"
                    "任务状态已落盘到 .gis_agent/runs/，产物与 trace 均保留，排查网关后重跑本任务即可"
                    "（同目录已有产物会被覆盖前备份）。"
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

    # ------------------------------------------------------- 断点续跑
    #: 这几种状态可以续（模型/网络中断、时限、轮次用尽），其余（completed 等）不续。
    RESUMABLE_STATUSES = ("llm_unavailable", "deadline_exceeded", "max_turns", "stopped", "failed")

    def find_checkpoint(self) -> tuple[str, str] | None:
        """在 ``.gis_agent/runs/`` 里找最近一次可续的运行。

        返回 ``(state_json, messages_json)``；没找到返回 None。
        """
        runs_dir = self.workspace / ".gis_agent" / "runs"
        if not runs_dir.is_dir():
            return None
        candidates: list[tuple[float, Path]] = []
        for path in runs_dir.glob("*.json"):
            if path.name.endswith(".messages.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(data.get("status") or "") not in self.RESUMABLE_STATUSES:
                continue
            messages = runs_dir / f"{path.stem}.messages.json"
            if not messages.exists():
                continue  # 没有消息历史就无法续跑（旧版本留下的状态）
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
        if not candidates:
            return None
        _mtime, latest = max(candidates, key=lambda item: item[0])
        return str(latest), str(runs_dir / f"{latest.stem}.messages.json")

    def resume_from_checkpoint(self) -> TaskState:
        """从最近一次未完成的运行继续跑（网关抖动 / 撞时限后不用从头再来）。

        与 ``resume()`` 的区别：那个用于 ``ask_user`` 暂停（状态在内存里）；
        这个从磁盘恢复——模型消息历史、清单、产物都在，接着往下跑。
        """
        found = self.find_checkpoint()
        if found is None:
            raise RuntimeError("没有可续跑的断点（.gis_agent/runs/ 下无未完成运行或缺少消息历史）")
        state_path, messages_path = found
        state = TaskState.load(state_path)
        try:
            messages = json.loads(Path(messages_path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"断点里的消息历史读取失败: {exc}") from exc
        if not isinstance(messages, list) or not messages:
            raise RuntimeError("断点里的消息历史为空，无法续跑")

        self.state = state
        self._messages = messages
        if self.context is not None:
            self.context.state = state
        # 断点的轮次可能已经用尽（max_turns）：续跑必须再给一份预算，
        # 否则 while turn < max_turns 直接不成立，续跑等于空转
        # （实测 14 届任务二a：60/60 续跑只把旧总结又打印了一遍）。
        self.config.max_turns = int(state.turn) + max(1, int(self.config.max_turns))
        # 续跑的 trace 写新文件（带 resumed 标记），状态仍回写原 checkpoint，
        # 这样反复续跑不会不断产生新的可续断点。
        if self.recorder is None:
            self.recorder = RunRecorder(new_run_id(), self.workspace / ".gis_agent" / "traces")
        self._deadline = (
            time.monotonic() + self.config.run_deadline_seconds
            if self.config.run_deadline_seconds and self.config.run_deadline_seconds > 0
            else None
        )
        state.status = "running"
        state.error = ""
        if self.recorder is not None:
            self.recorder.record(
                "resumed",
                {"from_run": state.run_id, "turn": state.turn, "messages": len(messages)},
            )
        self._emit("resumed", {"turn": state.turn, "messages": len(messages)})
        self._append_user(
            "[系统提示] 上一次运行中断了（模型网关不可用或超时），已从断点恢复。"
            "已完成的产物与清单都在，请接着未完成的步骤继续，不要从头重做。"
        )
        return self._continue()

    # ------------------------------------------------------- speed guardrails
    def _update_guards(self, actions: list[Action], observations: list[Observation]) -> None:
        """Exploration budget nudge + failure escalation counters."""
        # 收尾提醒：轮次用到 80% 还没调 finish 就催一次。
        # 实测 14 届任务四a：120 轮里 finish 调用 0 次——产物早就合格（断言 12/12），
        # 模型却一直在打磨，直到撞轮次上限，status 只能记成 max_turns。
        # 这类任务本该以 completed 收尾，预算也不该浪费在重复出图上。
        self._maybe_nudge_finish()
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

    def _maybe_nudge_finish(self) -> None:
        """轮次快用尽时提醒模型收尾（每个阈值只提一次）。"""
        limit = int(getattr(self.config, "max_turns", 0) or 0)
        if limit <= 0 or self.state is None:
            return
        turn = int(self.state.turn or 0)
        for threshold in (0.8, 0.95):
            if turn < limit * threshold or threshold in self._finish_nudges:
                continue
            self._finish_nudges.add(threshold)
            remaining = max(0, limit - turn)
            self._append_user(
                f"[系统提示] 已用 {turn}/{limit} 轮（剩 {remaining} 轮）。"
                "如果交付物已经齐备，请**立刻**调用 finish 说明成果与关键数值；"
                "不要再重复出图/重算——继续打磨会直接撞轮次上限，反而交不出结论。"
            )

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
            self.recorder.record(
                "tool_call",
                {"tool": action.tool, "args": self._safe_args(action.args, for_trace=True)},
                # 代码块是 trace 里最有价值的证据（复盘 / 配方蒸馏都靠它），
                # 不能被默认的 4000 字符上限切掉。
                max_str=20000,
            )
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
        """根目录意外项 + output/ 下过程残留，都及时提醒。"""
        try:
            strays = sorted(
                p.name
                for p in self.workspace.iterdir()
                if p.name not in _ALLOWED_ROOT and not p.name.startswith(".")
            )
        except Exception:
            return
        fresh = [name for name in strays if name not in self._hygiene_warned]
        if fresh:
            self._hygiene_warned.update(fresh)
            note = (
                "工作区根目录出现预期外的项："
                + ", ".join(fresh[:5])
                + "。产出必须写在 output/ 下；这通常是 Windows 路径反斜杠转义错误"
                "（如 `\"...\\data\\...\"`）导致，请核对路径（用原始字符串或正斜杠）并清理误建目录。"
            )
            observation.hint = (observation.hint + "\n" + note) if observation.hint else note
            self.recorder and self.recorder.record("workspace_hygiene", {"strays": fresh[:8]})
        # output/ 的残留要单独扫：根目录干净不代表交付目录干净
        self._warn_output_residue(observation)

    def _warn_output_residue(self, observation: Observation) -> None:
        """output/ 下的下划线临时件要**及时**提醒，别拖到最终验收。

        实测 14 届任务二a：模型把 `_mp_fix.shp/.dbf` 落在 output/ 里，一直到
        交付卫生断言才被抳住——那时 60 轮已经用尽，模型想清也来不及了。
        这里每轮扫一遍（只扫两层、跳过 .gdb），把清单塞进观察提示。
        """
        out = self.workspace / "output"
        if not out.is_dir():
            return
        found: list[str] = []
        try:
            for entry in os.scandir(out):
                if entry.is_file() and entry.name.startswith("_"):
                    found.append(entry.name)
                elif entry.is_dir() and not entry.name.lower().endswith(".gdb"):
                    for sub in os.scandir(entry.path):
                        if sub.is_file() and sub.name.startswith("_"):
                            found.append(f"{entry.name}/{sub.name}")
        except OSError:
            return
        fresh = sorted(n for n in found if n not in self._hygiene_warned)
        if not fresh:
            return
        self._hygiene_warned.update(fresh)
        note = (
            "output/ 下发现过程残留（下划线开头的临时件）："
            + "、".join(fresh[:5])
            + "。交付前必须删除（考卷要求提交时删除非必要过程数据），"
            "建议现在就用 arcpy.Delete_management / os.remove 清掉。"
        )
        observation.hint = (observation.hint + "\n" + note) if observation.hint else note
        self.recorder and self.recorder.record("output_residue", {"files": fresh[:8]})

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

        try:
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
        except Exception as exc:
            # 修复本身出错（最典型的是修复时模型网关断了）不能把整个 run 拖死：
            # 以前 LLMError 会一路冒到 runner，整次运行记成 runner_error、产物作废。
            # 修复是可选加速——交回原始失败观察，下一轮模型调用若也不通，
            # 循环会按 llm_unavailable 优雅退出（已跑出的产物不受影响）。
            logger.warning("repair failed: %s", exc)
            self.recorder.record("repair_error", {"error": str(exc)[:300]})
            self._emit("repair_error", {"error": str(exc)[:200]})
            return observation
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
        # 同一套证据被同一个理由连拒多次时，再拒一次也不可能变好（实测：组合任务
        # 连拒 4 次，每次 40~80s 的语义验收，产物一个字节都没变，最后 status=failed）。
        # 产物清单没变且拒绝理由完全一致 → 降级通过，但把原因写进 summary。
        signature = self._rejection_signature(verdict)
        artifacts_now = self._artifact_signature()
        repeated = bool(
            not verdict.get("pass")
            # 签名全空 = 没拿到任何具体理由，那就不存在“同一理由”，不许降级
            and any(signature)
            and self._last_rejection is not None
            and self._last_rejection == (signature, artifacts_now)
        )
        self._last_rejection = (signature, artifacts_now) if not verdict.get("pass") else None
        if repeated:
            verdict = dict(verdict)
            verdict["pass"] = True
            verdict["degraded"] = True
            verdict["summary"] = "验收降级通过（同一理由重复拒绝且产物未变）：" + str(verdict.get("summary", ""))
        self.recorder.record("verification", verdict)
        self._emit("verification", verdict)

        if verdict.get("pass"):
            self.state.status = "completed"
            self._build_deliverables(verdict)
            prefix = "验收降级通过" if verdict.get("degraded") else "验收通过"
            return (
                Observation(
                    ok=True,
                    summary=f"{prefix}。{observation.summary}",
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

    # ------------------------------------------------------------ deliverables
    def _build_deliverables(self, verdict: dict[str, Any]) -> None:
        """验收通过后生成单文件成果包（P0-4 步 d）。

        放在验收**之后**：报告里的"验证"段要写验收结论，提前生成只能是空的。
        任何失败都只记日志，不影响交付状态 —— 成果包是附加物。
        """
        if not getattr(self.config, "report_enabled", True) or self.state is None:
            return
        try:
            from ..report import build_deliverables

            result = build_deliverables(
                workspace=self.workspace,
                state=self.state,
                verdict=verdict,
                catalog=self.catalog,
                code_runner=getattr(self.context, "code_runner", None),
                trace_path=getattr(self.recorder, "jsonl_path", None),
                recipes=self.recipes,
            )
        except Exception as exc:  # pragma: no cover - 成果包不能拖垮交付
            logger.warning("成果包生成失败（不影响交付）: %s", exc)
            return
        if not result.paths:
            return
        for path in result.paths:
            self.state.add_artifact(path)
        payload = {
            "files": result.paths,
            "layers": result.layers,
            "images": result.images,
            "notes": result.notes,
        }
        self.recorder.record("deliverables", payload)
        self._emit("deliverables", payload)

    # ------------------------------------------------------------ requirements
    @staticmethod
    def _rejection_signature(verdict: dict[str, Any]) -> tuple:
        """拒绝理由的指纹（缺失项 + 结构问题 + 逐条需求 FAIL/PARTIAL）。

        只比 missing/problems 是不够的：验收的通过判据里 ``failed_reqs`` 单独一条
        就能否决，而它只出现在 ``semantic.per_requirement`` 里。漏了它，
        “第一次缺直方图、第二次坐标系不对”会是同一个签名 → 误判成同一问题而降级通过。
        """
        missing = tuple(sorted(str(item) for item in (verdict.get("missing") or [])))
        problems = tuple(sorted(str(item) for item in (verdict.get("problems") or [])))
        semantic = verdict.get("semantic") or {}
        failed = tuple(
            sorted(
                f"{item.get('id')}:{str(item.get('status', '')).upper()}"
                for item in (semantic.get("per_requirement") or [])
                if str(item.get("status", "")).upper() in {"FAIL", "PARTIAL"}
            )
        )
        return (missing, problems, failed)

    def _artifact_signature(self) -> tuple:
        """产物指纹：路径 + mtime/size（库内条目看库目录里所有文件）。

        只比路径时，“原地改了同一个文件”等于“没改”，降级判据会误判。
        """
        paths = [str(item) for item in (getattr(self.state, "artifacts", []) or [])]
        signer = getattr(self.verifier, "artifact_signature", None)
        if callable(signer):
            try:
                return tuple(signer(paths))
            except Exception:  # pragma: no cover - 指纹失败就退回路径比较
                pass
        return tuple(sorted(paths))

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

    # ------------------------------------------------------------ plan card
    def _plan_records(self) -> tuple[list[Any], int]:
        """给计划卡用的数据目录记录与登记总数（拿不到就空——计划卡不能拦住主流程）。"""
        if self.catalog is None:
            return [], 0
        try:
            prewarmer = self.catalog_prewarmer
            if prewarmer is not None and getattr(prewarmer, "busy", False):
                # 目录还在后台扫：最多等 3 秒。计划卡要的是“已登记”，
                # 不是“扫完”；为了它把启动拖住比少列两个数据集更亏。
                prewarmer.wait_ready(timeout=3.0)
            records = list(self.catalog.list_datasets(limit=200) or [])
        except Exception as exc:
            logger.warning("plan card catalog read failed: %s", exc)
            records = []
        registered = 0
        counter = getattr(self.catalog, "counts", None)
        if callable(counter):
            try:
                registered = sum(int(n) for n in (counter() or {}).values())
            except Exception:  # pragma: no cover - 计数失败就退回列表长度
                registered = 0
        return records, registered

    def _build_plan_card(self, goal: str, *,
                         requirements: list[Requirement] | None = None,
                         revisions: list[str] | None = None) -> Any:
        """按当前需求清单重建计划卡（纯渲染，不调模型）。"""
        from ..ui.plan_card import build_plan_card

        records, registered = self._plan_records()
        return build_plan_card(
            goal,
            requirements if requirements is not None else list(self.state.requirements),
            records,
            workspace=self.workspace,
            revisions=revisions or [],
            compliance=self.config.plan_compliance,
            registered=registered or None,
        )

    def _revise_plan(
        self, goal: str, revision: str, doc_paths: list[str] | None
    ) -> list[Requirement]:
        """把用户的一句话改动并回需求清单（重抽一次；失败就当成一条新需求）。"""
        amended = f"{goal}\n\n## 用户修正（以此为准）\n{revision}"
        revised: list[Requirement] = []
        if self.auto_extract_requirements:
            revised = self._extract_requirements(amended, doc_paths)
        if not revised:
            revised = list(self.state.requirements)
            revised.append(
                Requirement(
                    id=f"user{len(revised) + 1}",
                    text=revision,
                    acceptance="",
                    source="user",
                )
            )
        return revised

    def _confirm_plan(self, goal: str, doc_paths: list[str] | None) -> bool:
        """执行前发计划卡并等确认。返回 False = 用户取消，不该继续跑。

        非交互（或没注入闸门）时只发事件／记 trace，不读 stdin：CI 与脚本
        不能因为一张卡片而挂住。
        """
        if not self.config.plan_card:
            return True

        card = self._build_plan_card(goal)
        revisions: list[str] = []
        interactive = bool(getattr(self.plan_gate, "interactive", False))
        max_rounds = max(1, int(self.config.plan_max_revisions))
        shown = False

        for round_no in range(1, max_rounds + 1):
            self._emit_plan_card(card, interactive=interactive, round_no=round_no)
            shown = True

            if self.plan_gate is None:
                return True
            # chat 的闸门会真的挂起（引擎线程不读 stdin）：把暂停态同时写进
            # state 与 trace，界面/trace 都能回答“这次 run 在计划卡上等过”。
            pausing = bool(getattr(self.plan_gate, "pauses", False))
            try:
                if pausing:
                    self._enter_plan_pending(round_no, card)
                decision = self.plan_gate.ask(card)
            except Exception as exc:  # 闸门炸了不能把任务拖死（stdin 已关闭等）
                logger.warning("plan gate failed (continuing): %s", exc)
                self._emit("plan_card_auto_start", {"reason": f"计划确认不可用（{exc}），直接开始"})
                return True
            finally:
                if pausing:
                    self._leave_plan_pending()
            action = str(getattr(decision, "action", "start") or "start")
            revision = str(getattr(decision, "text", "") or "").strip()
            self.recorder.record("plan_decision", {"round": round_no, "action": action, "text": revision})
            if action == "cancel":
                logger.info("plan card cancelled by user")
                return False
            if action != "revise":
                return True

            if not revision:
                return True
            revised = self._revise_plan(goal, revision, doc_paths)
            if revised:
                self.state.requirements = list(revised)
            revisions.append(revision)
            self.recorder.record(
                "plan_revised",
                {"revision": revision, "requirements": self.state.requirements_digest()},
            )
            card = self._build_plan_card(goal, revisions=revisions)
            shown = False

        # 改到轮次上限还在改：按最后一版开始。最后一版没展示过就先发出去，
        # 否则跑的是一个用户没看过的计划。
        if not shown:
            self._emit_plan_card(card, interactive=False, round_no=max_rounds + 1)
        self._emit(
            "plan_card_auto_start",
            {"reason": f"已修改 {len(revisions)} 次，按当前计划开始"},
        )
        return True

    def _enter_plan_pending(self, round_no: int, card: Any) -> None:
        """进入 ``plan_pending`` 暂停态：run 停在计划闸门上。

        这里只改状态与 trace；**不读 stdin**。读键的是 UI 线程（chat 的
        prompt_toolkit 会话），引擎线程停在闸门自己的 Event 上。
        """
        if self.state is None:
            return
        self.state.status = "plan_pending"
        self.recorder.record("plan_pending", {"round": round_no, "text": card.render(interactive=True)})
        self._emit("plan_pending", {"round": round_no})

    def _leave_plan_pending(self) -> None:
        """离开暂停态（没进过就什么都不做）。

        本方法在 ``finally:`` 里调，所以**自己绝不能抛**：这里冒出的异常会顶掉
        刚刚拿到的答复，外层按“计划卡出错 → 当作已确认”继续 —— 用户按的 ``q``
        就被静默改成了 start（DAIR-38）。发事件失败只记日志，状态复位照旧。
        """
        if self.state is not None and self.state.status == "plan_pending":
            self.state.status = "running"
        try:
            self._emit("plan_decided", {})
        except Exception as exc:  # pragma: no cover - 防御性
            logger.warning("plan_decided emit failed (decision kept): %s", exc)

    def _emit_plan_card(self, card: Any, *, interactive: bool, round_no: int) -> None:
        """发一张计划卡（事件 + trace 同源，事后能回答“当时确认的是什么”）。"""
        payload = card.to_dict()
        payload["text"] = card.render(interactive=interactive)
        payload["round"] = round_no
        self.recorder.record("plan_card", payload)
        self._emit("plan_card", payload)

    def _confirm_plan_or_continue(self, goal: str, doc_paths: list[str] | None) -> bool:
        """计划卡永远不能拦住任务：自身出错就当“已确认”继续。

        fail-open 也要**看得见**（与内层闸门失败一致：trace 一条 + 屏幕一条），
        否则“计划卡被跳过了”只能去翻日志文件才知道（DAIR-38）。
        """
        try:
            return self._confirm_plan(goal, doc_paths)
        except Exception as exc:  # pragma: no cover - 防御性
            logger.warning("plan card failed (continuing): %s", exc)
            if self.recorder is not None:
                self.recorder.record("plan_card_failed", {"error": str(exc)[:500]})
            self._emit("plan_card_auto_start", {"reason": f"计划卡出错（{exc}），直接开始"})
            return True

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
            (self.config.role_preamble + "\n" if self.config.role_preamble else "")
            + "你是一个资深 GIS 智能体，目标是把用户的自然语言需求转化为可复现的 GIS 方法论并完成实际操作。\n\n"
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
            + self._delegate_hint()
            + "## 关于工具边界（省时间用）\n"
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

    def _delegate_hint(self) -> str:
        """有 delegate 工具时，明说什么时候该拆（否则模型一律自己干）。

        实测：glm-5.3-flash 在 28 轮的组合任务（两个互不依赖的交付）上也不会
        主动派子代理——工具描述写在 schema 里，模型仍然倾向自己做完。
        代价是长任务每轮重发整段历史（平方级）。所以这里给一条带门槛的指引：
        只有「≥2 块、每块预计 >10 轮」才拆。子代理自己看不到 delegate，
        这段对它不出现，免得它去找一个不存在的工具。
        """
        try:
            if self.registry.get("delegate") is None:
                return ""
        except Exception:
            return ""
        return (
            "9. **能拆才拆**：任务里有 **≥2 块互不依赖、每块预计 >10 轮** 的交付时"
            "（例如“4 张密度图”和“迁移图”），先 update_tasklist 拆开，再对其中一块调用 "
            "delegate（task 必须自包含：数据路径/字段/坐标系/输出位置/验收标准）。"
            "3~5 步的小活儿不要派——每个子代理的固定开销约 1.5 万 token，拆小了更贵。\n\n"
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
            limit = int(getattr(getattr(self, "config", None), "skills_max", 40) or 40)
            return "\n\n" + skills_prompt_section(found, max_skills=max(1, limit))
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
            config = getattr(self.llm, "config", None)
            # ``effective_vision`` 会把 vision_mode=auto 按模型名折算；
            # 老配置没有这个属性时退回原来的布尔开关。
            enabled = getattr(config, "effective_vision", None)
            if enabled is None:
                enabled = bool(getattr(config, "vision", False))
            if not enabled:
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

    def _safe_args(self, args: dict[str, Any], *, for_trace: bool = False) -> dict[str, Any]:
        """记录/推送用的参数副本。

        ``code`` 默认截到 400 字符——那是给**界面预览**用的。但 trace 还要被
        事后复盘和配方蒸馏读：实测蒸馏时只能看到每个代码块的前 400 字符，
        模型拼出来的“泛化配方”自然是垃圾。所以写 trace 时不截（由 recorder
        的 max_str 封顶）。
        """
        limit = 20000 if for_trace else 400
        safe = {}
        for key, value in (args or {}).items():
            if key == "code" and isinstance(value, str):
                safe["code"] = value[:limit] + ("..." if len(value) > limit else "")
            else:
                safe[key] = value
        return safe

    def _start_catalog_prewarm(self, *, kernel: bool = True) -> None:
        """启动后台预热：数据目录扫描 + 内核启动（幂等；无预热器时只起内核）。

        两者都是"用户已经发了任务、但还没用到"的资源：
        - 目录扫描：与首轮 LLM 调用重叠；
        - 内核启动（实测 11.6~17.6s，大头是 import arcpy）：与两者重叠，
          这样第一个 execute_code 到达时内核已经就绪。

        ``kernel=False``：只扫地目录。计划卡要在扫描后建（才有“用什么”），
        但卡没确认前用户随时可能取消，那时起内核就是白烧。
        """
        if self.catalog_prewarmer is not None:
            try:
                if self.catalog_prewarmer.start():
                    self._emit("catalog_scan", {"state": "started"})
                    self.recorder and self.recorder.record("catalog_scan", {"state": "started"})
            except Exception as exc:  # pragma: no cover - 扫描失败不影响任务
                logger.warning("catalog prewarm start failed: %s", exc)
        if kernel and self.code_runner is not None and hasattr(self.code_runner, "prewarm"):
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
        if self._persist_state:
            try:
                self.state.save(str(self.workspace / ".gis_agent" / "runs" / f"{self.state.run_id}.json"))
            except Exception as exc:
                logger.warning("failed to save run state: %s", exc)
            # 消息历史也要落盘，否则“续跑”无从恢复上下文（state 里只有清单/产物）。
            # 单独一个文件：messages 可能上百 KB，别把状态文件撞成日志。
            try:
                import json as _json

                target = self.workspace / ".gis_agent" / "runs" / f"{self.state.run_id}.messages.json"
                target.write_text(_json.dumps(self._messages, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                logger.warning("failed to save messages: %s", exc)
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
        if self.use_project_log and self._persist_state:
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
