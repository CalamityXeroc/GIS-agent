# -*- coding: utf-8 -*-
"""子代理：把一段自包含的子任务放进**独立对话上下文**里跑完，只把结论带回父会话。

为什么需要：长任务里所有中间过程都挤在一条消息列表里，每轮都要重发整段历史，
越跑越贵，还会撞 ``max_turns``（实测 4a 跑满 120 轮）。子代理复用同一个
``AgentLoop``，但给它自己的 messages / TaskState / trace —— **对话隔离**；
ArcPy 内核、workspace、CodeRunner 仍然共享：内核是单例 + 串行执行锁，开第二个
既要重新 ``import arcpy``（~15s）又会和 ArcGIS Pro 抢 GP，收益远小于代价。

对外只有一个入口：``SubAgentRunner``。``runner.build_engine`` 建好它、把
``runner.run`` 塞进 ``ctx.extra["subagent_factory"]``，``delegate`` 工具在调用时取用。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

from ..trace.recorder import RunRecorder
from .loop import AgentLoop
from .state import Observation
from .tools import EngineContext

logger = logging.getLogger(__name__)

DEFAULT_MAX_PER_RUN = 3
DEFAULT_MAX_TURNS = 30
MIN_MAX_TURNS = 5
MAX_MAX_TURNS = 60
DEFAULT_TIMEOUT_SECONDS = 900.0
#: 子代理的上下文软上限（比父小：子任务本来就该是短对话）
CHILD_CONTEXT_SOFT_LIMIT = 60_000
#: 子代理单次 execute_code 的超时（父默认 600s）。卡死的 GP 调用要尽快失败：
#: 600s 那一档实测让一次卡死吃掉 10 分钟，而子代理的产出往往不需要那么久。
DEFAULT_EXEC_TIMEOUT_SECONDS = 300.0
#: 子代理 system prompt 里最多列几个技能（父默认 40 个 / 13k 字符）。
CHILD_SKILLS_MAX = 12
#: 子代理不能用的工具：delegate 防递归，ask_user 会让子任务停下来等人
SUBAGENT_EXCLUDED_TOOLS = ("delegate", "ask_user")

ROLE_PREAMBLE = """## 你是子代理（subagent）

- 你看不到父代理的对话历史：**任务描述就是你的全部上下文**，数据路径/字段/坐标系/输出位置都在里面。
- 不要去猜父代理做过什么；缺信息先用 catalog_query / read_document 自己查，查不到再在 finish 里说明。
- 父代理只能看到你的 finish 摘要：摘要必须写「结论 + 产物路径 + 关键参数（坐标系/字段/阈值/分级）」，不要写过程流水账。
- 你与父代理共享同一个工作区和 ArcPy 内核：不要清空内核变量，不要删你没创建的产物；
  自己的变量一律加 `sa_` 前缀（如 `sa_pts` / `sa_df`），别覆盖父代理的同名变量。
- 你没有 ask_user（父在等你，不能停下来问人），也不能再派子代理。
"""


@dataclass
class SubtaskResult:
    """一个子代理的最终结果（父代理能看到的**全部**信息）。"""

    subtask_id: str
    status: str
    summary: str
    methodology: str = ""
    artifacts: list[str] = field(default_factory=list)
    turns: int = 0
    elapsed_s: float = 0.0
    tokens: int = 0
    trace: str = ""
    error: str = ""
    #: 子代理跑的时候内核被看门狗硬杀并重启过（父的内核变量已丢）
    kernel_restarted: bool = False
    #: 回给父的观察体上限（``engine.subagent_result_max_chars``）
    max_chars: int = 4000

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "status": self.status,
            "summary": self.summary,
            "methodology": self.methodology,
            "artifacts": list(self.artifacts),
            "turns": self.turns,
            "elapsed_s": round(self.elapsed_s, 1),
            "tokens": self.tokens,
            "trace": self.trace,
            "error": self.error,
            "kernel_restarted": self.kernel_restarted,
        }

    def to_observation(self) -> Observation:
        """转成父代理看到的那条工具返回。

        子代理的意义就是省上下文，所以这里对摘要做硬截断（结构字段保留，
        模型仍然拿得到产物路径与状态）；完整内容在 trace 里。
        """
        max_chars = int(self.max_chars or 4000)
        payload = self.to_dict()
        if len(payload["summary"]) > 1500:
            payload["summary"] = payload["summary"][:1500] + "…（摘要过长已截断）"
        text = json.dumps(payload, ensure_ascii=False)
        if len(text) > max_chars:
            payload["summary"] = payload["summary"][:600] + "…（已截断，完整内容见 trace）"
            text = json.dumps(payload, ensure_ascii=False)[:max_chars]
        head = "子代理完成" if self.ok else f"子代理未完成（{self.status}）"
        return Observation(
            ok=self.ok,
            summary=f"{head}：{self.summary[:600]}",
            data=payload,
            artifacts=list(self.artifacts),
            error=None if self.ok else {"type": "SubagentNotCompleted", "message": self.status or "unknown"},
            hint=(
                "注意：子代理执行期间内核被看门狗重启过，你在内核里的变量已丢失，需要重新建立。"
                if self.kernel_restarted
                else ("" if self.ok else "可以缩小范围重派一次，或改用 execute_code 自己完成这一段。")
            ),
        )


def _usage_total(client: Any) -> int:
    try:
        summary = client.usage_summary() or {}
    except Exception:
        return 0
    return int(summary.get("prompt_tokens", 0) or 0) + int(summary.get("completion_tokens", 0) or 0)


class SubAgentRunner:
    """一个父 loop 对应一个 runner（记录已派数量、转发子代理事件）。"""

    def __init__(
        self,
        parent: AgentLoop,
        *,
        max_per_run: int = DEFAULT_MAX_PER_RUN,
        max_turns: int = DEFAULT_MAX_TURNS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        exec_timeout_seconds: float = DEFAULT_EXEC_TIMEOUT_SECONDS,
        result_max_chars: int = 4000,
    ) -> None:
        self.parent = parent
        self.max_per_run = max(1, int(max_per_run))
        self.max_turns = self._clamp_turns(max_turns)
        self.timeout_seconds = float(timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
        self.exec_timeout = max(60.0, float(exec_timeout_seconds or DEFAULT_EXEC_TIMEOUT_SECONDS))
        self.result_max_chars = int(result_max_chars or 4000)
        self._used = 0

    # ------------------------------------------------------------------ public
    def run(
        self,
        *,
        task: str,
        acceptance: str = "",
        max_turns: int | None = None,
        tools: list[str] | None = None,
    ) -> SubtaskResult:
        """跑一段子任务；任何失败都折成 ``SubtaskResult``，不抛异常。"""
        self._used += 1
        sid = f"sa{self._used}"
        if self._used > self.max_per_run:
            return SubtaskResult(
                subtask_id=sid,
                status="refused",
                summary=f"本次运行最多派 {self.max_per_run} 个子代理，已达上限，剩下的请自己做。",
                error="subagent limit reached",
            )
        budget = self._clamp_turns(max_turns or self.max_turns)
        deadline_budget = self._deadline_budget()
        if deadline_budget < 60:
            result = SubtaskResult(
                subtask_id=sid,
                status="refused",
                summary=f"父任务剩余时间不足（{deadline_budget:.0f}s），子代理来不及跑完，请自己完成。",
                error="insufficient parent budget",
            )
            self._emit_end(sid, result)
            return result
        self._emit(
            "subagent_start",
            {"subtask_id": sid, "task": task[:200], "max_turns": budget, "timeout_s": round(deadline_budget)},
        )

        allowed = [n for n in self.parent.registry.names() if n not in SUBAGENT_EXCLUDED_TOOLS]
        if tools:
            # finish 永远保留：收窄 tools 时把它过滤掉，子代理就永远交不了差，
            # 只能撞 max_turns（docs/subagent.md 里的契约也是这么写的）。
            wanted = {str(x) for x in tools} | {"finish"}
            allowed = [n for n in allowed if n in wanted]
        if not allowed:
            result = SubtaskResult(
                subtask_id=sid,
                status="refused",
                summary="子代理的工具表为空（tools 收窄过度），请放宽 tools 或自己完成。",
                error="empty tool subset",
            )
            self._emit_end(sid, result)
            return result

        parent_run = str(getattr(getattr(self.parent, "state", None), "run_id", "") or "run")
        recorder = RunRecorder(f"{parent_run}__{sid}", self.parent.workspace / ".gis_agent" / "traces")
        config = replace(
            self.parent.config,
            max_turns=budget,
            run_deadline_seconds=deadline_budget,
            context_soft_limit_tokens=min(
                int(self.parent.config.context_soft_limit_tokens or CHILD_CONTEXT_SOFT_LIMIT),
                CHILD_CONTEXT_SOFT_LIMIT,
            ),
            # 子代理的成功运行不沉淀配方：它只是父任务的一段，蒸馏价值归父
            distill_after_success="off",
            # 子代理没有“人”可确认计划卡：打印出去也只是刷屏，直接关掉
            plan_card=False,
            # 子任务的产物由父代理统一交付：子层出一份“成果包”只会把半成品
            # 写成正式报告，且会和父的同名文件抢字
            report_enabled=False,
            role_preamble=ROLE_PREAMBLE,
            # 子代理的卡死要尽快失败，别用父那档 600s
            exec_timeout=min(float(self.parent.config.exec_timeout or 600.0), self.exec_timeout),
            # 技能目录 111 条占 13k 字符，子任务用不到那么多
            skills_max=CHILD_SKILLS_MAX,
        )
        # 不能直接复用父的 ctx：run() 会把 context.state 覆盖成子状态，
        # 父的工具随后读到的就是子状态了。
        context = replace(self.parent.context) if isinstance(self.parent.context, EngineContext) else self.parent.context
        # execute_code 的默认超时取的是 ctx.exec_timeout，不只是 config
        try:
            context.exec_timeout = min(float(getattr(context, "exec_timeout", 600.0) or 600.0), self.exec_timeout)
            # 模型可以自己传 timeout_seconds，这里兜住：子代理不允许超过自己的上限
            context.max_exec_timeout = self.exec_timeout
        except Exception:
            pass

        child = AgentLoop(
            workspace=self.parent.workspace,
            llm_client=self.parent.llm,
            registry=self.parent.registry.subset(allowed),
            context=context,
            code_runner=self.parent.code_runner,
            catalog=self.parent.catalog,
            # 验收交给父代理：子层的 finish 若跑父的 Verifier，会拿**父任务的整任务要求**
            # 去判一段子任务（还要对整个 output/ 做结构体检），子任务必然被拒 3 次后记 failed。
            # 子代理的产物由父的 verify_outputs / finish 统一验收。
            verifier=None,
            recipes=self.parent.recipes,
            recorder=recorder,
            config=config,
            on_event=self._child_event(sid),
            # 子代理的任务描述就是规格，不需要再花一次 LLM 往返抽取需求
            auto_extract_requirements=False,
            use_project_log=False,
            should_stop=self._parent_stopped,
        )

        requirements = (
            [{"id": "SA1", "text": acceptance, "acceptance": acceptance, "source": "delegate"}]
            if acceptance
            else None
        )
        tokens_before = _usage_total(self.parent.llm)
        kernel_before = getattr(self.parent.code_runner, "_kernel", None)
        started = time.monotonic()
        try:
            state = child.run(task, requirements=requirements, reset_runner=False, persist_state=False)
            status = str(state.status)
            summary = str(state.final_summary or "")
            methodology = str(state.methodology or "")
            artifacts = list(state.artifacts or [])
            turns = int(state.turn or 0)
            error = str(state.error or "")
        except Exception as exc:  # 子代理崩了不能拖垮父
            logger.exception("subagent %s crashed", sid)
            status, summary, methodology, artifacts, turns = "error", "", "", [], 0
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        if status != "completed" and not summary:
            summary = error or f"子代理状态 {status}，没有给出结论"

        result = SubtaskResult(
            subtask_id=sid,
            status=status,
            summary=summary,
            methodology=methodology,
            artifacts=artifacts,
            turns=turns,
            elapsed_s=elapsed,
            tokens=max(0, _usage_total(self.parent.llm) - tokens_before),
            trace=self._relative(recorder.jsonl_path),
            error=error,
            # 看门狗硬杀内核后 CodeRunner 会重建一个新的：父的内核变量已经没了，
            # 这个信号必须回给父，否则它会拿旧变量名继续写代码然后连错几次。
            kernel_restarted=(
                kernel_before is not None
                and getattr(self.parent.code_runner, "_kernel", None) is not kernel_before
            ),
            max_chars=self.result_max_chars,
        )
        self._emit_end(sid, result)
        return result

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _clamp_turns(value: Any) -> int:
        try:
            turns = int(value)
        except (TypeError, ValueError):
            turns = DEFAULT_MAX_TURNS
        return max(MIN_MAX_TURNS, min(MAX_MAX_TURNS, turns))

    def _deadline_budget(self) -> float:
        """子代理的墙钟预算 = min(自己的上限, 父的剩余 - 30s 收尾余量)。

        不夹的话，父只剩 3 分钟时子代理仍按 900s 跑：父会被子拖着一起超时，
        而超时后父已经没有余量做收尾了。
        """
        budget = float(self.timeout_seconds)
        deadline = getattr(self.parent, "_deadline", None)
        if deadline is not None:
            budget = min(budget, deadline - time.monotonic() - 30.0)
        return max(0.0, budget)

    def _parent_stopped(self) -> bool:
        state = getattr(self.parent, "state", None)
        return state is not None and str(getattr(state, "status", "running")) != "running"

    def _relative(self, path: Any) -> str:
        try:
            return str(path.relative_to(self.parent.workspace)).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        emit = getattr(self.parent, "_emit", None)
        if callable(emit):
            try:
                emit(event, payload)
            except Exception:  # pragma: no cover - 事件只是展示
                pass

    def _emit_end(self, sid: str, result: SubtaskResult) -> None:
        self._emit(
            "subagent_end",
            {
                "subtask_id": sid,
                "status": result.status,
                "summary": result.summary[:1200],
                "turns": result.turns,
                "elapsed_s": round(result.elapsed_s, 1),
                "artifacts": len(result.artifacts),
            },
        )

    def _child_event(self, sid: str):
        """把子代理的工具事件转成 UI 认的 ``subagent_*``（其余事件不转发，避免刷屏）。"""

        def _forward(kind: str, payload: dict[str, Any]) -> None:
            if kind == "tool_call":
                self._emit(
                    "subagent_tool",
                    {"subtask_id": sid, "tool": payload.get("tool"), "args": payload.get("args") or {}},
                )
            elif kind == "tool_result" and not payload.get("ok", True):
                self._emit(
                    "subagent_warn",
                    {"subtask_id": sid, "tool": payload.get("tool"), "summary": payload.get("summary") or ""},
                )

        return _forward
