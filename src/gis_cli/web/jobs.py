# -*- coding: utf-8 -*-
"""JobManager：**单 worker 线程 + deque 队列**，一个 run 一条状态机。

为什么必须单 worker（``docs/p2-console-plan.md`` §6）：``ArcpyKernel`` 是单例、
``llm_max_concurrency=1``、ArcGIS 也是单实例。并发跑两个 run 不是"更快"，
而是两个任务抢同一个内核，互相把对方的变量清掉。

状态机::

    queued ──▶ running ──▶ awaiting_plan ──▶ running …
       │          │                │
       │          └────────────────┴──▶ stopping ──▶ stopped / succeeded / failed / interrupted
       └──────────────────────────────────────▶ stopped / failed / interrupted

- ``stop``：置停止标记并**立即**把状态迁到 ``stopping``（同时推一条
  ``stop_requested`` SSE），界面不用等当前工具返回；引擎在下一个检查点
  退出（不硬杀子进程），worker 收到引擎结果后才转 ``stopped``。重复 stop 幂等。
  还在排队的 run 直接置 ``stopped`` 且**不再执行**。
- ``jobs.json``：每次状态迁移落盘（带节流）；服务重启时把 ``queued/running/
  awaiting_plan/stopping`` 一律标成 ``interrupted``，历史仍然可查。

本模块不 import 引擎（引擎只在真正跑任务时按需加载），也不 import ``gis_cli.ui``。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from pydantic import BaseModel

from .event_bus import EventBus
from .plan_gate import DEFAULT_TIMEOUT, ThreadingPlanGate

logger = logging.getLogger(__name__)

#: 持久化文件（相对 workspace）。
JOBS_FILE = Path(".gis_agent") / "jobs.json"

# ---------------------------------------------------------------- 状态机

QUEUED = "queued"
RUNNING = "running"
AWAITING_PLAN = "awaiting_plan"
#: 已收到 stop、引擎还没返回：界面上不能显示成「运行中」，也不能显示成「已停止」。
STOPPING = "stopping"
SUCCEEDED = "succeeded"
FAILED = "failed"
STOPPED = "stopped"
INTERRUPTED = "interrupted"

STATUSES = (QUEUED, RUNNING, AWAITING_PLAN, STOPPING, SUCCEEDED, FAILED, STOPPED, INTERRUPTED)
TERMINAL = frozenset({SUCCEEDED, FAILED, STOPPED, INTERRUPTED})

#: 合法迁移表：没列出来的一律 ``IllegalTransition``（单测直接盯这张表）。
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    QUEUED: frozenset({RUNNING, STOPPED, FAILED, INTERRUPTED}),
    RUNNING: frozenset({AWAITING_PLAN, STOPPING, SUCCEEDED, FAILED, STOPPED, INTERRUPTED}),
    AWAITING_PLAN: frozenset({RUNNING, STOPPING, FAILED, STOPPED, INTERRUPTED}),
    # stopping 的终态不是只有 stopped：stop 到达时引擎可能已经跑完（succeeded），
    # 也可能以别的错误结束（failed）。
    STOPPING: frozenset({SUCCEEDED, FAILED, STOPPED, INTERRUPTED}),
    SUCCEEDED: frozenset(),
    FAILED: frozenset(),
    STOPPED: frozenset(),
    INTERRUPTED: frozenset(),
}

#: 引擎 ``TaskState.status`` → job 状态。``completed`` 只代表"引擎跑完了"，
#: 过没过验收看 ``verification``（这一点与 P2 看板口径一致）。
ENGINE_STATUS_MAP: dict[str, str] = {
    "completed": SUCCEEDED,
    "cancelled": STOPPED,
    "stopped": STOPPED,
    "failed": FAILED,
    "deadline_exceeded": FAILED,
    "llm_unavailable": FAILED,
    "max_turns": FAILED,
    # 等用户补充信息 = 这一轮停住了，不是成功也不是失败
    "awaiting_user": INTERRUPTED,
    "plan_pending": INTERRUPTED,
}

#: 重启时会被改写成 ``interrupted`` 的状态。
_LIVE_STATUSES = frozenset({QUEUED, RUNNING, AWAITING_PLAN, STOPPING})


class JobError(Exception):
    """job 层错误的基类（路由层映射成 HTTP 状态码）。"""


class JobNotFound(JobError):
    pass


class JobConflict(JobError):
    """状态不允许这个操作（run 已结束 / 没有等待中的计划卡）。"""


class IllegalTransition(JobError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 数据模型


@dataclass
class Job:
    """一个 run 的可见状态。字段名与 ``GET /api/runs/{id}`` 的响应一致。"""

    id: str
    goal: str
    status: str = QUEUED
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    ended_at: str | None = None
    turns: int = 0
    tokens: int = 0
    repairs: int = 0
    model: str = ""
    current_tool: str = ""
    verdict: str | None = None
    error: str = ""
    resume_from: str | None = None
    engine_run_id: str | None = None
    trace_path: str | None = None

    # 提交时的一次性选项：max_turns 留空 = 用 engine.max_turns；plan_gate=False
    # = 照发计划卡但不等确认。只在本次进程内有效（重启后排队任务一律 interrupted）。
    max_turns: int | None = field(default=None, repr=False, compare=False)
    plan_gate: bool = field(default=True, repr=False, compare=False)

    # 运行时句柄（不进 jobs.json / 不进 API 响应）
    handle: Any = field(default=None, repr=False, compare=False)
    gate: Any = field(default=None, repr=False, compare=False)
    plan: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    stop_requested: bool = field(default=False, repr=False, compare=False)
    engine_status: str = field(default="", repr=False, compare=False)
    #: 引擎自己发过 run_end 没有；没发（建引擎就炸了）时由 JobManager 补一条
    saw_run_end: bool = field(default=False, repr=False, compare=False)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def transition(self, target: str) -> None:
        """状态迁移；非法迁移直接抛，不给"悄悄变成别的状态"留口子。"""
        if target == self.status:
            return
        if target not in ALLOWED_TRANSITIONS.get(self.status, frozenset()):
            raise IllegalTransition(f"{self.status} → {target} 不是合法迁移")
        self.status = target
        if target == RUNNING and not self.started_at:
            self.started_at = _now_iso()
        if target in TERMINAL:
            self.ended_at = _now_iso()

    def duration_s(self) -> float | None:
        begin = _parse_iso(self.started_at)
        if begin is None:
            return None
        end = _parse_iso(self.ended_at) or datetime.now(timezone.utc)
        return round(max(0.0, (end - begin).total_seconds()), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s(),
            "turns": self.turns,
            "tokens": self.tokens,
            "repairs": self.repairs,
            "model": self.model,
            "current_tool": self.current_tool,
            "verdict": self.verdict,
            "error": self.error,
            "resume_from": self.resume_from,
            "engine_run_id": self.engine_run_id,
            "trace_path": self.trace_path,
            "stop_requested": self.stop_requested,
            "plan": self.plan,
        }

    # 持久化只存"事实"，不存运行态句柄/计划卡（重启后那张卡已经没意义了）
    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "turns": self.turns,
            "tokens": self.tokens,
            "repairs": self.repairs,
            "model": self.model,
            "verdict": self.verdict,
            "error": self.error,
            "resume_from": self.resume_from,
            "engine_run_id": self.engine_run_id,
            "trace_path": self.trace_path,
        }

    @classmethod
    def from_record(cls, raw: dict[str, Any]) -> "Job":
        def _int(key: str) -> int:
            try:
                return int(raw.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        job = cls(
            id=str(raw.get("id") or ""),
            goal=str(raw.get("goal") or ""),
            status=str(raw.get("status") or QUEUED),
            created_at=str(raw.get("created_at") or _now_iso()),
            started_at=raw.get("started_at") or None,
            ended_at=raw.get("ended_at") or None,
            turns=_int("turns"),
            tokens=_int("tokens"),
            repairs=_int("repairs"),
            model=str(raw.get("model") or ""),
            verdict=raw.get("verdict") or None,
            error=str(raw.get("error") or ""),
            resume_from=raw.get("resume_from") or None,
            engine_run_id=raw.get("engine_run_id") or None,
            trace_path=raw.get("trace_path") or None,
        )
        return job


# ---------------------------------------------------------------- 运行句柄


class RunHandle(Protocol):
    """真实引擎/假引擎对外只需要这四个动作。"""

    def start(self) -> dict[str, Any]:
        """阻塞跑到结束（或暂停），返回 ``TaskState.to_dict()`` 形状的字典。"""

    def stop(self) -> bool:
        ...

    def inject(self, text: str) -> bool:
        ...

    def close(self) -> None:
        ...


RunnerFactory = Callable[[Job, Callable[[str, dict[str, Any]], None], ThreadingPlanGate], RunHandle]


def new_job_id() -> str:
    """``run_<时间>_<pid>_<进程内序号>``：trace 用同名格式，人眼能对上。"""
    global _JOB_SEQ
    with _JOB_SEQ_LOCK:
        _JOB_SEQ += 1
        seq = _JOB_SEQ
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{seq}"


_JOB_SEQ = 0
_JOB_SEQ_LOCK = threading.Lock()


# ---------------------------------------------------------------- 管理器


class JobManager:
    """排队、串行执行、持久化与三个控制钩子（stop / inject / plan）。"""

    def __init__(
        self,
        workspace: str | Path,
        *,
        bus: EventBus | None = None,
        runner: RunnerFactory | None = None,
        config_path: str | Path | None = None,
        plan_timeout: float = DEFAULT_TIMEOUT,
        jobs_path: str | Path | None = None,
        persist: bool = True,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.bus = bus or EventBus()
        self.config_path = config_path
        self.plan_timeout = float(plan_timeout)
        self._runner = runner or make_engine_runner(self.workspace, config_path)
        self._jobs_path = (
            Path(jobs_path) if jobs_path is not None else self.workspace / JOBS_FILE
        )
        self._persist_enabled = bool(persist)
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._worker: threading.Thread | None = None
        self._closed = False
        self._last_persist = 0.0
        self._persist_interval = 1.0
        self._load()

    # ------------------------------------------------------------- 生命周期
    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            if self._closed:
                raise JobError("JobManager 已关闭")
            self._worker = threading.Thread(
                target=self._worker_loop, name="gis-agent-job-worker", daemon=True
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait(0.5)
                if self._closed:
                    return
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status != QUEUED:
                    continue  # 排队期间被 stop 掉的，不再执行
            try:
                self._run_job(job)
            except Exception:  # pragma: no cover - 兜底：worker 不能死
                logger.exception("job %s crashed", job.id)
                self._safe_fail(job, "内部错误，见服务端日志")

    def close(self, timeout: float = 2.0) -> None:
        """停止接收新任务；正在跑的 run 在 ``jobs.json`` 里标成 ``interrupted``。"""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)
        with self._lock:
            for job in self._jobs.values():
                if job.status in _LIVE_STATUSES:
                    try:
                        job.transition(INTERRUPTED)
                        job.error = job.error or "服务关闭，运行已中断"
                    except IllegalTransition:  # pragma: no cover - 防御性
                        pass
            self._persist_locked(force=True)

    # ------------------------------------------------------------- 提交/查询
    def submit(
        self,
        goal: str,
        *,
        resume_from: str | None = None,
        max_turns: int | None = None,
        plan_gate: bool = True,
    ) -> Job:
        text = str(goal or "").strip()
        if not text:
            raise ValueError("goal 不能为空")
        if max_turns is not None:
            try:
                max_turns = int(max_turns)
            except (TypeError, ValueError):
                raise ValueError("max_turns 必须是整数") from None
            if max_turns < 1:
                raise ValueError("max_turns 必须 ≥ 1（留空表示用配置默认）")
        if resume_from:
            original = self.get(resume_from)
            if not original.terminal:
                raise JobConflict("原 run 还没结束，不能续跑")
            text = text or original.goal
        self._ensure_worker()
        job = Job(
            id=new_job_id(),
            goal=text,
            resume_from=resume_from,
            max_turns=max_turns,
            plan_gate=bool(plan_gate),
        )
        with self._cond:
            if self._closed:
                raise JobError("JobManager 已关闭")
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._persist_locked(force=True)
            self._cond.notify_all()
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(str(job_id))
            if job is None:
                raise JobNotFound(f"run 不存在: {job_id}")
            return job

    def get_or_none(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(str(job_id))

    def list_jobs(self, *, status: str | None = None, limit: int = 100) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            wanted = {item.strip() for item in str(status).split(",") if item.strip()}
            jobs = [job for job in jobs if job.status in wanted]
        jobs.sort(key=lambda item: str(item.created_at), reverse=True)
        return jobs[: max(1, int(limit))]

    def queue_position(self, job_id: str) -> int | None:
        with self._lock:
            try:
                return list(self._queue).index(job_id)
            except ValueError:
                return None

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue)

    def wait_for_idle(self, timeout: float = 30.0) -> bool:
        """等到没有排队/运行中的任务（测试与优雅退出用）。"""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                busy = any(
                    job.status in (QUEUED, RUNNING, AWAITING_PLAN, STOPPING) for job in self._jobs.values()
                )
                if not busy:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(min(remaining, 0.2))

    # ------------------------------------------------------------- 控制钩子
    def stop(self, job_id: str, reason: str = "用户中断") -> Job:
        """请求停止。重复调用幂等。

        正在跑（running / awaiting_plan）：立即迁到 ``stopping`` 并推 ``stop_requested``
        SSE，再调引擎 ``request_stop()``；引擎在下一个检查点退出，worker 拿到结果后
        由 ``_finish_job`` 转 ``stopped``（stop 到得晚、引擎已跑完则转 succeeded）。
        排队中：直接 ``stopped``，一步都不执行。
        """
        job = self.get(job_id)
        with self._lock:
            if job.terminal:
                raise JobConflict("run 已结束，不能停止")
            if job.status == STOPPING:
                return job  # 重复 stop：已经停过了，不再迁移、不再推事件
            job.stop_requested = True
            status = job.status
            handle = job.handle
            gate = job.gate
            if status in (RUNNING, AWAITING_PLAN):
                # 立刻进 stopping：界面/SSE 不该等当前工具返回（B4：以前要等到
                # 当前工具（最长 600s）返回，run 才变 stopped）。
                self._transition_locked(job, STOPPING)
        if status == QUEUED:
            # 还没轮到的 run 直接取消，worker 出队时看到状态不是 queued 就跳过
            with self._lock:
                self._transition_locked(job, STOPPED, error=reason)
            self.bus.push(job.id, "stop_requested", {"reason": reason})
            self._emit_run_end(job, STOPPED, error=reason)
            self.bus.close(job.id)
            self._notify_idle()
            return job
        if status in (RUNNING, AWAITING_PLAN):
            # SSE 立刻推：前端靠它把按钮切成「正在停止…」
            self.bus.push(job.id, "stop_requested", {"reason": reason})
        if status == AWAITING_PLAN and gate is not None:
            # 计划卡前停止：让引擎拿到 cancel，走正常的"未开始执行"退出路径
            gate.decide("cancel", reason)
        if handle is not None:
            try:
                handle.stop()
            except Exception:  # pragma: no cover - 引擎侧异常不影响 stop 请求
                logger.exception("stop failed for %s", job.id)
        return job

    def inject(self, job_id: str, text: str) -> Job:
        content = str(text or "").strip()
        if not content:
            raise ValueError("插话内容不能为空")
        job = self.get(job_id)
        with self._lock:
            handle = job.handle
            status = job.status
            if status not in (RUNNING, AWAITING_PLAN) or handle is None:
                raise JobConflict("run 未在运行，无法插话")
        if not handle.inject(content):
            raise JobConflict("引擎当前不接受插话（可能已到收尾阶段）")
        return job

    def decide_plan(self, job_id: str, action: str, text: str = "") -> Job:
        job = self.get(job_id)
        with self._lock:
            gate = job.gate
            if job.status != AWAITING_PLAN or gate is None or job.plan is None:
                raise JobConflict("当前没有等待决策的计划卡")
        try:
            accepted = gate.decide(action, text)
        except ValueError as exc:
            raise JobConflict(str(exc)) from exc
        if not accepted:
            raise JobConflict("计划卡已经决策过（或已超时自动开始）")
        return job

    def resume(self, job_id: str) -> Job:
        """从断点续跑：新建一个 run（trace 是新的，原 run 的历史保留）。"""
        original = self.get(job_id)
        if not original.terminal:
            raise JobConflict("run 还没结束，不能续跑")
        if original.status == SUCCEEDED:
            raise JobConflict("run 已成功结束，无需续跑")
        return self.submit(original.goal, resume_from=original.id)

    # ------------------------------------------------------------- 执行
    def _make_emitter(self, job: Job) -> Callable[[str, dict[str, Any]], None]:
        def emit(event: str, payload: dict[str, Any] | None = None) -> None:
            data = payload if isinstance(payload, dict) else {}
            self.bus.push(job.id, event, data)
            self._track(job, event, data)

        return emit

    def _track(self, job: Job, event: str, payload: dict[str, Any]) -> None:
        """从事件流累积状态条要的数字（不做任何渲染）。"""
        changed = False
        with self._lock:
            if event == "run_start":
                run_id = str(payload.get("run_id") or "")
                if run_id and job.engine_run_id != run_id:
                    job.engine_run_id = run_id
                    job.trace_path = str(
                        self.workspace / ".gis_agent" / "traces" / f"{run_id}.jsonl"
                    )
                    changed = True
            elif event == "turn":
                try:
                    job.turns = max(job.turns, int(payload.get("turn") or 0))
                except (TypeError, ValueError):
                    pass
                changed = True
            elif event == "llm_result":
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                try:
                    job.tokens += int((usage or {}).get("total_tokens") or 0)
                except (TypeError, ValueError):
                    pass
                model = str(payload.get("model") or "")
                if model:
                    job.model = model
                changed = True
            elif event == "tool_call":
                job.current_tool = str(payload.get("tool") or "")
                changed = True
            elif event == "tool_result":
                job.current_tool = ""
                changed = True
            elif event == "repair_attempt":
                job.repairs += 1
                changed = True
            elif event == "verification":
                job.verdict = "pass" if payload.get("pass") is True else "fail"
                changed = True
            elif event in {"run_end", "run_error"}:
                if event == "run_end":
                    job.saw_run_end = True
                job.engine_status = str(payload.get("status") or ("failed" if event == "run_error" else ""))
                # 实时 run_end 是 TaskState.to_dict()（turn 单数 / usage 汇总），
                # trace 里的 run_end 是 RunRecorder.finish()（turns / total_tokens）。
                turns_value = payload.get("turns", payload.get("turn"))
                if turns_value is not None:
                    try:
                        job.turns = int(turns_value or job.turns)
                    except (TypeError, ValueError):
                        pass
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                tokens_value = payload.get("total_tokens") or (usage or {}).get("total_tokens")
                if tokens_value is not None:
                    try:
                        job.tokens = int(tokens_value or job.tokens)
                    except (TypeError, ValueError):
                        pass
                job.current_tool = ""
                changed = True
        if changed:
            self._persist_throttled()

    def _run_job(self, job: Job) -> None:
        with self._lock:
            if job.status != QUEUED:
                return
            self._transition_locked(job, RUNNING)

        # plan_gate=False：不装闸门，引擎照发计划卡事件但不等确认（loop 里
        # ``plan_gate is None`` 直接返回 True）。
        gate: ThreadingPlanGate | None = None
        if job.plan_gate:
            gate = ThreadingPlanGate(
                timeout=self.plan_timeout,
                on_pending=lambda card: self._plan_pending(job, card),
                on_resolved=lambda decision, timed_out: self._plan_resolved(job, decision, timed_out),
            )
        with self._lock:
            job.gate = gate

        self.bus.push(job.id, "note", {"text": "启动引擎（加载数据目录/内核）…"})
        state: dict[str, Any] = {}
        handle: RunHandle | None = None
        try:
            handle = self._runner(job, self._make_emitter(job), gate)
            with self._lock:
                job.handle = handle
                stop_before_start = job.stop_requested
            if stop_before_start:
                # 建引擎期间就点了停止：一步都不执行
                state = {"status": "stopped", "error": "启动前已请求停止"}
            else:
                state = handle.start() or {}
        except Exception as exc:
            logger.exception("run %s failed", job.id)
            state = {"status": "failed", "error": str(exc)[:2000]}
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:  # pragma: no cover - 关引擎失败不影响状态落库
                    logger.exception("close failed for %s", job.id)
            with self._lock:
                job.handle = None
                job.gate = None
                job.plan = None
            self._finish_job(job, state)

    def _finish_job(self, job: Job, state: dict[str, Any]) -> None:
        raw_status = str((state or {}).get("status") or job.engine_status or "")
        target = ENGINE_STATUS_MAP.get(raw_status)
        if target is None:
            target = STOPPED if job.stop_requested else FAILED
        error = str((state or {}).get("error") or "")

        with self._lock:
            if job.terminal:
                return
            if (state or {}).get("turn") is not None:
                try:
                    job.turns = int(state.get("turn") or job.turns)
                except (TypeError, ValueError):
                    pass
            job.current_tool = ""
            try:
                self._transition_locked(job, target, error=error)
            except IllegalTransition:  # pragma: no cover - 并发 stop 兜底
                logger.warning("job %s: %s → %s rejected", job.id, job.status, target)
                return
            saw_run_end = job.saw_run_end
        # 引擎自己会发 run_end；异常/启动前停止这类路径没有，补一条让订阅者收尾。
        if not saw_run_end:
            self._emit_run_end(job, target, error=error)
        self.bus.close(job.id)
        self._notify_idle()

    def _plan_pending(self, job: Job, card: Any) -> None:
        payload: dict[str, Any] = {}
        try:
            if hasattr(card, "to_dict"):
                payload = dict(card.to_dict())
            if hasattr(card, "render"):
                # 引擎发事件时用 interactive=True 渲染（带按键提示），这里保持一致
                payload["text"] = card.render(interactive=True)
        except Exception:  # pragma: no cover - 计划卡是纯文本渲染，不该失败
            payload = {"text": str(card)}
        payload.setdefault("round", 1)
        payload["deadline"] = _iso_in(self.plan_timeout)
        with self._lock:
            job.plan = payload
            if job.status == RUNNING:
                self._transition_locked(job, AWAITING_PLAN)
        self._persist_throttled()

    def _plan_resolved(self, job: Job, decision: Any, timed_out: bool) -> None:
        with self._lock:
            job.plan = None
            if job.status == AWAITING_PLAN:
                self._transition_locked(job, RUNNING)
        if timed_out:
            self.bus.push(
                job.id,
                "plan_gate_timeout",
                {"reason": f"计划卡 {int(self.plan_timeout)}s 无决策，已自动开始"},
            )
        self._persist_throttled()

    def _safe_fail(self, job: Job, message: str) -> None:
        with self._lock:
            if job.terminal:
                return
            try:
                self._transition_locked(job, FAILED, error=message)
            except IllegalTransition:  # pragma: no cover
                return
        self._emit_run_end(job, FAILED, error=message)
        self.bus.close(job.id)
        self._notify_idle()

    def _emit_run_end(self, job: Job, status: str, *, error: str = "") -> None:
        self.bus.push(
            job.id,
            "run_end",
            {
                "status": status,
                "goal": job.goal,
                "turns": job.turns,
                "total_tokens": job.tokens,
                "error": error,
                "synthetic": True,
            },
        )

    # ------------------------------------------------------------- 状态/持久化
    def _transition_locked(self, job: Job, target: str, *, error: str | None = None) -> None:
        job.transition(target)
        if error:
            job.error = str(error)[:2000]
        self._persist_locked(force=True)

    def _notify_idle(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def _persist_throttled(self) -> None:
        if not self._persist_enabled:
            return
        now = time.monotonic()
        if now - self._last_persist < self._persist_interval:
            return
        with self._lock:
            self._persist_locked()

    def _persist_locked(self, *, force: bool = False) -> None:
        if not self._persist_enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_persist < self._persist_interval:
            return
        payload = {"version": 1, "jobs": [job.to_record() for job in self._jobs.values()]}
        try:
            self._jobs_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._jobs_path.with_name(self._jobs_path.name + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._jobs_path)
            self._last_persist = now
        except OSError as exc:  # pragma: no cover - 磁盘满/权限问题不该拖垮 worker
            logger.warning("jobs.json 写入失败: %s", exc)

    def _load(self) -> None:
        if not self._persist_enabled or not self._jobs_path.is_file():
            return
        try:
            data = json.loads(self._jobs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("jobs.json 读取失败（按空历史启动）: %s", exc)
            return
        records: Iterable[Any] = data.get("jobs", []) if isinstance(data, dict) else []
        restored: list[Job] = []
        for raw in records:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            try:
                job = Job.from_record(raw)
            except Exception:  # pragma: no cover - 单条坏记录跳过
                continue
            if job.status in _LIVE_STATUSES:
                # 进程已经死了，不可能还在跑：留个诚实的状态，不假装它还在运行
                job.status = INTERRUPTED
                job.ended_at = job.ended_at or _now_iso()
                job.error = job.error or "服务重启，运行已中断"
                job.current_tool = ""
            restored.append(job)
        with self._lock:
            for job in restored:
                self._jobs[job.id] = job
            self._persist_locked(force=True)


def _iso_in(seconds: float) -> str:
    from datetime import timedelta

    return (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, float(seconds)))).isoformat()


# ---------------------------------------------------------------- 真实引擎


class _EngineHandle:
    """把一个 ``EngineBundle`` 适配成 :class:`RunHandle`（照抄 CLI 的装配路径）。"""

    def __init__(self, bundle: Any, *, goal: str, resume: bool) -> None:
        self.bundle = bundle
        self.goal = goal
        self.resume = resume

    def start(self) -> dict[str, Any]:
        loop = self.bundle.loop
        state = loop.resume_from_checkpoint() if self.resume else loop.run(self.goal)
        if hasattr(state, "to_dict"):
            return dict(state.to_dict())
        return dict(state)

    def stop(self) -> bool:
        try:
            return bool(self.bundle.loop.request_stop())
        except Exception:  # pragma: no cover
            return False

    def inject(self, text: str) -> bool:
        try:
            return bool(self.bundle.loop.inject_user_message(text))
        except Exception:  # pragma: no cover
            return False

    def close(self) -> None:
        self.bundle.close()


def engine_overrides_for(job: Job) -> dict[str, Any] | None:
    """每-run 覆盖：只有页面显式填了 max_turns 才覆盖 ``engine.max_turns``。"""
    if job.max_turns is None:
        return None
    return {"max_turns": int(job.max_turns)}


def make_engine_runner(workspace: str | Path, config_path: str | Path | None = None) -> RunnerFactory:
    """真实引擎装配：与 ``agent/cli.py:_run_engine_task()`` 同一条路径。

    引擎只在第一次真正跑任务时 import（``gis_cli.engine`` 会拉起 openai/arcpy
    相关模块），所以只读控制台仍能在无 arcpy 环境里启动。
    """
    ws = Path(workspace)

    def _factory(job: Job, emit: Callable[[str, dict[str, Any]], None], gate: ThreadingPlanGate | None) -> RunHandle:
        import importlib

        from ..engine.runner import build_engine

        # 引擎日志落 .gis_agent/logs/engine.log（照 CLI）；``gis_cli.ui`` 的包初始化
        # 会连带 import rich，所以用 importlib 而不是 import 语句 —— 保持
        # ``gis_cli.web`` 的静态 import 面干净（见 tests/web/test_import_guard.py）。
        logging_setup = importlib.import_module("gis_cli.ui.logging_setup")
        logging_setup.setup_engine_logging(ws)

        bundle = build_engine(
            workspace=ws,
            config_path=config_path,
            refresh_catalog=True,
            on_event=emit,
            engine_overrides=engine_overrides_for(job),
            plan_gate=gate,
        )
        return _EngineHandle(bundle, goal=job.goal, resume=bool(job.resume_from))

    return _factory


# ---------------------------------------------------------------- HTTP 路由


#: FastAPI 用 get_type_hints 解析注解，闭包里的类解析不到 —— 请求体模型必须放模块级。
class CreateRun(BaseModel):
    goal: str
    resume_from: str | None = None
    #: 留空 = 用 ``engine.max_turns``（页面上的「最大轮次」）
    max_turns: int | None = None
    #: False = 计划卡照发但不等确认，直接开始
    plan_gate: bool = True


class InjectBody(BaseModel):
    text: str


class PlanBody(BaseModel):
    action: str = "start"
    text: str = ""


#: 引擎 run_end.status → 控制台 run 状态。直接复用 live 口径（ENGINE_STATUS_MAP），
#: 否则同一个 run 重启前后徽章会变（`cancelled` 被当成 `interrupted`、
#: `deadline_exceeded`/`max_turns` 被当成中断而不是失败），DAIR-271 P2。
_TRACE_STATUS: dict[str, str] = {**ENGINE_STATUS_MAP, "completed": SUCCEEDED}

#: run_end 之外的错误事件（按出现顺序，后面的覆盖前面的）。
_TRACE_ERROR_EVENTS = {"llm_error", "run_error", "deadline_exceeded"}


def trace_run_detail(workspace: str | Path, run_id: str) -> dict[str, Any] | None:
    """用 trace 拼一份**只读** run 详情；trace 也不存在时返回 None。

    ``/runs`` 列表的行来自 trace（``run_id`` 是 trace 文件名），而详情路由原本只认
    ``jobs.json`` 里当前进程的 run —— 重启前/直接粘贴 trace id 的详情页于是 404。
    字段名与 Job.to_dict() 对齐，多一个 ``readonly: True`` 供前端隐藏控制按钮；
    时间线由 ``/api/runs/{id}/events`` 从 trace 回放，接口不用再吐一遍事件。
    """
    from . import trace_api

    path = Path(workspace).resolve() / trace_api.TRACE_DIR / f"{run_id}.jsonl"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None

    # 用 read_events + summarize 而不是 load_trace：详情页不需要 .md 报告，
    # 后者会把整份报告也读进内存再丢掉。
    events = trace_api.read_events(workspace, run_id)
    trace = trace_api.summarize(run_id, events, mtime)

    error = ""
    for event in events:
        name = event.get("event")
        payload = event.get("payload") or {}
        if name in _TRACE_ERROR_EVENTS:
            error = str(payload.get("error") or payload.get("reason") or "") or error
    # 「多次验收未通过」/坏配置收尾这类只在 run_end.payload.error 里带原因的 run：
    # 不兜这一条，详情页的 error 就是空的（DAIR-271 建议 3）。
    for event in reversed(events):
        if event.get("event") == "run_end":
            payload = event.get("payload") or {}
            error = str(payload.get("error") or "") or error
            break
    models = trace.get("models") or {}
    started_at = trace.get("started_at")
    return {
        "id": run_id,
        "goal": str(trace.get("goal") or ""),
        "status": _TRACE_STATUS.get(str(trace.get("status") or ""), INTERRUPTED),
        "created_at": started_at,
        "started_at": started_at,
        "ended_at": trace.get("finished_at"),
        "duration_s": trace.get("duration_s"),
        "turns": int(trace.get("turns") or 0),
        "tokens": int(trace.get("tokens") or 0),
        "repairs": int(trace.get("repairs") or 0),
        "model": max(models, key=models.get) if models else "",
        "current_tool": "",
        "verdict": trace.get("verdict"),
        "error": error[:500],
        "resume_from": None,
        "engine_run_id": run_id,
        "trace_path": str(
            Path(workspace).resolve() / trace_api.TRACE_DIR / f"{run_id}.jsonl"
        ),
        "stop_requested": False,
        "plan": None,
        #: 只读标记：前端据此隐藏停止/续跑/插话/计划卡
        "readonly": True,
        "source": "trace",
        "last_seq": 0,
    }


def build_router(manager: JobManager):
    """``/api/runs`` 的五个路由 + 三个控制钩子（token 中间件在 ``app.py``）。"""
    from fastapi import APIRouter, HTTPException

    def _error(exc: JobError) -> HTTPException:
        status = 404 if isinstance(exc, JobNotFound) else 409
        return HTTPException(status_code=status, detail=str(exc))

    router = APIRouter()

    @router.post("/api/runs", status_code=202)
    def create_run(body: CreateRun) -> dict[str, Any]:
        try:
            job = manager.submit(
                body.goal,
                resume_from=body.resume_from,
                max_turns=body.max_turns,
                plan_gate=body.plan_gate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except JobError as exc:
            raise _error(exc) from exc
        return {"run": job.to_dict()}

    @router.get("/api/runs")
    def list_runs(status: str | None = None, limit: int = 50) -> dict[str, Any]:
        jobs = manager.list_jobs(status=status, limit=limit)
        return {
            "items": [job.to_dict() for job in jobs],
            "count": len(jobs),
            "queue": manager.queue_length,
        }

    @router.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            job = manager.get(run_id)
        except JobNotFound as exc:
            # 服务重启过 / 列表里点进来的是 trace id：jobs.json 不认这些 run，
            # 但 trace 还在磁盘上。回退成只读详情，别让用户以为数据丢了（DAIR-270）。
            detail = trace_run_detail(manager.workspace, run_id)
            if detail is None:
                raise _error(exc) from exc
            return detail
        payload = job.to_dict()
        payload["last_seq"] = manager.bus.last_seq(job.id)
        return payload

    @router.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> dict[str, Any]:
        try:
            job = manager.stop(run_id)
        except JobError as exc:
            raise _error(exc) from exc
        return {"ok": True, "run": job.to_dict()}

    @router.post("/api/runs/{run_id}/inject")
    def inject_run(run_id: str, body: InjectBody) -> dict[str, Any]:
        try:
            job = manager.inject(run_id, body.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except JobError as exc:
            raise _error(exc) from exc
        return {"ok": True, "run": job.to_dict()}

    @router.post("/api/runs/{run_id}/plan")
    def decide_plan(run_id: str, body: PlanBody) -> dict[str, Any]:
        try:
            job = manager.decide_plan(run_id, body.action, body.text)
        except JobError as exc:
            raise _error(exc) from exc
        return {"ok": True, "run": job.to_dict()}

    @router.post("/api/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str) -> dict[str, Any]:
        try:
            job = manager.resume(run_id)
        except JobError as exc:
            raise _error(exc) from exc
        return {"run": job.to_dict()}

    return router


__all__ = [
    "ALLOWED_TRANSITIONS",
    "AWAITING_PLAN",
    "ENGINE_STATUS_MAP",
    "FAILED",
    "INTERRUPTED",
    "IllegalTransition",
    "Job",
    "JobConflict",
    "JobError",
    "JobManager",
    "JobNotFound",
    "build_router",
    "QUEUED",
    "RUNNING",
    "STATUSES",
    "STOPPED",
    "SUCCEEDED",
    "TERMINAL",
    "make_engine_runner",
    "new_job_id",
]
