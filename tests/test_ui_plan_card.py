# -*- coding: utf-8 -*-
"""分析计划卡（P0-3）：5 行人话 + 执行前确认闸门。

不需要 ArcGIS：计划卡纯渲染，闸门只读注入的输入函数。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gis_cli.datacatalog.store import DatasetRecord
from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.state import Requirement
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.trace.recorder import RunRecorder
from gis_cli.ui.plan_card import (
    ChatPlanGate,
    PlanCard,
    PlanGate,
    build_plan_card,
    parse_decision,
)

GOAL = (
    "按街道边界汇总人口.csv 里的老年人口占比并出专题图，"
    "结果存到 output/老年人口分布.gdb"
)


def _records(workspace: Path) -> list[DatasetRecord]:
    return [
        DatasetRecord(path=str(workspace / "input" / "街道边界.shp"), kind="vector", name="街道边界"),
        DatasetRecord(path=str(workspace / "input" / "人口.csv"), kind="table", name="人口"),
    ]


def _reqs() -> list[Requirement]:
    return [
        Requirement(id="r1", text="按街道汇总老年人口", acceptance="要素数 12 个", source="extracted"),
        Requirement(id="r2", text="计算老年人口占比", acceptance="占比合计 100%", source="extracted"),
    ]


# ------------------------------------------------------------------ 渲染


def test_card_has_five_rows_and_data_traces_back_to_catalog(tmp_path: Path):
    card = build_plan_card(GOAL, _reqs(), _records(tmp_path), workspace=tmp_path)
    rows = card.rows()

    assert [label for label, _ in rows] == ["做什么", "用什么", "输出", "怎么验", "合规"]
    data_line = dict(rows)["用什么"]
    # 数据来源必须能追到 catalog 记录（相对路径 + 已登记数）
    assert "input/街道边界.shp" in data_line and "input/人口.csv" in data_line
    assert "已登记 2 个数据集" in data_line
    # 输出从任务描述里取，且不能把输入数据当产出
    outputs = dict(rows)["输出"]
    assert "output/老年人口分布.gdb" in outputs
    assert "input/" not in outputs
    # 怎么验来自需求里的验收标准
    assert "要素数 12 个" in dict(rows)["怎么验"]
    assert "占比合计 100%" in dict(rows)["怎么验"]

    text = card.render(interactive=True)
    for label in ("做什么", "用什么", "输出", "怎么验", "合规"):
        assert f"{label}：" in text
    assert "┌ 计划确认" in text and "[s]/Enter 开始" in text


def test_card_wont_invent_outputs(tmp_path: Path):
    """没扫过 catalog（首次任务）时不能把已有数据当产出，也不能把裸后缀当路径。"""
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "用地.tif").write_bytes(b"x")

    card = build_plan_card(
        "把用地.tif 重分类后导出到 output/结果.tif，另外把统计结果存成 结果.gdb，再导出为 .shp",
        [],
        [],  # catalog 为空：只剩磁盘侧判据
        workspace=tmp_path,
    )
    outputs = dict(card.rows())["输出"]
    items = [x.strip() for x in outputs.split(" + ")]
    assert "output/结果.tif" in items
    assert "用地.tif" not in items, "已有数据（磁盘上真实存在）不能当成产出"
    assert "结果.gdb" not in items, "没有目录的文件名说不清是输入还是产物，不写"
    assert ".shp" not in items, "只有后缀没有文件名的 token 不是路径"


def test_card_without_catalog_data_says_so(tmp_path: Path):
    card = build_plan_card(GOAL, [], [], workspace=tmp_path)
    data_line = dict(card.rows())["用什么"]
    assert "执行时先确认" in data_line
    assert dict(card.rows())["怎么验"] == "按需求清单逐条核对"
    assert "未确认，直接开始" in card.render()  # 非交互：只打印，不等输入


def test_card_lists_inputs_even_when_goal_has_no_filename(tmp_path: Path):
    """外行不会写全文件名：对不上名字时退回 input/ 下已登记的数据，但要标成候选。"""
    card = build_plan_card("给这些数据做个地图", [], _records(tmp_path), workspace=tmp_path)
    data_line = dict(card.rows())["用什么"]
    assert "街道边界.shp" in data_line and "人口.csv" in data_line
    assert "候选" in data_line, "猜出来的数据不能冒充“任务点名要的”"


# ------------------------------------------------------------------ 闸门


def test_gate_non_interactive_never_reads_stdin(tmp_path: Path):
    def _boom(_prompt: str) -> str:  # pragma: no cover - 调用即失败
        raise AssertionError("非交互模式不能读 stdin（CI / 脚本会被挂住）")

    gate = PlanGate(interactive=False, input_fn=_boom)
    decision = gate.ask(build_plan_card(GOAL, _reqs(), _records(tmp_path), workspace=tmp_path))
    assert decision.action == "start"


def test_gate_needs_tty_on_both_ends(monkeypatch):
    """stdin 是 TTY 但 stdout 被重定向（``loop "…" | tee run.log``）时不能等人：
    提示进了管道，屏幕上一个字都看不到，进程却停在原地等 Enter。"""
    import io as _io

    from gis_cli.ui import plan_card as pc

    class _Tty(_io.StringIO):
        def isatty(self) -> bool:
            return True

    # tty 判定已收口到 ``console_env``（DAIR-38）：直接打真实流，三处共用同一份实现
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("sys.stdout", _io.StringIO())
    assert PlanGate().interactive is False

    monkeypatch.setattr("sys.stdout", _Tty())
    assert PlanGate().interactive is True
    assert pc.can_ask() is True


def test_gate_edit_then_start(tmp_path: Path):
    answers = iter(["e", "按街道统计，不要出图", ""])
    asked: list[str] = []

    def _input(prompt: str) -> str:
        asked.append(prompt)
        return next(answers)

    gate = PlanGate(interactive=True, input_fn=_input)
    card = build_plan_card(GOAL, _reqs(), _records(tmp_path), workspace=tmp_path)

    first = gate.ask(card)
    assert (first.action, first.text) == ("revise", "按街道统计，不要出图")
    assert asked[1].startswith("怎么改")  # 追问一句改动，而不是让用户自己拼格式

    second = gate.ask(card)
    assert second.action == "start"


def test_gate_accepts_plain_sentence_as_revision(tmp_path: Path):
    gate = PlanGate(interactive=True, input_fn=lambda _p: "只统计城关镇")
    decision = gate.ask(build_plan_card(GOAL, _reqs(), _records(tmp_path), workspace=tmp_path))
    assert decision.action == "revise" and decision.text == "只统计城关镇"


def test_gate_quit_cancels(tmp_path: Path):
    gate = PlanGate(interactive=True, input_fn=lambda _p: "q")
    decision = gate.ask(build_plan_card(GOAL, _reqs(), _records(tmp_path), workspace=tmp_path))
    assert decision.action == "cancel"


# ------------------------------------------------------------------ 循环钩子


class ScriptedLLM:
    class _Cfg:
        model = "mock"

    config = _Cfg()

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content=json.dumps({"tool": "finish", "args": {"summary": "done"}}), model="mock")


class CodecSpy:
    """转发给真 codec，只记下每轮**真的发出去**的 messages。

    断言“模型看到了什么”必须看这里，而不是 trace 里的 ``system_prompt.requirements``
    （那是 digest）：哪天 ``_system_prompt()`` 不再拼 digest，断言 trace 的用例仍会绿。
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self.seen: list[list[dict]] = []

    def protocol_for(self, model: str):
        return self._real.protocol_for(model)

    def call(self, messages, tools, **kwargs):
        self.seen.append([dict(message) for message in messages])
        return self._real.call(messages, tools, **kwargs)


class StubVerifier:
    def verify(self, state):
        return {"pass": True, "summary": "验收通过"}


class FakeCatalog:
    def __init__(self, records, counts=None):
        self._records = records
        self._counts = counts

    def list_datasets(self, **_kwargs):
        return list(self._records)

    def counts(self):
        if self._counts is None:
            raise RuntimeError("no counts")
        return dict(self._counts)


class FakePrewarmer:
    """记录“扫描在卡之前就开始了”，并验证取消时不预热内核。"""

    def __init__(self, trace: list[str]):
        self.trace = trace
        self.busy = False

    def start(self) -> bool:
        self.trace.append("scan.start")
        return True

    def wait_ready(self, timeout=None):  # noqa: ARG002 - 与真对象同签名
        return None


class TraceKernel:
    def __init__(self, trace: list[str]):
        self.trace = trace

    def prewarm(self) -> None:
        self.trace.append("kernel.prewarm")


class ScriptedGate:
    """按脚本回答的闸门（记录每次拿到的卡，便于断言"改完重渲染"）。"""

    def __init__(self, decisions, interactive=True):
        self.decisions = list(decisions)
        self.cards: list[PlanCard] = []
        self.interactive = interactive

    def ask(self, card):
        from gis_cli.ui.plan_card import PlanDecision

        self.cards.append(card)
        return self.decisions.pop(0) if self.decisions else PlanDecision("start")


def _requirements_payload(*texts: str) -> LLMResponse:
    payload = {"requirements": [{"id": f"r{i}", "text": t, "acceptance": f"{t}已核对"} for i, t in enumerate(texts, 1)]}
    return LLMResponse(content=json.dumps(payload, ensure_ascii=False), model="mock")


def _finish(summary: str = "完成") -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id="c_finish", name="finish", arguments={"summary": summary}, raw_arguments="{}")],
        model="mock",
    )


def _make_loop(tmp_path: Path, responses, *, gate=None, catalog=None, events=None, code_runner=None,
               prewarmer=None, **cfg):
    llm = ScriptedLLM(responses)
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=llm,
        registry=build_default_registry(),
        context=EngineContext(workspace=tmp_path),
        code_runner=code_runner,
        catalog=catalog,
        verifier=StubVerifier(),
        recorder=RunRecorder("run_plan", tmp_path / "traces"),
        config=LoopConfig(max_turns=6, **cfg),
        on_event=(lambda name, payload: events.append((name, payload))) if events is not None else None,
        catalog_prewarmer=prewarmer,
        plan_gate=gate,
    )
    return loop, llm


def test_loop_prints_card_and_runs_without_gate(tmp_path: Path):
    """没注入闸门（无头 / 子代理）：卡照打，绝不阻塞。"""
    events: list[tuple[str, dict]] = []
    loop, llm = _make_loop(
        tmp_path,
        [_requirements_payload("按街道汇总老年人口"), _finish()],
        catalog=FakeCatalog(_records(tmp_path)),
        events=events,
    )
    state = loop.run(GOAL)

    assert state.status == "completed"
    cards = [payload for name, payload in events if name == "plan_card"]
    assert cards, "计划卡必须发出来（非 TTY 也要打印）"
    text = cards[0]["text"]
    for label in ("做什么", "用什么", "输出", "怎么验", "合规"):
        assert f"{label}：" in text
    assert "input/街道边界.shp" in text
    # trace 里也要留下这张卡（事后能回答"当时确认的是什么"）
    trace = (tmp_path / "traces" / "run_plan.jsonl").read_text(encoding="utf-8")
    assert '"plan_card"' in trace


def test_loop_gate_revision_reruns_extraction_and_rerenders(tmp_path: Path):
    from gis_cli.ui.plan_card import PlanDecision

    gate = ScriptedGate([PlanDecision("revise", "只统计城关镇"), PlanDecision("start")])
    events: list[tuple[str, dict]] = []
    loop, llm = _make_loop(
        tmp_path,
        [
            _requirements_payload("按街道汇总老年人口"),  # 首次抽取
            _requirements_payload("只统计城关镇的老年人口"),  # 改完重抽
            _finish(),
        ],
        gate=gate,
        catalog=FakeCatalog(_records(tmp_path)),
        events=events,
    )
    state = loop.run(GOAL)

    assert state.status == "completed"
    assert len(gate.cards) == 2, "改完之后必须重新渲染并再确认一次"
    assert "修改：只统计城关镇" in gate.cards[1].rows()[0][1]
    # 改动真的并回了需求清单（模型后续看到的是改过的清单）
    assert any("城关镇" in r.text for r in state.requirements)
    assert any("城关镇" in json.dumps(payload, ensure_ascii=False) for name, payload in events if name == "plan_card")


def test_loop_gate_cancel_stops_before_first_turn(tmp_path: Path):
    """取消后一轮都不跑（需求抽取在闸门之前，那一次往返已经花掉了）。"""
    from gis_cli.ui.plan_card import PlanDecision

    gate = ScriptedGate([PlanDecision("cancel")])
    events: list[tuple[str, dict]] = []
    loop, llm = _make_loop(tmp_path, [_requirements_payload("x")], gate=gate, events=events)
    state = loop.run(GOAL)

    assert state.status == "cancelled"
    assert "取消" in state.error
    assert not [name for name, _ in events if name in ("turn", "tool_call")], "取消了就不该进第一轮"
    assert any(name == "plan_cancelled" for name, _ in events)


def test_loop_gate_revision_cap_shows_final_card(tmp_path: Path):
    """改到上限还在改：不能跑一个用户没看过的计划（最后一版必须发出来）。"""
    from gis_cli.ui.plan_card import PlanDecision

    gate = ScriptedGate([PlanDecision("revise", f"第{i}次改") for i in range(1, 5)])
    events: list[tuple[str, dict]] = []
    loop, _ = _make_loop(
        tmp_path,
        [_requirements_payload("按街道汇总"), _requirements_payload("改1"), _requirements_payload("改2"), _finish()],
        gate=gate,
        events=events,
        plan_max_revisions=2,
    )
    state = loop.run(GOAL)

    assert state.status == "completed"
    cards = [payload for name, payload in events if name == "plan_card"]
    # 第 1 次展示 + 第 1 次改后展示 + 到上限后再补发最后一版
    assert len(cards) == 3
    assert "第2次改" in cards[-1]["text"]
    assert any(name == "plan_card_auto_start" for name, _ in events)


def test_loop_plan_card_can_be_disabled(tmp_path: Path):
    events: list[tuple[str, dict]] = []
    loop, _ = _make_loop(
        tmp_path,
        [_requirements_payload("x"), _finish()],
        events=events,
        plan_card=False,
    )
    state = loop.run(GOAL)

    assert state.status == "completed"
    assert not [name for name, _ in events if name == "plan_card"]


def test_plan_card_event_renders_as_plan_unit():
    """UI 层要认得这个事件，否则卡片在终端里根本不出现。"""
    from gis_cli.ui import events as ui_events

    unit = ui_events.to_unit("plan_card", {"text": "┌ 计划确认\n│ 做什么：x"})
    assert unit is not None and unit.kind == "plan"
    assert "做什么" in unit.text


def test_loop_reports_registered_total_from_catalog_counts(tmp_path: Path):
    """“已登记 N 个”要算全，不能只数本次列出的那几条（列表有 limit）。"""
    catalog = FakeCatalog(_records(tmp_path), counts={"vector": 2, "table": 5})
    events: list[tuple[str, dict]] = []
    loop, _ = _make_loop(tmp_path, [_requirements_payload("按街道汇总"), _finish()], catalog=catalog, events=events)
    loop.run(GOAL)

    card = [payload for name, payload in events if name == "plan_card"][0]
    assert "已登记 7 个数据集" in card["text"]


def test_loop_scans_catalog_before_card_and_prewarms_kernel_after(tmp_path: Path):
    """卡要看得到刚扫出来的数据；但内核不能赶在确认之前预热（取消就白烧）。"""
    trace_scan: list[str] = []
    trace_kernel: list[str] = []
    events: list[tuple[str, dict]] = []

    loop, _ = _make_loop(
        tmp_path,
        [_requirements_payload("按街道汇总"), _finish()],
        catalog=FakeCatalog(_records(tmp_path)),
        events=events,
        prewarmer=FakePrewarmer(trace_scan),
        code_runner=TraceKernel(trace_kernel),
    )
    # 用事件顺序卡住“扫描 → 卡 → 内核”这个顺序
    loop.run(GOAL)

    names = [name for name, _ in events]
    assert names.index("catalog_scan") < names.index("plan_card")
    assert trace_kernel == ["kernel.prewarm"]


def test_loop_cancel_does_not_prewarm_kernel(tmp_path: Path):
    from gis_cli.ui.plan_card import PlanDecision

    trace_kernel: list[str] = []
    gate = ScriptedGate([PlanDecision("cancel")])
    loop, _ = _make_loop(
        tmp_path,
        [_requirements_payload("x")],
        gate=gate,
        code_runner=TraceKernel(trace_kernel),
    )
    state = loop.run(GOAL)

    assert state.status == "cancelled"
    assert trace_kernel == [], "按 q 取消不该已经起过内核"


# ------------------------------------------------------------------ 暂停态（DAIR-30）

def test_card_never_claims_confirmation_before_it_happens(tmp_path: Path):
    """没按过键就不能写“已（自动）确认”：chat 以前就是这么糊弄过去的。"""
    card = build_plan_card(GOAL, _reqs(), _records(tmp_path), workspace=tmp_path)

    auto = card.render()  # 非交互 / --yes / 没注入闸门
    assert "未确认，直接开始" in auto
    assert "已自动确认" not in auto and "已确认" not in auto

    waiting = card.render(interactive=True)
    for word in ("[s]/Enter 开始", "[e] 修改", "[q] 取消"):
        assert word in waiting
    assert "已确认" not in waiting, "确认前底栏不得出现“已确认”"


def test_parse_decision_words_are_shared_by_both_gates():
    """loop 的 input 与 chat 的输入框必须认同一套动作词。"""
    assert parse_decision("").action == "start"  # 卡片上写着 Enter 开始
    for word in ("s", "S", "start", "开始", "确认"):
        assert parse_decision(word).action == "start"
    for word in ("q", "Q", "quit", "取消", "退出"):
        assert parse_decision(word).action == "cancel"
    for word in ("e", "edit", "修改"):
        assert parse_decision(word) is None, "e 只是开个头：还要再问一句怎么改"
    plain = parse_decision("只统计城关镇")
    assert (plain.action, plain.text) == ("revise", "只统计城关镇")


def test_chat_gate_parks_engine_thread_and_never_reads_stdin(tmp_path: Path):
    """chat 闸门：引擎线程停在 Event 上，等主线程（UI）答复。"""
    card = build_plan_card(GOAL, _reqs(), _records(tmp_path), workspace=tmp_path)
    gate = ChatPlanGate()
    assert gate.pauses is True
    answered: dict[str, object] = {}

    worker = threading.Thread(target=lambda: answered.setdefault("decision", gate.ask(card)), daemon=True)
    worker.start()

    deadline = time.monotonic() + 5
    while gate.pending is None and time.monotonic() < deadline:
        time.sleep(0.01)
    request = gate.pending
    assert request is not None and request.card is card
    assert worker.is_alive(), "没答复之前引擎线程必须还停着（还没开始执行）"

    assert request.resolve("revise", "只统计城关镇") is True
    assert request.resolve("start") is False, "重复按键不能覆盖第一个答案"
    worker.join(timeout=5)
    decision = answered["decision"]
    assert (decision.action, decision.text) == ("revise", "只统计城关镇")
    assert gate.pending is None


def test_chat_gate_non_interactive_degrades_without_blocking(tmp_path: Path):
    """非 TTY（管道 / 重定向）：不问，直接开始，行为与没有闸门时一致。"""
    card = build_plan_card(GOAL, _reqs(), _records(tmp_path), workspace=tmp_path)
    gate = ChatPlanGate(interactive=False)

    assert gate.ask(card).action == "start"
    assert gate.pending is None
    assert gate.pauses is False, "非交互降级不能发一个没人会答的暂停态"
    assert gate.cancel_pending() is False  # 没有挂起的请求


def test_chat_gate_cancel_pending_wakes_the_engine(tmp_path: Path):
    """Esc / 退出会话时要把闸门判成 cancel，否则线程一直等下去。"""
    card = build_plan_card(GOAL, _reqs(), _records(tmp_path), workspace=tmp_path)
    gate = ChatPlanGate()
    got: dict[str, object] = {}
    worker = threading.Thread(target=lambda: got.setdefault("decision", gate.ask(card)), daemon=True)
    worker.start()
    deadline = time.monotonic() + 5
    while gate.pending is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert gate.cancel_pending() is True
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert got["decision"].action == "cancel"


def _respond_from_script(gate: ChatPlanGate, replies: list[tuple[str, str]], peek=None):
    """模拟主线程：每次出现待答请求先看一眼再按脚本答复（daemon 线程）。"""

    def _run() -> None:
        deadline = time.monotonic() + 10
        while replies and time.monotonic() < deadline:
            request = gate.pending
            if request is None or request.answered:
                time.sleep(0.005)
                continue
            if peek is not None:
                peek(request)
            request.resolve(*replies.pop(0))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def test_loop_plan_pending_puts_the_revision_into_the_system_prompt(tmp_path: Path):
    """引擎真的进 ``plan_pending``；`e` 改的那句必须进第一轮 system prompt。"""
    holder: dict[str, Any] = {}
    planned_status: list[str] = []

    def _peek(_request):
        loop = holder.get("loop")
        if loop is not None:
            planned_status.append(str(getattr(loop.state, "status", "")))

    gate = ChatPlanGate()
    responder = _respond_from_script(gate, [("revise", "只统计城关镇"), ("start", "")], peek=_peek)
    events: list[tuple[str, dict]] = []
    loop, _llm = _make_loop(
        tmp_path,
        [
            _requirements_payload("按街道汇总老年人口"),  # 首次抽取
            _requirements_payload("只统计城关镇的老年人口"),  # e 改完重抽
            _finish(),
        ],
        gate=gate,
        catalog=FakeCatalog(_records(tmp_path)),
        events=events,
    )
    holder["loop"] = loop
    spy = CodecSpy(loop.codec)
    loop.codec = spy
    state = loop.run(GOAL)
    responder.join(timeout=10)

    assert state.status == "completed"
    assert planned_status and planned_status[0] == "plan_pending", "等人按键时 state 必须是暂停态"
    names = [name for name, _ in events]
    assert names.count("plan_pending") == 2, "两次确认都要进暂停态（e 改完必须再确认一次）"
    assert names.index("plan_pending") < names.index("plan_decided")
    assert not [name for name in names if name == "plan_card_auto_start"]

    records = [
        json.loads(line)
        for line in (tmp_path / "traces" / "run_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_event: dict[str, list[dict]] = {}
    for record in records:
        by_event.setdefault(record["event"], []).append(record["payload"])

    assert len(by_event["plan_pending"]) == 2
    assert by_event["plan_decision"][0]["action"] == "revise"
    assert by_event["plan_decision"][1]["action"] == "start"
    # 第二张卡上要能看到“修改：…”（重渲染，不是只存着）
    assert "修改：只统计城关镇" in by_event["plan_pending"][1]["text"]
    # **关键**：确认之后拼的 system prompt 里确实带着改过的那一句。
    # 断言口径：codec 真收到的第一条消息（模型看到的那一版），不是 trace 里的 digest。
    assert spy.seen, "引擎必须真的调过模型"
    assert "城关镇" in spy.seen[0][0]["content"]
    assert any("城关镇" in requirement.text for requirement in state.requirements)


def test_loop_plan_pending_is_only_used_by_the_chat_gate(tmp_path: Path):
    """loop 自己的 input 闸门（``PlanGate``）不算暂停态：它就在主线程读键盘。"""
    from gis_cli.ui.plan_card import PlanDecision

    gate = ScriptedGate([PlanDecision("start")])
    events: list[tuple[str, dict]] = []
    loop, _ = _make_loop(tmp_path, [_requirements_payload("x"), _finish()], gate=gate, events=events)
    loop.run(GOAL)
    assert not [name for name, _ in events if name in ("plan_pending", "plan_decided")]


def test_loop_gate_crash_does_not_block_the_run(tmp_path: Path):
    """闸门炸了（stdin 被关等）只能降级为“直接开始”，不能把任务拖死。"""

    class BoomGate:
        interactive = True

        def ask(self, card):  # noqa: ARG002
            raise ValueError("I/O operation on closed file")

    events: list[tuple[str, dict]] = []
    loop, _ = _make_loop(tmp_path, [_requirements_payload("x"), _finish()], gate=BoomGate(), events=events)
    state = loop.run(GOAL)

    assert state.status == "completed"
    assert any("计划确认不可用" in str(payload.get("reason", "")) for name, payload in events if name == "plan_card_auto_start")


def test_plan_decided_emit_failure_keeps_the_user_decision(tmp_path: Path, monkeypatch):
    """``finally`` 里发事件失败不能把用户刚按的决定吃掉（``q`` 被当成 start）。"""
    gate = ChatPlanGate()
    responder = _respond_from_script(gate, [("cancel", "")])
    loop, _ = _make_loop(tmp_path, [_requirements_payload("x")], gate=gate)
    real_emit = loop._emit

    def _flaky_emit(name: str, payload: dict) -> None:
        if name == "plan_decided":
            raise RuntimeError("rich 写终端失败")
        real_emit(name, payload)

    monkeypatch.setattr(loop, "_emit", _flaky_emit)
    state = loop.run(GOAL)
    responder.join(timeout=10)

    assert state.status == "cancelled", "按了 q 就必须取消：退出暂停态失败不能改判成 start"


def test_plan_card_fail_open_is_recorded_and_visible(tmp_path: Path, monkeypatch):
    """兜底 fail-open 要留痕：trace 一条 + 屏幕一条，不能只写 logger（DAIR-38）。"""
    events: list[tuple[str, dict]] = []
    loop, _ = _make_loop(tmp_path, [_requirements_payload("x"), _finish()], events=events)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("卡片渲染炸了")

    monkeypatch.setattr(loop, "_confirm_plan", _boom)
    state = loop.run(GOAL)

    assert state.status == "completed", "计划卡出错只能降级为直接开始"
    records = [
        json.loads(line)
        for line in (tmp_path / "traces" / "run_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed = [record for record in records if record["event"] == "plan_card_failed"]
    assert failed and "卡片渲染炸了" in failed[0]["payload"]["error"]
    reasons = [payload.get("reason", "") for name, payload in events if name == "plan_card_auto_start"]
    assert any("计划卡出错" in reason for reason in reasons)
