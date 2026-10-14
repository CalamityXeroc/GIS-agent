# -*- coding: utf-8 -*-
"""界面层纯逻辑单测（不需要真实 TTY）。

覆盖：事件→渲染单元映射、参数摘要、状态行累积、时长/token 格式化、
主题模式判定与 NO_COLOR、渲染组件冒烟、折叠输出缓存。
"""

from __future__ import annotations

import io
import os

import pytest

from gis_cli.ui import events as ev
from gis_cli.ui import render as rd
from gis_cli.ui import theme as th


# --------------------------------------------------------------- 事件映射


def test_tool_call_becomes_step_with_args_summary():
    unit = ev.to_unit("tool_call", {"tool": "run_recipe", "args": {"recipe_id": "kernel_density", "cell_size": 100}})
    assert unit is not None
    assert unit.kind == "step"
    assert unit.text == "run_recipe"
    assert "kernel_density" in unit.lines[0]
    assert "cell_size=100" in unit.lines[0]


def test_tool_call_code_arg_is_summarised_not_dumped():
    code = "import arcpy\n" + "x = 1\n" * 500
    unit = ev.to_unit("tool_call", {"tool": "execute_code", "args": {"code": code}})
    assert unit is not None
    line = unit.lines[0]
    assert "import arcpy" in line
    assert len(line) < 120  # 不能把整段代码塞进摘要


def test_tool_result_marks_failure():
    ok = ev.to_unit("tool_result", {"tool": "execute_code", "ok": True, "summary": "执行成功"})
    bad = ev.to_unit("tool_result", {"tool": "execute_code", "ok": False, "summary": "KeyError: 'x'"})
    assert ok is not None and ok.ok is True
    assert bad is not None and bad.ok is False
    assert "KeyError" in bad.text


def test_first_turn_is_suppressed_but_later_turns_are_shown():
    assert ev.to_unit("turn", {"turn": 1}) is None
    unit = ev.to_unit("turn", {"turn": 4, "tasklist": "- [x] 投影"})
    assert unit is not None and unit.kind == "turn"
    assert "第 4 轮" in unit.text


def test_verification_and_repair_and_error_units():
    verify = ev.to_unit("verification", {"pass": False, "summary": "验收未通过：缺产出"})
    assert verify is not None and verify.kind == "verify" and verify.ok is False
    repair = ev.to_unit("repair_attempt", {"attempt": 2, "error": "NameError: name 'dem' is not defined"})
    assert repair is not None and repair.kind == "warn" and "第 2 次" in repair.text
    err = ev.to_unit("llm_error", {"error": "Connection error"})
    assert err is not None and err.kind == "error"


def test_turn_header_keeps_the_number():
    """轮次分隔线要显示数字（曾经 split 取末段得到"轮"）。"""
    console = th.make_console("dark", no_color=True, force_terminal=False, width=80)
    buf = io.StringIO()
    console.file = buf
    console.print(rd.render_unit(ev.RenderUnit("turn", "第 12 轮")))
    out = buf.getvalue()
    assert "12" in out and "第 轮" not in out


def test_status_tracker_accumulates_usage_from_apply_only_once():
    """用量只在 apply 里累加一次（曾因调用点重复 add_usage 而翻倍）。"""
    tracker = ev.StatusTracker()
    payload = {"model": "m", "actions": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    tracker.apply("llm_result", payload)
    tracker.apply("llm_result", payload)
    assert tracker.tokens_in == 20 and tracker.tokens_out == 10


def test_run_end_unit_carries_artifacts_and_status():
    unit = ev.to_unit(
        "run_end",
        {
            "status": "completed",
            "artifacts": ["output/成果图/密度_1982_1986.jpg"],
            "tasklist_digest": "- [x] 核密度\n- [x] 出图",
            "methodology": "IDW 幂 2",
            "final_summary": "已交付 4 张图",
        },
    )
    assert unit is not None and unit.kind == "final" and unit.ok is True
    body = "\n".join(unit.lines)
    assert "密度_1982_1986.jpg" in body and "IDW" in body and "已交付" in body


def test_run_end_unit_builds_tasklist_from_array():
    """引擎 to_dict() 给的是 tasklist 数组（没有 digest）——卡片要自己能拼。"""
    unit = ev.to_unit(
        "run_end",
        {
            "status": "completed",
            "artifacts": ["output/a.jpg"],
            "tasklist": [
                {"id": "t1", "title": "清洗点位", "status": "done"},
                {"id": "t2", "title": "核密度", "status": "in_progress"},
                {"id": "t3", "title": "出图", "status": "failed"},
            ],
        },
    )
    assert unit is not None
    body = "\n".join(unit.lines)
    assert "- [x] 清洗点位" in body
    assert "- [>] 核密度" in body
    assert "- [!] 出图" in body


def test_cleanup_error_strips_gateway_noise():
    raw = (
        'Non-retryable LLM error (402): {"code": "INSUFFICIENT_BALANCE", '
        '"message": "余额不足", "data": {"retryAfterSeconds": 52}, '
        '"traceId": "trace_f94e693b-690c-41f5-8b93-550dde250105"}'
    )
    text = ev.cleanup_error(raw)
    assert "余额不足" in text
    assert "traceId" not in text and "retryAfterSeconds" not in text
    assert len(text) < 80


def test_cleanup_error_truncates_long_plain_text():
    text = ev.cleanup_error("x" * 500, 60)
    assert len(text) == 60 and text.endswith("…")


def test_run_end_wraps_long_summary():
    unit = ev.to_unit("run_end", {"status": "failed", "error": "很长的说明" * 40})
    assert unit is not None
    assert all(len(line) <= 96 for line in unit.lines)


def test_unknown_event_is_ignored():
    assert ev.to_unit("something_new", {"x": 1}) is None


# --------------------------------------------------------------- 格式化


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0s"), (9.4, "9s"), (59, "59s"), (60, "1m00s"), (372, "6m12s"), (3600, "1h00m"), (7999, "2h13m")],
)
def test_format_duration(seconds, expected):
    assert ev.format_duration(seconds) == expected


@pytest.mark.parametrize("count,expected", [(0, "0"), (999, "999"), (1000, "1.0k"), (38200, "38.2k"), (1_400_000, "1.4M")])
def test_format_tokens(count, expected):
    assert ev.format_tokens(count) == expected


def test_describe_args_handles_lists_dicts_and_long_text():
    text = ev.describe_args("t", {"items": [1, 2, 3], "opts": {"a": 1}, "note": "x" * 200})
    assert "items=3项" in text and "opts=1键" in text
    assert len(text) < 200


# --------------------------------------------------------------- 状态行


def test_status_tracker_accumulates_and_renders_footer():
    tracker = ev.StatusTracker(model="glm-5.3-flash")
    tracker.apply("run_start", {"goal": "g"})
    tracker.apply("turn", {"turn": 3})
    tracker.apply("llm_result", {"model": "glm-5.3-flash", "actions": [{"tool": "run_recipe"}]})
    tracker.add_usage({"prompt_tokens": 1200, "completion_tokens": 300})
    tracker.apply("tool_call", {"tool": "run_recipe", "args": {}})
    tracker.apply("tool_result", {"tool": "run_recipe", "ok": True, "summary": "ok"})
    tracker.apply("repair_attempt", {"attempt": 1})
    tracker.apply("verification", {"pass": True, "summary": "验收通过"})

    assert tracker.turn == 3
    assert tracker.tool_count == 1
    assert tracker.repair_count == 1
    assert tracker.total_tokens == 1500
    assert tracker.verify_pass is True

    footer = tracker.footer()
    assert "glm-5.3-flash" in footer and "第 3 轮" in footer
    assert "↑1.2k" in footer and "↓300" in footer  # 输入/输出分开显示

    tracker.apply("run_end", {"status": "completed"})
    assert tracker.status == "completed"
    assert "已完成" in tracker.footer()


def test_status_tracker_footer_truncates_to_width():
    tracker = ev.StatusTracker(model="a-very-long-model-name-for-testing")
    tracker.apply("run_start", {})
    text = tracker.footer(width=30, show_action=False)
    assert len(text) <= 30


# --------------------------------------------------------------- 主题


def test_detect_theme_mode_explicit_and_env(monkeypatch):
    monkeypatch.delenv("GIS_AGENT_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert th.detect_theme_mode("light") == "light"
    assert th.detect_theme_mode("dark") == "dark"
    assert th.detect_theme_mode() == "dark"  # 默认

    monkeypatch.setenv("GIS_AGENT_THEME", "light")
    assert th.detect_theme_mode() == "light"

    monkeypatch.delenv("GIS_AGENT_THEME")
    monkeypatch.setenv("COLORFGBG", "0;15")  # 浅底终端
    assert th.detect_theme_mode() == "light"
    monkeypatch.setenv("COLORFGBG", "15;0")  # 深底终端
    assert th.detect_theme_mode() == "dark"


def test_no_color_env_and_console(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert th.no_color_requested() is False
    monkeypatch.setenv("NO_COLOR", "1")
    assert th.no_color_requested() is True

    console = th.make_console("dark", no_color=True, force_terminal=False)
    assert console.no_color is True
    buf = io.StringIO()
    console.file = buf
    console.print("[gis.ok]完成[/]")
    assert "完成" in buf.getvalue()


def test_build_theme_has_all_roles():
    from gis_cli.ui.theme import _DARK, _LIGHT

    dark = th.build_theme("dark")
    light = th.build_theme("light")
    for role in _DARK:
        assert role in dark.styles
        assert role in light.styles


# --------------------------------------------------------------- 渲染组件


@pytest.mark.parametrize(
    "unit",
    [
        ev.RenderUnit("step", "run_recipe", lines=["recipe_id=kernel_density"], meta={}),
        ev.RenderUnit("result", "执行成功", ok=True),
        ev.RenderUnit("result", "报错", ok=False),
        ev.RenderUnit("verify", "验收通过", ok=True),
        ev.RenderUnit("warn", "自动修复第 1 次"),
        ev.RenderUnit("error", "模型调用失败", lines=["timeout"]),
        ev.RenderUnit("turn", "第 2 轮"),
        ev.RenderUnit("note", "决策 1 项"),
        ev.RenderUnit("final", "任务完成", lines=["产出：", "  a.jpg"], ok=True),
    ],
)
def test_render_unit_smoke(unit):
    console = th.make_console("dark", no_color=True, force_terminal=False, width=100)
    console.print(rd.render_unit(unit, elapsed=1.2))


def test_banner_and_result_card_and_footer_render():
    console = th.make_console("dark", no_color=True, force_terminal=False, width=100)
    buf = io.StringIO()
    console.file = buf

    console.print(
        rd.banner(
            model="glm-5.3-flash",
            api_base="https://tokenrhythm.studio/v1",
            workspace="E:/x/workspace",
            catalog="2 图层",
            recipes=47,
            session_id="20261003-1045",
            kernel="ready",
            version="0.7.1",
        )
    )
    console.print(
        rd.result_card(
            {
                "status": "completed",
                "artifacts": ["output/a.jpg"],
                "tasklist_digest": "- [x] 核密度",
                "methodology": "IDW",
                "final_summary": "done",
                "turn": 12,
            }
        )
    )
    tracker = ev.StatusTracker(model="m", status="running", turn=2)
    console.print(rd.footer_text(tracker, width=90))
    console.print(rd.hint_line(busy=True))

    out = buf.getvalue()
    assert "glm-5.3-flash" in out and "tokenrhythm.studio" in out
    assert "a.jpg" in out and "done" in out
    assert "中断" in out and "明细" in out


def test_result_card_handles_empty_artifacts():
    console = th.make_console("dark", no_color=True, force_terminal=False, width=80)
    console.print(rd.result_card({"status": "failed", "artifacts": [], "error": "无产出"}))


# --------------------------------------------------------------- 折叠缓存


def test_tool_output_store_keeps_recent_and_reports_latest():
    store = rd.ToolOutputStore(limit=3)
    for i in range(5):
        store.add(f"t{i}", f"body{i}")
    assert len(store) == 3
    assert store.latest() == ("t4", "body4")
    assert [title for title, _ in store.all()] == ["t2", "t3", "t4"]

    store.add("empty", "")
    assert store.latest() == ("t4", "body4")  # 空内容不入栈
    store.clear()
    assert len(store) == 0


# --------------------------------------------------------------- 诊断


def test_theme_diagnostics_shape(monkeypatch):
    monkeypatch.setenv("GIS_AGENT_THEME", "light")
    info = th.theme_diagnostics()
    assert info["mode"] == "light"
    assert "no_color" in info and "term" in info


# --------------------------------------------------------------- 转录视图（新）


class _CaptureConsole:
    """把 rich 渲染结果收集成纯文本，便于断言转录形态。"""

    def __init__(self, width: int = 96) -> None:
        import io as _io

        from rich.console import Console as _Console

        self.buffer = _io.StringIO()
        self.width = width
        self._console = _Console(
            file=self.buffer,
            width=width,
            no_color=True,
            force_terminal=False,
            highlight=False,
            theme=th.build_theme("dark"),
        )

    def print(self, *args, **kwargs) -> None:  # noqa: D102
        self._console.print(*args, **kwargs)

    @property
    def text(self) -> str:
        return self.buffer.getvalue()


def _view(width: int = 96):
    console = _CaptureConsole(width)
    tracker = ev.StatusTracker(model="deepseek-flash")
    store = rd.ToolOutputStore()
    return rd.TranscriptView(console, tracker, store), console, store


def test_consecutive_tool_calls_are_merged_into_one_line():
    """12 次目录查询必须收成一行 ×12，而不是 12 组 └。"""
    view, console, _ = _view()
    for index in range(12):
        view.handle("tool_call", {"tool": "catalog_query", "args": {"keyword": f"kw{index}"}})
        view.handle("tool_result", {"tool": "catalog_query", "ok": True, "summary": f"数据目录查询：{index} 条匹配"})
    view.flush()

    out = console.text
    assert "catalog_query ×12" in out
    assert out.count("catalog_query") == 1, "同一工具连续调用必须合并成一行"
    assert "└" not in out, "成功结果内联在工具行，不应再出现 └"
    assert "数据目录查询" in out  # 结果摘要内联可见


def test_tool_group_flushes_when_other_tool_starts():
    view, console, _ = _view()
    view.handle("tool_call", {"tool": "catalog_query", "args": {}})
    view.handle("tool_result", {"tool": "catalog_query", "ok": True, "summary": "查询完成"})
    view.handle("tool_call", {"tool": "execute_code", "args": {"code": "print(1)"}})
    view.handle("tool_result", {"tool": "execute_code", "ok": True, "summary": "执行成功"})
    view.flush()

    out = console.text
    assert "catalog_query" in out and "execute_code" in out
    assert out.count("⏺") == 2  # 两个不同的工具组，各一行
    assert "└" not in out  # 成功结果内联，不需要 └ 续行


def test_thought_block_is_expanded_with_left_bar_and_final_corner():
    view, console, _ = _view()
    reasoning = "First inspect the catalog, then project the layers to EPSG:4548 before analysis."
    view.handle("llm_result", {"model": "m", "actions": [{"tool": "catalog_query", "reasoning": reasoning}]})
    view.flush()

    out = console.text
    assert "◆ 思考" in out
    assert "First inspect the catalog" in out  # 默认展开，能看到完整思考
    assert "└ " in out  # 最后一行收口（单行思考只有 └）


def test_thought_falls_back_to_model_content_when_no_reasoning():
    view, console, _ = _view()
    view.handle("llm_result", {"model": "m", "actions": [{"tool": "x", "reasoning": "", "content": "我打算先核对字段"}]})
    view.flush()
    assert "我打算先核对字段" in console.text


def test_answer_is_separated_from_tool_result():
    """finish 的正文要变成独立的 ● Agent 块，不能埋在工具行里。"""
    view, console, _ = _view()
    view.handle("tool_call", {"tool": "finish", "args": {"summary": "统计完成"}})
    view.handle("tool_result", {"tool": "finish", "ok": True, "summary": "验收通过。已交付 output/结果.csv"})
    view.flush()

    out = console.text
    assert "● Agent" in out
    assert "已交付 output/结果.csv" in out
    assert "验收通过。" not in out  # 前缀被剥离
    assert "└" not in out  # 答复不再以工具结果形式出现


def test_long_tool_summary_is_folded_to_one_line():
    view, console, store = _view()
    view.handle("tool_call", {"tool": "execute_code", "args": {"code": "x=1"}})
    view.handle("tool_result", {"tool": "execute_code", "ok": True, "summary": "执行成功：" + "很长" * 200})
    view.flush()

    out = console.text
    lines = [line for line in out.splitlines() if "执行成功" in line]
    assert lines and len(lines[0]) < 110  # 只留一行，不铺开
    assert len(store) == 1  # 明细进了折叠区（Ctrl+O）


def test_warn_line_is_bounded_and_points_to_details():
    view, console, store = _view()
    view.handle("repair_attempt", {"attempt": 2, "error": "报错根因：" + "很长的解释" * 60})
    view.flush()

    out = console.text
    lines = [line for line in out.splitlines() if "自动修复" in line]
    assert lines and len(lines[0]) < 110
    assert "Ctrl+O 明细" in out
    assert len(store) == 1


def test_turn_rule_has_no_duplicate_marker():
    view, console, _ = _view()
    view.handle("turn", {"turn": 3})
    view.flush()
    first = console.text.splitlines()[0]
    assert "第 3 轮" in first
    assert first.count("·") == 1  # 不再出现 "── · 第 2 轮"
    assert not first.startswith("─")


def test_user_block_is_dimensioned_and_wrapped():
    text = rd.user_block("统计 animal_points.shp 里 NAME 字段各物种的点位数，并写出 CSV 文件")
    console = _CaptureConsole(60)
    console.print(text)
    out = console.text
    assert "▸" in out and "animal_points" in out


def test_wrap_lines_never_splits_words_or_leaves_orphans():
    line = "1. Giraffe — 1792 2. Gazelle — 1159 3. Impala — 361 4. Bison — 136"
    wrapped = rd.wrap_lines(line, 40)
    assert all(len(item) <= 40 for item in wrapped)
    joined = " ".join(wrapped)
    assert joined.split() == line.split()  # 词没被劈开
    assert not any(item.strip().endswith("—") for item in wrapped)


def test_wrap_lines_counts_cjk_as_two_columns():
    wrapped = rd.wrap_lines("中文" * 60, 20)
    assert all(rd.disp_width(item) <= 20 for item in wrapped)


def test_short_uses_display_width():
    assert rd.disp_width(rd._short("中文" * 100, 20)) <= 20
    assert rd._short("abc", 10) == "abc"
