# -*- coding: utf-8 -*-
"""Tests for loop speed guardrails: exploration budget + failure escalation."""
from __future__ import annotations

import json
from pathlib import Path

from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.loop import AgentLoop, LoopConfig
from gis_cli.engine.tools import EngineContext, build_default_registry
from gis_cli.trace.recorder import RunRecorder


class ScriptedLLM:
    class _Cfg:
        model = "mock"

    config = _Cfg()

    def __init__(self, responses):
        self.responses = list(responses)
        self.task_type_history: list[str] = []

    def chat(self, messages, **kwargs):
        # surface the requested task_type so tests can assert escalation
        self.last_task_type = kwargs.get("task_type")
        self.task_type_history.append(self.last_task_type)
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(
            content=json.dumps({"tool": "finish", "args": {"summary": "done"}}), model="mock"
        )


class StubVerifier:
    def verify(self, state):
        return {"pass": True, "summary": "ok"}


def _tc(name, args=None):
    args = args or {}
    return LLMResponse(
        tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=args, raw_arguments=json.dumps(args))],
        model="mock",
    )


def _make(tmp_path: Path, responses):
    llm = ScriptedLLM(responses)
    loop = AgentLoop(
        workspace=tmp_path,
        llm_client=llm,
        registry=build_default_registry(),
        context=EngineContext(workspace=tmp_path),
        verifier=StubVerifier(),
        recorder=RunRecorder("run_guard", tmp_path / "traces"),
        config=LoopConfig(max_turns=12, max_diagnosis_turns=2, escalate_after_failures=2),
        auto_extract_requirements=False,
    )
    return loop, llm


def test_diagnosis_budget_triggers_nudge(tmp_path: Path):
    responses = [
        _tc("catalog_query", {"query": "a"}),
        _tc("read_document", {"path": "input/doc.txt"}),
        _tc("list_recipes", {}),
        _tc("update_tasklist", {"items": [{"id": "t1", "title": "干活", "status": "running"}]}),
        _tc("finish", {"summary": "完成"}),
    ]
    loop, llm = _make(tmp_path, responses)
    state = loop.run("测试")
    # 3 consecutive diagnosis turns > budget 2 -> nudge must have been injected
    nudges = [m for m in loop._messages if m["role"] == "user" and "侦查已连续" in str(m.get("content"))]
    assert nudges, "expected exploration-budget nudge"
    assert state.status == "completed"


def test_escalation_after_consecutive_failures(tmp_path: Path):
    responses = [
        # two consecutive failed turns -> escalate to agent_strong
        _tc("catalog_query", {"query": "x"}),
        _tc("catalog_query", {"query": "x"}),
        _tc("update_tasklist", {"items": [{"id": "t1", "title": "恢复", "status": "running"}]}),
        _tc("finish", {"summary": "完成"}),
    ]
    # make catalog_query fail: no catalog in context -> tool returns ok? build_default_registry
    # catalog_query returns ok=False when ctx.catalog is None.
    loop, llm = _make(tmp_path, responses)
    loop.run("测试")
    # turn1/turn2 failures (agent), turn3 must be escalated to agent_strong,
    # then success resets back to agent for the finish turn.
    assert llm.task_type_history[:3] == ["agent", "agent", "agent_strong"]
    assert llm.task_type_history[-1] == "agent"


# ---------------------------------------------------------------- recipe hint
class _FakeRecipe:
    def __init__(self, rid, description="", triggers=()):
        self.id = rid
        self.name = rid
        self.description = description
        self.triggers = list(triggers)

    def match_score(self, query: str) -> int:
        text = (query or "").strip().lower()
        if not text:
            return 1
        tokens = [t for t in __import__("re").split(r"[\s,，。;；/]+", text) if t]
        score = 0
        for token in tokens:
            if token in self.id.lower() or token in self.name.lower():
                score += 6
            if token in self.description.lower():
                score += 3
            for trigger in self.triggers:
                tl = trigger.lower()
                if token and (token in tl or tl in token):
                    score += 4
        return score


class _FakeRecipeStore:
    def __init__(self, recipes):
        self._recipes = {r.id: r for r in recipes}

    def get(self, rid):
        return self._recipes.get(rid)

    def all(self):
        return list(self._recipes.values())


def _hint_ctx(recipes):
    from types import SimpleNamespace

    return SimpleNamespace(recipes=recipes)


def test_recipe_hint_by_code_marker():
    from gis_cli.engine.tools import _recipe_hint

    ctx = _hint_ctx(_FakeRecipeStore([_FakeRecipe("graduated_colors_map")]))
    hint = _recipe_hint(ctx, "制作高亮地图", "p = arcpy.mp.ArcGISProject(...)\nlyt.exportToJPEG(out)")
    assert "graduated_colors_map" in hint and "run_recipe" in hint
    # marker absent -> no hint
    assert _recipe_hint(ctx, "随便算算", "x = 1 + 1") == ""


def test_recipe_hint_by_description_score():
    from gis_cli.engine.tools import _recipe_hint

    store = _FakeRecipeStore(
        [_FakeRecipe("空间连接", description="空间连接汇总统计", triggers=[])]
    )
    ctx = _hint_ctx(store)
    hint = _recipe_hint(ctx, "空间连接", "")
    assert "空间连接" in hint and "run_recipe" in hint
    # weak match below threshold -> no hint
    assert _recipe_hint(ctx, "检查一下", "") == ""


def test_recipe_hint_no_recipes():
    from gis_cli.engine.tools import _recipe_hint

    assert _recipe_hint(_hint_ctx(None), "制作高亮地图", "exportToJPEG") == ""
    assert _recipe_hint(_hint_ctx(None), "", "x=1") == ""


def test_office_files_count_as_outputs(tmp_path):
    """Word/PPT/PDF 这类非 GIS 产出也要被登记与发现。

    实测 bug：tools 与 verifier 各有一份扩展名清单，都只列 GIS/图像格式、
    **没有 .docx**。于是“写个 Word 文档”的任务产出永远登记不上、也发现不了，
    验收一直报“没有任何产出文件”。
    """
    from gis_cli.engine.tools import extract_outputs

    payload = {
        "a": str(tmp_path / "output" / "报告.docx"),
        "b": str(tmp_path / "output" / "汇报.pptx"),
        "c": str(tmp_path / "output" / "说明.pdf"),
    }
    found = [p.replace("\\", "/") for p in extract_outputs(payload, tmp_path)]
    assert any(p.endswith("报告.docx") for p in found), found
    assert any(p.endswith("汇报.pptx") for p in found), found
    assert any(p.endswith("说明.pdf") for p in found), found


def test_verifier_discovers_office_files(tmp_path):
    """verifier 的产出发现要与 tools 用同一份扩展名清单。"""
    from gis_cli.engine.verifier import Verifier

    out = tmp_path / "output"
    out.mkdir(parents=True)
    (out / "GIS简介.docx").write_bytes(b"PK\x03\x04 fake")
    (out / "~$GIS简介.docx").write_bytes(b"office lock")

    found = [Path(p).name for p in Verifier(workspace=tmp_path)._discover_outputs()]
    assert "GIS简介.docx" in found
    assert not any(name.startswith("~$") for name in found), "Office 临时文件不算产出"


def test_extract_outputs_excludes_input_and_temp(tmp_path):
    """输入/中间目录里的路径不能当产物。

    实测 bug：模型习惯把读过的输入也写进 set_result，被登记成产物后
    验收报“产出不在 output 目录内”直接判失败（benchmark 4b 就栽在这）。
    """
    from gis_cli.engine.tools import extract_outputs

    payload = {
        "checked": str(tmp_path / "input" / "animal_points.shp"),
        "data": [
            str(tmp_path / "output" / "counts.csv"),
            str(tmp_path / "temp" / "scratch.gdb" / "junk"),
        ],
        "rel": "input/dem.tif",
    }
    found = [p.replace("\\", "/") for p in extract_outputs(payload, tmp_path)]
    assert any(p.endswith("output/counts.csv") for p in found), found
    assert not [p for p in found if "/input/" in p or p.startswith("input/")], found
    assert not [p for p in found if "/temp/" in p], found


def test_extract_outputs_keeps_normal_results(tmp_path):
    """排除输入不能误伤正常产物（含 gdb 内部要素路径）。"""
    from gis_cli.engine.tools import extract_outputs

    payload = {
        "a": str(tmp_path / "output" / "result_data.gdb" / "study_area"),
        "b": str(tmp_path / "output" / "map.png"),
        "c": "not-a-path",
        "d": str(tmp_path / "output" / "插值精度.md"),
    }
    found = [p.replace("\\", "/") for p in extract_outputs(payload, tmp_path)]
    assert any("result_data.gdb/study_area" in p for p in found), found
    assert any(p.endswith("map.png") for p in found), found
    assert all("not-a-path" not in p for p in found), found


def test_extract_outputs_drops_paths_outside_workspace(tmp_path):
    """workspace 之外的路径不算产出（回归：14 届 4a-1 空转到 130 轮）。

    实测：模型把 check_map_project 回参里的 ArcGIS 模板工程与用户工程一并写进
    set_result，被登记成产物后验收每轮都报“产出不在 output 目录内”，
    断言 7/7 全过却 status=max_turns（跑了 40 分钟）。
    """
    from gis_cli.engine.tools import extract_outputs

    outside = tmp_path.parent / (tmp_path.name + "_elsewhere")
    payload = {
        "junk": [
            r"E:\ArcGISPro3.6\Resources\ArcToolBox\Services\routingservices\data\Blank.aprx",
            r"C:\Users\<user>\Documents\ArcGIS\Projects\MyProject\MyProject.aprx",
            str(outside / "other.gdb" / "feature"),
        ],
        "good": str(tmp_path / "output" / "成果图" / "密度_1982_1986.jpg"),
    }
    found = [p.replace("\\", "/") for p in extract_outputs(payload, tmp_path)]
    assert any(p.endswith("密度_1982_1986.jpg") for p in found), found
    assert not [p for p in found if "Blank.aprx" in p or "MyProject" in p], found
    assert not [p for p in found if "elsewhere" in p], found


def test_extract_outputs_keeps_relative_paths(tmp_path):
    """回归：相对路径的产物不能被丢掉（14 届 4a-1 空转的直接原因）。

    实测：模型 finish 时写的是 ``output\\result_data.gdb\\kde_1982_1986``，
    而 ``Path(text).resolve()`` 把相对路径按**当前工作目录**展开（不是 workspace），
    ``relative_to(workspace)`` 直接抛异常 → 产物被当成外部路径丢掉。
    后果：验收的产物清单里只有 ``output/result_data.gdb`` 目录本身，
    四个栅格一条都没有，语义验收只能说“缺失栅格元数据”，agent 空转。
    """
    from gis_cli.engine.tools import extract_outputs

    (tmp_path / "output" / "成果图").mkdir(parents=True)
    (tmp_path / "output" / "result_data.gdb").mkdir(parents=True)
    (tmp_path / "output" / "成果图" / "密度_1982_1986.spec.json").write_text(
        "{}", encoding="utf-8"
    )
    payload = [
        r"output\result_data.gdb\kde_1982_1986",
        r"output\result_data.gdb\kde_1987_1991",
        r"output\成果图\密度_1982_1986.spec.json",
    ]
    found = extract_outputs(payload, tmp_path)
    assert len(found) == 3, found
    # 登记的必须是**绝对路径**：验收可能跑在别的进程/内核里，
    # 相对路径到那边会解析到别的目录。
    assert all(Path(p).is_absolute() for p in found), found
    assert any("kde_1982_1986" in p for p in found), found
    assert any("spec.json" in p for p in found), found


def test_backup_skips_arcgis_lock_files(tmp_path):
    """回归：备份 GDB 时不能去复制 .sr.lock/.rd.lock。

    实测：这些锁文件是 ArcGIS 写数据时的瞬时产物，copytree 把它们列进去后
    很快就被删掉，复制到一半报 [WinError 2]，刷一整排警告还留下半成品备份。
    """
    from gis_cli.runtime.code_runner import backup_paths

    gdb = tmp_path / "result_data.gdb"
    gdb.mkdir()
    (gdb / "a00000001.gdbtable").write_bytes(b"data")
    (gdb / "keep.txt").write_text("x", encoding="utf-8")
    (gdb / "kde_1982_1986.CALAMITY-XEROC.1234.5678.sr.lock").write_bytes(b"")
    (gdb / "GDB_Items.CALAMITY-XEROC.1234.rd.lock").write_bytes(b"")

    created = backup_paths([str(gdb)], tmp_path / "backups")
    assert created, "应该产出备份目录"
    copied = {p.name for p in Path(created[0]).iterdir()}
    assert "a00000001.gdbtable" in copied, copied
    assert "keep.txt" in copied, copied
    assert not [n for n in copied if n.endswith(".lock")], copied


def test_backup_skips_readonly_and_large_dirs(tmp_path):
    """备份不能把只读输入目录整份复制（实测 input/ 被备份 100 次共 2 GB）。"""
    from gis_cli.engine.tools import _referenced_existing_paths

    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "big.tif").write_bytes(b"x" * 1024)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "result.gdb").mkdir()
    code = f'p = r"{tmp_path / "input"}"\nq = r"{tmp_path / "output" / "result.gdb"}"\narcpy.management.Delete(q)\n'
    found = _referenced_existing_paths(code, tmp_path)
    assert str(tmp_path / "input") not in found  # 只读输入目录跳过
    assert str(tmp_path / "output" / "result.gdb") in found  # 结果目录仍备份


def test_backup_handles_relative_paths_without_ctx(tmp_path):
    """回归：备份扫描碰到以 / 开头的引号串不能抛 NameError。

    实测 bug（14 届 4a-1）：_referenced_existing_paths 签名只有 (code, workspace)，
    内部却引用了不存在的 ctx。_QUOTED_PATH 会把 ``'/query'`` 这类以 / 开头的
    普通字符串当成路径，于是 _resolve_path(ctx, ...) 直接抛 NameError——
    而它跑在“破坏性代码先备份”这一步，报错会中断整个 execute_code。
    旧测试全用绝对路径，恰好绕过了那一行。
    """
    from gis_cli.engine.tools import _referenced_existing_paths

    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "keep.gdb").mkdir()
    # 以 / 开头的引号串（模型代码里很常见，如请求路径、正则）就会落到那一行
    code = 'url = "/query"\narcpy.management.Delete("output/keep.gdb")\n'

    found = _referenced_existing_paths(code, tmp_path)  # 不传 ctx：以前这里就崩
    assert isinstance(found, list)

    class _Ctx:
        workspace = tmp_path

    # 传 ctx 也不能崩（走 _resolve_path 分支）
    assert isinstance(_referenced_existing_paths(code, tmp_path, _Ctx()), list)
