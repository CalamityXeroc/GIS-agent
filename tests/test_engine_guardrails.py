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
    from gis_cli.runtime.code_runner import backup_targets

    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "big.tif").write_bytes(b"x" * 1024)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "result.gdb").mkdir()
    code = (
        f'p = r"{tmp_path / "input"}"\n'
        f'q = r"{tmp_path / "output" / "result.gdb"}"\n'
        "arcpy.management.Delete(q)\n"
    )
    found = backup_targets(code, tmp_path)
    assert str(tmp_path / "input") not in found  # 只读输入目录跳过
    assert str(tmp_path / "output" / "result.gdb") in found  # 结果目录仍备份


def test_backup_handles_relative_paths_and_non_path_strings(tmp_path):
    """回归：备份扫描碰到以 / 开头的引号串不能抛异常。

    实测 bug（14 届 4a-1）：旧实现把 ``'/query'`` 这类以 / 开头的普通字符串
    当成路径，在“破坏性代码先备份”这一步抛 NameError，直接中断整个
    execute_code。旧测试全用绝对路径，恰好绕过了那一行。
    """
    from gis_cli.runtime.code_runner import backup_targets

    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "keep.gdb").mkdir()
    # 以 / 开头的引号串（模型代码里很常见，如请求路径、正则）
    code = 'url = "/query"\narcpy.management.Delete("output/keep.gdb")\n'

    found = backup_targets(code, tmp_path)  # 不能抛
    assert isinstance(found, list)
    assert str(tmp_path / "output" / "keep.gdb") in found, found


# ------------------------------------------------ 上下文压缩（抖动回归）


def _loop_for_compaction(**overrides):
    """造一个只用于测压缩逻辑的 loop（不触网、不跑代码）。"""
    from gis_cli.engine.loop import AgentLoop, LoopConfig

    config = LoopConfig(**overrides)
    return AgentLoop(
        workspace=".",
        llm_client=None,
        registry=None,
        context=None,
        config=config,
        auto_extract_requirements=False,
        use_project_log=False,
    )


def _image_message(payload_chars=533_000):
    """一条带图的消息（模拟 400KB JPG 的 base64）。"""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "看这张图"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + "A" * payload_chars},
            },
        ],
    }


def test_image_blocks_use_fixed_token_cost():
    """图片必须按固定成本估算 —— 不能把 base64 当文本算。

    实测坑：一张 400KB 图 base64 有 53 万字符，旧公式算出 26.7 万 token
    （软上限 12 万）→ 一附图就必触发压缩，而压缩本身是一次 LLM 调用。
    """
    loop = _loop_for_compaction(image_token_cost=1500, context_soft_limit_tokens=120_000)
    loop._messages = [_image_message()]
    estimated = loop._estimate_tokens()
    assert estimated < 10_000, f"单张图不应被估成 {estimated} token"


def test_image_heavy_conversation_does_not_compact_every_turn():
    """带图长对话不应反复触发压缩（滞回 + 固定成本一起生效）。"""
    loop = _loop_for_compaction(
        image_token_cost=1500,
        context_soft_limit_tokens=120_000,
        compact_hysteresis=1.2,
        compact_min_messages=8,
    )
    # 30 条消息、每条 3 张图：真实 token 远低于上限
    loop._messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
    for _ in range(30):
        loop._messages.append(_image_message())
        loop._messages.append({"role": "assistant", "content": "看完了"})

    summarized: list[int] = []

    class _Recorder:
        def record(self, event, payload=None, duration_ms=0):
            if event == "context_compacted":
                summarized.append(payload.get("summarized_messages"))
            return None

    loop.recorder = _Recorder()
    loop._messages_backup = list(loop._messages)
    for _ in range(30):  # 模拟连续 30 轮检查
        loop._compact_if_needed()
    assert not summarized, f"不该发生压缩（实际 {len(summarized)} 次）"
    assert len(loop._messages) == 62


def test_compaction_has_hysteresis_above_soft_limit():
    """刚过软上限不压；超过滞回阈值才压。"""
    loop = _loop_for_compaction(
        context_soft_limit_tokens=1_000,
        compact_hysteresis=2.0,
        compact_min_messages=2,
        keep_recent_messages=2,
    )
    calls: list[int] = []
    loop._summarize = lambda msgs: (calls.append(len(msgs)), "摘要")[1]

    # 估算约 1000+ token（每条 200 字符 → 100 token），但不到 2000 的滞回线
    loop._messages = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": "x" * 200} for _ in range(8)
    ]
    assert loop._estimate_tokens() < 2_000
    loop._compact_if_needed()
    assert not calls, "刚过软上限（未达滞回线）不该压"

    # 加长到超过滞回线 → 应触发一次
    loop._messages += [{"role": "user", "content": "x" * 400} for _ in range(8)]
    assert loop._estimate_tokens() >= 2_000
    loop._compact_if_needed()
    assert len(calls) == 1, "超过滞回线应压一次"
    loop._compact_if_needed()
    assert len(calls) == 1, "压完就应安静下来（不再连续压）"


def test_compaction_skips_when_nothing_worth_summarizing():
    """可压消息太少时不花那次 LLM 调用。"""
    loop = _loop_for_compaction(
        context_soft_limit_tokens=10,
        compact_hysteresis=1.0,
        compact_min_messages=8,
        keep_recent_messages=2,
    )
    calls: list[int] = []
    loop._summarize = lambda msgs: (calls.append(len(msgs)), "摘要")[1]
    loop._messages = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": "x" * 200} for _ in range(4)
    ]
    loop._compact_if_needed()
    assert not calls, "只有 4 条可压消息，不值得一次 LLM 往返"


# ------------------------------------------------ 卫生判据参数可配（P3-13）
def test_hygiene_params_come_from_config(tmp_path):
    """卫生判据必须可配：以前 verifier 硬编码 0 / 5.0，与配方侧两套口径。

    后果：同一份交付物「配方判不过、验收判过」，模型被两边不一致的结论
    来回拉扯（实测 14 届交付卫生阶段）。
    """
    from gis_cli.engine.verifier import Verifier

    v = Verifier(
        workspace=tmp_path,
        hygiene_max_duplicate_families=2,
        hygiene_ratio_limit=1.5,
        hygiene_forbid_suffixes=[".PKL", ".Tmp"],
    )
    assert v.hygiene_max_duplicate_families == 2
    assert v.hygiene_ratio_limit == 1.5
    assert v.hygiene_forbid_suffixes == [".pkl", ".tmp"], "后缀应统一小写"


def test_hygiene_params_default_unchanged(tmp_path):
    """不配时行为与以前完全一致（0 个重复家族、5 倍体积比）。"""
    from gis_cli.engine.verifier import Verifier

    v = Verifier(workspace=tmp_path)
    assert v.hygiene_max_duplicate_families == 0
    assert v.hygiene_ratio_limit == 5.0
    assert v.hygiene_forbid_suffixes is None


def test_hygiene_params_read_from_engine_config(tmp_path):
    """engine 配置段里的键要真正传到 Verifier。"""
    from gis_cli.engine.runner import build_engine

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "llm_config.json").write_text(
        '{"model": "m", "api_key": "k", "api_base": "http://x",'
        ' "engine": {"hygiene_check": "block",'
        ' "hygiene_max_duplicate_families": 3,'
        ' "hygiene_ratio_limit": 2.5,'
        ' "hygiene_forbid_suffixes": [".pkl", ".log"]}}',
        encoding="utf-8",
    )
    bundle = build_engine(workspace=tmp_path, refresh_catalog=True)
    try:
        v = bundle.verifier
        assert v.hygiene_mode == "block"
        assert v.hygiene_max_duplicate_families == 3
        assert v.hygiene_ratio_limit == 2.5
        assert v.hygiene_forbid_suffixes == [".pkl", ".log"]
    finally:
        bundle.close()


# ------------------------------------------------- 已删的中间件不该拖死验收
def test_stale_artifacts_are_pruned_not_reported(tmp_path):
    """模型中途建的中间产物（tmp_train / idw_p1…）删掉后仍留在 artifacts 里。

    验收器跑在**无 arcpy 的 venv** 里时判不了 GDB 内路径，一律"假设存在"，
    随后跑在内核里的 structural_check 又报"产出不存在"——两边打架的结果是
    验收永远不过，agent 空转到轮次上限（14 届任务二a 实测：60 轮耗尽后
    模型改去建 result_data_v2.gdb，产物全对但路径违约）。
    """
    from types import SimpleNamespace

    from gis_cli.engine.verifier import Verifier

    out = tmp_path / "output"
    out.mkdir()
    real = out / "报告.md"
    real.write_text("ok", encoding="utf-8")
    gdb = out / "result_data.gdb"
    gdb.mkdir()

    stale = str(gdb / "tmp_train")
    alive = str(gdb / "meteo_points")

    class _Runner:
        """模拟内核：GDB 内只有 meteo_points，tmp_train 已删。"""

        def run(self, code, timeout=None, workspace=None):
            records = [
                {"path": stale, "exists": False},
                {"path": alive, "exists": True},
                {"path": str(real), "exists": True},
            ]
            return SimpleNamespace(ok=True, result={"records": records})

    v = Verifier(workspace=tmp_path, code_runner=_Runner())
    state = SimpleNamespace(
        artifacts=[stale, alive, str(real)],
        requirements=[],
        methodology="",
        final_summary="",
        tasklist=[],
        goal="test",
    )
    verdict = v.verify(state)

    joined = " ".join(verdict.get("problems") or [])
    assert "产出不存在" not in joined, f"不该把已删中间件报成缺失: {joined}"
    assert state.artifacts == [alive, str(real)], f"应静默剔除失效产物: {state.artifacts}"


def test_all_artifacts_stale_is_still_a_failure(tmp_path):
    """全都失效 != 清理中间件，必须报“没有产出”。"""
    from types import SimpleNamespace

    from gis_cli.engine.verifier import Verifier

    out = tmp_path / "output"
    out.mkdir()
    gone = str(out / "result_data.gdb" / "tmp_raster")

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            return SimpleNamespace(ok=True, result={"records": [{"path": gone, "exists": False}]})

    v = Verifier(workspace=tmp_path, code_runner=_Runner())
    state = SimpleNamespace(
        artifacts=[gone], requirements=[], methodology="",
        final_summary="", tasklist=[], goal="test",
    )
    verdict = v.verify(state)
    assert any("没有任何产出文件" in p for p in (verdict.get("problems") or []))


def test_finish_nudge_fires_near_turn_limit(tmp_path):
    """轮次用到 80%/95% 还没 finish 时要催一次收尾。

    实测 14 届任务四a：120 轮里 finish 调用 0 次——产物早就合格（断言 12/12），
    模型却一直在打磨，最后撞 max_turns，status 只能记成 max_turns。
    """
    from gis_cli.engine.loop import AgentLoop, LoopConfig

    loop = AgentLoop.__new__(AgentLoop)
    loop.config = LoopConfig(max_turns=100)
    loop._finish_nudges = set()
    loop._messages = []

    class _State:
        turn = 50

    loop.state = _State()
    loop._append_user = lambda text: loop._messages.append(text)

    loop._maybe_nudge_finish()          # 50% → 不催
    assert not loop._messages

    _State.turn = 81                    # 81% → 催一次
    loop._maybe_nudge_finish()
    assert len(loop._messages) == 1
    assert "finish" in loop._messages[0]

    loop._maybe_nudge_finish()          # 同阈值不重复
    assert len(loop._messages) == 1

    _State.turn = 96                    # 96% → 再催一次
    loop._maybe_nudge_finish()
    assert len(loop._messages) == 2
def test_repair_exception_does_not_kill_run(tmp_path):
    """修复过程本身报错（典型：修复时网关断了）不能让整个 run 崩。

    实测 14 届任务二a 独立跑：第 10 轮时反代抖动，修复里的 llm_client.chat
    抛 LLMError，一路冒到 runner，整次运行记成 runner_error（0 轮 / 0 秒），
    已跑出的产物白费。
    """
    from types import SimpleNamespace

    from gis_cli.engine.loop import AgentLoop
    from gis_cli.engine.state import Observation

    class _Boom:
        def run(self, **kw):
            raise RuntimeError("All models failed")

    loop = AgentLoop.__new__(AgentLoop)
    loop.repairer = _Boom()
    loop.config = SimpleNamespace(exec_timeout=600.0)
    loop.workspace = tmp_path
    loop.state = SimpleNamespace(goal="g", add_artifact=lambda p: None)
    loop.recorder = SimpleNamespace(record=lambda *a, **k: None)
    loop._emit = lambda *a, **k: None
    loop._recipes_digest = lambda text: ""
    loop._catalog_digest = lambda: ""
    loop.use_error_memory = False
    loop.catalog = None
    loop._check_workspace_hygiene = lambda obs: None
    loop._collect_design_note = lambda obs: None

    class _Registry:
        def execute(self, tool, args, ctx):
            return Observation(ok=False, summary="代码执行失败", error={"type": "E", "message": "x"})

    loop.registry = _Registry()
    loop.context = None

    action = SimpleNamespace(kind="tool", tool="execute_code",
                             args={"code": "print(1)", "description": "d"}, parse_error=None)
    out = loop._run_code_with_repair(action)   # 不抛 = 通过
    assert isinstance(out, Observation) and out.ok is False


def test_verify_sees_outputs_that_were_never_registered(tmp_path):
    """模型用 csv 模块写出的文件不在 artifacts 里，验收也必须看得见。

    实测组合任务：``output/分布中心.csv`` 一直在盘上，但工具只从 set_result /
    代码里的字面量路径提产物，它从没进过 artifacts → 语义验收每次都报“缺失该
    CSV”，模型反复补证也无法自证（4 次 finish 连拒的主因之一）。
    """
    from types import SimpleNamespace

    from gis_cli.engine.verifier import Verifier

    out = tmp_path / "output"
    out.mkdir()
    declared = out / "报告.md"
    declared.write_text("ok", encoding="utf-8")
    csv_path = out / "分布中心.csv"
    csv_path.write_text("YEAR,X\n1982,1\n", encoding="utf-8")
    gdb = out / "result_data.gdb"
    gdb.mkdir()
    child = gdb / "year_centers"

    seen: list[str] = []

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            paths = [str(declared), str(csv_path), str(child)]
            seen.extend(paths)
            return SimpleNamespace(ok=True, result={"records": [{"path": p, "exists": True} for p in paths]})

    v = Verifier(workspace=tmp_path, code_runner=_Runner())
    state = SimpleNamespace(
        artifacts=[str(declared), str(child)],
        requirements=[], methodology="", final_summary="", tasklist=[], goal="test",
    )
    verdict = v.verify(state)

    paths = [str(r.get("path")) for r in verdict["structural"]]
    assert str(csv_path) in paths, f"未登记的产出必须被发现: {paths}"
    assert str(csv_path) in state.artifacts
    # 已声明库内具体条目时不再补整个 .gdb 目录（目录只是弱证据）
    assert str(gdb) not in paths, paths


def test_structural_check_is_cached_until_outputs_change(tmp_path):
    """同一批产物重复验收不该反复问内核（实测 25 条要 14~25s，一个 run 验 4 次）。"""
    from types import SimpleNamespace

    from gis_cli.engine.verifier import Verifier

    out = tmp_path / "output"
    out.mkdir()
    raster = out / "kde.tif"
    raster.write_bytes(b"x")

    calls = {"n": 0}

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            calls["n"] += 1
            return SimpleNamespace(ok=True, result={"records": [{"path": str(raster), "exists": True}]})

    v = Verifier(workspace=tmp_path, code_runner=_Runner())
    v.structural_check([str(raster)])
    v.structural_check([str(raster)])
    assert calls["n"] == 1, "产物没变时不该再跑内核"

    import os

    os.utime(raster, (raster.stat().st_atime, raster.stat().st_mtime + 5))
    v.structural_check([str(raster)])
    assert calls["n"] == 2, "产物变了必须重新体检"


def test_verify_code_enumerates_gdb_contents():
    """GDB 只回一个 kind=Workspace 的壳等于什么都没证——必须把库里的数据集列出来。"""
    from gis_cli.engine.verifier import _VERIFY_CODE

    assert "_workspace_contents" in _VERIFY_CODE
    assert 'rec["contents"]' in _VERIFY_CODE
    assert "ListRasters" in _VERIFY_CODE and "ListFeatureClasses" in _VERIFY_CODE
    # 图片不该被当成栅格去读值域（Describe 对 .jpg 也回 RasterDataset）
    assert '"Image"' in _VERIFY_CODE


def test_semantic_verdict_is_cached_when_material_unchanged(tmp_path):
    """同一批产物 + 同一份 summary，语义验收只该问一次模型。

    ``verify_outputs``（工具）和 finish 的 ``verification`` 走的是同一个
    ``verify()``，实测组合任务 4 次 verify_outputs 共 206s，四次材料完全一样。
    """
    from types import SimpleNamespace

    from gis_cli.engine.verifier import Verifier

    out = tmp_path / "output"
    out.mkdir()
    report = out / "报告.md"
    report.write_text("ok", encoding="utf-8")

    calls = {"n": 0}

    class _LLM:
        class _Cfg:
            effective_vision = False

        config = _Cfg()

        def chat(self, messages, **kwargs):
            calls["n"] += 1
            return SimpleNamespace(
                content='{"pass": true, "per_requirement": [], "missing": [], "repair_hint": ""}',
                model="mock",
            )

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            return SimpleNamespace(ok=True, result={"records": [{"path": str(report), "exists": True}]})

    v = Verifier(workspace=tmp_path, code_runner=_Runner(), llm_client=_LLM())
    state = SimpleNamespace(
        artifacts=[str(report)], requirements=[SimpleNamespace(to_dict=lambda: {"id": "r1", "text": "报告"})],
        methodology="", final_summary="已完成", tasklist=[], goal="测试",
    )
    first = v.verify(state)
    second = v.verify(state)

    assert calls["n"] == 1, "材料没变不该再问一次模型"
    assert first["pass"] is True and second["pass"] is True

    # 摘要变了 = 材料变了，必须重新验收
    state.final_summary = "已完成，另附说明"
    v.verify(state)
    assert calls["n"] == 2


def test_discovery_ignores_previous_run_outputs(tmp_path):
    """持久 workspace 里上次的产物不能算成这次的（DAIR-6 阻塞项 2）。

    基准每 run 一个 workspace，交互式 workspace/output 从不按 run 清理。
    历史产物并进 artifacts 会一路喂给卫生检查、识图质检和语义验收。
    """
    import os
    import time
    from types import SimpleNamespace

    from gis_cli.engine.verifier import Verifier

    out = tmp_path / "output"
    out.mkdir()
    old = out / "关键统计汇总.csv"
    old.write_text("x", encoding="utf-8")
    old_time = time.time() - 86400
    os.utime(old, (old_time, old_time))
    fresh = out / "本次报告.md"
    fresh.write_text("ok", encoding="utf-8")
    (out / "tmp_train.tif").write_bytes(b"x")

    v = Verifier(workspace=tmp_path)
    found = v._discover_outputs(since=time.time() - 60)
    names = [Path(p).name for p in found]

    assert "本次报告.md" in names
    assert "关键统计汇总.csv" not in names, f"历史产物不该被发现: {names}"
    assert "tmp_train.tif" not in names, f"过程件不该被发现: {names}"

    # verify() 端到端：历史 CSV 不能进产物清单
    state = SimpleNamespace(
        artifacts=[str(fresh)], requirements=[], methodology="",
        final_summary="", tasklist=[], goal="t",
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    )
    v.verify(state)
    assert str(old) not in state.artifacts


def test_structural_key_sees_inplace_gdb_edits(tmp_path):
    """原地改库内条目（追加要素）必须让缓存失效（DAIR-6 阻塞项 3）。

    只看 `.gdb` 目录自身的 mtime 不够：改数据文件不动目录时间戳。
    """
    import os
    import time
    from types import SimpleNamespace

    from gis_cli.engine.verifier import Verifier

    gdb = tmp_path / "output" / "result_data.gdb"
    gdb.mkdir(parents=True)
    (gdb / "gdb").write_bytes(b"")
    table = gdb / "a00000001.gdbtable"
    table.write_bytes(b"x" * 10)

    calls = {"n": 0}

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            calls["n"] += 1
            return SimpleNamespace(ok=True, result={"records": [{"path": str(gdb / "pts"), "exists": True, "feature_count": calls["n"]}]})

    v = Verifier(workspace=tmp_path, code_runner=_Runner())
    target = str(gdb / "pts")
    v.structural_check([target])
    v.structural_check([target])
    assert calls["n"] == 1, "产物没变时缓存应命中"

    # 原地追加数据：改库内数据文件，不动 gdb 目录本身
    time.sleep(0.01)
    table.write_bytes(b"x" * 99)
    os.utime(table, (time.time() + 5, time.time() + 5))
    v.structural_check([target])
    assert calls["n"] == 2, "库内条目变了必须重新体检"

    # 目录形态（``output/x.gdb`` 本身就会被 _discover_outputs 当产物）也要失效：
    # 目录自身的 mtime 不随库内数据文件变化，只比它等于没比。
    folder = [str(gdb)]
    before = v._structural_key(folder)
    table.write_bytes(b"x" * 5000)
    os.utime(table, (time.time() + 30, time.time() + 30))
    assert v._structural_key(folder) != before, "目录形态的库内改动必须让缓存失效"
