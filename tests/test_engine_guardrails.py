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
