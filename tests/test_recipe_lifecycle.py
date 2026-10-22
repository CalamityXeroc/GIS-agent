# -*- coding: utf-8 -*-
"""候选配方门禁：槽位一致性 + 硬编码路径。

蒸馏出来的配方是模型对一次成功运行的**泛化猜测**。最常见的两种坏配方是
把任务特定的绝对路径写死（换台机器就废）、以及参数槽位自相矛盾（调用时才炸）。
两者都能纯静态挡住，不需要跑 GIS。
"""
from __future__ import annotations

from pathlib import Path

from gis_cli.recipes.lifecycle import RecipeLifecycle, _assertion_slots, _slots_in
from gis_cli.recipes.schema import Recipe, RecipeParam
from gis_cli.recipes.store import RecipeLibrary


def _lifecycle(tmp_path: Path) -> RecipeLifecycle:
    lib = RecipeLibrary(user_dir=tmp_path / "recipes")
    lib.load()
    return RecipeLifecycle(lib, tmp_path)


def _good_recipe(**overrides) -> Recipe:
    data = dict(
        id="clip_by_mask",
        name="按掩膜裁切",
        code_template=(
            "import arcpy\n"
            "arcpy.analysis.Clip({{input_fc}}, {{mask_fc}}, {{output_fc}})\n"
            "set_result({'output': {{output_fc}}})\n"
        ),
        params={
            "input_fc": RecipeParam("input_fc", "string", True, "input/a.shp", "输入"),
            "mask_fc": RecipeParam("mask_fc", "string", True, "input/m.shp", "掩膜"),
            "output_fc": RecipeParam("output_fc", "string", True, "output/clip.gdb/clip", "输出"),
        },
        validation=[{"type": "vector_exists", "path": "{{output_fc}}"}],
    )
    data.update(overrides)
    return Recipe(**data)


def test_slots_helper_reads_both_forms():
    assert _slots_in("a = {{x}}\nb = '@@y@@'") == {"x", "y"}
    assert _assertion_slots([{"path": "{{out}}", "value": {"k": "@@f@@"}}]) == {"out", "f"}


def test_good_recipe_passes(tmp_path: Path):
    gate = _lifecycle(tmp_path).static_gate(_good_recipe())
    assert gate.ok, gate.summary()
    names = {c.name for c in gate.checks if c.ok}
    assert {"required_fields", "renders", "valid_python", "no_hardcoded_paths", "slots_declared"} <= names


def test_hardcoded_windows_path_is_rejected(tmp_path: Path):
    """写死盘符路径的配方换台机器就废，必须拦在候选目录里。"""
    recipe = _good_recipe(
        code_template=(
            "import arcpy\n"
            "arcpy.analysis.Clip(r" + repr("E:\\data\\a.shp") + ", {{mask_fc}}, {{output_fc}})\n"
        )
    )
    gate = _lifecycle(tmp_path).static_gate(recipe)
    assert not gate.ok
    hard = [c for c in gate.checks if c.name == "no_hardcoded_paths"][0]
    assert not hard.ok and "绝对路径" in hard.detail


def test_source_workspace_path_is_rejected(tmp_path: Path):
    """源运行目录即使写成相对/正斜杠形式也要拦。"""
    ws = tmp_path / "ws"
    recipe = _good_recipe(
        code_template="import arcpy\np = " + repr(str(ws)) + "\nset_result({'x': p})\n"
    )
    gate = _lifecycle(tmp_path).static_gate(recipe, source_workspace=str(ws))
    hard = [c for c in gate.checks if c.name == "no_hardcoded_paths"][0]
    assert not hard.ok and "源工作区" in hard.detail


def test_undeclared_slot_is_rejected(tmp_path: Path):
    """代码里用 {{x}} 但 params 没声明 → 调用时才炸，必须静态拦。"""
    recipe = _good_recipe(
        code_template="import arcpy\narcpy.analysis.Clip({{input_fc}}, {{missing_param}}, {{output_fc}})\n"
    )
    gate = _lifecycle(tmp_path).static_gate(recipe)
    slot_check = [c for c in gate.checks if c.name == "slots_declared"][0]
    assert not slot_check.ok and "missing_param" in slot_check.detail


def test_required_param_without_default_is_rejected(tmp_path: Path):
    """没默认值 → 无法回放、也无法被 agent 直接调用。"""
    recipe = _good_recipe()
    recipe.params["mask_fc"].default = None
    gate = _lifecycle(tmp_path).static_gate(recipe)
    check = [c for c in gate.checks if c.name == "params_have_defaults"][0]
    assert not check.ok and "mask_fc" in check.detail


def test_invalid_python_is_rejected(tmp_path: Path):
    recipe = _good_recipe(code_template="def broken(:\n  pass\n")
    gate = _lifecycle(tmp_path).static_gate(recipe)
    assert not [c for c in gate.checks if c.name == "valid_python"][0].ok


def test_missing_assertions_is_rejected(tmp_path: Path):
    """没有 validation 的配方无法自证，不能算"已验证的方法论"。"""
    gate = _lifecycle(tmp_path).static_gate(_good_recipe(validation=[]))
    assert not [c for c in gate.checks if c.name == "assertions_present"][0].ok


def test_assertion_slot_must_be_declared(tmp_path: Path):
    recipe = _good_recipe(validation=[{"type": "vector_exists", "path": "{{ghost}}"}])
    gate = _lifecycle(tmp_path).static_gate(recipe)
    check = [c for c in gate.checks if c.name == "assertions_use_declared_params"][0]
    assert not check.ok and "ghost" in check.detail


def test_unused_param_is_only_a_warning(tmp_path: Path):
    """声明了没用到的参数只是警告——可能是留给调用方覆盖的。"""
    recipe = _good_recipe()
    recipe.params["spare"] = RecipeParam("spare", "string", False, "x", "备用")
    gate = _lifecycle(tmp_path).static_gate(recipe)
    assert gate.ok, gate.summary()
    assert any("spare" in w for w in gate.warnings)


def test_candidate_save_and_adopt(tmp_path: Path):
    lc = _lifecycle(tmp_path)
    recipe = _good_recipe()
    path = lc.save_candidate(recipe, provenance="task:demo")
    assert path.exists()
    assert lc.list_candidates()[0]["gate_ok"] is True

    target, note = lc.adopt(recipe.id)
    assert target is not None and target.exists(), note
    # 采纳后库要能立刻看到（下一次 build_engine 靠的就是这个目录）
    assert lc.library.get(recipe.id) is not None


def test_adopt_refuses_when_gate_fails(tmp_path: Path):
    """候选在磁盘上被改坏时，采纳要重新过门禁并拒绝。"""
    lc = _lifecycle(tmp_path)
    lc.save_candidate(_good_recipe(), provenance="task:demo")
    # 直接往磁盘写一个“用了未声明槽位”的坏候选（模拟手改 / 旧版本留下的文件）
    broken = _good_recipe(
        code_template="import arcpy\narcpy.analysis.Clip({{ghost}}, {{mask_fc}}, {{output_fc}})\n"
    )
    broken.to_yaml(lc._candidate_yaml("clip_by_mask"))

    target, note = lc.adopt("clip_by_mask")
    assert target is None and "门禁未通过" in note, note


def test_adopt_avoids_id_collision(tmp_path: Path):
    lc = _lifecycle(tmp_path)
    lc.save_candidate(_good_recipe(), provenance="task:a")
    first, _ = lc.adopt("clip_by_mask")
    assert first is not None

    other = _good_recipe()
    other.description = "另一个变体"
    lc.save_candidate(other, provenance="task:b")
    second, note = lc.adopt("clip_by_mask")
    assert second is not None and second.name == "clip_by_mask-2.yaml", note


# ------------------------------------------------ 蒸馏结果的净化与工作区哨兵
def test_sanitize_removes_double_quoted_slots():
    """模型常写 `ws = "'{{workspace}}'"`，而槽位渲染时自带引号 → 语法错。"""
    from gis_cli.recipes.lifecycle import sanitize_distilled

    recipe = _good_recipe(
        code_template="ws = \"'{{input_fc}}'\"\nimport arcpy\narcpy.analysis.Clip({{input_fc}}, {{mask_fc}}, {{output_fc}})\n"
    )
    notes = sanitize_distilled(recipe)
    assert "{{input_fc}}" in recipe.code_template
    assert "'{{input_fc}}'" not in recipe.code_template
    assert any("引号" in n for n in notes)


def test_sanitize_replaces_source_workspace_with_sentinel(tmp_path: Path):
    """参数默认值是本机绝对路径 → 换成 @workspace@（运行时自动填）。"""
    from gis_cli.recipes.lifecycle import sanitize_distilled
    from gis_cli.recipes.schema import WORKSPACE_SENTINEL

    ws = tmp_path / "ws"
    recipe = _good_recipe()
    recipe.params["workspace"] = RecipeParam("workspace", "string", True, str(ws), "工作区")
    notes = sanitize_distilled(recipe, source_workspace=str(ws))
    assert recipe.params["workspace"].default == WORKSPACE_SENTINEL
    assert any("workspace" in n for n in notes)


def test_workspace_sentinel_is_filled_at_run_time(tmp_path: Path):
    """``@workspace@`` 必须在执行时换成真实工作区，否则配方找不到 input/。"""
    from types import SimpleNamespace

    from gis_cli.recipes.schema import WORKSPACE_SENTINEL
    from gis_cli.recipes.store import RecipeLibrary

    seen: dict[str, str] = {}

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            seen["code"] = code
            return SimpleNamespace(ok=True, stdout="", stderr="", result={"ok": True},
                                   error=None, display_images=[])

    lib = RecipeLibrary(user_dir=tmp_path / "recipes", code_runner=_Runner())
    lib.load()
    recipe = _good_recipe(
        code_template="ws = {{workspace}}\nset_result({'ok': True})\n",
        params={"workspace": RecipeParam("workspace", "string", True, WORKSPACE_SENTINEL, "工作区")},
        validation=[],
    )
    lib.save(recipe)

    ctx = SimpleNamespace(workspace=tmp_path, code_runner=_Runner(), exec_timeout=10)
    lib.run(recipe.id, {}, ctx)
    # 渲染出的代码里路径是 repr 形式（反斜杠被转义一层），比较前先归一
    normalized = seen["code"].replace("\\\\", "\\")
    assert str(tmp_path).replace("/", "\\") in normalized, seen["code"]
    assert WORKSPACE_SENTINEL not in seen["code"]


def test_collect_successful_code_prefers_artifact_producers(tmp_path: Path):
    """挑片段要优先“产出过产物”的，而不是简单取最后几个。"""
    import json

    from gis_cli.recipes.distiller import RecipeDistiller

    trace = tmp_path / "t.jsonl"
    events = [
        {"event": "tool_call", "payload": {"tool": "execute_code", "args": {"code": "A = 1  # 干活的"}}},
        {"event": "tool_result", "payload": {"tool": "execute_code", "ok": True, "artifacts": ["out/a.gdb"]}},
        {"event": "tool_call", "payload": {"tool": "execute_code", "args": {"code": "B = 2  # 小核验"}}},
        {"event": "tool_result", "payload": {"tool": "execute_code", "ok": True, "artifacts": []}},
        {"event": "tool_call", "payload": {"tool": "execute_code", "args": {"code": "C = 3  # 失败过"}}},
        {"event": "tool_result", "payload": {"tool": "execute_code", "ok": False, "artifacts": []}},
    ]
    trace.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8")

    snippets = RecipeDistiller.collect_successful_code(trace)
    assert snippets[0].startswith("A = 1"), snippets
    assert all(not s.startswith("C = 3") for s in snippets), "失败的片段不该被选"


# ------------------------------------------- 配方的断言路径与产物登记
def test_assertion_relative_path_is_resolved_against_workspace(tmp_path: Path):
    """蒸馏配方的断言路径常是 ``output/...`` 相对形式。

    内核的 CWD 不是工作区，直接交给 arcpy 会一律 False（实测：配方执行成功、
    断言 0/4，四条全红）。
    """
    from gis_cli.recipes.store import RecipeLibrary

    ws = tmp_path / "ws"
    assert RecipeLibrary._absolutize("output/a.gdb/fc", ws) == str(ws / "output/a.gdb/fc")
    absolute = str(ws / "output" / "a.gdb" / "fc")
    assert RecipeLibrary._absolutize(absolute, ws) == absolute
    assert RecipeLibrary._absolutize("", ws) == ""
    assert RecipeLibrary._absolutize("output/x", None) == "output/x"


def test_recipe_artifacts_fall_back_to_existence_assertions(tmp_path: Path):
    """配方只 print 不 set_result 时，用“通过的存在性断言”当产物清单。

    否则引擎认为“没有产物”，finish 会被验收驳回（实测：配方断言 4/4 通过，
    但 state.artifacts 为空 → 多次验收未通过 → status=failed）。
    """
    from types import SimpleNamespace

    from gis_cli.recipes.store import RecipeLibrary

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            # 模拟“代码没调 set_result”：result 为空
            return SimpleNamespace(ok=True, stdout="done", stderr="", result=None,
                                   error=None, display_images=[])

    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    target = ws / "output" / "x.txt"
    target.write_text("ok", encoding="utf-8")

    lib = RecipeLibrary(user_dir=tmp_path / "recipes", code_runner=_Runner())
    lib.load()
    recipe = _good_recipe(
        code_template="print('done')\n",
        params={"out": RecipeParam("out", "string", True, "output/x.txt", "输出")},
        validation=[{"type": "file_exists", "path": "{{out}}"}],
    )
    lib.save(recipe)

    ctx = SimpleNamespace(workspace=ws, code_runner=_Runner(), exec_timeout=10)
    obs = lib.run(recipe.id, {}, ctx)
    assert obs.ok, obs.summary
    assert obs.artifacts, "应从事断里兜底推出产物"
    assert str(target) in obs.artifacts[0]


# ------------------------------------------------------ 门禁 B：回放验证
class _ExecRunner:
    """真执行代码（不涉 arcpy），用于测 file_exists 类断言的回放。"""

    def __init__(self):
        self.codes: list[str] = []

    def run(self, code, timeout=None, workspace=None):
        from types import SimpleNamespace

        self.codes.append(code)
        ns: dict = {}
        try:
            exec(compile(code, "<recipe>", "exec"), ns)
            return SimpleNamespace(ok=True, stdout="", stderr="", result=ns.get("_result"),
                                   error=None, display_images=[])
        except Exception as exc:  # pragma: no cover - 反例用
            return SimpleNamespace(ok=False, stdout="", stderr="", result=None,
                                   error={"type": type(exc).__name__, "message": str(exc)},
                                   display_images=[])


def _replay_recipe(**overrides) -> Recipe:
    code = (
        "import os\n"
        "os.makedirs(os.path.dirname({{out}}), exist_ok=True)\n"
        "with open({{out}}, 'w', encoding='utf-8') as f:\n"
        "    f.write('ok')\n"
    )
    data = dict(
        id="write_marker",
        name="写标记文件",
        code_template=code,
        params={"out": RecipeParam("out", "string", True, "output/marker.txt", "输出文件")},
        validation=[{"type": "file_exists", "path": "{{out}}"}],
    )
    data.update(overrides)
    return Recipe(**data)


def test_replay_passes_and_writes_only_to_scratch(tmp_path: Path):
    """回放要在隔离目录跑，不能覆盖 output/ 下的原产物。"""
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    original = ws / "output" / "marker.txt"
    original.write_text("原始内容", encoding="utf-8")

    lib = RecipeLibrary(user_dir=ws / "recipes")
    lib.load()
    runner = _ExecRunner()
    lc = RecipeLifecycle(lib, ws, code_runner=runner)
    lc.save_candidate(_replay_recipe(), provenance="test")

    res = lc.replay("write_marker", keep=True)
    assert res.ok, res.summary()
    # 原产物没被改
    assert original.read_text(encoding="utf-8") == "原始内容"
    # 代码实际写到了 replay 目录
    assert "replay" in runner.codes[0]
    assert (ws / ".gis_agent" / "recipes" / "replay" / "write_marker" / "marker.txt").exists()


def test_replay_fails_when_assertion_mismatch(tmp_path: Path):
    """断言写错（内容对不上）→ 回放必须失败，并指出是哪条。"""
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    lib = RecipeLibrary(user_dir=ws / "recipes")
    lib.load()
    lc = RecipeLifecycle(lib, ws, code_runner=_ExecRunner())
    recipe = _replay_recipe()
    recipe.validation = [{"type": "file_contains", "path": "{{out}}", "value": "绝对不可能出现的串"}]
    lc.save_candidate(recipe, provenance="test")

    res = lc.replay("write_marker")
    assert not res.ok
    failed = [c for c in res.checks if not c.ok]
    assert failed and failed[0].name.startswith("assert:"), res.summary()


def test_replay_refuses_without_recognizable_output_param(tmp_path: Path):
    """识别不出输出参数就拒绝回放——宁可不错，也不要覆盖原产物。"""
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    lib = RecipeLibrary(user_dir=ws / "recipes")
    lib.load()
    lc = RecipeLifecycle(lib, ws, code_runner=_ExecRunner())
    recipe = _replay_recipe()
    recipe.params = {"target_path": RecipeParam("target_path", "string", True, "data/marker.txt", "输入")}
    recipe.code_template = recipe.code_template.replace("{{out}}", "{{target_path}}")
    recipe.validation = [{"type": "file_exists", "path": "{{target_path}}"}]
    lc.save_candidate(recipe, provenance="test")

    res = lc.replay("write_marker")
    assert not res.ok
    assert any(c.name == "output_params" and not c.ok for c in res.checks), res.summary()


def test_replay_refuses_when_static_gate_fails(tmp_path: Path):
    """静态门禁都过不了，不该浪费时间去跑回放。"""
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    lib = RecipeLibrary(user_dir=ws / "recipes")
    lib.load()
    runner = _ExecRunner()
    lc = RecipeLifecycle(lib, ws, code_runner=runner)
    lc.save_candidate(_replay_recipe(validation=[]), provenance="test")  # 无断言

    res = lc.replay("write_marker")
    assert not res.ok
    assert runner.codes == [], "静态门禁失败时不该执行代码"


def test_adopt_refuses_near_duplicate(tmp_path: Path):
    """同一方法论反复蒸馏会产出一堆近义配方（实测两个 csv_group_sum_* 并存）。"""
    lc = _lifecycle(tmp_path)
    first = _good_recipe(id="clip_by_mask_v2", name="按掩膜裁切要素", description="用掩膜面裁切要素类并入库")
    lc.save_candidate(first, provenance="a")
    target, _ = lc.adopt("clip_by_mask_v2")
    assert target is not None

    near = _good_recipe(id="mask_clip_features", name="按掩膜裁切入库", description="按掩膜面裁切要素并写入 GDB")
    lc.save_candidate(near, provenance="b")
    target2, note = lc.adopt("mask_clip_features")
    assert target2 is None and "相似配方" in note, note


def test_output_param_detected_for_bare_dir_name(tmp_path: Path):
    """蒸馏出来的配方常把输出目录写成裸 ``output``（无分隔符/无后缀）。

    按“值长得像路径”判会漏掉它 → 回放直接放弃。这里钉住：裸目录名也要认。
    """
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    original = ws / "output" / "summary.md"
    original.write_text("原产物", encoding="utf-8")

    code = (
        "import os\n"
        "os.makedirs({{out_dir}}, exist_ok=True)\n"
        "with open(os.path.join({{out_dir}}, 'summary.md'), 'w', encoding='utf-8') as f:\n"
        "    f.write('ok')\n"
    )
    recipe = Recipe(
        id="bare_output_dir",
        name="裸输出目录",
        code_template=code,
        params={"out_dir": RecipeParam("out_dir", "string", True, "output", "输出目录")},
        validation=[{"type": "file_exists", "path": "{{out_dir}}/summary.md"}],
    )
    lib = RecipeLibrary(user_dir=ws / "recipes")
    lib.load()
    runner = _ExecRunner()
    lc = RecipeLifecycle(lib, ws, code_runner=runner)
    lc.save_candidate(recipe, provenance="test")

    res = lc.replay("bare_output_dir", keep=True)
    assert res.ok, res.summary()
    # 原产物没被动过，回放写进了隔离目录
    assert original.read_text(encoding="utf-8") == "原产物"
    assert (ws / ".gis_agent" / "recipes" / "replay" / "bare_output_dir" / "summary.md").exists()


# ------------------------------------------- 断言：@@slot@@ 与 file_matches
def test_assertion_resolves_raw_slots(tmp_path: Path):
    """断言里的 @@param@@ 也要替换（只认 {{param}} 会把字面量当期望值）。"""
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "t.md").write_text("地类 | 记录数\n耕地 | 3\n", encoding="utf-8")
    lib = RecipeLibrary(user_dir=ws / "recipes")
    lib.load()
    recipe = Recipe(
        id="raw_slot",
        name="raw slot",
        code_template="x = {{group}}\n",
        params={"group": RecipeParam("group", "string", True, "地类", "")},
        validation=[{"type": "file_contains", "path": "output/t.md", "value": "@@group@@ | 记录数"}],
    )
    results = lib._run_assertions(recipe, {"group": "地类"}, None, workspace=ws)
    assert results[0].ok, results[0].detail


def test_file_matches_uses_regex(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "total.txt").write_text("11286.75\n", encoding="utf-8")
    lib = RecipeLibrary(user_dir=ws / "recipes")
    lib.load()
    ok = lib._run_assertions(
        Recipe(id="r", name="r", validation=[{"type": "file_matches", "path": "output/total.txt", "value": r"^\d+\.\d{2}$"}]),
        {}, None, workspace=ws,
    )
    assert ok[0].ok, ok[0].detail
    bad = lib._run_assertions(
        Recipe(id="r", name="r", validation=[{"type": "file_matches", "path": "output/total.txt", "value": r"^\d{3,}$"}]),
        {}, None, workspace=ws,
    )
    assert not bad[0].ok


# --------------------------------------- 参数类型：声明 integer 但给字符串
def test_render_coerces_declared_types():
    """``type: integer`` + 默认值 ``"2"`` 要渲染成 ``2``，不能是 ``'2'``。

    实测蒸馏：模型把 decimals 声明成 integer 却给 "2"，渲染出
    ``round(total, '2')`` → 回放 TypeError。类型是契约，渲染前按它收敛。
    """
    recipe = Recipe(
        id="coerce",
        name="coerce",
        code_template="a = round(1.234, {{decimals}})\nb = {{flag}}\n",
        params={
            "decimals": RecipeParam("decimals", "integer", False, "2", ""),
            "flag": RecipeParam("flag", "boolean", False, "true", ""),
        },
    )
    code = recipe.render({})
    assert "round(1.234, 2)" in code
    assert "b = True" in code
    ns: dict = {}
    exec(code, ns)
    assert ns["a"] == 1.23 and ns["b"] is True


def test_render_leaves_uncoercible_values_alone():
    """转不了就原样保留，不要静默换值。"""
    recipe = Recipe(
        id="coerce_bad",
        name="coerce_bad",
        code_template="x = {{n}}\n",
        params={"n": RecipeParam("n", "integer", False, "两个", "")},
    )
    assert "x = '两个'" in recipe.render({})


def test_adopt_twice_is_idempotent(tmp_path: Path):
    """采纳会把 version 升到 1.0、provenance 加 ``| adopted``。

    重复采纳同一份内容时，这两个字段的差异不该让“已存在相同配方”失效，
    否则每采纳一次就多一个 ``-2`` 副本（实测）。
    """
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    lib = RecipeLibrary(user_dir=ws / "recipes")
    lib.load()
    lc = RecipeLifecycle(lib, ws, code_runner=_ExecRunner())
    lc.save_candidate(_replay_recipe(), provenance="test")

    first, note1 = lc.adopt("write_marker")
    assert first is not None and first.exists()
    second, note2 = lc.adopt("write_marker")
    assert second == first, note2
    assert "已存在相同配方" in note2
    assert not (ws / "recipes" / "write_marker-2.yaml").exists()
