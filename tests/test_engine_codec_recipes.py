# -*- coding: utf-8 -*-
"""Unit tests for tool codec and recipe schema (no ArcGIS required)."""
from __future__ import annotations

from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.tool_codec import (
    Action,
    actions_from_json,
    actions_from_native,
    extract_json_object,
)
from gis_cli.recipes.schema import Recipe, RecipeParam


def test_extract_json_object_variants():
    assert extract_json_object('{"tool": "x", "args": {}}')["tool"] == "x"
    assert extract_json_object('前缀```json\n{"tool": "y"}\n```后缀')["tool"] == "y"
    assert extract_json_object("no json here") is None


def test_actions_from_native():
    response = LLMResponse(
        tool_calls=[
            ToolCall(id="c1", name="execute_code", arguments={"code": "print(1)"}, raw_arguments='{"code": "print(1)"}')
        ],
        model="m",
    )
    actions = actions_from_native(response)
    assert len(actions) == 1
    assert actions[0].kind == "tool"
    assert actions[0].tool == "execute_code"
    assert actions[0].args["code"] == "print(1)"


def test_actions_from_json_top_level_args():
    response = LLMResponse(content='{"tool": "catalog_query", "query": "city", "thought": "查数据"}')
    actions = actions_from_json(response)
    assert actions[0].tool == "catalog_query"
    assert actions[0].args == {"query": "city"}
    assert "查数据" in actions[0].reasoning


def test_actions_from_json_final():
    response = LLMResponse(content="我完成了")
    actions = actions_from_json(response)
    assert actions[0].kind == "final"


def test_recipe_render_literal_and_raw():
    recipe = Recipe(
        id="r",
        name="r",
        code_template="path = {{input_path}}\nname = '@@field@@'\nset_result({'output': {{input_path}}})",
        params={
            "input_path": RecipeParam("input_path", "string", required=True),
            "field": RecipeParam("field", "string", default="pop"),
        },
    )
    code = recipe.render({"input_path": r"E:\data\a.shp"})
    assert "path = 'E:\\\\data\\\\a.shp'" in code
    assert "name = 'pop'" in code


def test_recipe_required_param_error():
    recipe = Recipe(id="r", name="r", params={"x": RecipeParam("x", "string", required=True)})
    try:
        recipe.render({})
    except ValueError as exc:
        assert "x" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_recipe_match_score():
    recipe = Recipe(id="buffer_dissolve", name="缓冲区", description="缓冲区融合", triggers=["缓冲区", "buffer"])
    assert recipe.match_score("做缓冲区") > 0
    assert recipe.match_score("完全无关") == 0


def test_recipe_validation_slot_resolution():
    """Validation paths use {{param}} and must resolve to a bare value."""
    from gis_cli.recipes.store import _resolve

    out = _resolve("{{output_path}}", {"output_path": r"E:\data\a.gdb\county"})
    assert out == r"E:\data\a.gdb\county"
    assert "{" not in out and "}" not in out
    # integer slot inside a value string
    assert _resolve("{{target_srid}}", {"target_srid": 4508}) == "4508"


def test_all_builtin_recipes_load_and_render():
    """Every shipped recipe must parse and render without leftover slots."""
    from gis_cli.recipes.store import RecipeLibrary

    lib = RecipeLibrary()
    count = lib.load()
    assert count >= 8, f"expected >=8 builtin recipes, got {count}"
    for recipe in lib.all():
        params = {}
        for name, spec in recipe.params.items():
            if spec.default is not None:
                params[name] = spec.default
            elif spec.type in {"integer", "number"}:
                params[name] = 1
            else:
                params[name] = f"dummy_{name}"
        code = recipe.render(params)
        assert isinstance(code, str) and code.strip(), recipe.id
        assert "{{" not in code and "@@" not in code, f"unresolved slot in {recipe.id}"
        # every recipe must declare at least one validation assertion
        assert recipe.validation, f"{recipe.id} has no validation"


def test_validation_list_values_resolve_slots():
    """List-typed assertion values (fields_present) must resolve {{slots}} too."""
    from gis_cli.recipes.store import RecipeLibrary, _resolve

    lib = RecipeLibrary()
    recipe = lib.get("polygon_neighbor_stats")
    assert recipe is not None
    resolved = recipe.resolve_params({
        "input_path": r"E:\d\a.shp",
        "output_path": r"E:\d\out.gdb\b",
        "value_field": "Y2020",
    })
    raw = next(v for v in recipe.validation if v.get("type") == "fields_present")
    target = _resolve(str(raw.get("path", "")), resolved)
    expected = [_resolve(str(v), resolved) for v in raw.get("value", [])]
    assert expected == ["NB_MEAN", "ABOVE_MEAN"], expected
    assert "{" not in target and "{" not in "".join(expected)
