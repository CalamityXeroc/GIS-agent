from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baml_client import b  # noqa: E402


def test_shim_infer_intent_greeting_returns_general() -> None:
    result = b.infer_gis_intent(
        message="你好，你是什么模型",
        valid_intents=["execute_task", "confirm_action", "preview", "cancel", "query_status", "tool_help", "general"],
    )
    assert isinstance(result, dict)
    assert result.get("intent") == "general"


def test_shim_infer_intent_unknown_defaults_to_general() -> None:
    result = b.infer_gis_intent(
        message="嗯",
        valid_intents=["execute_task", "confirm_action", "preview", "cancel", "query_status", "tool_help", "general"],
    )
    assert isinstance(result, dict)
    assert result.get("intent") == "general"


def test_shim_infer_intent_scan_task_returns_execute_task() -> None:
    result = b.infer_gis_intent(
        message="继续扫描 input 并制作行政区划图",
        valid_intents=["execute_task", "confirm_action", "preview", "cancel", "query_status", "tool_help", "general"],
    )
    assert isinstance(result, dict)
    assert result.get("intent") == "execute_task"


def test_shim_infer_intent_retry_phrase_not_general() -> None:
    result = b.infer_gis_intent(
        message="你再试试，换个方法试试？我需要你完成这个任务",
        valid_intents=["execute_task", "confirm_action", "preview", "cancel", "query_status", "tool_help", "general"],
    )
    assert isinstance(result, dict)
    assert result.get("intent") in {"execute_task", "confirm_action"}


def test_shim_infer_intent_followup_input_phrase_not_general() -> None:
    result = b.infer_gis_intent(
        message="就在input里面呀，继续之前的任务即可",
        valid_intents=["execute_task", "confirm_action", "preview", "cancel", "query_status", "tool_help", "general"],
    )
    assert isinstance(result, dict)
    assert result.get("intent") in {"execute_task", "confirm_action"}


def test_shim_generate_plan_returns_none_for_runtime_fallback() -> None:
    result = b.generate_gis_plan(task_description="制作行政区划图", context={"input_folder": "workspace/input"})
    assert result is None
