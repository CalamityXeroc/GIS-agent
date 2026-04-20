"""Project-local BAML client shim.

This module provides an importable `baml_client.b` object with the core
capabilities expected by the agent runtime:
- infer_gis_intent
- refine_gis_task
- generate_gis_plan
- suggest_gis_recovery

It is intentionally lightweight and deterministic. When a generated BAML client
is later introduced, this module can be replaced by the generated package
without changing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _pick_intent(message: str, valid_intents: list[str]) -> str:
    text = (message or "").strip().lower()
    # Greeting / identity questions should be treated as general chat.
    general_markers = [
        "你好", "您好", "hello", "hi", "hey",
        "你是谁", "你是什么", "什么模型", "哪个模型", "model",
        "能做什么", "可以做什么", "功能",
    ]
    if any(marker in text for marker in general_markers):
        if "general" in valid_intents:
            return "general"

    status_markers = ["扫好了吗", "完成了吗", "进度", "状态", "结果", "好了没", "status"]
    if any(marker in text for marker in status_markers):
        if "query_status" in valid_intents:
            return "query_status"

    preferred = [
        (
            "confirm_action",
            [
                "确认", "执行", "开始", "ok", "yes", "1", "继续执行", "开始执行",
                "再试", "再试试", "重试", "retry", "换个方法", "换个方式", "试试别的",
            ],
        ),
        ("cancel", ["取消", "停止", "中止", "cancel"]),
        ("tool_help", ["帮助", "怎么用", "help"]),
        (
            "execute_task",
            [
                "规划", "任务", "处理", "导出", "run", "execute", "扫描", "扫", "制图", "出图", "绘制",
                "地图", "行政区划", "图层", "合并", "投影", "分析", "继续之前", "继续任务", "继续上一个",
                "就在input", "之前的文件夹", "完成这个任务", "arcpy",
            ],
        ),
    ]
    for intent, markers in preferred:
        if any(marker in text for marker in markers):
            if intent in valid_intents:
                return intent

    # Safe default must be "general" to avoid accidental task execution.
    if "general" in valid_intents:
        return "general"
    return valid_intents[0] if valid_intents else "general"


def _normalize_task(message: str) -> str:
    value = (message or "").strip()
    return value if value else "请根据当前上下文继续执行 GIS 任务"


def _default_plan(task_description: str) -> dict[str, Any] | None:
    # Return None to let runtime planner use real LLM/rule planning.
    # Avoid forcing a low-quality fixed one-step scan plan.
    _ = task_description
    return None


def _default_recovery() -> dict[str, Any] | None:
    # Return None to let runtime use built-in recovery strategies.
    return None


@dataclass
class _BamlShim:
    def infer_gis_intent(self, message: str, valid_intents: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        intents = [str(x).strip() for x in (valid_intents or []) if str(x).strip()]
        return {"intent": _pick_intent(message, intents if intents else ["general"])}

    def classify_gis_intent(self, message: str, valid_intents: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        return self.infer_gis_intent(message=message, valid_intents=valid_intents)

    def detect_gis_intent(self, message: str, valid_intents: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        return self.infer_gis_intent(message=message, valid_intents=valid_intents)

    def refine_gis_task(self, message: str, **kwargs: Any) -> str:
        return _normalize_task(message)

    def normalize_gis_task(self, message: str, **kwargs: Any) -> str:
        return self.refine_gis_task(message=message)

    def rewrite_gis_task(self, message: str, **kwargs: Any) -> str:
        return self.refine_gis_task(message=message)

    def generate_gis_plan(self, task_description: str, context: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any] | None:
        # Keep a deterministic baseline plan. Planner can still fall back/augment.
        return _default_plan(task_description)

    def create_gis_plan(self, task_description: str, context: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any] | None:
        return self.generate_gis_plan(task_description=task_description, context=context)

    def generate_execution_plan(self, task_description: str, context: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any] | None:
        return self.generate_gis_plan(task_description=task_description, context=context)

    def suggest_gis_recovery(
        self,
        plan: dict[str, Any] | None = None,
        failed_step: dict[str, Any] | None = None,
        remaining_steps: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        return _default_recovery()

    def generate_recovery_plan(self, **kwargs: Any) -> dict[str, Any] | None:
        return self.suggest_gis_recovery(**kwargs)

    def create_recovery_plan(self, **kwargs: Any) -> dict[str, Any] | None:
        return self.suggest_gis_recovery(**kwargs)


b = _BamlShim()
