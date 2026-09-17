# -*- coding: utf-8 -*-
"""Distill successful runs into candidate GIS methodology recipes.

The loop between "did it once" and "can do it every time" closes here: after a
successful run the executed code, goal, and methodology are turned into a
candidate Recipe (with validation hooks), saved for human review rather than
silently entering the builtin library.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..engine.tool_codec import extract_json_object
from .schema import Recipe

logger = logging.getLogger(__name__)


class RecipeDistiller:
    """Turn a successful run into a candidate recipe."""

    def __init__(self, llm_client: Any, library: Any):
        self.llm_client = llm_client
        self.library = library

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def collect_successful_code(trace_jsonl: str | Path, *, max_snippets: int = 6, snippet_chars: int = 6000) -> list[str]:
        """Extract successful execute_code snippets from a run trace."""
        snippets: list[str] = []
        path = Path(trace_jsonl)
        if not path.exists():
            return snippets
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                payload = event.get("payload", {})
                if event.get("event") == "tool_call" and payload.get("tool") == "execute_code":
                    code = str((payload.get("args") or {}).get("code", "") or "")
                    if code.strip():
                        snippets.append(code[:snippet_chars])
        except Exception as exc:
            logger.warning("trace read failed: %s", exc)
        return snippets[-max_snippets:]

    # ------------------------------------------------------------------ distill
    def distill(
        self,
        *,
        goal: str,
        methodology: str,
        final_summary: str,
        code_snippets: list[str],
        artifacts: list[str],
    ) -> Recipe | None:
        """Ask the model to generalize the successful run into a recipe."""
        if not code_snippets:
            return None
        code_block = "\n\n---\n\n".join(code_snippets)
        prompt = (
            "你是 GIS 方法论专家。下面是一次成功完成的 GIS 任务的最终代码。请把它泛化为一条可复用的\n"
            "GIS 方法论配方：把任务特定的值改成 {{参数}} 槽位（字符串参数在代码里写作 {{param}}，\n"
            "代码/表达式内嵌文本写作 @@param@@），并给出触发词、参数说明与验收断言。\n\n"
            f"## 任务目标\n{goal}\n\n"
            f"## 采用的方法论\n{methodology}\n\n"
            f"## 结果摘要\n{final_summary[:2000]}\n\n"
            f"## 成功代码\n```python\n{code_block[:14000]}\n```\n\n"
            "只输出一个 JSON 对象：\n"
            "{\n"
            '  "id": "snake_case_id",\n'
            '  "name": "中文名",\n'
            '  "description": "一句话用途",\n'
            '  "domain": "projection|spatial_analysis|cartography|data_quality|data_management|spatial_statistics",\n'
            '  "triggers": ["触发词"],\n'
            '  "params": {"参数名": {"type": "string|integer|boolean", "required": true, "default": null, "description": "..."}},\n'
            '  "steps": ["步骤"],\n'
            '  "code_template": "泛化后的完整代码",\n'
            '  "validation": [{"type": "vector_exists|feature_count_gt|crs_wkid|fields_present|file_exists|image_not_blank", "path": "{{参数}}", "value": ...}],\n'
            '  "pitfalls": ["坑位"],\n'
            '  "examples": ["示例"]\n'
            "}\n"
            "注意：code_template 中除参数槽位外，其余花括号（如字典、f-string）保持原样，不要转义。"
        )
        try:
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": "你只输出一个 JSON 对象。"},
                    {"role": "user", "content": prompt},
                ],
                task_type="requirements",
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning("distill LLM call failed: %s", exc)
            return None
        payload = extract_json_object(response.content)
        if not isinstance(payload, dict):
            logger.warning("distill response unparseable")
            return None
        try:
            recipe = Recipe.from_dict(payload)
        except Exception as exc:
            logger.warning("distilled recipe invalid: %s", exc)
            return None
        if not recipe.id or not recipe.code_template:
            return None
        return recipe

    # --------------------------------------------------------------------- save
    def save_candidate(self, recipe: Recipe, out_dir: str | Path, provenance: str = "") -> Path:
        recipe.provenance = provenance or "distilled"
        recipe.version = "0.1-draft"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{recipe.id}.yaml"
        recipe.to_yaml(path)
        return path
