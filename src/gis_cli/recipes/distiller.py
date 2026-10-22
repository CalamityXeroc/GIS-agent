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
    def collect_successful_code(
        trace_jsonl: str | Path, *, max_snippets: int = 6, snippet_chars: int = 6000
    ) -> list[str]:
        """从 trace 里挑出**最有代表性**的 execute_code 片段。

        以前直接取最后 6 个 —— 实测那 6 个往往是最后的核验小脚本（总共
        2318 字符），真正干活的代码在前面。现在按 tool_call/tool_result 配对，
        优先选**成功且产出过产物**的，其次成功的，同档取靠后的（最终版）。
        """
        path = Path(trace_jsonl)
        if not path.exists():
            return []
        pending: str | None = None
        scored: list[tuple[int, int, str]] = []
        index = 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                kind = event.get("event")
                payload = event.get("payload") or {}
                if kind == "tool_call" and payload.get("tool") == "execute_code":
                    pending = str((payload.get("args") or {}).get("code", "") or "")
                    continue
                if kind != "tool_result" or payload.get("tool") != "execute_code":
                    continue
                index += 1
                code = pending or ""
                pending = None
                if not code.strip():
                    continue
                ok = bool(payload.get("ok"))
                artifacts = payload.get("artifacts") or []
                # 2 = 成功且产出过产物（真正干活的代码），1 = 成功，0 = 失败
                priority = 2 if (ok and artifacts) else (1 if ok else 0)
                if priority == 0:
                    continue
                scored.append((priority, index, code[:snippet_chars]))
        except Exception as exc:
            logger.warning("trace read failed: %s", exc)
            return []
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [code for _priority, _index, code in scored[:max_snippets]]

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
            '  "params": {"参数名": {"type": "string|integer|boolean", "required": true, "default": "本次运行实际用的值", "description": "..."}},\n'
            '  "steps": ["步骤"],\n'
            '  "code_template": "泛化后的完整代码",\n'
            '  "validation": [{"type": "vector_exists|feature_count_gt|crs_wkid|fields_present|file_exists|file_contains|file_matches|image_not_blank", "path": "{{参数}}", "value": ...}],\n'
            '  "pitfalls": ["坑位"],\n'
            '  "examples": ["示例"]\n'
            "}\n"
            "## 硬性要求（违反会被门禁拦下）\n"
            "1. **每个必填参数都要给 `default`，填本次运行实际用的那个值**。配方要能被直接调用，\n"
            "   也要能用默认值回放验证。\n"
            "2. **代码里不得出现写死的绝对路径**（如 `E:\\\\data\\\\x.shp`、`/home/...`、\n"
            "   `benchmark/runs/...`）。输入/输出路径一律做成参数槽位。\n"
            "3. `id` 用小写蛇形（`^[a-z][a-z0-9_]*$`），不要中文/大写/空格。\n"
            "4. `validation` 里的 `path` 只能引用上面声明过的参数槽位。\n"
            "   断言类型必须与**产物类型**匹配：\n"
            "   - 矢量（.gdb 里的要素类 / .shp）：vector_exists / feature_count_gt / crs_wkid / fields_present\n"
            "   - 栅格（.tif / .img / 库内栅格）：raster_exists / raster_not_constant\n"
            "   - 文本/报告/图片：只用 file_exists / file_contains / file_matches（正则），\n"
            "     不要用 fields_present 这类矢量断言（回放时会直接失败）\n"
            "   - 只校验文本的**格式**（如两位小数、日期、固定前缀）用 file_matches，value 写正则；\n"
            "     校验某个字面字符串出现用 file_contains（**不要**把正则塞进 file_contains）\n"
            "5. **参数不要超过 10 个**。只把“换一个任务时真的会变”的东西做成参数：\n"
            "   输入/输出路径、字段名、编码、目标坐标系、关键阈值。\n"
            "   任务特有的常量（某次数据里异常块的经纬度范围、只本次用得上的清理开关、\n"
            "   报告标题）直接写进代码，不要做成参数——否则调用方要填 30 个值，\n"
            "   等于没泛化。\n"
            "6. 工作区根路径用 `{{workspace}}` 参数，默认值写 `@workspace@`（运行时自动填）。\n"
            "7. `{{param}}` 渲染出来**已经是带引号的 Python 字面量**，不要再自己包一层引号：\n"
            "   写 `ws = {{workspace}}`，**不要**写 `ws = \"'{{workspace}}'\"`。\n"
            "   只有**已经在引号内部**时才用 `@@param@@`（原样替换，不加引号），如 `\"output/@@name@@.txt\"`；\n"
            "   独立作为值或表达式时必须用 `{{param}}` —— `@@x@@.split(',')` 渲染后是语法错误（实测）。\n"
            "   同样地，**下标/函数实参/比较**里也用 `{{param}}`：`r[@@group_field@@]` 会渲染成\n"
            "   `r[dlmc]` → NameError（实测），正确写法是 `r[{{group_field}}]`。\n"
            "8. code_template 中除参数槽位外，其余花括号（如字典、f-string）保持原样，不要转义。\n"
            "9. **代码结尾必须调 `set_result({...})`**，把产物路径放进 `output` / `artifacts` 字段\n"
            "   （如 `set_result({'output': out_fc, 'artifacts': [out_fc, report_path]})`）。\n"
            "   引擎靠它收集交付物；只 print 不 set_result 会被当成“没有产物”。\n"
        )
        try:
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": "你只输出一个 JSON 对象。"},
                    {"role": "user", "content": prompt},
                ],
                task_type="requirements",
                temperature=0.1,
                # 配方含完整代码模板，JSON 很容易上万字符。默认 max_tokens 不够时
                # finish_reason=length，JSON 断在字符串中间 → extract_json_object 失败
                # （实测 14 届蒸馏：5082 字符处被截）。
                max_tokens=32000,
                # 蒸馏是后台增强，不该把网关抖动的时间算进用户等待
                budget_seconds=180.0,
            )
        except Exception as exc:
            logger.warning("distill LLM call failed: %s", exc)
            return None
        payload = extract_json_object(response.content)
        if not isinstance(payload, dict):
            # 两种失败都要重试一次：被截断（finish_reason=length），以及"正常结束
            # 但不是合法 JSON"（模型加了解释文字 / 代码块 / 非法转义）。实测后者
            # 偶发，不重试就等于白丢一次蒸馏（2026-10-22 端到端验证首跑即命中）。
            truncated = str(getattr(response, "finish_reason", "")) == "length"
            hint = (
                "## 重要：上一次输出被 token 上限截断了\n"
                "请精简 code_template：去掉注释、去掉调试 print、合并重复逻辑，"
                "但必须保证 JSON 完整闭合、代码语法正确。"
                if truncated
                else "## 重要：上一次输出不是合法 JSON（夹了解释文字/代码块/非法转义）\n"
                "这次只输出一个 JSON 对象：不要前言、不要结语、不要 markdown 代码块；"
                "代码里的换行写成 \\n，双引号写成 \\\"。"
            )
            logger.info("distill output unusable (truncated=%s), retrying", truncated)
            try:
                retry = self.llm_client.chat(
                    [
                        {"role": "system", "content": "你只输出一个 JSON 对象，不要 markdown 代码块。"},
                        {"role": "user", "content": prompt + "\n\n" + hint},
                    ],
                    task_type="requirements",
                    temperature=0.1,
                    max_tokens=32000,
                    budget_seconds=180.0,
                )
                payload = extract_json_object(retry.content)
            except Exception as exc:
                logger.warning("distill retry failed: %s", exc)
        if not isinstance(payload, dict):
            logger.warning("distill response unparseable (finish_reason=%s)", getattr(response, "finish_reason", ""))
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
