# -*- coding: utf-8 -*-
"""Code repair loop: feed errors back to the model and retry.

This is the single highest-leverage reliability feature: instead of treating a
failed ``execute_code`` as terminal, the engine asks the model to fix the code
using the real traceback and the real data schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .state import Observation
from .tool_codec import extract_json_object

logger = logging.getLogger(__name__)


@dataclass
class RepairAttempt:
    """One repair round."""

    attempt: int
    error: str
    code: str
    thought: str = ""
    ok: bool = False
    observation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "error": self.error,
            "thought": self.thought,
            "ok": self.ok,
            "code_preview": self.code[:1200],
            "observation": self.observation,
        }


def _extract_code_block(text: str) -> str:
    """Extract the largest fenced code block from a model reply."""
    import re

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return ""
    return max(blocks, key=len).strip()


class CodeRepairer:
    """Bounded self-repair for execute_code failures."""

    def __init__(
        self,
        *,
        llm_client: Any,
        code_runner: Any,
        max_attempts: int = 4,
    ):
        self.llm_client = llm_client
        self.code_runner = code_runner
        self.max_attempts = max(1, int(max_attempts))

    # ------------------------------------------------------------------ public
    def run(
        self,
        *,
        intent: str,
        code: str,
        observation: Observation,
        catalog_digest: str = "",
        recipes_digest: str = "",
        timeout: float = 600.0,
        workspace: str | None = None,
        on_attempt: Any = None,
    ) -> tuple[Observation, str, list[RepairAttempt]]:
        """Repair failed code until it works or attempts are exhausted."""
        attempts: list[RepairAttempt] = []
        current_code = code
        current_obs = observation
        seen_fingerprints: set[str] = set()

        for attempt in range(1, self.max_attempts + 1):
            if current_obs.ok:
                return current_obs, current_code, attempts
            fingerprint = self._fingerprint(current_obs)
            if fingerprint in seen_fingerprints:
                logger.info("repair stopped: repeated identical failure")
                break
            seen_fingerprints.add(fingerprint)

            fix = self._ask_for_fix(
                intent=intent,
                code=current_code,
                observation=current_obs,
                catalog_digest=catalog_digest,
                recipes_digest=recipes_digest,
                history=attempts,
            )
            record = RepairAttempt(
                attempt=attempt,
                error=current_obs.summary,
                code=str(fix.get("code", "") or ""),
                thought=str(fix.get("thought", "") or ""),
            )
            if fix.get("give_up") or not record.code.strip():
                record.observation = {"give_up": True, "reason": fix.get("reason", "")}
                attempts.append(record)
                if on_attempt:
                    on_attempt(record)
                current_obs = Observation(
                    ok=False,
                    summary=f"修复放弃：{fix.get('reason') or '模型未能给出可用修复代码'}",
                    data=current_obs.data,
                    error=current_obs.error,
                    hint=str(fix.get("reason", "")),
                )
                break

            result = self.code_runner.run(record.code, timeout=timeout, workspace=workspace)
            record.ok = bool(result.ok)
            record.observation = {
                "stdout": (result.stdout or "")[-2000:],
                "error": result.error,
            }
            attempts.append(record)
            if on_attempt:
                on_attempt(record)

            if result.ok:
                from .tools import extract_outputs

                artifacts = extract_outputs(result.result, workspace or ".")
                current_obs = Observation(
                    ok=True,
                    summary=f"修复成功（第 {attempt} 次）：{record.thought[:120]}",
                    data={"stdout": (result.stdout or "")[-4000:], "result": result.result},
                    artifacts=artifacts,
                    images=result.display_images[:3],
                )
                current_code = record.code
                return current_obs, current_code, attempts

            current_obs = Observation(
                ok=False,
                summary=f"修复后仍失败: [{(result.error or {}).get('type')}] {(result.error or {}).get('message')}",
                data={"stdout": (result.stdout or "")[-2000:], "stderr": (result.stderr or "")[-1500:]},
                error=result.error,
                hint=current_obs.hint,
            )
            current_code = record.code

        return current_obs, current_code, attempts

    # ------------------------------------------------------------------ prompt
    def _ask_for_fix(
        self,
        *,
        intent: str,
        code: str,
        observation: Observation,
        catalog_digest: str,
        recipes_digest: str,
        history: list[RepairAttempt],
    ) -> dict[str, Any]:
        error = observation.error or {}
        history_text = "\n".join(
            f"- 第{a.attempt}次: {a.thought[:160]} -> {'成功' if a.ok else '仍失败'}" for a in history
        ) or "（无）"
        prompt = (
            "你是一名 GIS/ArcPy 专家。下面这段 ArcPy 代码执行失败了，请修正它。\n\n"
            f"## 任务意图\n{intent}\n\n"
            f"## 失败代码\n```python\n{code[:8000]}\n```\n\n"
            f"## 错误\n类型: {error.get('type', '')}\n消息: {error.get('message', '')}\n"
            f"traceback:\n{(error.get('traceback') or '')[-2500:]}\n\n"
            f"## 标准输出（尾部）\n{(observation.data or {}).get('stdout', '')[-1500:] if isinstance(observation.data, dict) else ''}\n\n"
            f"## 数据目录（真实字段名/坐标系，禁止臆造）\n{catalog_digest[:3500]}\n\n"
            f"## 可参考的已验证配方\n{recipes_digest[:2500]}\n\n"
            f"## 之前的修复尝试\n{history_text}\n\n"
            "要求：\n"
            "1. 只修正错误，不要改变任务的业务目标。\n"
            "2. 字段名必须来自上面的数据目录；坐标系必须明确。\n"
            "3. 用 arcpy.Exists 检查输入，输出路径写到 workspace/output 下。\n"
            "4. 用 set_result({...}) 返回关键结果与输出路径。\n"
            "5. 如果确实无法修复，给出 give_up 与原因。\n\n"
            "只输出一个 JSON 对象：\n"
            '{"thought": "修正思路", "code": "完整可执行代码", "give_up": false, "reason": ""}'
        )
        response = self.llm_client.chat(
            [
                {"role": "system", "content": "你只输出一个 JSON 对象，不要 markdown 代码块。"},
                {"role": "user", "content": prompt},
            ],
            task_type="code_repair",
            temperature=0.1,
        )
        payload = extract_json_object(response.content)
        if isinstance(payload, dict):
            return payload

        # Reasoning models may burn the token budget on reasoning_content and
        # return an empty/partial answer. Retry once with a bigger budget, then
        # fall back to extracting a fenced code block.
        if not (response.content or "").strip() or response.finish_reason == "length":
            try:
                retry = self.llm_client.chat(
                    [
                        {"role": "system", "content": "你只输出一个 JSON 对象。直接给结论，不要长篇推理。"},
                        {"role": "user", "content": prompt},
                    ],
                    task_type="code_repair",
                    temperature=0.1,
                    max_tokens=32000,
                )
                payload = extract_json_object(retry.content)
                if isinstance(payload, dict):
                    return payload
                response = retry
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("repair retry failed: %s", exc)

        code = _extract_code_block(response.content or "")
        if code:
            return {"thought": "从回复的代码块中提取", "code": code}
        return {"give_up": True, "reason": f"修复响应无法解析: {response.content[:300]}"}

    @staticmethod
    def _fingerprint(observation: Observation) -> str:
        error = observation.error or {}
        text = f"{error.get('type', '')}|{error.get('message', '')}"
        return " ".join(text.split()).lower()[:300]
