# -*- coding: utf-8 -*-
"""End-to-end smoke test of the engine loop with a scripted mock LLM.

Runs the real persistent kernel + catalog + verifier, but replaces the model
with a deterministic script so the loop can be tested without network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

from gis_cli.engine.llm_client import LLMResponse, ToolCall
from gis_cli.engine.runner import build_engine
from gis_cli.engine.state import Observation


class ScriptedMockLLM:
    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls = 0

    class _Cfg:
        model = "mock-model"

    config = _Cfg()

    def chat(self, messages, **kwargs):
        self.calls += 1
        joined = json.dumps(messages, ensure_ascii=False)
        if "验收专家" in joined:
            return LLMResponse(
                content=json.dumps(
                    {"pass": True, "per_requirement": [], "missing": [], "repair_hint": ""},
                    ensure_ascii=False,
                ),
                model="mock-model",
            )
        if self.script:
            return self.script.pop(0)
        return LLMResponse(content=json.dumps({"tool": "finish", "args": {"summary": "done"}}, ensure_ascii=False), model="mock-model")


def tc(name: str, args: dict) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=args, raw_arguments=json.dumps(args, ensure_ascii=False))],
        model="mock-model",
    )


def main() -> int:
    workspace = Path(r"E:\BST_lab\GIS_CLI\workspace")
    bundle = build_engine(workspace=workspace, refresh_catalog=False)
    mock = ScriptedMockLLM(
        [
            tc("update_tasklist", {"items": [
                {"id": "t1", "title": "扫描并确认数据", "status": "running"},
                {"id": "t2", "title": "生成并执行代码", "status": "pending"},
                {"id": "t3", "title": "验证并交付", "status": "pending"},
            ]}),
            tc("catalog_query", {"query": "city"}),
            tc("execute_code", {
                "code": (
                    "import arcpy\n"
                    "out = r'E:\\BST_lab\\GIS_CLI\\workspace\\output\\_engine_smoke.txt'\n"
                    "open(out, 'w', encoding='utf-8').write('hello engine')\n"
                    "set_result({'output': out, 'message': 'ok'})\n"
                ),
                "description": "写一个冒烟测试文件",
            }),
            tc("update_tasklist", {"items": [
                {"id": "t1", "title": "扫描并确认数据", "status": "done"},
                {"id": "t2", "title": "生成并执行代码", "status": "done"},
                {"id": "t3", "title": "验证并交付", "status": "running"},
            ]}),
            tc("finish", {
                "summary": "冒烟测试完成，已产出 _engine_smoke.txt",
                "methodology": "无（冒烟测试）",
                "artifacts": [r"E:\BST_lab\GIS_CLI\workspace\output\_engine_smoke.txt"],
            }),
        ]
    )
    bundle.loop.llm = mock
    bundle.loop.codec.client = mock
    bundle.loop.repairer.llm_client = mock
    bundle.verifier.llm_client = mock

    state = bundle.loop.run("冒烟测试：写一个文件并交付")
    print("STATUS:", state.status)
    print("TURNS:", state.turn)
    print("TASKLIST:")
    print(state.tasklist_digest())
    print("ARTIFACTS:", state.artifacts)
    print("SUMMARY:", state.final_summary)
    print("TRACE:", bundle.loop.recorder.jsonl_path)
    bundle.close()
    ok = state.status == "completed" and state.artifacts
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
