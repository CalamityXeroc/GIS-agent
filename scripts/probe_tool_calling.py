# -*- coding: utf-8 -*-
"""Probe native tool calling for the configured models."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")
from gis_cli.engine.llm_client import EngineLLMClient, EngineLLMConfig

cfg = EngineLLMConfig.from_file(r"E:\BST_lab\GIS_CLI\config\llm_config.json")
client = EngineLLMClient(cfg)

tools = [
    {
        "type": "function",
        "function": {
            "name": "catalog_query",
            "description": "查询数据目录中的数据集信息",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "关键词"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "执行 ArcPy 代码",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
]

messages = [
    {"role": "system", "content": "你是 GIS 智能体。先查数据目录，再写代码。"},
    {"role": "user", "content": "请先查询名字包含 city 的数据集，只调用一次工具。"},
]

for model in ["GLM-5.2", "Deepseek-v4-flash"]:
    print(f"\n=== {model} ===")
    try:
        resp = client.chat(messages, tools=tools, model=model, task_type="agent", max_tokens=2000)
        print("content:", (resp.content or "")[:200])
        print("reasoning:", (resp.reasoning or "")[:120])
        print("tool_calls:", [(c.name, c.arguments) for c in resp.tool_calls])
        print("finish:", resp.finish_reason, "usage:", resp.usage)
    except Exception as exc:
        print("ERROR:", type(exc).__name__, str(exc)[:400])
