# -*- coding: utf-8 -*-
"""旧引擎（已下线，仅供追溯）。

## 这里是什么

两套 plan-then-execute 引擎，都已被 ``gis_cli.engine`` 的有界循环取代：

- ``orchestrator/``：TaskOrchestrator 系（意图识别 → 计划 → 执行 → 恢复），
  顶层 ``gis-cli`` 命令（``task-*``）的实现在这里。
- ``agent_legacy/``：GISAgent 系（planner + executor + memory + BAML 适配），
  ``gis-agent chat-legacy`` 的实现在这里。
- ``core_tools/``：两套引擎共用的工具框架（pydantic + ToolRegistry）、
  10 个 Tool 类、旧技能格式（``code_template``）、Web 界面。

## 为什么保留而不直接删

**观察期**：确认新引擎在 14 届真题全部任务上稳定后，再整体删除。
git 历史里也有完整记录，保留目录只是让"回退"变成一次 ``git mv`` 而不是
从历史里捞文件。

## 关键：不参与打包

``pyproject.toml`` 的 ``[tool.setuptools.packages.find]`` 里有
``exclude = ["gis_cli.legacy*"]`` —— 这些代码**不会**被安装到用户环境，
也不会被 CI 加载。它们只存在于源码仓库里。

## 不要在这里加新功能

需要改行为请改 ``gis_cli.engine`` / ``gis_cli.runtime`` / ``gis_cli.ui``。
"""

from __future__ import annotations

__all__: list[str] = []
