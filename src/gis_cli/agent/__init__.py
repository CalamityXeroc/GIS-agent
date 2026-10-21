"""GIS Agent 包。

## 为什么要惰性导出（重要）

这个包以前在 ``__init__.py`` 里**立即** import 全部旧引擎模块（``agent`` /
``planner`` / ``executor`` / ``llm`` / ``model_adaptation`` …）。后果：

    import gis_cli.agent.interactive      # 新引擎的交互入口
      → 触发 gis_cli.agent.__init__
      → 触发 agent.agent → core → pydantic / langchain / litellm

也就是说**新引擎明明一行都不用旧引擎，却被拖着必须装 langchain**。
实测：在"没有 pydantic"的环境里 `import gis_cli.agent.interactive` 直接
ImportError，而新引擎根本不需要它。

现在改成 PEP 562 的模块级 ``__getattr__`` 惰性导出：
- 访问 ``gis_cli.agent.Memory`` 这类旧名字时才真正 import 对应模块；
- 新引擎（``interactive`` / ``cli`` / ``evaluation`` 的新引擎部分）完全不
  触发旧引擎加载；
- 旧代码的 ``from gis_cli.agent import X`` 写法保持可用（向后兼容）。

旧引擎整体下线后，这个文件的旧名字映射表会一起删掉。
"""

from __future__ import annotations

from typing import Any

#: 旧名字 → 所在模块（**惰性**加载，只有真被访问时才 import）。
_LEGACY_EXPORTS: dict[str, str] = {
    # agent
    "GISAgent": "..legacy.agent_legacy.agent",
    "AgentConfig": "..legacy.agent_legacy.agent",
    "AgentResponse": "..legacy.agent_legacy.agent",
    # memory
    "Memory": "..legacy.agent_legacy.memory",
    "ConversationTurn": "..legacy.agent_legacy.memory",
    "MemoryStore": "..legacy.agent_legacy.memory",
    "Role": "..legacy.agent_legacy.memory",
    "StructuredMemoryItem": "..legacy.agent_legacy.memory",
    "ReflectionItem": "..legacy.agent_legacy.memory",
    # state
    "LangGraphStateManager": "..legacy.agent_legacy.state_manager",
    "WorkflowState": "..legacy.agent_legacy.state_manager",
    # planner
    "AgentPlanner": "..legacy.agent_legacy.planner",
    "Plan": "..legacy.agent_legacy.planner",
    "PlanStep": "..legacy.agent_legacy.planner",
    "StepStatus": "..legacy.agent_legacy.planner",
    # executor
    "Executor": "..legacy.agent_legacy.executor",
    "ExecutionResult": "..legacy.agent_legacy.executor",
    "ExecutionTrace": "..legacy.agent_legacy.executor",
    "ExecutionMode": "..legacy.agent_legacy.executor",
    "ExecutionAdapter": "..legacy.agent_legacy.execution_adapter",
    # prompts
    "SystemPrompts": "..legacy.agent_legacy.prompts",
    "PromptBuilder": "..legacy.agent_legacy.prompts",
    "build_agent_prompt": "..legacy.agent_legacy.prompts",
    # llm
    "LLMClient": "..legacy.agent_legacy.llm",
    "LLMConfig": "..legacy.agent_legacy.llm",
    "OpenAIClient": "..legacy.agent_legacy.llm",
    "UnifiedModelClient": "..legacy.agent_legacy.llm",
    "LiteLLMClient": "..legacy.agent_legacy.llm",
    "MockLLMClient": "..legacy.agent_legacy.llm",
    "create_llm_client": "..legacy.agent_legacy.llm",
    # model adaptation
    "PromptAdapter": "..legacy.agent_legacy.model_adaptation",
    "PlanStandardizer": "..legacy.agent_legacy.model_adaptation",
    "AdaptationMetrics": "..legacy.agent_legacy.model_adaptation",
    "BAMLBridge": "..legacy.agent_legacy.model_adaptation",
    # context
    "ContextHub": "..legacy.agent_legacy.context_hub",
    "ContextSnapshot": "..legacy.agent_legacy.context_hub",
}

#: 新引擎/共用部分：这些模块**不依赖** pydantic/langchain/litellm，
#: 可以直接导出（``vector_retriever`` 被 ``engine.project_log`` 用到）。
from .vector_retriever import (  # noqa: E402  (放在映射表后便于阅读)
    BaseVectorRetriever,
    HashingVectorRetriever,
    RetrieverMatch,
)

__all__ = [
    # 共用（新引擎也用）
    "BaseVectorRetriever",
    "HashingVectorRetriever",
    "RetrieverMatch",
    # 旧引擎（惰性；下线后删除）
    *_LEGACY_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    """PEP 562：访问旧名字时才 import 对应模块（见模块 docstring）。"""
    module_name = _LEGACY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
