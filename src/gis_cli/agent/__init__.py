"""GIS Agent module - Complete AI agent for GIS workflows.

This module provides a full-featured agent that can:
- Understand natural language GIS requests
- Plan multi-step workflows using tools and skills
- Execute plans with error handling and recovery
- Maintain conversation context and memory
- Learn from feedback and improve over time
"""

from .agent import GISAgent, AgentConfig, AgentResponse
from .memory import Memory, ConversationTurn, MemoryStore, Role, StructuredMemoryItem, ReflectionItem
from .vector_retriever import BaseVectorRetriever, HashingVectorRetriever, RetrieverMatch
from .state_manager import LangGraphStateManager, WorkflowState
from .evaluation import (
    AgentEvaluationLoop,
    EvaluationMetrics,
    EvaluationReport,
    MultiModelBenchmark,
    MultiModelBenchmarkReport,
    ModelBenchmarkMetrics,
)
from .context_hub import ContextHub, ContextSnapshot
from .planner import AgentPlanner, Plan, PlanStep, StepStatus
from .executor import Executor, ExecutionResult, ExecutionTrace, ExecutionMode
from .execution_adapter import ExecutionAdapter
from .prompts import SystemPrompts, PromptBuilder, build_agent_prompt
from .llm import LLMClient, LLMConfig, OpenAIClient, UnifiedModelClient, LiteLLMClient, MockLLMClient, create_llm_client
from .model_adaptation import PromptAdapter, PlanStandardizer, AdaptationMetrics, BAMLBridge

__all__ = [
    # Agent
    "GISAgent",
    "AgentConfig",
    "AgentResponse",
    # Memory
    "Memory",
    "ConversationTurn",
    "MemoryStore",
    "Role",
    "StructuredMemoryItem",
    "ReflectionItem",
    "BaseVectorRetriever",
    "HashingVectorRetriever",
    "RetrieverMatch",
    "LangGraphStateManager",
    "WorkflowState",
    "AgentEvaluationLoop",
    "EvaluationMetrics",
    "EvaluationReport",
    "MultiModelBenchmark",
    "MultiModelBenchmarkReport",
    "ModelBenchmarkMetrics",
    "ContextHub",
    "ContextSnapshot",
    # Planner
    "AgentPlanner",
    "Plan",
    "PlanStep",
    "StepStatus",
    # Executor
    "Executor",
    "ExecutionAdapter",
    "ExecutionResult",
    "ExecutionTrace",
    "ExecutionMode",
    # Prompts
    "SystemPrompts",
    "PromptBuilder",
    "build_agent_prompt",
    # LLM
    "LLMClient",
    "LLMConfig",
    "OpenAIClient",
    "UnifiedModelClient",
    "LiteLLMClient",
    "MockLLMClient",
    "create_llm_client",
    "PromptAdapter",
    "PlanStandardizer",
    "AdaptationMetrics",
    "BAMLBridge",
]
