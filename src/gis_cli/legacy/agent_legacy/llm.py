"""LLM integration for GIS Agent.

Provides a unified interface to various LLM providers:
- OpenAI API
- Azure OpenAI
- Local models (Ollama, etc.)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator


def _to_bool(value: Any, default: bool) -> bool:
    """Parse loose truthy/falsy values from env/json sources."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_str_list(value: Any, default: list[str]) -> list[str]:
    """Parse list-like values from env/json sources into non-empty str list."""
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default)
        # Try JSON array first.
        if text.startswith("["):
            try:
                payload = json.loads(text)
                if isinstance(payload, list):
                    return [str(v).strip() for v in payload if str(v).strip()]
            except Exception:
                pass
        # Fallback: comma-separated list.
        return [item.strip() for item in text.split(",") if item.strip()]
    return list(default)


def _clean_model_name(model: str, provider: str = "", api_base: str = "") -> str:
    """Normalize model names with provider-aware behavior.

    Some OpenAI-compatible gateways (e.g. siliconflow) require the full model
    identifier with provider prefix (such as ``Qwen/Qwen3.5-397B-A17B``).
    """
    cleaned = (model or "").strip()
    if "/" not in cleaned:
        return cleaned

    provider_lower = (provider or "").strip().lower()
    api_base_lower = (api_base or "").strip().lower()

    # Keep full model id for known OpenAI-compatible gateways.
    # Note: DeepSeek official API does NOT use prefixed model names.
    # SiliconFlow/Qwen etc. use org/model-name format, so keep prefix for them.
    preserve_prefix_providers = {
        "siliconflow", "qwen", "minimax", "zhipu", "glm", "openrouter"
    }
    if provider_lower in preserve_prefix_providers:
        return cleaned
    if provider_lower == "deepseek":
        # Strip org/ prefix AND lowercase — DeepSeek API rejects mixed-case
        # e.g. deepseek-ai/DeepSeek-V4-Flash → deepseek-v4-flash
        return cleaned.split("/")[-1].lower()

    # For non-official OpenAI bases, prefer preserving full id to avoid 400 model-not-found.
    if api_base_lower and "api.openai.com" not in api_base_lower:
        return cleaned

    # Default behavior for official OpenAI-style model aliases.
    return cleaned.split("/")[-1]


def _model_profile(provider: str, model: str) -> dict[str, Any]:
    """Return provider/model-specific default inference parameters."""
    p = (provider or "").strip().lower()
    m = (model or "").strip().lower()

    # Conservative defaults (best for deterministic planning outputs).
    profile = {
        "temperature": 0.5,
        "max_tokens": 2048,
        "timeout": 60,
        "retry_count": 1,
    }

    if "qwen" in m or p in {"qwen", "siliconflow"}:
        profile.update({"temperature": 0.3, "max_tokens": 2500, "timeout": 120, "retry_count": 2})
    elif "deepseek" in m or p == "deepseek":
        profile.update({"temperature": 0.2, "max_tokens": 3000, "timeout": 120, "retry_count": 2})
    elif "gpt" in m or p in {"openai", "azure"}:
        profile.update({"temperature": 0.5, "max_tokens": 2048, "timeout": 90, "retry_count": 1})
    elif "claude" in m or p == "anthropic":
        profile.update({"temperature": 0.5, "max_tokens": 2048, "timeout": 90, "retry_count": 1})
    elif "minimax" in m or p == "minimax":
        profile.update({"temperature": 0.3, "max_tokens": 2500, "timeout": 120, "retry_count": 2})

    return profile


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    
    model: str = "gpt-4"
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    provider: str = "openai"
    fallback_models: list[str] = field(default_factory=list)
    retry_count: int = 1
    enable_prompt_optimizer: bool = True
    enable_baml_standardizer: bool = True
    provider_models: dict[str, list[str]] = field(default_factory=dict)
    routing_rules: dict[str, list[str]] = field(default_factory=dict)
    baml_functions: dict[str, list[str]] = field(default_factory=dict)
    enable_baml_preflight: bool = True
    baml_preflight_strict: bool = False
    baml_required_capabilities: list[str] = field(default_factory=lambda: ["intent", "task_refine", "planning", "recovery"])
    enable_baml_builtin_fallback: bool = True
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables."""
        model = os.environ.get("GIS_LLM_MODEL", "gpt-4")
        provider = os.environ.get("GIS_LLM_PROVIDER", "openai")
        profile = _model_profile(provider, model)

        return cls(
            model=model,
            api_key=os.environ.get("GIS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("GIS_LLM_API_BASE") or os.environ.get("OPENAI_API_BASE"),
            temperature=float(os.environ.get("GIS_LLM_TEMPERATURE", str(profile["temperature"]))),
            max_tokens=int(os.environ.get("GIS_LLM_MAX_TOKENS", str(profile["max_tokens"]))),
            timeout=int(os.environ.get("GIS_LLM_TIMEOUT", str(profile["timeout"]))),
            provider=provider,
            fallback_models=[m.strip() for m in os.environ.get("GIS_LLM_FALLBACK_MODELS", "").split(",") if m.strip()],
            retry_count=max(1, int(os.environ.get("GIS_LLM_RETRY_COUNT", str(profile["retry_count"])))),
            enable_prompt_optimizer=_to_bool(os.environ.get("GIS_ENABLE_PROMPT_OPTIMIZER"), True),
            enable_baml_standardizer=_to_bool(os.environ.get("GIS_ENABLE_BAML_STANDARDIZER"), True),
            provider_models=_load_json_object_from_env("GIS_LLM_PROVIDER_MODELS"),
            routing_rules=_load_json_object_from_env("GIS_LLM_ROUTING_RULES"),
            baml_functions=_load_json_object_from_env("GIS_BAML_FUNCTIONS"),
            enable_baml_preflight=_to_bool(os.environ.get("GIS_ENABLE_BAML_PREFLIGHT"), True),
            baml_preflight_strict=_to_bool(os.environ.get("GIS_BAML_PREFLIGHT_STRICT"), False),
            baml_required_capabilities=_to_str_list(
                os.environ.get("GIS_BAML_REQUIRED_CAPABILITIES"),
                ["intent", "task_refine", "planning", "recovery"],
            ),
            enable_baml_builtin_fallback=_to_bool(os.environ.get("GIS_ENABLE_BAML_BUILTIN_FALLBACK"), True),
        )
    
    @classmethod
    def from_file(cls, path: str) -> "LLMConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = data.get("model", "gpt-4")
        provider = data.get("provider", "openai")
        profile = _model_profile(provider, model)
        
        return cls(
            model=model,
            api_key=data.get("api_key"),
            api_base=data.get("api_base"),
            temperature=data["temperature"] if "temperature" in data else profile["temperature"],
            max_tokens=data["max_tokens"] if "max_tokens" in data else profile["max_tokens"],
            timeout=data["timeout"] if "timeout" in data else profile["timeout"],
            provider=provider,
            fallback_models=data.get("fallback_models", []) if isinstance(data.get("fallback_models"), list) else [],
            retry_count=max(1, int(data["retry_count"])) if "retry_count" in data else profile["retry_count"],
            enable_prompt_optimizer=_to_bool(data.get("enable_prompt_optimizer"), True),
            enable_baml_standardizer=_to_bool(data.get("enable_baml_standardizer"), True),
            provider_models=_normalize_model_map(data.get("provider_models")),
            routing_rules=_normalize_model_map(data.get("routing_rules")),
            baml_functions=_normalize_model_map(data.get("baml_functions")),
            enable_baml_preflight=_to_bool(data.get("enable_baml_preflight"), True),
            baml_preflight_strict=_to_bool(data.get("baml_preflight_strict"), False),
            baml_required_capabilities=_to_str_list(
                data.get("baml_required_capabilities"),
                ["intent", "task_refine", "planning", "recovery"],
            ),
            enable_baml_builtin_fallback=_to_bool(data.get("enable_baml_builtin_fallback"), True),
        )


def _normalize_model_map(value: Any) -> dict[str, list[str]]:
    """Normalize map fields like routing_rules/provider_models to list[str] values."""
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        if isinstance(raw_value, str):
            items = [raw_value]
        elif isinstance(raw_value, list):
            items = [str(v).strip() for v in raw_value if str(v).strip()]
        else:
            continue
        if items:
            normalized[key] = items
    return normalized


def _load_json_object_from_env(env_name: str) -> dict[str, list[str]]:
    """Load a dict-like json object from env var and normalize to model map."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return _normalize_model_map(payload)


@dataclass
class ChatMessage:
    """A chat message."""
    role: str  # system, user, assistant, tool
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None


@dataclass
class ChatCompletion:
    """Response from chat completion."""
    content: str
    role: str = "assistant"
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> str:
        """Send chat messages and get a response."""
        pass
    
    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> ChatCompletion:
        """Get a full chat completion with metadata."""
        pass
    
    def stream(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response."""
        # Default implementation: yield full response
        yield self.chat(messages, **kwargs)


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self._client = None
    
    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    timeout=self.config.timeout  # Add timeout support
                )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
        return self._client
    
    def chat(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> str:
        """Send chat messages and get a response."""
        completion = self.chat_completion(messages, **kwargs)
        return completion.content
    
    def chat_completion(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> ChatCompletion:
        """Get a full chat completion with metadata."""
        # Convert messages to dict format
        msg_list = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                msg_list.append({"role": msg.role, "content": msg.content})
            else:
                msg_list.append(msg)
        
        # Call API
        response = self.client.chat.completions.create(
            model=_clean_model_name(
                kwargs.get("model", self.config.model),
                provider=self.config.provider,
                api_base=self.config.api_base or "",
            ),
            messages=msg_list,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
        )
        
        choice = response.choices[0]
        return ChatCompletion(
            content=choice.message.content or "",
            role=choice.message.role,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
        )
    
    def stream(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response."""
        msg_list = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                msg_list.append({"role": msg.role, "content": msg.content})
            else:
                msg_list.append(msg)
        
        response = self.client.chat.completions.create(
            model=_clean_model_name(
                kwargs.get("model", self.config.model),
                provider=self.config.provider,
                api_base=self.config.api_base or "",
            ),
            messages=msg_list,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without actual API calls."""
    
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["这是一个模拟响应。"]
        self.call_count = 0
        self.messages_log: list[list[dict]] = []
    
    def chat(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> str:
        """Return mock response."""
        self.messages_log.append(messages)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def chat_completion(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> ChatCompletion:
        """Return mock completion."""
        return ChatCompletion(
            content=self.chat(messages, **kwargs),
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )


class LocalLLMClient(LLMClient):
    """Client for local LLM servers (Ollama, LM Studio, etc.)."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def chat(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> str:
        """Send chat to local LLM server."""
        import urllib.request
        import json
        
        msg_list = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                msg_list.append({"role": msg.role, "content": msg.content})
            else:
                msg_list.append(msg)
        
        # Ollama API format
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": msg_list,
            "stream": False
        }
        
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("message", {}).get("content", "")
    
    def chat_completion(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> ChatCompletion:
        """Get completion from local LLM."""
        return ChatCompletion(content=self.chat(messages, **kwargs))


class UnifiedModelClient(LLMClient):
    """Unified model client with task-aware routing and graceful fallback.

    This is the preferred client for the new model-agnostic architecture.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()

    def _normalize_messages(self, messages: list[dict | ChatMessage]) -> list[dict[str, Any]]:
        msg_list: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                msg_list.append({"role": msg.role, "content": msg.content})
            else:
                msg_list.append(msg)
        return msg_list

    def _candidate_models(self, explicit_model: str | None = None, task_type: str | None = None) -> list[str]:
        candidates: list[str] = []

        def append_model(model_name: str | None) -> None:
            ms = str(model_name or "").strip()
            if ms and ms not in candidates:
                candidates.append(ms)

        append_model(explicit_model)

        task_key = str(task_type or "").strip().lower()
        if task_key and self.config.routing_rules:
            for route_model in self.config.routing_rules.get(task_key, []):
                append_model(route_model)

        append_model(self.config.model)

        provider_key = (self.config.provider or "").strip().lower()
        if provider_key and self.config.provider_models:
            for provider_model in self.config.provider_models.get(provider_key, []):
                append_model(provider_model)

        for fallback in self.config.fallback_models:
            append_model(fallback)

        return candidates

    def _model_call_variants(self, candidate: str) -> list[str]:
        """Return provider-specific model naming variants for LiteLLM.

        Some OpenAI-compatible providers (e.g. siliconflow) require an explicit
        provider prefix like `openai/<model>` even when `api_base` is set.
        """
        raw = str(candidate or "").strip()
        if not raw:
            return []

        variants: list[str] = [raw]

        known_prefixes = (
            "openai/",
            "azure/",
            "anthropic/",
            "bedrock/",
            "vertex_ai/",
            "gemini/",
            "huggingface/",
            "ollama/",
        )
        if raw.startswith(known_prefixes):
            return variants

        provider = (self.config.provider or "").strip().lower()
        openai_compatible = {
            "siliconflow", "openai", "azure", "deepseek", "minimax", "zhipu", "glm", "qwen"
        }
        if provider in openai_compatible:
            variants = [f"openai/{raw}", raw]

        # de-duplicate while preserving order
        deduped: list[str] = []
        for item in variants:
            if item not in deduped:
                deduped.append(item)
        return deduped

    def chat_complete(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> ChatCompletion:
        """Compatibility API for architecture docs expecting chat_complete."""
        return self.chat_completion(messages, **kwargs)

    def chat(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> str:
        completion = self.chat_completion(messages, **kwargs)
        return completion.content

    def chat_completion(
        self,
        messages: list[dict | ChatMessage],
        **kwargs
    ) -> ChatCompletion:
        try:
            from litellm import completion
        except ImportError as exc:
            # Keep runtime available on minimal environments by falling back
            # to direct OpenAI-compatible client when LiteLLM is unavailable.
            fallback_client = OpenAIClient(self.config)
            return fallback_client.chat_completion(messages, **kwargs)

        msg_list = self._normalize_messages(messages)
        retries = max(1, int(kwargs.get("retry_count", self.config.retry_count)))
        candidate_models = self._candidate_models(
            kwargs.get("model"),
            task_type=kwargs.get("task_type"),
        )
        if not candidate_models:
            candidate_models = ["gpt-4o-mini"]

        last_error: Exception | None = None
        for _ in range(retries):
            for candidate in candidate_models:
                for model_variant in self._model_call_variants(candidate):
                    try:
                        response = completion(
                            model=model_variant,
                            messages=msg_list,
                            api_key=kwargs.get("api_key", self.config.api_key),
                            api_base=kwargs.get("api_base", self.config.api_base),
                            temperature=kwargs.get("temperature", self.config.temperature),
                            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                            timeout=kwargs.get("timeout", self.config.timeout),
                        )
                        choice = response.choices[0]
                        message = choice.message
                        content = getattr(message, "content", "") or ""
                        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"
                        usage = {
                            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if getattr(response, "usage", None) else 0,
                            "completion_tokens": getattr(response.usage, "completion_tokens", 0) if getattr(response, "usage", None) else 0,
                            "total_tokens": getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else 0,
                        }
                        return ChatCompletion(
                            content=content,
                            role=getattr(message, "role", "assistant") or "assistant",
                            finish_reason=finish_reason,
                            usage=usage,
                        )
                    except Exception as exc:  # pragma: no cover - provider-specific errors
                        last_error = exc
                        continue

        if last_error is not None:
            raise last_error

        return ChatCompletion(content="")


class LiteLLMClient(UnifiedModelClient):
    """Backward-compatible alias of UnifiedModelClient.

    Existing imports can continue using LiteLLMClient during migration.
    """


def create_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """Create appropriate LLM client based on configuration.
    
    Args:
        config: LLM configuration. If None, uses environment variables.
        
    Returns:
        LLMClient instance
    """
    config = config or LLMConfig.from_env()
    
    if not config.api_key:
        # No API key, use mock client
        return MockLLMClient()
    
    if "localhost" in (config.api_base or "") or "127.0.0.1" in (config.api_base or ""):
        # Local server
        return LocalLLMClient(base_url=config.api_base, model=config.model)

    provider = (config.provider or "openai").strip().lower()

    # For single-provider OpenAI-compatible gateways, prefer direct OpenAI client.
    # This avoids LiteLLM startup network lookups (e.g., model cost map fetch warnings)
    # while preserving the same chat-completions behavior.
    has_advanced_routing = bool(config.fallback_models or config.provider_models or config.routing_rules)
    openai_compatible = {"openai", "azure", "siliconflow", "qwen", "deepseek", "minimax", "zhipu", "glm", "gemini"}
    if provider in openai_compatible and provider != "litellm" and not has_advanced_routing:
        return OpenAIClient(config)

    # Prefer unified client when multi-model routing/fallback is configured.
    if provider in {
        "litellm", "openai", "azure", "anthropic", "siliconflow", "qwen",
        "deepseek", "minimax", "zhipu", "glm", "gemini"
    }:
        try:
            return UnifiedModelClient(config)
        except Exception:
            return OpenAIClient(config)
    
    # Default to OpenAI
    return OpenAIClient(config)
