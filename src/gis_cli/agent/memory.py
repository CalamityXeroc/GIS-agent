"""Agent memory system.

Provides conversation history, context storage, and memory persistence.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from enum import Enum

from .vector_retriever import BaseVectorRetriever, HashingVectorRetriever


class Role(str, Enum):
    """Conversation roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: Role
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional metadata
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
        if self.tool_name:
            data["tool_name"] = self.tool_name
        if self.tool_input:
            data["tool_input"] = self.tool_input
        if self.tool_output is not None:
            data["tool_output"] = self.tool_output
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            role=Role(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tool_name=data.get("tool_name"),
            tool_input=data.get("tool_input"),
            tool_output=data.get("tool_output")
        )
    
    @classmethod
    def user(cls, content: str) -> "ConversationTurn":
        """Create a user turn."""
        return cls(role=Role.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "ConversationTurn":
        """Create an assistant turn."""
        return cls(role=Role.ASSISTANT, content=content)
    
    @classmethod
    def system(cls, content: str) -> "ConversationTurn":
        """Create a system turn."""
        return cls(role=Role.SYSTEM, content=content)
    
    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any
    ) -> "ConversationTurn":
        """Create a tool call turn."""
        return cls(
            role=Role.TOOL,
            content=f"Called {tool_name}",
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output
        )


@dataclass
class StructuredMemoryItem:
    """Structured memory item for durable facts and task context."""

    memory_id: str
    memory_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: int = 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "metadata": self.metadata,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredMemoryItem":
        timestamp_raw = data.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_raw) if timestamp_raw else datetime.now(timezone.utc)
        return cls(
            memory_id=str(data.get("memory_id") or f"mem_{uuid.uuid4().hex[:10]}"),
            memory_type=str(data.get("memory_type") or "note"),
            content=str(data.get("content") or ""),
            metadata=dict(data.get("metadata") or {}),
            importance=max(1, min(int(data.get("importance") or 1), 5)),
            timestamp=timestamp,
        )


@dataclass
class ReflectionItem:
    """Reflection note distilled from execution outcomes."""

    reflection_id: str
    trigger: Literal["success", "failure", "manual"]
    issue: str
    lesson: str
    action_hint: str = ""
    related_tools: list[str] = field(default_factory=list)
    confidence: float = 0.6
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reflection_id": self.reflection_id,
            "trigger": self.trigger,
            "issue": self.issue,
            "lesson": self.lesson,
            "action_hint": self.action_hint,
            "related_tools": self.related_tools,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReflectionItem":
        trigger = str(data.get("trigger") or "manual")
        if trigger not in {"success", "failure", "manual"}:
            trigger = "manual"
        timestamp_raw = data.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_raw) if timestamp_raw else datetime.now(timezone.utc)
        return cls(
            reflection_id=str(data.get("reflection_id") or f"refl_{uuid.uuid4().hex[:10]}"),
            trigger=trigger,  # type: ignore[arg-type]
            issue=str(data.get("issue") or ""),
            lesson=str(data.get("lesson") or ""),
            action_hint=str(data.get("action_hint") or ""),
            related_tools=list(data.get("related_tools") or []),
            confidence=max(0.1, min(float(data.get("confidence") or 0.6), 1.0)),
            timestamp=timestamp,
        )


@dataclass
class MemoryStore:
    """Persistent storage for agent memory."""
    
    path: Path
    
    def __post_init__(self):
        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
    
    def save_conversation(
        self,
        session_id: str,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
        tool_history: list[dict[str, Any]] | None = None,
        structured_memories: list[StructuredMemoryItem] | None = None,
        reflections: list[ReflectionItem] | None = None,
    ) -> None:
        """Save conversation to disk."""
        file_path = self.path / f"{session_id}.json"
        data = {
            "session_id": session_id,
            "turns": [t.to_dict() for t in turns],
            "context": context or {},
            "tool_history": tool_history or [],
            "structured_memories": [m.to_dict() for m in (structured_memories or [])],
            "reflections": [r.to_dict() for r in (reflections or [])],
            "saved_at": datetime.now(timezone.utc).isoformat()
        }
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def load_conversation(self, session_id: str) -> dict[str, Any]:
        """Load conversation from disk."""
        file_path = self.path / f"{session_id}.json"
        if not file_path.exists():
            return {
                "turns": [],
                "context": {},
                "tool_history": [],
                "structured_memories": [],
                "reflections": [],
            }
        
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return {
            "turns": [ConversationTurn.from_dict(t) for t in data.get("turns", [])],
            "context": dict(data.get("context") or {}),
            "tool_history": list(data.get("tool_history") or []),
            "structured_memories": [
                StructuredMemoryItem.from_dict(item)
                for item in data.get("structured_memories", [])
                if isinstance(item, dict)
            ],
            "reflections": [
                ReflectionItem.from_dict(item)
                for item in data.get("reflections", [])
                if isinstance(item, dict)
            ],
        }
    
    def list_sessions(self) -> list[str]:
        """List all saved sessions."""
        return [f.stem for f in self.path.glob("*.json")]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        file_path = self.path / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False


@dataclass
class Memory:
    """Agent memory system.
    
    Maintains:
    - Conversation history
    - Working context (current task state)
    - Tool call history
    - User preferences
    """
    
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    tool_history: list[dict] = field(default_factory=list)
    structured_memories: list[StructuredMemoryItem] = field(default_factory=list)
    reflections: list[ReflectionItem] = field(default_factory=list)
    
    # Optional persistent storage
    store: MemoryStore | None = None
    vector_retriever: BaseVectorRetriever | None = None
    
    # Memory limits
    max_turns: int = 100
    max_tool_history: int = 50
    max_structured_memories: int = 200
    max_reflections: int = 120

    def __post_init__(self) -> None:
        if self.vector_retriever is None:
            self.vector_retriever = HashingVectorRetriever()
    
    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.turns.append(ConversationTurn.user(content))
        self._trim_history()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.turns.append(ConversationTurn.assistant(content))
        self._trim_history()
    
    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.turns.append(ConversationTurn.system(content))
        self._trim_history()
    
    def add_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any
    ) -> None:
        """Record a tool call."""
        turn = ConversationTurn.tool_call(tool_name, tool_input, tool_output)
        self.turns.append(turn)
        
        # Also add to tool history
        self.tool_history.append({
            "tool": tool_name,
            "input": tool_input,
            "output": tool_output,
            "timestamp": turn.timestamp.isoformat()
        })
        
        self._trim_history()
        self._trim_tool_history()

    def add_structured_memory(
        self,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: int = 1,
    ) -> StructuredMemoryItem:
        """Add structured memory fact."""
        item = StructuredMemoryItem(
            memory_id=f"mem_{uuid.uuid4().hex[:10]}",
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=max(1, min(int(importance), 5)),
        )
        self.structured_memories.append(item)
        self._index_structured_memory(item)
        self._trim_structured_memories()
        return item

    def add_reflection(
        self,
        trigger: Literal["success", "failure", "manual"],
        issue: str,
        lesson: str,
        action_hint: str = "",
        related_tools: list[str] | None = None,
        confidence: float = 0.6,
    ) -> ReflectionItem:
        """Add reflection note from execution feedback."""
        item = ReflectionItem(
            reflection_id=f"refl_{uuid.uuid4().hex[:10]}",
            trigger=trigger,
            issue=issue,
            lesson=lesson,
            action_hint=action_hint,
            related_tools=related_tools or [],
            confidence=max(0.1, min(float(confidence), 1.0)),
        )
        self.reflections.append(item)
        self._index_reflection(item)
        self._trim_reflections()
        return item

    def _index_structured_memory(self, item: StructuredMemoryItem) -> None:
        if self.vector_retriever is None:
            return
        text = f"{item.content} {json.dumps(item.metadata, ensure_ascii=False)}"
        self.vector_retriever.upsert(
            f"structured:{item.memory_id}",
            text,
            metadata={
                "kind": "structured",
                "memory_id": item.memory_id,
                "memory_type": item.memory_type,
                "importance": item.importance,
            },
        )

    def _index_reflection(self, item: ReflectionItem) -> None:
        if self.vector_retriever is None:
            return
        text = f"{item.issue} {item.lesson} {item.action_hint} {' '.join(item.related_tools)}"
        self.vector_retriever.upsert(
            f"reflection:{item.reflection_id}",
            text,
            metadata={
                "kind": "reflection",
                "reflection_id": item.reflection_id,
                "trigger": item.trigger,
                "confidence": item.confidence,
            },
        )

    def _rebuild_vector_index(self) -> None:
        if self.vector_retriever is None:
            return
        self.vector_retriever.clear()
        for item in self.structured_memories:
            self._index_structured_memory(item)
        for item in self.reflections:
            self._index_reflection(item)
    
    def _trim_history(self) -> None:
        """Trim conversation history to max_turns."""
        if len(self.turns) > self.max_turns:
            # Keep system messages and recent turns
            system_turns = [t for t in self.turns if t.role == Role.SYSTEM]
            other_turns = [t for t in self.turns if t.role != Role.SYSTEM]
            keep_count = self.max_turns - len(system_turns)
            self.turns = system_turns + other_turns[-keep_count:]
    
    def _trim_tool_history(self) -> None:
        """Trim tool history to max_tool_history."""
        if len(self.tool_history) > self.max_tool_history:
            self.tool_history = self.tool_history[-self.max_tool_history:]

    def _trim_structured_memories(self) -> None:
        """Keep important and recent structured memories."""
        if len(self.structured_memories) <= self.max_structured_memories:
            return
        ranked = sorted(
            self.structured_memories,
            key=lambda x: (x.importance, x.timestamp),
            reverse=True,
        )
        self.structured_memories = sorted(
            ranked[: self.max_structured_memories],
            key=lambda x: x.timestamp,
        )

    def _trim_reflections(self) -> None:
        """Trim reflections to max count by confidence and recency."""
        if len(self.reflections) <= self.max_reflections:
            return
        ranked = sorted(
            self.reflections,
            key=lambda x: (x.confidence, x.timestamp),
            reverse=True,
        )
        self.reflections = sorted(
            ranked[: self.max_reflections],
            key=lambda x: x.timestamp,
        )

    def get_structured_memories(
        self,
        memory_type: str | None = None,
        top_k: int = 20,
    ) -> list[StructuredMemoryItem]:
        """Get recent structured memories, optionally filtered by type."""
        items = self.structured_memories
        if memory_type:
            items = [m for m in items if m.memory_type == memory_type]
        return items[-max(top_k, 0):]

    def search_structured_memories(self, query: str, top_k: int = 5) -> list[StructuredMemoryItem]:
        """Hybrid vector + keyword search on structured memories."""
        q = query.lower().strip()
        if not q:
            return []
        indexed_results: list[StructuredMemoryItem] = []
        if self.vector_retriever is not None:
            vector_hits = self.vector_retriever.search(query, top_k=max(top_k * 2, 6))
            by_id = {m.memory_id: m for m in self.structured_memories}
            for hit in vector_hits:
                if hit.metadata.get("kind") != "structured":
                    continue
                mem_id = hit.metadata.get("memory_id")
                if isinstance(mem_id, str) and mem_id in by_id:
                    indexed_results.append(by_id[mem_id])

        scored: list[tuple[int, StructuredMemoryItem]] = []
        for item in self.structured_memories:
            text = f"{item.content} {json.dumps(item.metadata, ensure_ascii=False)}".lower()
            score = 0
            for token in q.split():
                if token and token in text:
                    score += 1
            if score > 0:
                score += item.importance
                scored.append((score, item))
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        keyword_results = [item for _, item in scored[: max(top_k * 2, 6)]]
        merged: list[StructuredMemoryItem] = []
        seen: set[str] = set()
        for item in indexed_results + keyword_results:
            if item.memory_id in seen:
                continue
            seen.add(item.memory_id)
            merged.append(item)
            if len(merged) >= max(top_k, 0):
                break
        return merged

    def get_reflection_hints(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve reflection hints relevant to current query."""
        q = query.lower().strip()
        vector_ranked: list[ReflectionItem] = []
        if self.vector_retriever is not None and q:
            by_id = {r.reflection_id: r for r in self.reflections}
            vector_hits = self.vector_retriever.search(query, top_k=max(top_k * 2, 6))
            for hit in vector_hits:
                if hit.metadata.get("kind") != "reflection":
                    continue
                refl_id = hit.metadata.get("reflection_id")
                if isinstance(refl_id, str) and refl_id in by_id:
                    vector_ranked.append(by_id[refl_id])

        scored: list[tuple[float, ReflectionItem]] = []
        for item in self.reflections:
            text = f"{item.issue} {item.lesson} {item.action_hint} {' '.join(item.related_tools)}".lower()
            hit = 0
            for token in q.split():
                if token and token in text:
                    hit += 1
            if hit > 0:
                scored.append((hit + item.confidence, item))
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        keyword_ranked = [item for _, item in scored[: max(top_k * 2, 6)]]
        merged: list[ReflectionItem] = []
        seen: set[str] = set()
        for item in vector_ranked + keyword_ranked:
            if item.reflection_id in seen:
                continue
            seen.add(item.reflection_id)
            merged.append(item)
            if len(merged) >= max(top_k, 0):
                break

        results = merged
        return [
            {
                "trigger": item.trigger,
                "issue": item.issue,
                "lesson": item.lesson,
                "action_hint": item.action_hint,
                "related_tools": item.related_tools,
                "confidence": item.confidence,
            }
            for item in results
        ]
    
    def get_conversation_for_llm(self, include_tools: bool = True) -> list[dict]:
        """Get conversation history formatted for LLM."""
        messages = []
        for turn in self.turns:
            if turn.role == Role.TOOL and not include_tools:
                continue
            
            msg = {"role": turn.role.value, "content": turn.content}
            if turn.role == Role.TOOL and turn.tool_output is not None:
                msg["tool_output"] = turn.tool_output
            messages.append(msg)
        return messages
    
    def get_last_n_turns(self, n: int = 5) -> list[ConversationTurn]:
        """Get last N conversation turns."""
        return self.turns[-n:] if self.turns else []
    
    def get_last_user_message(self) -> str | None:
        """Get the last user message."""
        for turn in reversed(self.turns):
            if turn.role == Role.USER:
                return turn.content
        return None
    
    def get_tool_results(self, tool_name: str | None = None) -> list[dict]:
        """Get tool call results, optionally filtered by tool name."""
        if tool_name:
            return [h for h in self.tool_history if h["tool"] == tool_name]
        return self.tool_history
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.context.get(key, default)
    
    def clear_context(self) -> None:
        """Clear all context."""
        self.context.clear()
    
    def save(self) -> None:
        """Save memory to persistent storage."""
        if self.store:
            self.store.save_conversation(
                self.session_id,
                self.turns,
                context=self.context,
                tool_history=self.tool_history,
                structured_memories=self.structured_memories,
                reflections=self.reflections,
            )
    
    def load(self) -> None:
        """Load memory from persistent storage."""
        if self.store:
            payload = self.store.load_conversation(self.session_id)
            self.turns = payload.get("turns", [])
            self.context = payload.get("context", {})
            self.tool_history = payload.get("tool_history", [])
            self.structured_memories = payload.get("structured_memories", [])
            self.reflections = payload.get("reflections", [])
            self._rebuild_vector_index()
    
    def clear(self) -> None:
        """Clear all memory."""
        self.turns.clear()
        self.context.clear()
        self.tool_history.clear()
        self.structured_memories.clear()
        self.reflections.clear()
        if self.vector_retriever is not None:
            self.vector_retriever.clear()
    
    def summary(self) -> dict[str, Any]:
        """Get memory summary."""
        return {
            "session_id": self.session_id,
            "total_turns": len(self.turns),
            "user_messages": sum(1 for t in self.turns if t.role == Role.USER),
            "assistant_messages": sum(1 for t in self.turns if t.role == Role.ASSISTANT),
            "tool_calls": len(self.tool_history),
            "context_keys": list(self.context.keys()),
            "structured_memories": len(self.structured_memories),
            "reflections": len(self.reflections),
        }
