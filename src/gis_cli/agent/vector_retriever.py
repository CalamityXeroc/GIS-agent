"""Local vector retrieval primitives for agent memory.

This module provides a lightweight hashing-vector retriever so the agent can
perform vector-like semantic lookup without external embedding services.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrieverMatch:
    """Single retrieval result."""

    record_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorRetriever:
    """Minimal vector retriever interface."""

    def upsert(self, record_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def delete(self, record_id: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[RetrieverMatch]:
        raise NotImplementedError


@dataclass
class _VectorRecord:
    record_id: str
    text: str
    metadata: dict[str, Any]
    vector: dict[int, float]
    norm: float


class HashingVectorRetriever(BaseVectorRetriever):
    """Deterministic hashing-based vector retriever.

    - No external model dependency
    - Stable vectors across process restarts
    - Works for mixed Chinese/English GIS text
    """

    def __init__(self, dimension: int = 384):
        self.dimension = max(64, int(dimension))
        self._records: dict[str, _VectorRecord] = {}

    def upsert(self, record_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        sparse = self._encode(text)
        norm = math.sqrt(sum(v * v for v in sparse.values())) or 1.0
        self._records[record_id] = _VectorRecord(
            record_id=record_id,
            text=text,
            metadata=metadata or {},
            vector=sparse,
            norm=norm,
        )

    def delete(self, record_id: str) -> None:
        self._records.pop(record_id, None)

    def clear(self) -> None:
        self._records.clear()

    def search(self, query: str, top_k: int = 5) -> list[RetrieverMatch]:
        if not query.strip() or not self._records:
            return []
        q_vec = self._encode(query)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: list[RetrieverMatch] = []
        for rec in self._records.values():
            score = self._cosine_sparse(q_vec, q_norm, rec.vector, rec.norm)
            if score <= 0.0:
                continue
            scored.append(
                RetrieverMatch(
                    record_id=rec.record_id,
                    score=score,
                    text=rec.text,
                    metadata=rec.metadata,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: max(1, int(top_k))]

    def _encode(self, text: str) -> dict[int, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        sparse: dict[int, float] = {}
        for token in tokens:
            idx = self._token_index(token)
            sparse[idx] = sparse.get(idx, 0.0) + 1.0

        # L2-friendly tf normalization
        total = float(len(tokens))
        if total > 0:
            for key in list(sparse.keys()):
                sparse[key] = sparse[key] / total
        return sparse

    def _token_index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
        return int(digest[:8], 16) % self.dimension

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower().strip()
        if not lowered:
            return []

        tokens: list[str] = []
        # English/number tokens
        tokens.extend(re.findall(r"[a-z0-9_]+", lowered))

        # Chinese char tokens + bigrams for better phrase sensitivity
        cjk_chars = [c for c in lowered if "\u4e00" <= c <= "\u9fff"]
        tokens.extend(cjk_chars)
        for i in range(len(cjk_chars) - 1):
            tokens.append(cjk_chars[i] + cjk_chars[i + 1])

        return tokens

    def _cosine_sparse(
        self,
        query_vec: dict[int, float],
        query_norm: float,
        doc_vec: dict[int, float],
        doc_norm: float,
    ) -> float:
        if not query_vec or not doc_vec:
            return 0.0

        # Iterate on smaller sparse map for efficiency
        if len(query_vec) > len(doc_vec):
            query_vec, doc_vec = doc_vec, query_vec
            query_norm, doc_norm = doc_norm, query_norm

        dot = 0.0
        for idx, qv in query_vec.items():
            dv = doc_vec.get(idx)
            if dv is None:
                continue
            dot += qv * dv
        if dot <= 0:
            return 0.0
        return dot / (query_norm * doc_norm)
