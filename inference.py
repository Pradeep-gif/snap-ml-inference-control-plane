from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class Request:
    request_id: str
    vector: tuple[float, ...]
    deadline_ms: int


@dataclass(frozen=True)
class ModelVersion:
    name: str
    weight: int
    batch_runtime_ms: int


class InferenceControlPlane:
    def __init__(self, versions: Iterable[ModelVersion], max_batch_size: int = 8):
        self.versions = tuple(versions)
        self.max_batch_size = max_batch_size
        self.queue: deque[Request] = deque()
        self.metrics: Counter[str] = Counter()
        if not self.versions or sum(v.weight for v in self.versions) <= 0:
            raise ValueError("at least one weighted model version is required")

    def route(self, request_id: str) -> ModelVersion:
        slot = sum(request_id.encode()) % sum(v.weight for v in self.versions)
        for version in self.versions:
            if slot < version.weight:
                return version
            slot -= version.weight
        return self.versions[-1]

    def submit(self, request: Request, estimated_queue_ms: int) -> bool:
        version = self.route(request.request_id)
        if estimated_queue_ms + version.batch_runtime_ms > request.deadline_ms:
            self.metrics["rejected_slo"] += 1
            return False
        self.queue.append(request)
        self.metrics["accepted"] += 1
        return True

    def next_batch(self) -> tuple[Request, ...]:
        batch = tuple(self.queue.popleft() for _ in range(min(len(self.queue), self.max_batch_size)))
        self.metrics["completed"] += len(batch)
        return batch


class VectorIndex:
    def __init__(self, vectors: dict[str, tuple[float, ...]]):
        self.vectors = vectors

    def search(self, query: tuple[float, ...], k: int) -> list[str]:
        return [key for key, _ in sorted(
            ((key, cosine(query, vector)) for key, vector in self.vectors.items()),
            key=lambda item: item[1], reverse=True)[:k]]

    def recall_at_k(self, queries: dict[str, tuple[float, ...]], truth: dict[str, set[str]], k: int) -> float:
        hits = sum(bool(set(self.search(vector, k)) & truth[name]) for name, vector in queries.items())
        return hits / len(queries) if queries else 0.0


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match")
    denominator = sqrt(sum(x * x for x in left)) * sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0
