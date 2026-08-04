"""Test doubles for the retrieval seam, shared by the prefetch tests."""

from __future__ import annotations

import threading

from wikipedia.prefetch import AgentState
from wikipedia.types import SearchResult


def make_result(source_id: str, title: str) -> SearchResult:
    return SearchResult(
        source_id=source_id,
        score=0.9,
        url=f"https://example.com/{source_id}",
        title=title,
        text=f"body of {title}",
    )


class FakeVector:
    """What FakeEncoder hands to FakeDatabase, carrying the query it came from."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> FakeVector:
        self.calls.append(text)
        return FakeVector(text)


class FakeDatabase:
    """Records every search and returns results derived from the query."""

    def __init__(
        self, *, gate: threading.Event | None = None, gate_query: str | None = None
    ) -> None:
        self.searches: list[str] = []
        self._lock = threading.Lock()
        self._gate = gate
        self._gate_query = gate_query

    def search(self, vector: FakeVector, *, limit: int = 10) -> list[SearchResult]:
        with self._lock:
            self.searches.append(vector.text)
        if self._gate is not None and vector.text == self._gate_query:
            self._gate.wait(timeout=5)
        return [
            make_result(f"{vector.text}#{index}", f"{vector.text} {index}")
            for index in range(limit)
        ]


class ScriptedPredictor:
    """Returns a canned prediction per hop count, so hits and misses are chosen."""

    def __init__(self, *predictions: str | None) -> None:
        self.predictions = list(predictions)
        self.states: list[AgentState] = []

    def predict(self, state: AgentState) -> str | None:
        self.states.append(state)
        index = len(state.hops) - 1
        if 0 <= index < len(self.predictions):
            return self.predictions[index]
        return None
