from types import SimpleNamespace

from agent.retrieval import TimedRetriever, render_search_results
from wikipedia.types import SearchResult


class Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


class Encoder:
    def encode(self, text: str) -> list[float]:
        assert text == "query"
        return [1.0]


class Database:
    def search(self, vector: list[float], *, limit: int) -> list[SearchResult]:
        assert vector == [1.0]
        assert limit == 2
        return [
            SearchResult("a", 0.9, "https://a", "A", "abcdef"),
            SearchResult("b", 0.8, "https://b", "B", "xy"),
        ]


def test_retrieval_separates_timings_and_preserves_exact_visible_text() -> None:
    call = TimedRetriever(
        Encoder(), Database(), top_k=2, max_chars_per_result=4, clock_ns=Clock()
    ).search("query")

    assert call["encode_duration_ms"] == 1
    assert call["search_duration_ms"] == 1
    assert call["results"][0] == {
        "rank": 1,
        "source_id": "a",
        "title": "A",
        "url": "https://a",
        "score": 0.9,
        "snippet": "abcd",
        "original_length": 6,
        "truncated": True,
    }
    assert '"snippet":"abcd"' in render_search_results(call)
    assert "abcdef" not in render_search_results(call)
