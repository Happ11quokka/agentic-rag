import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fakes import (
    FakeDatabase,
    FakeEncoder,
    FakeVector,
    ScriptedPredictor,
    make_result as _result,
)

from wikipedia.encoder import search_text
from wikipedia.prefetch import (
    AgentState,
    HTTPPredictor,
    PrefetchingRetriever,
    PrefetchStats,
    ReplayPredictor,
    normalize_query,
)
from wikipedia.types import SearchResult


def test_normalize_query_ignores_case_and_surrounding_whitespace() -> None:
    assert normalize_query("  What Causes Auroras?  ") == normalize_query(
        "what causes auroras?"
    )


def test_normalize_query_collapses_internal_whitespace() -> None:
    assert normalize_query("what  causes\tauroras") == normalize_query(
        "what causes auroras"
    )


def test_normalize_query_keeps_distinct_queries_distinct() -> None:
    assert normalize_query("aurora borealis") != normalize_query("aurora australis")


def test_normalize_query_rejects_blank_text() -> None:
    with pytest.raises(ValueError):
        normalize_query("   ")


def test_agent_state_extend_appends_a_hop_without_mutating_the_original() -> None:
    state = AgentState(question="What causes auroras?")
    extended = state.extend("aurora", [_result("1", "Aurora")])

    assert state.hops == ()
    assert len(extended.hops) == 1
    assert extended.hops[0].query == "aurora"
    assert extended.hops[0].titles == ("Aurora",)
    assert extended.question == state.question


def test_retrieve_without_a_predictor_searches_once_and_returns_the_plain_result() -> None:
    database, encoder = FakeDatabase(), FakeEncoder()
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, limit=3) as retriever:
        results = retriever.retrieve("aurora", state)

    assert database.searches == ["aurora"]
    assert results == search_text(database, FakeEncoder(), "aurora", limit=3)


def test_retrieve_without_a_predictor_starts_no_threads() -> None:
    database, encoder = FakeDatabase(), FakeEncoder()
    before = threading.active_count()

    with PrefetchingRetriever(database, encoder) as retriever:
        retriever.retrieve("aurora", AgentState(question="Q"))
        assert threading.active_count() == before

    assert retriever.stats.attempts == 0


def _await_search(database: FakeDatabase, text: str, timeout: float = 5.0) -> None:
    """Block until the prefetch thread has begun searching `text`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if text in list(database.searches):
            return
        time.sleep(0.005)
    raise AssertionError(f"prefetch never searched {text!r}; saw {database.searches}")


def test_the_drafter_is_synced_with_the_hop_that_just_completed() -> None:
    database, encoder = FakeDatabase(), FakeEncoder()
    predictor = ScriptedPredictor("borealis")

    with PrefetchingRetriever(database, encoder, predictor=predictor, limit=2) as retriever:
        retriever.retrieve("aurora", AgentState(question="What causes auroras?"))
        _await_search(database, "borealis")

    synced = predictor.states[0]
    assert synced.question == "What causes auroras?"
    assert [hop.query for hop in synced.hops] == ["aurora"]
    assert synced.hops[0].titles == ("aurora 0", "aurora 1")


def test_a_correct_prediction_is_served_without_searching_again() -> None:
    database, encoder = FakeDatabase(), FakeEncoder()
    predictor = ScriptedPredictor("borealis")
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, predictor=predictor, limit=2) as retriever:
        first = retriever.retrieve("aurora", state)
        _await_search(database, "borealis")
        second = retriever.retrieve("borealis", state.extend("aurora", first))

    assert database.searches == ["aurora", "borealis"]
    assert retriever.stats.hits == 1
    assert second == search_text(FakeDatabase(), FakeEncoder(), "borealis", limit=2)


def test_a_prediction_matches_regardless_of_case_and_spacing() -> None:
    database, encoder = FakeDatabase(), FakeEncoder()
    predictor = ScriptedPredictor("Aurora  Borealis")
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, predictor=predictor) as retriever:
        first = retriever.retrieve("aurora", state)
        _await_search(database, "Aurora  Borealis")
        retriever.retrieve("aurora borealis", state.extend("aurora", first))

    assert retriever.stats.hits == 1


def test_a_wrong_prediction_returns_the_same_results_as_no_prefetch() -> None:
    database, encoder = FakeDatabase(), FakeEncoder()
    predictor = ScriptedPredictor("wrong guess")
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, predictor=predictor, limit=2) as retriever:
        first = retriever.retrieve("aurora", state)
        _await_search(database, "wrong guess")
        second = retriever.retrieve("borealis", state.extend("aurora", first))

    assert retriever.stats.hits == 0
    assert second == search_text(FakeDatabase(), FakeEncoder(), "borealis", limit=2)


@pytest.mark.parametrize(
    "predictor",
    [
        ScriptedPredictor(None),
        ScriptedPredictor("   "),
        ScriptedPredictor("aurora"),
    ],
    ids=["none", "blank", "repeat-of-current"],
)
def test_unusable_predictions_are_not_prefetched(predictor: ScriptedPredictor) -> None:
    database, encoder = FakeDatabase(), FakeEncoder()

    with PrefetchingRetriever(database, encoder, predictor=predictor) as retriever:
        retriever.retrieve("aurora", AgentState(question="Q"))

    assert database.searches == ["aurora"]


def test_a_drafter_that_raises_does_not_break_retrieval() -> None:
    class ExplodingPredictor:
        def predict(self, state: AgentState) -> str | None:
            raise RuntimeError("draft server is down")

    database, encoder = FakeDatabase(), FakeEncoder()
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, predictor=ExplodingPredictor(), limit=2) as retriever:
        first = retriever.retrieve("aurora", state)
        second = retriever.retrieve("borealis", state.extend("aurora", first))

    assert database.searches == ["aurora", "borealis"]
    assert second == search_text(FakeDatabase(), FakeEncoder(), "borealis", limit=2)


def test_a_prefetch_that_raises_falls_back_to_a_fresh_search() -> None:
    class FlakyDatabase(FakeDatabase):
        def search(self, vector: FakeVector, *, limit: int = 10) -> list[SearchResult]:
            if vector.text == "borealis" and "borealis" not in self.searches:
                self.searches.append(vector.text)
                raise RuntimeError("milvus rpc failed")
            return super().search(vector, limit=limit)

    database, encoder = FlakyDatabase(), FakeEncoder()
    predictor = ScriptedPredictor("borealis")
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, predictor=predictor, limit=2) as retriever:
        first = retriever.retrieve("aurora", state)
        _await_search(database, "borealis")
        second = retriever.retrieve("borealis", state.extend("aurora", first))

    assert retriever.stats.hits == 0
    assert second == search_text(FakeDatabase(), FakeEncoder(), "borealis", limit=2)


def test_caches_are_not_shared_between_retrievers() -> None:
    database, encoder = FakeDatabase(), FakeEncoder()
    predictor = ScriptedPredictor("borealis")
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, predictor=predictor) as episode_a:
        first = episode_a.retrieve("aurora", state)
        _await_search(database, "borealis")

    with PrefetchingRetriever(database, encoder) as episode_b:
        episode_b.retrieve("borealis", state.extend("aurora", first))

    assert database.searches == ["aurora", "borealis", "borealis"]
    assert episode_b.stats.hits == 0


def test_a_hit_on_an_in_flight_prefetch_waits_instead_of_searching_again() -> None:
    gate = threading.Event()
    database = FakeDatabase(gate=gate, gate_query="borealis")
    encoder = FakeEncoder()
    predictor = ScriptedPredictor("borealis")
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, predictor=predictor, limit=2) as retriever:
        first = retriever.retrieve("aurora", state)
        _await_search(database, "borealis")

        done = threading.Event()
        captured: list[list[SearchResult]] = []

        def ask() -> None:
            captured.append(retriever.retrieve("borealis", state.extend("aurora", first)))
            done.set()

        asker = threading.Thread(target=ask)
        asker.start()
        time.sleep(0.05)
        assert not done.is_set(), "the target should be waiting on the in-flight prefetch"
        gate.set()
        asker.join(timeout=5)

    assert database.searches == ["aurora", "borealis"]
    assert retriever.stats.hits == 1
    assert captured[0] == search_text(FakeDatabase(), FakeEncoder(), "borealis", limit=2)


def test_hidden_time_is_what_the_prefetch_finished_before_the_target_asked() -> None:
    class SlowDatabase(FakeDatabase):
        def search(self, vector: FakeVector, *, limit: int = 10) -> list[SearchResult]:
            time.sleep(0.05)
            return super().search(vector, limit=limit)

    database, encoder = SlowDatabase(), FakeEncoder()
    predictor = ScriptedPredictor("borealis")
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, predictor=predictor, limit=2) as retriever:
        first = retriever.retrieve("aurora", state)
        _await_search(database, "borealis")
        time.sleep(0.2)
        second_started = time.perf_counter()
        retriever.retrieve("borealis", state.extend("aurora", first))
        observed_ms = (time.perf_counter() - second_started) * 1000

    assert retriever.stats.hits == 1
    assert observed_ms < 25, "a finished prefetch should be returned without waiting"
    assert retriever.stats.hidden_ms >= 40, retriever.stats.hidden_ms
    assert retriever.stats.wait_ms < 25


def test_a_prefetch_the_agent_never_asked_for_is_counted_as_wasted() -> None:
    database, encoder = FakeDatabase(), FakeEncoder()
    predictor = ScriptedPredictor("never asked for")

    with PrefetchingRetriever(database, encoder, predictor=predictor) as retriever:
        retriever.retrieve("aurora", AgentState(question="Q"))
        _await_search(database, "never asked for")

    assert retriever.stats.wasted == 1
    assert retriever.stats.hits == 0


def test_a_miss_racing_an_unfinished_prefetch_is_recorded_as_contended() -> None:
    gate = threading.Event()
    database = FakeDatabase(gate=gate, gate_query="wrong guess")
    encoder = FakeEncoder()
    predictor = ScriptedPredictor("wrong guess")
    state = AgentState(question="Q")

    with PrefetchingRetriever(database, encoder, predictor=predictor) as retriever:
        first = retriever.retrieve("aurora", state)
        _await_search(database, "wrong guess")
        retriever.retrieve("borealis", state.extend("aurora", first))
        gate.set()

    assert retriever.stats.contended == 1
    assert retriever.stats.hits == 0


def test_hit_rate_reports_zero_before_any_retrieval() -> None:
    assert PrefetchStats().hit_rate == 0.0


def test_replay_predictor_returns_the_next_recorded_query() -> None:
    predictor = ReplayPredictor(["aurora", "borealis", "solar wind"])
    state = AgentState(question="Q").extend("aurora", [_result("1", "Aurora")])

    assert predictor.predict(state) == "borealis"


def test_replay_predictor_returns_none_past_the_end_of_the_recording() -> None:
    predictor = ReplayPredictor(["aurora"])
    state = AgentState(question="Q").extend("aurora", [_result("1", "Aurora")])

    assert predictor.predict(state) is None


def test_replay_predictor_misses_on_the_hops_it_was_told_to_miss() -> None:
    predictor = ReplayPredictor(["aurora", "borealis"], wrong_hops=(1,))
    state = AgentState(question="Q").extend("aurora", [_result("1", "Aurora")])

    prediction = predictor.predict(state)

    assert prediction is not None
    assert normalize_query(prediction) != normalize_query("borealis")


def test_replay_predictor_can_charge_the_drafter_for_thinking_time() -> None:
    predictor = ReplayPredictor(["aurora", "borealis"], think_seconds=0.05)
    state = AgentState(question="Q").extend("aurora", [_result("1", "Aurora")])

    started = time.perf_counter()
    predictor.predict(state)

    assert time.perf_counter() - started >= 0.05


def test_http_predictor_extracts_the_query_from_a_search_action() -> None:
    calls: list[dict[str, object]] = []

    def transport(url: str, payload: dict[str, object], timeout: float) -> dict:
        calls.append({"url": url, "payload": payload, "timeout": timeout})
        return {"choices": [{"message": {"content": "Thought: ...\nAction: search[solar wind]"}}]}

    predictor = HTTPPredictor(
        "http://localhost:8001", model="llama-3.2-1b", transport=transport
    )
    state = AgentState(question="What causes auroras?").extend(
        "aurora", [_result("1", "Aurora")]
    )

    assert predictor.predict(state) == "solar wind"
    assert calls[0]["url"] == "http://localhost:8001/v1/chat/completions"


def test_http_predictor_prompt_carries_the_question_and_the_synced_hops() -> None:
    captured: list[dict] = []

    def transport(url: str, payload: dict, timeout: float) -> dict:
        captured.append(payload)
        return {"choices": [{"message": {"content": "Action: search[x]"}}]}

    predictor = HTTPPredictor("http://localhost:8001", model="m", transport=transport)
    state = AgentState(question="What causes auroras?").extend(
        "aurora", [_result("1", "Aurora Borealis")]
    )
    predictor.predict(state)

    prompt = "\n".join(
        str(message["content"]) for message in captured[0]["messages"]
    )
    assert "What causes auroras?" in prompt
    assert "aurora" in prompt
    assert "Aurora Borealis" in prompt


@pytest.mark.parametrize(
    "content", ["I am not sure what to search for.", "", "Action: finish[done]"]
)
def test_http_predictor_returns_none_when_no_search_is_proposed(content: str) -> None:
    predictor = HTTPPredictor(
        "http://localhost:8001",
        model="m",
        transport=lambda url, payload, timeout: {
            "choices": [{"message": {"content": content}}]
        },
    )

    assert predictor.predict(AgentState(question="Q")) is None


def test_http_predictor_reports_a_dead_draft_server_as_no_prediction() -> None:
    def transport(url: str, payload: dict, timeout: float) -> dict:
        raise OSError("connection refused")

    predictor = HTTPPredictor("http://localhost:8001", model="m", transport=transport)

    assert predictor.predict(AgentState(question="Q")) is None


def test_http_predictor_does_not_mistake_research_for_a_search_action() -> None:
    predictor = HTTPPredictor(
        "http://localhost:8001",
        model="m",
        transport=lambda url, payload, timeout: {
            "choices": [{"message": {"content": "Action: research[aurora]"}}]
        },
    )

    assert predictor.predict(AgentState(question="Q")) is None


def test_http_predictor_default_transport_speaks_to_a_real_server() -> None:
    """Exercise the urllib path, which every injected-transport test skips."""
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            received.append(
                {
                    "path": self.path,
                    "content_type": self.headers["Content-Type"],
                    "body": json.loads(self.rfile.read(length).decode("utf-8")),
                }
            )
            payload = json.dumps(
                {"choices": [{"message": {"content": "Action: search[solar wind]"}}]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        predictor = HTTPPredictor(
            f"http://127.0.0.1:{server.server_port}", model="test-model", timeout=5
        )
        prediction = predictor.predict(AgentState(question="What causes auroras?"))
    finally:
        server.shutdown()
        server.server_close()

    assert prediction == "solar wind"
    assert received[0]["path"] == "/v1/chat/completions"
    assert received[0]["content_type"] == "application/json"
    assert received[0]["body"]["model"] == "test-model"
    assert received[0]["body"]["stream"] is False


def test_http_predictor_survives_a_server_that_is_not_listening() -> None:
    predictor = HTTPPredictor("http://127.0.0.1:1", model="m", timeout=2)

    assert predictor.predict(AgentState(question="Q")) is None


def test_concurrent_prefetch_threads_do_not_lose_stat_increments() -> None:
    """attempts/errors are incremented off the agent's thread, by more than one.

    A miss leaves its speculative search running while the next hop spawns
    another, so two prefetch threads overlap. These counters are experiment
    output, and a lost increment would understate the work silently.
    """
    barrier = threading.Barrier(24)

    class RacingPredictor:
        def predict(self, state: AgentState) -> str | None:
            barrier.wait(timeout=10)
            raise RuntimeError("always fails, so attempts and errors both move")

    retriever = PrefetchingRetriever(
        FakeDatabase(), FakeEncoder(), predictor=RacingPredictor()
    )
    threads = [
        threading.Thread(
            target=retriever._predict_and_prefetch,
            args=(AgentState(question="Q"), "answered"),
        )
        for _ in range(24)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert retriever.stats.attempts == 24
    assert retriever.stats.errors == 24
