"""Speculative prefetch of the next retrieval, driven by synced agent state.

The target model blocks on every retrieval. On an on-disk index that wait is
seconds, and while it waits the CPU and GPU are idle — so the retrieval the
agent will issue *next* can be run during the decode that precedes it, and
returned from cache when the agent asks for it.

Prefetch changes only *when* a search happens, never *what* it returns: a
prefetched result is used only when the predicted query normalizes to exactly
the query the agent issued. Anything else falls through to the ordinary path.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .encoder import search_text
from .types import SearchResult

DEFAULT_WAIT_TIMEOUT = 900.0


def normalize_query(text: str) -> str:
    """Canonical cache key for a query.

    The one place a key is derived, for writer and reader alike. If the two
    sides ever disagreed, a prefetch would either never be found (harmless but
    silently disables the experiment) or would return another query's passages
    (breaks the guarantee that results match the baseline).

    Deliberately conservative: case and whitespace only. Stripping punctuation
    would fold genuinely different queries onto one key.
    """
    if not isinstance(text, str):
        raise TypeError("query must be a string")
    normalized = " ".join(text.split()).casefold()
    if not normalized:
        raise ValueError("query must not be blank")
    return normalized


@dataclass(frozen=True, slots=True)
class Hop:
    """One completed retrieval: what was asked, and what came back."""

    query: str
    titles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentState:
    """Immutable snapshot of the agent, as handed to the drafter.

    Immutable because the drafter reads it on another thread while the target
    keeps going; a shared mutable transcript would let the drafter predict from
    a half-written step.
    """

    question: str
    hops: tuple[Hop, ...] = field(default=())

    def extend(self, query: str, results: Sequence[SearchResult]) -> AgentState:
        hop = Hop(query=query, titles=tuple(result.title for result in results))
        return AgentState(question=self.question, hops=(*self.hops, hop))


class QueryPredictor(Protocol):
    """Guesses the query the agent will issue next, from synced state."""

    def predict(self, state: AgentState) -> str | None: ...


@dataclass(slots=True)
class PrefetchEntry:
    """One speculative search: reserved before it starts, filled when it ends."""

    key: str
    # The exact string the prefetch embedded. The key is normalized so near-
    # identical predictions do not each start their own search, but the encoder
    # is case- and whitespace-sensitive, so only the raw string may be served.
    query: str
    started: float
    done: threading.Event
    results: list[SearchResult] | None = None
    finished: float | None = None
    consumed: bool = False

    @property
    def elapsed_ms(self) -> float:
        if self.finished is None:
            return 0.0
        return (self.finished - self.started) * 1000


class PrefetchCache:
    """Speculative results for one episode.

    Scoped to a single episode on purpose. A cache that outlived the episode
    would need a test proving one question's passages cannot surface in
    another's; bounding the lifetime leaves nothing to prove.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, PrefetchEntry] = {}

    def reserve(self, key: str, query: str) -> PrefetchEntry | None:
        """Claim `key`, or return None if it is already cached or in flight.

        Reserving before the search starts is what lets a target that arrives
        mid-flight wait on the running search instead of starting a second one.
        """
        with self._lock:
            if key in self._entries:
                return None
            entry = PrefetchEntry(
                key=key,
                query=query,
                started=time.perf_counter(),
                done=threading.Event(),
            )
            self._entries[key] = entry
            return entry

    def get(self, key: str) -> PrefetchEntry | None:
        with self._lock:
            return self._entries.get(key)

    def entries(self) -> list[PrefetchEntry]:
        with self._lock:
            return list(self._entries.values())

    def in_flight(self) -> int:
        with self._lock:
            return sum(1 for entry in self._entries.values() if not entry.done.is_set())


@dataclass(slots=True)
class PrefetchStats:
    attempts: int = 0
    hits: int = 0
    misses: int = 0
    errors: int = 0
    contended: int = 0
    wasted: int = 0
    hidden_ms: float = 0.0
    wait_ms: float = 0.0
    search_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class PrefetchingRetriever:
    """Retrieval seam that syncs state to a drafter on every retrieval.

    Sync is a side effect of retrieving rather than a separate call the agent
    loop has to remember. A forgotten sync would silently drop that hop back to
    the baseline path while the run still recorded prefetch as enabled.
    """

    def __init__(
        self,
        database: Any,
        encoder: Any,
        *,
        predictor: QueryPredictor | None = None,
        limit: int = 5,
        wait_timeout: float = DEFAULT_WAIT_TIMEOUT,
    ) -> None:
        self.database = database
        self.encoder = encoder
        self.predictor = predictor
        self.limit = limit
        self.wait_timeout = wait_timeout
        self.stats = PrefetchStats()
        self.cache = PrefetchCache()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        # Guards stats. Everything the agent's own thread records is serial,
        # but a miss leaves its speculative search running while the next hop
        # spawns another, so two prefetch threads can bump these at once.
        self._stats_lock = threading.Lock()

    def retrieve(self, query: str, state: AgentState) -> list[SearchResult]:
        """Answer `query`, then sync the resulting state to the drafter."""
        key = normalize_query(query)
        results = self._take_prefetched(key, query)
        if results is None:
            results = self._search_now(query)
        self._sync(state.extend(query, results), answered=key)
        return results

    def _take_prefetched(self, key: str, query: str) -> list[SearchResult] | None:
        entry = self.cache.get(key)
        if entry is None:
            return None
        if entry.query != query:
            # Same cache key, different string. The prefetch embedded its own
            # wording, so its passages are not necessarily the ones this query
            # would return -- serving them would change the answer, which is the
            # one thing prefetch is not allowed to do.
            return None
        started = time.perf_counter()
        completed = entry.done.wait(timeout=self.wait_timeout)
        waited_ms = (time.perf_counter() - started) * 1000
        if not completed or entry.results is None:
            # A failed or hung prefetch must not become a failed retrieval.
            return None
        entry.consumed = True
        self.stats.hits += 1
        self.stats.wait_ms += waited_ms
        # Waiting on a search that started earlier is never worse than starting
        # a fresh one, so whatever it had already spent is time the target did
        # not spend.
        self.stats.hidden_ms += max(0.0, entry.elapsed_ms - waited_ms)
        return list(entry.results)

    def _search_now(self, query: str) -> list[SearchResult]:
        if self.cache.in_flight():
            # A prediction that missed is still holding the disk while the
            # target searches for real. On a medium bounded by random IOPS that
            # is a tax on the very hop prefetch was supposed to help.
            self.stats.contended += 1
        started = time.perf_counter()
        results = search_text(self.database, self.encoder, query, limit=self.limit)
        self.stats.misses += 1
        self.stats.search_ms += (time.perf_counter() - started) * 1000
        return results

    def _sync(self, state: AgentState, *, answered: str) -> None:
        if self.predictor is None:
            return
        thread = threading.Thread(
            target=self._predict_and_prefetch,
            args=(state, answered),
            name="wikipedia-prefetch",
            daemon=True,
        )
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def _predict_and_prefetch(self, state: AgentState, answered: str) -> None:
        # Every failure here degrades to a miss. The drafter is an accelerator;
        # it must never be able to break or alter the retrieval it speculates on.
        with self._stats_lock:
            self.stats.attempts += 1
        try:
            predicted = self.predictor.predict(state)  # type: ignore[union-attr]
        except Exception:
            with self._stats_lock:
                self.stats.errors += 1
            return
        if not predicted:
            return
        try:
            key = normalize_query(predicted)
        except (TypeError, ValueError):
            return
        if key == answered:
            # Caching the query just answered would post a hit no drafter
            # earned and would spend the disk twice on one retrieval.
            return
        entry = self.cache.reserve(key, predicted)
        if entry is None:
            return
        try:
            entry.results = search_text(
                self.database, self.encoder, predicted, limit=self.limit
            )
        except Exception:
            with self._stats_lock:
                self.stats.errors += 1
        finally:
            entry.finished = time.perf_counter()
            entry.done.set()

    def close(self) -> None:
        """Wait out in-flight prefetches so the stats describe a finished run."""
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=self.wait_timeout)
        self.stats.wasted = sum(
            1
            for entry in self.cache.entries()
            if entry.results is not None and not entry.consumed
        )

    def __enter__(self) -> PrefetchingRetriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# \b so that "research[...]" is not read as a search action. A drafter that
# proposes something else must produce no prefetch at all, not a wrong one.
_SEARCH_ACTION = re.compile(r"\bsearch\s*\[\s*(.+?)\s*\]", re.IGNORECASE | re.DOTALL)


class ReplayPredictor:
    """Replays a recorded query sequence.

    This is the *ceiling* of the mechanism, not a measurement of a drafter: a
    perfect replay hits every hop by construction. It exists to measure what a
    hit is worth on this storage medium, and — via ``wrong_hops`` — what a miss
    costs while a wasted prefetch is still on the disk.

    ``think_seconds`` charges the prediction the time a real drafter would take.
    Without it the prefetch starts sooner than any model could have started it,
    which flatters the result.
    """

    def __init__(
        self,
        queries: Sequence[str],
        *,
        wrong_hops: Sequence[int] = (),
        think_seconds: float = 0.0,
    ) -> None:
        self.queries = list(queries)
        self.wrong_hops = set(wrong_hops)
        self.think_seconds = think_seconds

    def predict(self, state: AgentState) -> str | None:
        if self.think_seconds:
            time.sleep(self.think_seconds)
        index = len(state.hops)
        if index >= len(self.queries):
            return None
        if index in self.wrong_hops:
            return f"deliberate miss at hop {index}"
        return self.queries[index]


class HTTPPredictor:
    """Drafter behind an OpenAI-compatible chat endpoint.

    Works against llama-server and Ollama alike. Every failure — unreachable
    server, malformed reply, no proposed search — returns None, which the
    retriever treats as "no prefetch". A drafter must never be able to break the
    retrieval it is speculating about.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        timeout: float = 10.0,
        max_tokens: int = 48,
        temperature: float = 0.0,
        transport: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._transport = transport or _post_json

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def build_prompt(self, state: AgentState) -> str:
        lines = [f"Question: {state.question}"]
        for index, hop in enumerate(state.hops, 1):
            lines.append(f"Action {index}: search[{hop.query}]")
            lines.append(f"Observation {index}: {'; '.join(hop.titles)}")
        lines.append(
            "Give only the next line, in the form: Action: search[<query>]"
        )
        return "\n".join(lines)

    def predict(self, state: AgentState) -> str | None:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You continue a ReAct transcript. Reply with exactly one "
                        "line: Action: search[<query>]"
                    ),
                },
                {"role": "user", "content": self.build_prompt(state)},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        try:
            response = self._transport(self.url, payload, self.timeout)
            content = response["choices"][0]["message"]["content"]
        except Exception:
            return None
        if not isinstance(content, str):
            return None
        match = _SEARCH_ACTION.search(content)
        if match is None:
            return None
        predicted = " ".join(match.group(1).split())
        return predicted or None


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    import json
    import urllib.request

    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))
