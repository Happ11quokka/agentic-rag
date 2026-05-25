"""llama.cpp metrics polling and parsing.

Parsers (parse_metrics, parse_slots, PollSample) plus background polling
thread (MetricsCollector) for capturing llama-server /metrics and /slots
snapshots in a ring buffer with bracket-aware slicing.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class PollSample:
    t: float
    prefill_s_total: float
    decode_s_total: float
    prefill_tokens_total: int
    decode_tokens_total: int
    n_decode_total: int
    n_tokens_max: int
    requests_processing: int
    is_processing: bool
    n_prompt_tokens: int
    n_prompt_tokens_processed: int
    n_prompt_tokens_cache: int

    @classmethod
    def from_endpoints(cls, *, t: float, metrics: dict, slots: dict) -> "PollSample":
        return cls(
            t=t,
            prefill_s_total=metrics.get("llamacpp:prompt_seconds_total", 0.0),
            decode_s_total=metrics.get("llamacpp:tokens_predicted_seconds_total", 0.0),
            prefill_tokens_total=int(metrics.get("llamacpp:prompt_tokens_total", 0)),
            decode_tokens_total=int(metrics.get("llamacpp:tokens_predicted_total", 0)),
            n_decode_total=int(metrics.get("llamacpp:n_decode_total", 0)),
            n_tokens_max=int(metrics.get("llamacpp:n_tokens_max", 0)),
            requests_processing=int(metrics.get("llamacpp:requests_processing", 0)),
            is_processing=slots.get("is_processing", False),
            n_prompt_tokens=slots.get("n_prompt_tokens", 0),
            n_prompt_tokens_processed=slots.get("n_prompt_tokens_processed", 0),
            n_prompt_tokens_cache=slots.get("n_prompt_tokens_cache", 0),
        )


def parse_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus text format. Skips comments and HELP/TYPE lines."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name, value = parts[0], parts[-1]
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def parse_slots(slots_json: list[dict]) -> dict[str, Any]:
    """Take the first slot (we use --parallel 1). Returns flat dict."""
    if not slots_json:
        return {
            "is_processing": False,
            "n_prompt_tokens": 0,
            "n_prompt_tokens_processed": 0,
            "n_prompt_tokens_cache": 0,
        }
    s = slots_json[0]
    return {
        "is_processing": bool(s.get("is_processing", False)),
        "n_prompt_tokens": int(s.get("n_prompt_tokens", 0)),
        "n_prompt_tokens_processed": int(s.get("n_prompt_tokens_processed", 0)),
        "n_prompt_tokens_cache": int(s.get("n_prompt_tokens_cache", 0)),
    }


import threading
import time
from collections import deque
from typing import Optional
import requests


class MetricsCollector:
    """Background polling thread for llama-server /metrics and /slots.

    Lifecycle:
        c = MetricsCollector("http://localhost:8000").start()
        # ... run sweep ...
        samples = c.slice(t_start, t_end)
        c.stop()

    `slice(a, b)` returns all PollSample within [a, b] PLUS the boundary samples
    (one before a, one after b) when available -- required for correct delta
    computation when the window is shorter than the polling interval (see spec
    section 7.4).
    """

    def __init__(self, base_url: str, *, interval_s: float = 0.1, maxlen: int = 1_800_000):
        self.base_url = base_url.rstrip("/")
        self.interval_s = interval_s
        self._buf: deque[PollSample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._session = requests.Session()

    def start(self) -> "MetricsCollector":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MetricsCollector")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                m_text = self._session.get(f"{self.base_url}/metrics", timeout=2).text
                s_json = self._session.get(f"{self.base_url}/slots", timeout=2).json()
                sample = PollSample.from_endpoints(
                    t=t,
                    metrics=parse_metrics(m_text),
                    slots=parse_slots(s_json),
                )
                with self._lock:
                    self._buf.append(sample)
            except Exception:
                # transient network errors -- skip this tick
                pass
            self._stop.wait(self.interval_s)

    def slice(self, t_start: float, t_end: float) -> list[PollSample]:
        """Return samples in [t_start, t_end] plus bracket samples (one before, one after)."""
        with self._lock:
            buf = list(self._buf)
        if not buf:
            return []
        in_window = [s for s in buf if t_start <= s.t <= t_end]
        before = [s for s in buf if s.t < t_start]
        after = [s for s in buf if s.t > t_end]
        result: list[PollSample] = []
        if before:
            result.append(before[-1])
        result.extend(in_window)
        if after:
            result.append(after[0])
        return result

    def detect_kv_eviction(self, samples: list[PollSample]) -> bool:
        """Spec section 11: return True if n_prompt_tokens_cache decreased while
        is_processing=True.

        Such a decrease signals llama.cpp evicted cached prefix tokens to fit a
        longer prompt -- relevant for LATS at long contexts.
        """
        prev = None
        for s in samples:
            if prev is not None and s.is_processing and prev.is_processing:
                if s.n_prompt_tokens_cache < prev.n_prompt_tokens_cache:
                    return True
            prev = s
        return False

    def detect_no_decode_progress(
        self, *, window_samples: int = 30
    ) -> bool:
        """Spec section 9.5 #2: return True if decode_tokens_total did not
        increase across the last `window_samples` polls while is_processing=True.

        Caller checks this from the run_one watchdog every poll interval.
        """
        with self._lock:
            buf = list(self._buf)
        if len(buf) < window_samples:
            return False
        recent = buf[-window_samples:]
        if not all(s.is_processing for s in recent):
            return False
        return recent[-1].decode_tokens_total == recent[0].decode_tokens_total
