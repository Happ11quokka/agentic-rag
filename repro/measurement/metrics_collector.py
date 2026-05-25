"""llama.cpp metrics polling and parsing.

Polling thread implementation in Task 17; this file is parser-only for now.
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
