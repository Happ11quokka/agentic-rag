from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fanoutqa.dataset import Question

from .common import format_metric, summarize

DEFAULT_TASKS = 10
DEFAULT_REPETITIONS = 3
SYSTEM_PROMPT = (
    "Use thinking mode. /think\n\n"
    "Answer the user's question directly. Preserve requested lists or mappings."
)


def benchmark_messages(question: Question) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.question},
    ]


def role_metric_rows(
    calls_by_role: dict[str, list[dict[str, Any]]],
) -> list[list[object]]:
    rows: list[list[object]] = []
    for role, metrics in role_metric_summary(calls_by_role).items():
        for label, key in (
            ("TTFT (ms)", "ttft_ms"),
            ("decode (tok/s)", "decode_tokens_per_second"),
        ):
            stats = metrics[key]
            rows.append(
                [
                    role,
                    label,
                    stats["count"],
                    format_metric(stats["mean"]),
                    format_metric(stats["median"]),
                    format_metric(stats["p95_worst"]),
                ]
            )
    return rows


def role_metric_summary(
    calls_by_role: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for role in ("main", "draft"):
        calls = calls_by_role[role]
        ttft = summarize(
            call["timing"]["ttft_ms"]
            for call in calls
            if call["timing"].get("ttft_ms") is not None
        )
        throughput = summarize(
            (
                call["timing"]["decode_tokens_per_second"]
                for call in calls
                if call["timing"].get("decode_tokens_per_second") is not None
            ),
            higher_is_better=True,
        )
        summary[role] = {
            "ttft_ms": asdict(ttft),
            "decode_tokens_per_second": asdict(throughput),
        }
    return summary
