"""Paired baseline-vs-prefetch runs of multi-hop retrieval.

Measures what a prefetch hit is worth on the storage medium under test, and what
a miss costs while a wasted speculative search is still holding the disk. The
two arms must return identical passages -- that is checked, not assumed, because
it is the property that keeps accuracy out of the comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bundle import BundlePaths
from .encoder import Encoder
from .inspect import inspect_bundle, open_milvus
from .milvus_runtime import DEFAULT_MILVUS_URI
from .prefetch import (
    AgentState,
    HTTPPredictor,
    PrefetchingRetriever,
    PrefetchStats,
    QueryPredictor,
    ReplayPredictor,
)

ARMS = ("baseline", "prefetch")


@dataclass(frozen=True, slots=True)
class Episode:
    """A question and the sequence of searches an agent issues to answer it."""

    question: str
    queries: tuple[str, ...]


# Hand-written chains rather than agent traces: this measures the prefetch
# mechanism, not a drafter's accuracy. Queries are distinct across the whole set
# so that no hop can be answered from a page cache another hop warmed.
DEFAULT_EPISODES: tuple[Episode, ...] = (
    Episode(
        question="What causes auroras?",
        queries=(
            "solar wind charged particles",
            "earth magnetosphere field lines",
            "oxygen nitrogen emission colours in the upper atmosphere",
        ),
    ),
    Episode(
        question="Why did the Roman Republic become an empire?",
        queries=(
            "roman republic civil wars first century BC",
            "julius caesar crossing the rubicon",
            "augustus principate constitutional settlement",
        ),
    ),
    Episode(
        question="How do mRNA vaccines work?",
        queries=(
            "lipid nanoparticle delivery of nucleic acids",
            "spike protein antigen presentation",
            "adaptive immunity memory b cells",
        ),
    ),
    Episode(
        question="How are black holes detected?",
        queries=(
            "x-ray binaries accretion disc emission",
            "gravitational wave interferometer detection",
            "event horizon telescope interferometry imaging",
        ),
    ),
    Episode(
        question="How did the printing press change Europe?",
        queries=(
            "gutenberg movable type invention",
            "vernacular literacy spread in early modern europe",
            "protestant reformation pamphlet circulation",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class HopRecord:
    query: str
    latency_ms: float
    hit: bool
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArmResult:
    arm: str
    question: str
    hops: tuple[HopRecord, ...]
    stats: PrefetchStats
    retrieval_ms: float

    @property
    def hit_count(self) -> int:
        return sum(1 for hop in self.hops if hop.hit)


@dataclass(frozen=True, slots=True)
class Comparison:
    identical: bool
    divergent_hops: tuple[str, ...]
    baseline_retrieval_ms: float
    prefetch_retrieval_ms: float
    hidden_ms: float

    @property
    def saved_ms(self) -> float:
        return self.baseline_retrieval_ms - self.prefetch_retrieval_ms


def arm_order(index: int) -> tuple[str, str]:
    """Which arm runs first for episode `index`.

    Alternating matters on an on-disk index: whichever arm runs second inherits
    a page cache the first one warmed. Alternating does not remove that
    advantage -- the second arm always gets it -- it stops the advantage
    accruing to the same arm every episode. With an odd episode count the
    balance is imperfect by one episode, so report the count.
    """
    return (ARMS[1], ARMS[0]) if index % 2 else ARMS


def run_episode(
    episode: Episode,
    database: Any,
    encoder: Any,
    *,
    arm: str,
    predictor: QueryPredictor | None = None,
    limit: int = 5,
    decode_seconds: float = 0.0,
    timer: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> ArmResult:
    """Run one episode end to end and record every hop."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if arm == "baseline":
        predictor = None

    hops: list[HopRecord] = []
    retrieval_ms = 0.0
    state = AgentState(question=episode.question)
    with PrefetchingRetriever(
        database, encoder, predictor=predictor, limit=limit
    ) as retriever:
        for index, query in enumerate(episode.queries):
            hits_before = retriever.stats.hits
            started = timer()
            results = retriever.retrieve(query, state)
            latency_ms = (timer() - started) * 1000
            retrieval_ms += latency_ms
            state = state.extend(query, results)
            hops.append(
                HopRecord(
                    query=query,
                    latency_ms=latency_ms,
                    hit=retriever.stats.hits > hits_before,
                    source_ids=tuple(result.source_id for result in results),
                )
            )
            # Stands in for the target model decoding the next thought. This is
            # the window a prefetch has to work in, so it is an input to the
            # measurement and must be reported alongside it.
            if decode_seconds and index < len(episode.queries) - 1:
                sleep(decode_seconds)
        stats = retriever.stats
    return ArmResult(
        arm=arm,
        question=episode.question,
        hops=tuple(hops),
        stats=stats,
        retrieval_ms=retrieval_ms,
    )


def compare_arms(baseline: ArmResult, prefetch: ArmResult) -> Comparison:
    """Compare what the two arms returned, and price the difference.

    Divergence is a bug signal, not a result: prefetch is only ever allowed to
    change when a search happens. It is reported rather than raised, so a run
    still produces its numbers alongside the warning. The comparison is over
    passage ids, so it would not catch a change in score or in passage text at
    the same id.
    """
    divergent = tuple(
        left.query
        for left, right in zip(baseline.hops, prefetch.hops, strict=True)
        if left.source_ids != right.source_ids
    )
    return Comparison(
        identical=not divergent,
        divergent_hops=divergent,
        baseline_retrieval_ms=baseline.retrieval_ms,
        prefetch_retrieval_ms=prefetch.retrieval_ms,
        hidden_ms=prefetch.stats.hidden_ms,
    )


def select_episodes(count: int, episodes: Sequence[Episode] | None = None) -> list[Episode]:
    pool = list(episodes) if episodes else list(DEFAULT_EPISODES)
    if count > len(pool):
        raise ValueError(
            f"asked for {count} episodes but only {len(pool)} are defined; "
            "reusing one would answer its hops from the page cache the first "
            "pass warmed"
        )
    return pool[:count]


def format_report(
    episodes: Sequence[tuple[Episode, ArmResult, ArmResult, Comparison]],
    *,
    decode_seconds: float,
    drafter: str,
) -> str:
    lines: list[str] = []
    total_baseline = total_prefetch = total_hidden = 0.0
    hits = hops = contended = wasted = 0
    for episode, baseline, prefetch, comparison in episodes:
        lines.append(f"\n{episode.question}")
        for base_hop, pre_hop in zip(baseline.hops, prefetch.hops, strict=True):
            mark = "HIT " if pre_hop.hit else "miss"
            lines.append(
                f"  {mark} {base_hop.latency_ms:9.1f} ms -> {pre_hop.latency_ms:9.1f} ms"
                f"   {base_hop.query}"
            )
        total_baseline += baseline.retrieval_ms
        total_prefetch += prefetch.retrieval_ms
        total_hidden += prefetch.stats.hidden_ms
        hits += prefetch.hit_count
        hops += len(prefetch.hops)
        contended += prefetch.stats.contended
        wasted += prefetch.stats.wasted
        if not comparison.identical:
            lines.append(
                f"  DIVERGENCE: {', '.join(comparison.divergent_hops)} returned "
                "different passages than baseline — this is a bug, not a result"
            )

    saved = total_baseline - total_prefetch
    lines.append("")
    lines.append(f"drafter:          {drafter}")
    lines.append(f"decode gap:       {decode_seconds:.2f} s per hop (simulated)")
    lines.append(f"hit rate:         {hits}/{hops}")
    lines.append(f"retrieval total:  baseline {total_baseline:,.0f} ms -> prefetch {total_prefetch:,.0f} ms")
    lines.append(f"saved:            {saved:,.0f} ms ({saved / total_baseline * 100:.1f}%)" if total_baseline else "saved: n/a")
    lines.append(f"hidden (accounted): {total_hidden:,.0f} ms")
    lines.append(f"contended misses: {contended}   wasted prefetches: {wasted}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-hop retrieval twice per episode -- once plain, once with a "
            "drafter prefetching the next query -- and compare"
        )
    )
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--search-list", type=int)
    parser.add_argument("--uri", help=f"Milvus URI (default: {DEFAULT_MILVUS_URI})")
    parser.add_argument("--collection")
    parser.add_argument("--load-timeout", type=float)
    parser.add_argument(
        "--decode-seconds",
        type=float,
        default=1.0,
        help=(
            "pause between hops standing in for the target model's decode. This "
            "is the window a prefetch has to work in, so it bounds the result and "
            "is reported with it"
        ),
    )
    parser.add_argument(
        "--wrong-hops",
        type=int,
        nargs="*",
        default=(),
        metavar="HOP",
        help="hop indices the replay drafter should deliberately mispredict",
    )
    parser.add_argument(
        "--think-seconds",
        type=float,
        default=0.0,
        help="charge the replay drafter this much inference time per prediction",
    )
    parser.add_argument(
        "--draft-url",
        help=(
            "OpenAI-compatible drafter endpoint (llama-server or Ollama). Without "
            "it the drafter is a perfect replay of the episode, which measures the "
            "mechanism's ceiling rather than a model's accuracy"
        ),
    )
    parser.add_argument("--draft-model", default="llama-3.2-1b-instruct")
    parser.add_argument("--draft-timeout", type=float, default=10.0)
    parser.add_argument(
        "--json", type=Path, help="also write the raw per-hop records here"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = BundlePaths.resolve(args.bundle_dir)
        print(f"bundle: {paths.bundle_dir}", flush=True)
        summary = inspect_bundle(paths.bundle_dir, backend="milvus")
        episodes = select_episodes(args.episodes)

        database = open_milvus(
            summary,
            uri=args.uri,
            collection=args.collection,
            search_list=args.search_list,
            load_timeout=args.load_timeout,
        )
        with database:
            print("encoder: loading BGE-M3", flush=True)
            encoder = Encoder(summary.paths.bundle_dir, require_complete=False)

            if args.draft_url:
                drafter = f"{args.draft_model} at {args.draft_url}"
            else:
                drafter = (
                    "replay (perfect recall — a ceiling, not a drafter measurement)"
                )
            print(f"drafter: {drafter}")
            print(
                f"episodes: {len(episodes)}, decode gap {args.decode_seconds:.2f} s, "
                f"limit {args.limit}",
                flush=True,
            )

            records: list[tuple[Episode, ArmResult, ArmResult, Comparison]] = []
            for index, episode in enumerate(episodes):
                results: dict[str, ArmResult] = {}
                for arm in arm_order(index):
                    print(f"  [{index + 1}/{len(episodes)}] {arm}", flush=True)
                    results[arm] = run_episode(
                        episode,
                        database,
                        encoder,
                        arm=arm,
                        predictor=_make_predictor(args, episode),
                        limit=args.limit,
                        decode_seconds=args.decode_seconds,
                    )
                comparison = compare_arms(results["baseline"], results["prefetch"])
                records.append(
                    (episode, results["baseline"], results["prefetch"], comparison)
                )

        print(
            format_report(
                records, decode_seconds=args.decode_seconds, drafter=drafter
            )
        )
        if args.json:
            args.json.write_text(
                json.dumps(
                    [
                        {
                            "question": episode.question,
                            "baseline": [asdict(hop) for hop in baseline.hops],
                            "prefetch": [asdict(hop) for hop in prefetch.hops],
                            "stats": asdict(prefetch.stats),
                            "identical": comparison.identical,
                        }
                        for episode, baseline, prefetch, comparison in records
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"wrote {args.json}")
    except Exception as exc:
        parser.error(str(exc))


def _make_predictor(args: argparse.Namespace, episode: Episode) -> QueryPredictor:
    if args.draft_url:
        return HTTPPredictor(
            args.draft_url, model=args.draft_model, timeout=args.draft_timeout
        )
    return ReplayPredictor(
        episode.queries,
        wrong_hops=tuple(args.wrong_hops or ()),
        think_seconds=args.think_seconds,
    )


if __name__ == "__main__":
    main()
