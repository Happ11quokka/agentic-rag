import pytest
from fakes import FakeDatabase, FakeEncoder

from wikipedia.prefetch import ReplayPredictor
from wikipedia.prefetch_bench import (
    DEFAULT_EPISODES,
    Comparison,
    Episode,
    arm_order,
    compare_arms,
    format_report,
    run_episode,
    select_episodes,
)

EPISODE = Episode(
    question="What causes auroras?",
    queries=("solar wind", "magnetosphere", "oxygen emission lines"),
)


def _run(arm: str, database: FakeDatabase, **kwargs: object) -> object:
    predictor = ReplayPredictor(EPISODE.queries) if arm == "prefetch" else None
    kwargs.setdefault("decode_seconds", 0.0)
    return run_episode(
        EPISODE,
        database,
        FakeEncoder(),
        arm=arm,
        predictor=predictor,
        limit=2,
        **kwargs,
    )


def test_arm_order_alternates_so_cache_warming_does_not_favour_one_arm() -> None:
    assert arm_order(0) != arm_order(1)
    assert arm_order(0) == arm_order(2)
    assert set(arm_order(0)) == {"baseline", "prefetch"}


def test_default_episodes_never_repeat_a_query() -> None:
    queries = [query for episode in DEFAULT_EPISODES for query in episode.queries]

    assert len(set(queries)) == len(queries)


def test_default_episodes_are_multi_hop() -> None:
    assert DEFAULT_EPISODES
    assert all(len(episode.queries) >= 2 for episode in DEFAULT_EPISODES)


def test_run_episode_records_one_hop_per_query_in_order() -> None:
    result = _run("baseline", FakeDatabase())

    assert [hop.query for hop in result.hops] == list(EPISODE.queries)
    assert result.arm == "baseline"


def test_baseline_searches_every_hop_and_never_reports_a_hit() -> None:
    database = FakeDatabase()

    result = _run("baseline", database)

    assert database.searches == list(EPISODE.queries)
    assert [hop.hit for hop in result.hops] == [False, False, False]


def test_a_replayed_drafter_serves_every_hop_after_the_first_from_prefetch() -> None:
    result = _run("prefetch", FakeDatabase())

    assert [hop.hit for hop in result.hops] == [False, True, True]
    assert result.stats.hits == 2


def test_compare_arms_passes_when_both_arms_return_the_same_passages() -> None:
    database = FakeDatabase()

    comparison = compare_arms(_run("baseline", database), _run("prefetch", database))

    assert comparison.identical is True
    assert comparison.divergent_hops == ()


def test_compare_arms_flags_a_hop_whose_passages_differ() -> None:
    class DriftingDatabase(FakeDatabase):
        """Returns different passages the second time a query is searched."""

        def search(self, vector, *, limit: int = 10):
            seen = self.searches.count(vector.text)
            results = super().search(vector, limit=limit)
            if vector.text == "magnetosphere" and seen:
                return list(reversed(results))
            return results

    database = DriftingDatabase()

    comparison = compare_arms(_run("baseline", database), _run("prefetch", database))

    assert comparison.identical is False
    assert comparison.divergent_hops == ("magnetosphere",)


def test_run_episode_pauses_between_hops_to_stand_in_for_decode() -> None:
    slept: list[float] = []

    _run("baseline", FakeDatabase(), decode_seconds=0.25, sleep=slept.append)

    assert slept == [0.25, 0.25], "one decode gap between hops, none after the last"


def test_run_episode_rejects_an_arm_it_does_not_know() -> None:
    with pytest.raises(ValueError):
        _run("neither", FakeDatabase())


def test_report_shouts_when_the_two_arms_disagree() -> None:
    database = FakeDatabase()
    baseline, prefetch = _run("baseline", database), _run("prefetch", database)
    divergent = Comparison(
        identical=False,
        divergent_hops=("magnetosphere",),
        baseline_retrieval_ms=100.0,
        prefetch_retrieval_ms=40.0,
        hidden_ms=60.0,
    )

    report = format_report(
        [(EPISODE, baseline, prefetch, divergent)],
        decode_seconds=1.0,
        drafter="replay",
    )

    assert "DIVERGENCE" in report
    assert "magnetosphere" in report


def test_report_states_the_simulated_decode_gap_it_depended_on() -> None:
    database = FakeDatabase()
    baseline, prefetch = _run("baseline", database), _run("prefetch", database)
    comparison = compare_arms(baseline, prefetch)

    report = format_report(
        [(EPISODE, baseline, prefetch, comparison)],
        decode_seconds=2.5,
        drafter="replay",
    )

    assert "2.50 s" in report
    assert "simulated" in report
    assert "DIVERGENCE" not in report


def test_select_episodes_refuses_to_recycle_an_episode() -> None:
    with pytest.raises(ValueError):
        select_episodes(len(DEFAULT_EPISODES) + 1)
