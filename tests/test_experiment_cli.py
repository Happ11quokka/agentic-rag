import pytest

from experiment import (
    cli,
    independent_run,
    parallel,
    parallel_scheduled,
    prefetched_toolcall,
    tooluse,
    vectordb,
)
from experiment.common import (
    DEFAULT_MODEL_KEYS,
    MODEL_SPECS,
    ExperimentError,
    prompt_positive_int,
)


def test_choose_experiment_accepts_number_after_invalid_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["wrong", "1"])

    selected = cli.choose_experiment(input_fn=lambda _: next(answers))

    assert selected.name == "parallel"
    output = capsys.readouterr().out
    assert "parallel" in output
    assert "Invalid experiment selection" in output


def test_choose_experiment_accepts_name() -> None:
    assert cli.choose_experiment(input_fn=lambda _: "vectordb").name == "vectordb"
    assert (
        cli.choose_experiment(input_fn=lambda _: "independent-run").name
        == "independent-run"
    )
    assert (
        cli.choose_experiment(input_fn=lambda _: "prefetched-toolcall").name
        == "prefetched-toolcall"
    )
    assert (
        cli.choose_experiment(input_fn=lambda _: "parallel-scheduled").name
        == "parallel-scheduled"
    )


def test_scheduled_experiment_is_immediately_after_parallel() -> None:
    assert [item.name for item in cli.EXPERIMENTS[:2]] == [
        "parallel",
        "parallel-scheduled",
    ]
    assert cli.choose_experiment(input_fn=lambda _: "2").name == "parallel-scheduled"


def test_run_experiment_dispatches_with_same_input_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    experiment = cli.Experiment(
        "test",
        "test",
        lambda *, input_fn, model_specs: calls.append((input_fn, model_specs)),
    )
    selected_models = {
        "main": MODEL_SPECS["qwen3-0.6b-q8-0"],
        "draft": MODEL_SPECS["qwen3-14b-q4-k-m"],
    }
    monkeypatch.setattr(cli, "choose_experiment", lambda **kwargs: experiment)
    monkeypatch.setattr(cli, "choose_model_specs", lambda **kwargs: selected_models)
    monkeypatch.setattr(cli.sys, "argv", ["run-experiment"])

    def input_fn(_: str) -> str:
        return ""

    cli.run_experiment(input_fn=input_fn)

    assert calls == [(input_fn, selected_models)]


def test_choose_model_accepts_defaults_numbers_and_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.choose_model("main", input_fn=lambda _: "").key == DEFAULT_MODEL_KEYS[
        "main"
    ]
    assert cli.choose_model("draft", input_fn=lambda _: "").key == DEFAULT_MODEL_KEYS[
        "draft"
    ]
    assert cli.choose_model("main", input_fn=lambda _: "4").key == "qwen3-0.6b-q8-0"
    assert (
        cli.choose_model("draft", input_fn=lambda _: "qwen3-14b-q4-k-m").key
        == "qwen3-14b-q4-k-m"
    )
    assert "target" in capsys.readouterr().out


def test_choose_model_retries_invalid_input() -> None:
    answers = iter(["wrong", "2"])
    assert cli.choose_model("draft", input_fn=lambda _: next(answers)).key == (
        "qwen3-4b-q4-k-m"
    )


def test_choose_model_specs_allows_cross_role_and_duplicate_selection() -> None:
    cross_answers = iter(["4", "1"])
    cross = cli.choose_model_specs(input_fn=lambda _: next(cross_answers))
    assert cross["main"].key == "qwen3-0.6b-q8-0"
    assert cross["draft"].key == "qwen3-14b-q4-k-m"

    duplicate_answers = iter(["2", "2"])
    duplicate = cli.choose_model_specs(input_fn=lambda _: next(duplicate_answers))
    assert duplicate["main"] is duplicate["draft"]


def test_choose_model_reports_eof() -> None:
    def no_input(_: str) -> str:
        raise EOFError

    with pytest.raises(ExperimentError, match="model selection"):
        cli.choose_model("main", input_fn=no_input)


def test_vectordb_dispatch_skips_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    experiment = cli.Experiment(
        "test", "test", lambda *, input_fn: calls.append(input_fn), uses_models=False
    )
    monkeypatch.setattr(cli, "choose_experiment", lambda **kwargs: experiment)
    monkeypatch.setattr(
        cli,
        "choose_model_specs",
        lambda **kwargs: pytest.fail("vectordb must not prompt for models"),
    )
    monkeypatch.setattr(cli.sys, "argv", ["run-experiment"])

    def input_fn(_: str) -> str:
        return ""

    cli.run_experiment(input_fn=input_fn)

    assert calls == [input_fn]


def test_run_experiment_rejects_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "argv", ["run-experiment", "parallel"])
    with pytest.raises(ExperimentError, match="accepts no arguments"):
        cli.run_experiment()


def test_prompt_positive_int_uses_default_and_retries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["bad", "-1", "3"])
    assert prompt_positive_int("Count", 10, input_fn=lambda _: next(answers)) == 3
    assert capsys.readouterr().out.count("positive integer") == 2
    assert prompt_positive_int("Count", 10, input_fn=lambda _: "") == 10


@pytest.mark.parametrize(
    ("module", "defaults"),
    [
        (parallel, (parallel.DEFAULT_TASKS, parallel.DEFAULT_REPETITIONS)),
        (
            parallel_scheduled,
            (parallel_scheduled.DEFAULT_TASKS, parallel_scheduled.DEFAULT_REPETITIONS),
        ),
        (
            independent_run,
            (independent_run.DEFAULT_TASKS, independent_run.DEFAULT_REPETITIONS),
        ),
        (tooluse, (tooluse.DEFAULT_TASKS, tooluse.DEFAULT_REPETITIONS)),
        (
            prefetched_toolcall,
            (
                prefetched_toolcall.DEFAULT_TASKS,
                prefetched_toolcall.DEFAULT_REPETITIONS,
            ),
        ),
        (vectordb, (vectordb.DEFAULT_QUERIES, vectordb.DEFAULT_REPETITIONS)),
    ],
)
def test_experiment_prompts_forward_defaults(
    module: object,
    defaults: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        module, "run", lambda first, second, **kwargs: calls.append((first, second))
    )
    module.prompt_and_run(input_fn=lambda _: "")  # type: ignore[attr-defined]
    assert calls == [defaults]
