import pytest

from experiment import cli, independent_run, parallel, tooluse, vectordb
from experiment.common import ExperimentError, prompt_positive_int


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


def test_run_experiment_dispatches_with_same_input_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    experiment = cli.Experiment(
        "test", "test", lambda *, input_fn: calls.append(input_fn)
    )
    monkeypatch.setattr(cli, "choose_experiment", lambda **kwargs: experiment)
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
            independent_run,
            (independent_run.DEFAULT_TASKS, independent_run.DEFAULT_REPETITIONS),
        ),
        (tooluse, (tooluse.DEFAULT_TASKS, tooluse.DEFAULT_REPETITIONS)),
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
        module, "run", lambda first, second: calls.append((first, second))
    )
    module.prompt_and_run(input_fn=lambda _: "")  # type: ignore[attr-defined]
    assert calls == [defaults]
