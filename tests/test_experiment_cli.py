from pathlib import Path

import pytest

from experiment.setup import cli


def test_available_experiments_are_sorted(tmp_path: Path) -> None:
    (tmp_path / "z.toml").write_text("", encoding="utf-8")
    (tmp_path / "a.toml").write_text("", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("", encoding="utf-8")

    assert [path.name for path in cli.available_experiments(tmp_path)] == [
        "a.toml",
        "z.toml",
    ]


def test_choose_experiment_accepts_number_after_invalid_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment = tmp_path / "parallel.toml"
    experiment.write_text(
        'schema_version = 1\ndescription = "Sequential phases"\n', encoding="utf-8"
    )
    answers = iter(["wrong", "1"])

    selected = cli.choose_experiment([experiment], input_fn=lambda _: next(answers))

    assert selected == experiment
    output = capsys.readouterr().out
    assert "parallel" in output
    assert "Invalid experiment selection" in output


def test_choose_experiment_accepts_name(tmp_path: Path) -> None:
    experiment = tmp_path / "parallel.toml"
    experiment.write_text("schema_version = 1\n", encoding="utf-8")

    assert cli.choose_experiment(
        [experiment], input_fn=lambda _: "parallel"
    ) == experiment


def test_run_experiment_forwards_selected_config_and_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = tmp_path / "parallel.toml"
    calls: list[list[str] | None] = []
    monkeypatch.setattr(cli, "available_experiments", lambda: [experiment])
    monkeypatch.setattr(cli, "choose_experiment", lambda _: experiment)
    monkeypatch.setattr(cli.orchestrate, "main", lambda args=None: calls.append(args))
    monkeypatch.setattr(cli.sys, "argv", ["run-experiment", "--limit", "1"])

    cli.run_experiment()

    assert calls == [["--config", str(experiment), "--limit", "1"]]


def test_download_models_resolves_both_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli.orchestrate,
        "load_config",
        lambda: {"models": {"main": {"file": "main"}, "draft": {"file": "draft"}}},
    )

    def resolve(role: str, model: object, models_dir: Path):
        calls.append(role)
        return tmp_path / f"{role}.gguf", {"duration_ms": 1.0}

    monkeypatch.setattr(cli.orchestrate, "resolve_model", resolve)

    cli.download_models()

    assert calls == ["main", "draft"]
