import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from experiment.setup.orchestrate import (
    SetupError,
    completed_keys,
    read_trace_rows,
    repair_trailing_jsonl,
    run_phases,
)


def test_main_and_draft_server_phases_never_overlap() -> None:
    events: list[str] = []

    @contextmanager
    def server(role: str):
        events.append(f"start:{role}")
        try:
            yield role
        finally:
            events.append(f"stop:{role}")

    run_phases(
        ["main", "draft"],
        server,
        lambda role, value: events.append(f"run:{role}:{value}"),
    )

    assert events == [
        "start:main",
        "run:main:main",
        "stop:main",
        "start:draft",
        "run:draft:draft",
        "stop:draft",
    ]


def test_resume_keys_ignore_malformed_trailing_line(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps({"model_role": "main", "question": {"id": "q"}}) + "\n{" ,
        encoding="utf-8",
    )

    assert completed_keys(path) == {("main", "q")}
    assert repair_trailing_jsonl(path) is True
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_malformed_middle_line_is_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text("{\n{}\n", encoding="utf-8")

    with pytest.raises(SetupError, match="non-trailing"):
        read_trace_rows(path)
