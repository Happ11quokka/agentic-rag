import json

import pytest

from experiment.artifacts import RunArtifacts


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_run_artifacts_write_crash_safe_records_and_summary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr("experiment.artifacts.os.fsync", fsync_calls.append)

    with RunArtifacts("parallel", {"setting": "값"}, results_dir=tmp_path) as run:
        run.append({"question": {"id": "q1"}, "content": "응답"})
        run.append({"question": {"id": "q2"}, "content": "answer"})
        run.write_summary({"metrics": {"count": 2}})

    manifest = json.loads(run.manifest_path.read_text())
    rows = _read_jsonl(run.traces_path)
    summary = json.loads(run.summary_path.read_text())

    assert manifest["status"] == "completed"
    assert manifest["record_count"] == 2
    assert manifest["setting"] == "값"
    assert [row["record_index"] for row in rows] == [1, 2]
    assert rows[0]["content"] == "응답"
    assert summary["record_count"] == 2
    assert summary["metrics"] == {"count": 2}
    assert len(fsync_calls) >= 5


def test_run_artifacts_keep_completed_rows_when_run_fails(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="broken"):
        with RunArtifacts("tooluse", {}, results_dir=tmp_path) as run:
            run.append({"question": {"id": "q1"}})
            raise RuntimeError("broken")

    manifest = json.loads(run.manifest_path.read_text())
    rows = _read_jsonl(run.traces_path)

    assert manifest["status"] == "failed"
    assert manifest["record_count"] == 1
    assert manifest["error"] == {"type": "RuntimeError", "message": "broken"}
    assert rows[0]["question"]["id"] == "q1"
    assert not run.summary_path.exists()


def test_run_artifact_directories_are_unique(tmp_path) -> None:
    first = RunArtifacts("parallel", {}, results_dir=tmp_path)
    second = RunArtifacts("parallel", {}, results_dir=tmp_path)
    try:
        assert first.run_dir != second.run_dir
    finally:
        first.__exit__(None, None, None)
        second.__exit__(None, None, None)
