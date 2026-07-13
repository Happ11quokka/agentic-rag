from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import wikipedia.download as module


class FakeApi:
    def dataset_info(self, repo_id: str, revision: str):
        assert repo_id == module.DATASET_REPOSITORY
        return SimpleNamespace(sha="resolved-dataset")

    def model_info(self, repo_id: str, revision: str):
        assert repo_id == module.MODEL_REPOSITORY
        return SimpleNamespace(sha="resolved-model")

    def list_repo_files(self, repo_id: str, revision: str, repo_type: str):
        assert repo_type == "dataset"
        return ["data/fr/x.parquet", "data/en/02.parquet", "data/en/01.parquet"]


def test_download_filters_and_manifest_is_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[tool.uv.workspace]\nmembers=[]\n", encoding="utf-8")
    output = tmp_path / "bundle"
    calls = []

    def marker(path):
        target = repo / ".wikipedia.local.toml"
        target.write_text(str(path), encoding="utf-8")
        return target

    monkeypatch.setattr(module, "write_bundle_marker", marker)

    def snapshot(**kwargs):
        assert (repo / ".wikipedia.local.toml").exists()
        assert not (output / "manifest.json").exists()
        calls.append(kwargs)
        local = Path(kwargs["local_dir"])
        if kwargs["repo_type"] == "dataset":
            for name in kwargs["allow_patterns"]:
                target = local / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"parquet")
        else:
            (local / "config.json").write_text("{}", encoding="utf-8")

    manifest = module.download_bundle(
        output, max_shards=1, token="secret", api=FakeApi(), snapshot=snapshot
    )
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["allow_patterns"] == ["data/en/01.parquet"]
    assert calls[1]["repo_type"] == "model"
    assert "onnx/**" in calls[1]["ignore_patterns"]
    assert manifest["partial"] is True
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"


def test_failed_download_leaves_no_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(module, "write_bundle_marker", lambda path: None)
    output = tmp_path / "bundle"

    def fail(**kwargs):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        module.download_bundle(output, api=FakeApi(), snapshot=fail)
    assert not (output / "manifest.json").exists()
