from pathlib import Path

import pytest

from wikipedia import bundle


def _write_workspace(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["wikipedia"]\n',
        encoding="utf-8",
    )


def _use_source_clone(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = root / "wikipedia/src/wikipedia/bundle.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    monkeypatch.setattr(bundle, "__file__", str(source))


def test_repository_root_prefers_source_clone_over_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_clone = tmp_path / "source-clone"
    other_workspace = tmp_path / "other-workspace"
    _write_workspace(source_clone)
    _write_workspace(other_workspace)
    _use_source_clone(source_clone, monkeypatch)
    monkeypatch.chdir(other_workspace)

    assert bundle.find_repository_root() == source_clone


def test_clone_marker_is_written_and_reused_with_documented_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_clone = tmp_path / "source-clone"
    other_workspace = tmp_path / "other-workspace"
    remembered = tmp_path / "remembered"
    environment = tmp_path / "environment"
    explicit = tmp_path / "explicit"
    for path in (source_clone, other_workspace):
        _write_workspace(path)
    for path in (remembered, environment, explicit):
        path.mkdir()
    _use_source_clone(source_clone, monkeypatch)
    monkeypatch.chdir(other_workspace)
    monkeypatch.delenv(bundle.ENV_NAME, raising=False)

    marker = bundle.write_bundle_marker(remembered)

    assert marker == source_clone / bundle.MARKER_NAME
    assert bundle.BundlePaths.resolve().bundle_dir == remembered.resolve()

    monkeypatch.setenv(bundle.ENV_NAME, str(environment))
    assert bundle.BundlePaths.resolve().bundle_dir == environment.resolve()
    assert bundle.BundlePaths.resolve(explicit).bundle_dir == explicit.resolve()
