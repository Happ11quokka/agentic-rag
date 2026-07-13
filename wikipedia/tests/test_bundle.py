from __future__ import annotations

from pathlib import Path

import pytest

from wikipedia.bundle import BundleError, BundlePaths, write_bundle_marker


def workspace(path: Path) -> Path:
    path.mkdir()
    (path / "pyproject.toml").write_text("[tool.uv.workspace]\nmembers = []\n", encoding="utf-8")
    return path


def test_resolution_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = workspace(tmp_path / "repo")
    marker_bundle = tmp_path / "marker"
    env_bundle = tmp_path / "environment"
    explicit_bundle = tmp_path / "explicit"
    for path in (marker_bundle, env_bundle, explicit_bundle):
        path.mkdir()
    write_bundle_marker(marker_bundle, root)
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setenv("WIKIPEDIA_BUNDLE_DIR", str(env_bundle))
    assert BundlePaths.resolve(explicit_bundle).bundle_dir == explicit_bundle.resolve()
    assert BundlePaths.resolve().bundle_dir == env_bundle.resolve()
    monkeypatch.delenv("WIKIPEDIA_BUNDLE_DIR")
    assert BundlePaths.resolve().bundle_dir == marker_bundle.resolve()


def test_marker_errors_are_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = workspace(tmp_path / "repo")
    monkeypatch.chdir(root)
    with pytest.raises(BundleError, match="wikipedia-download"):
        BundlePaths.resolve()
    (root / ".wikipedia.local.toml").write_text(
        'schema_version = 1\nbundle_dir = "relative"\n', encoding="utf-8"
    )
    with pytest.raises(BundleError, match="absolute"):
        BundlePaths.resolve()
    (root / ".wikipedia.local.toml").write_text("not = [valid", encoding="utf-8")
    with pytest.raises(BundleError, match="Malformed"):
        BundlePaths.resolve()
    missing = tmp_path / "missing"
    (root / ".wikipedia.local.toml").write_text(
        f'schema_version = 1\nbundle_dir = "{missing}"\n', encoding="utf-8"
    )
    with pytest.raises(BundleError, match="does not exist"):
        BundlePaths.resolve()


def test_marker_update_does_not_touch_old_bundle(tmp_path: Path) -> None:
    root = workspace(tmp_path / "repo")
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    sentinel = old / "keep"
    sentinel.write_text("yes", encoding="utf-8")
    write_bundle_marker(old, root)
    write_bundle_marker(new, root)
    assert sentinel.read_text(encoding="utf-8") == "yes"
    assert str(new.resolve()) in (root / ".wikipedia.local.toml").read_text(encoding="utf-8")
