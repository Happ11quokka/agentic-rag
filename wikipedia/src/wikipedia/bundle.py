from __future__ import annotations

import json
import os
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MARKER_NAME = ".wikipedia.local.toml"
ENV_NAME = "WIKIPEDIA_BUNDLE_DIR"


class BundleError(RuntimeError):
    pass


def find_repository_root(start: Path | None = None) -> Path:
    candidates = [Path.cwd() if start is None else Path(start)]
    candidates.append(Path(__file__).resolve())
    for candidate in candidates:
        current = candidate.resolve()
        if current.is_file():
            current = current.parent
        for directory in (current, *current.parents):
            pyproject = directory / "pyproject.toml"
            if not pyproject.is_file():
                continue
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                continue
            if "workspace" in data.get("tool", {}).get("uv", {}):
                return directory
    raise BundleError("Cannot find root uv workspace (pyproject.toml with [tool.uv.workspace]).")


def _absolute_path(value: str | Path, source: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise BundleError(f"{source} must contain an absolute path, got: {value!s}")
    return raw.resolve()


@dataclass(frozen=True, slots=True)
class BundlePaths:
    bundle_dir: Path
    dataset_dir: Path
    model_dir: Path
    manifest_path: Path
    state_dir: Path

    @classmethod
    def from_dir(cls, bundle_dir: str | Path) -> BundlePaths:
        root = _absolute_path(bundle_dir, "bundle_dir")
        return cls(
            bundle_dir=root,
            dataset_dir=root / "dataset" / "data" / "en",
            model_dir=root / "models" / "bge-m3",
            manifest_path=root / "manifest.json",
            state_dir=root / "state",
        )

    @classmethod
    def resolve(cls, bundle_dir: str | Path | None = None) -> BundlePaths:
        if bundle_dir is not None:
            return cls.from_dir(bundle_dir)
        environment = os.environ.get(ENV_NAME)
        if environment:
            return cls.from_dir(_absolute_path(environment, ENV_NAME))
        root = find_repository_root()
        marker = root / MARKER_NAME
        if marker.exists():
            try:
                data = tomllib.loads(marker.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise BundleError(f"Malformed bundle marker {marker}: {exc}") from exc
            if data.get("schema_version") != 1:
                raise BundleError(f"Unsupported or missing schema_version in {marker}")
            value = data.get("bundle_dir")
            if not isinstance(value, str) or not value:
                raise BundleError(f"Missing bundle_dir in {marker}")
            paths = cls.from_dir(_absolute_path(value, str(marker)))
            if not paths.bundle_dir.is_dir():
                raise BundleError(
                    f"Configured Wikipedia bundle does not exist: {paths.bundle_dir}. "
                    "Run `uv run wikipedia-ingest BACKEND --bundle-dir /absolute/path`."
                )
            return paths
        raise BundleError(
            "Wikipedia bundle is not configured. Run `uv run wikipedia-ingest BACKEND "
            "--bundle-dir /absolute/path`."
        )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_bundle_marker(bundle_dir: str | Path, repository_root: Path | None = None) -> Path:
    resolved = _absolute_path(bundle_dir, "--output-dir")
    root = find_repository_root() if repository_root is None else repository_root.resolve()
    escaped = str(resolved).replace("\\", "\\\\").replace('"', '\\"')
    marker = root / MARKER_NAME
    _atomic_text(marker, f'schema_version = 1\nbundle_dir = "{escaped}"\n')
    return marker


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_manifest(paths: BundlePaths, *, require_complete: bool = True) -> dict[str, Any]:
    try:
        manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if require_complete:
            raise BundleError(
                f"Bundle manifest is missing or incomplete at {paths.manifest_path}. "
                "Resume with `uv run wikipedia-ingest BACKEND`."
            ) from exc
        raise
    if manifest.get("schema_version") != 1:
        raise BundleError(f"Unsupported bundle manifest at {paths.manifest_path}.")
    if require_complete and manifest.get("status") != "complete":
        raise BundleError(
            f"Bundle manifest is incomplete at {paths.manifest_path}. "
            "Resume with `uv run wikipedia-ingest BACKEND`."
        )
    return manifest
