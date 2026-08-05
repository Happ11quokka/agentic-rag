from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4

from fanoutqa.dataset import DEV_GIT_BLOB_SHA1, DEV_REVISION, Question

from .common import ROOT, SEED, ModelSpec

SCHEMA_VERSION = 1
RESULTS_DIR = ROOT / "experiment" / "results"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _git_metadata() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {"commit": None, "dirty": None}
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def run_metadata(
    *,
    task_count: int,
    repetitions: int,
    questions: list[Question],
    model_specs: Mapping[str, ModelSpec],
    model_paths: dict[str, Path],
    llama_cpp: dict[str, Any],
    generation: dict[str, Any],
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_count": task_count,
        "repetitions": repetitions,
        "seed": SEED,
        "dataset": {
            "name": "FanOutQA dev",
            "revision": DEV_REVISION,
            "git_blob_sha1": DEV_GIT_BLOB_SHA1,
        },
        "questions": [question.agent_value() for question in questions],
        "generation": generation,
        "models": {
            role: {
                "catalog_key": model_specs[role].key,
                "label": model_specs[role].label,
                "repo": model_specs[role].repo,
                "revision": model_specs[role].revision,
                "filename": model_specs[role].filename,
                "bytes": model_specs[role].bytes,
                "sha256": model_specs[role].sha256,
                "path": str(path),
            }
            for role, path in model_paths.items()
        },
        "llama_cpp": llama_cpp,
        "git": _git_metadata(),
        "parameters": parameters or {},
    }


class RunArtifacts:
    def __init__(
        self,
        experiment: str,
        metadata: dict[str, Any],
        *,
        results_dir: Path | None = None,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_id = f"{timestamp}-{uuid4().hex[:8]}"
        self.experiment = experiment
        self.run_dir = (results_dir or RESULTS_DIR) / experiment / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.run_dir / "manifest.json"
        self.traces_path = self.run_dir / "traces.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.record_count = 0
        self.manifest = {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment": experiment,
            "status": "running",
            "created_at": _now(),
            "completed_at": None,
            "record_count": 0,
            "trace_file": self.traces_path.name,
            "summary_file": self.summary_path.name,
            "recording": {"warmup_calls": False, "server_logs": False},
        }
        _atomic_json(self.manifest_path, self.manifest)
        print(f"results: {self.run_dir}", flush=True)

    def __enter__(self) -> RunArtifacts:
        return self

    def append(self, value: dict[str, Any]) -> None:
        record_index = self.record_count + 1
        row = {
            **value,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "record_index": record_index,
            "recorded_at": _now(),
        }
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.traces_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self.record_count = record_index

    def write_summary(self, value: dict[str, Any]) -> None:
        _atomic_json(
            self.summary_path,
            {
                **value,
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "experiment": self.experiment,
                "record_count": self.record_count,
            },
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.manifest["status"] = "completed"
        elif issubclass(exc_type, KeyboardInterrupt):
            self.manifest["status"] = "interrupted"
        else:
            self.manifest["status"] = "failed"
        self.manifest["completed_at"] = _now()
        self.manifest["record_count"] = self.record_count
        if exc is not None:
            self.manifest["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        _atomic_json(self.manifest_path, self.manifest)
