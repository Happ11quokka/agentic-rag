from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from fanoutqa.dataset import Question, load_dev, select_questions
from wikipedia.bundle import BundleError, BundlePaths, load_manifest
from wikipedia.download import model_is_downloaded
from wikipedia.qdrant import QdrantConfig, QdrantVectorDB
from wikipedia.qdrant_runtime import DEFAULT_QDRANT_URL, ensure_qdrant

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "experiment" / "models"
HOST = "127.0.0.1"
MAIN_PORT = 8080
DRAFT_PORT = 8081
MINIMUM_LLAMA_BUILD = 8360
SERVER_STARTUP_TIMEOUT_SECONDS = 180.0
REQUEST_TIMEOUT_SECONDS = 180.0
SEED = 42
CONTEXT_SIZE = 16_384
MAX_TOKENS_PER_TURN = 1_536
MAX_SEARCHES = 8
RETRIEVAL_TOP_K = 3
RETRIEVAL_MAX_CHARS = 1_400
FIXED_LLM_WARMUP = "Reply with only: warm"
FIXED_RETRIEVAL_WARMUP = "English Wikipedia"
INGEST_STABILITY_SECONDS = 2.0

GENERATION = {
    "max_tokens": MAX_TOKENS_PER_TURN,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "seed": SEED,
}


class ExperimentError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    role: str
    repo: str
    revision: str
    filename: str
    bytes: int
    sha256: str

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.role / self.filename


MODEL_SPECS = {
    "main": ModelSpec(
        role="main",
        repo="Qwen/Qwen3-14B-GGUF",
        revision="530227a7d994db8eca5ab5ced2fb692b614357fd",
        filename="Qwen3-14B-Q4_K_M.gguf",
        bytes=9_001_752_960,
        sha256="500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
    ),
    "draft": ModelSpec(
        role="draft",
        repo="Qwen/Qwen3-4B-GGUF",
        revision="bc640142c66e1fdd12af0bd68f40445458f3869b",
        filename="Qwen3-4B-Q4_K_M.gguf",
        bytes=2_497_280_256,
        sha256="7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
    ),
}


@dataclass(frozen=True, slots=True)
class Stats:
    count: int
    mean: float | None
    median: float | None
    p95_worst: float | None
    standard_deviation: float | None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(values: Iterable[float], *, higher_is_better: bool = False) -> Stats:
    """Summarize values with a direction-aware 95% worst-case bound.

    For lower-is-better metrics, ``p95_worst`` is numeric p95: 95% of observations
    are at or below it. For higher-is-better metrics, it is numeric p5: 95% of
    observations are at or above it.
    """
    selected = [float(value) for value in values]
    if not selected:
        return Stats(0, None, None, None, None)
    mean = sum(selected) / len(selected)
    variance = sum((value - mean) ** 2 for value in selected) / len(selected)
    worst_fraction = 0.05 if higher_is_better else 0.95
    return Stats(
        count=len(selected),
        mean=mean,
        median=percentile(selected, 0.5),
        p95_worst=percentile(selected, worst_fraction),
        standard_deviation=math.sqrt(variance),
    )


def format_metric(value: float | None, *, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    values = [
        [str(value) for value in headers],
        *[[str(value) for value in row] for row in rows],
    ]
    widths = [max(len(row[index]) for row in values) for index in range(len(headers))]

    def line(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print(line(values[0]))
    print(line(["-" * width for width in widths]))
    for row in values[1:]:
        print(line(row))


class Progress:
    def __init__(
        self,
        label: str,
        total: int,
        *,
        stream: Any = sys.stdout,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.label = label
        self.total = total
        self.stream = stream
        self.clock = clock
        self.started = clock()
        self.last_printed = 0
        self.tty = bool(getattr(stream, "isatty", lambda: False)())

    def update(self, completed: int, detail: str = "") -> None:
        completed = min(completed, self.total)
        if not self.tty and completed not in {1, self.total}:
            milestone = max(1, math.ceil(self.total / 20))
            if completed - self.last_printed < milestone:
                return
        elapsed = max(0.0, self.clock() - self.started)
        eta = elapsed / completed * (self.total - completed) if completed else 0.0
        suffix = f" | {detail}" if detail else ""
        message = (
            f"[{self.label}] {completed}/{self.total} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s{suffix}"
        )
        if self.tty:
            end = "\n" if completed >= self.total else "\r"
            print(message.ljust(100), end=end, file=self.stream, flush=True)
        else:
            print(message, file=self.stream, flush=True)
        self.last_printed = completed


def prompt_positive_int(
    label: str,
    default: int,
    *,
    input_fn: Callable[[str], str] = input,
) -> int:
    while True:
        try:
            answer = input_fn(f"{label} [{default}]: ").strip()
        except EOFError:
            raise ExperimentError(
                "interactive experiment options require input"
            ) from None
        if not answer:
            return default
        try:
            value = int(answer)
        except ValueError:
            value = 0
        if value > 0:
            return value
        print("Enter a positive integer.")


def select_benchmark_questions(count: int) -> list[Question]:
    questions = load_dev()
    try:
        return select_questions(questions, limit=count, seed=SEED)
    except ValueError as exc:
        raise ExperimentError(str(exc)) from exc


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(spec: ModelSpec, *, hash_file: bool = True) -> Path:
    path = spec.path
    if not path.is_file():
        raise ExperimentError(
            f"{spec.role} model is missing: {path}\n"
            "Prepare it with `uv run download-models`, then retry."
        )
    actual_size = path.stat().st_size
    if actual_size != spec.bytes:
        raise ExperimentError(
            f"{spec.role} model size mismatch at {path}: got {actual_size}; "
            f"expected {spec.bytes}. Remove the bad file and run `uv run download-models`."
        )
    if hash_file:
        actual_sha = sha256_file(path)
        if actual_sha != spec.sha256:
            raise ExperimentError(
                f"{spec.role} model checksum mismatch at {path}: got {actual_sha}; "
                f"expected {spec.sha256}. Remove the bad file and run `uv run download-models`."
            )
    return path.resolve()


def require_models(roles: Iterable[str] = ("main", "draft")) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for role in roles:
        print(f"setup: validating {role} model", flush=True)
        paths[role] = validate_model(MODEL_SPECS[role])
    return paths


def llama_server_binary() -> tuple[str, dict[str, Any]]:
    binary = os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
    if binary is None:
        raise ExperimentError(
            "llama-server is not installed or on PATH. Install llama.cpp or set "
            "LLAMA_SERVER=/absolute/path/to/llama-server."
        )
    result = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=False
    )
    output = (result.stdout + result.stderr).strip()
    match = re.search(r"version:\s*(\d+)", output)
    if result.returncode != 0 or match is None:
        raise ExperimentError(
            f"could not determine llama.cpp build from `{binary} --version`"
        )
    build = int(match.group(1))
    if build < MINIMUM_LLAMA_BUILD:
        raise ExperimentError(
            f"llama.cpp build {build} is too old; build {MINIMUM_LLAMA_BUILD} or newer is required"
        )
    return binary, {"build": build, "raw": output}


def server_arguments(
    binary: str,
    model_path: Path,
    port: int,
    *,
    reasoning_budget: int | None = None,
) -> list[str]:
    arguments = [
        binary,
        "--model",
        str(model_path),
        "--host",
        HOST,
        "--port",
        str(port),
        "--jinja",
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
        "--ctx-size",
        str(CONTEXT_SIZE),
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--parallel",
        "1",
        "--no-context-shift",
        "--no-cache-prompt",
        "--metrics",
        "--slots",
        "--n-gpu-layers",
        "all",
        "--seed",
        str(SEED),
    ]
    if reasoning_budget is not None:
        arguments.extend(
            [
                "--reasoning-budget",
                str(reasoning_budget),
                "--reasoning-budget-message",
                "Proceed with the single next action now.",
            ]
        )
    return arguments


class ModelServer(AbstractContextManager["ModelServer"]):
    def __init__(
        self,
        role: str,
        binary: str,
        model_path: Path,
        port: int,
        *,
        startup_timeout_seconds: float = SERVER_STARTUP_TIMEOUT_SECONDS,
        reasoning_budget: int | None = None,
    ) -> None:
        self.role = role
        self.base_url = f"http://{HOST}:{port}"
        self.arguments = server_arguments(
            binary,
            model_path,
            port,
            reasoning_budget=reasoning_budget,
        )
        self.startup_timeout_seconds = startup_timeout_seconds
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path: Path | None = None
        self.log_handle: Any = None

    def __enter__(self) -> ModelServer:
        print(
            f"setup: starting {self.role} model server at {self.base_url}", flush=True
        )
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=0.5)
        except httpx.HTTPError:
            pass
        else:
            raise ExperimentError(
                f"port for {self.role} model is already serving at {self.base_url} "
                f"(HTTP {response.status_code}); stop that service and retry"
            )
        descriptor, name = tempfile.mkstemp(
            prefix=f"experiment-{self.role}-", suffix=".log"
        )
        self.log_path = Path(name)
        self.log_handle = os.fdopen(descriptor, "ab", buffering=0)
        try:
            self.process = subprocess.Popen(
                self.arguments,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            self.stop()
            raise ExperimentError(
                f"could not start {self.role} llama-server: {exc}"
            ) from exc
        deadline = time.monotonic() + self.startup_timeout_seconds
        detail = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                detail = f"llama-server exited with code {self.process.returncode}"
                break
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    print(f"setup: {self.role} model server ready", flush=True)
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        tail = self._log_tail()
        self.stop()
        suffix = f"\nserver log tail:\n{tail}" if tail else ""
        raise ExperimentError(
            (
                detail
                or f"{self.role} llama-server did not become healthy before timeout"
            )
            + suffix
        )

    def _log_tail(self, lines: int = 20) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return ""
        try:
            return "\n".join(
                self.log_path.read_text(errors="replace").splitlines()[-lines:]
            )
        except OSError:
            return ""

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        if self.log_path is not None:
            self.log_path.unlink(missing_ok=True)
            self.log_path = None

    def __exit__(self, *args: object) -> None:
        self.stop()


@dataclass(slots=True)
class WikipediaEnvironment(AbstractContextManager["WikipediaEnvironment"]):
    paths: BundlePaths
    manifest: dict[str, Any]
    database: QdrantVectorDB
    points_count: int
    incomplete: bool

    def __exit__(self, *args: object) -> None:
        self.database.close()


def _local_ingestion_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return []
    current = os.getpid()
    active: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == current:
            continue
        command = parts[1]
        if "wikipedia-ingest" in command or "wikipedia.cli" in command:
            active.append(stripped)
    return active


def _collection_points(database: QdrantVectorDB) -> int:
    info = database.client.get_collection(database.config.collection_name)
    count = getattr(info, "points_count", None)
    if not isinstance(count, int) or count < 1:
        raise ExperimentError(
            f"Qdrant collection is empty: {database.config.collection_name}"
        )
    return count


def prepare_wikipedia(
    *,
    require_idle: bool = True,
    stability_seconds: float = INGEST_STABILITY_SECONDS,
) -> WikipediaEnvironment:
    try:
        paths = BundlePaths.resolve()
        manifest = load_manifest(paths, require_complete=False)
    except (BundleError, OSError) as exc:
        raise ExperimentError(f"Wikipedia bundle is unavailable: {exc}") from exc
    if not paths.bundle_dir.is_dir():
        raise ExperimentError(
            f"Wikipedia bundle volume is not mounted: {paths.bundle_dir}. Mount it and retry."
        )
    if not model_is_downloaded(paths):
        raise ExperimentError(
            f"BGE-M3 model is incomplete under {paths.model_dir}. Resume preparation with "
            "`uv run wikipedia-ingest qdrant`."
        )
    saved = manifest.get("qdrant")
    if not isinstance(saved, dict):
        raise ExperimentError(
            "Wikipedia manifest has no Qdrant configuration. Prepare it with "
            "`uv run wikipedia-ingest qdrant`."
        )
    url = str(saved.get("url") or DEFAULT_QDRANT_URL)
    try:
        runtime = ensure_qdrant(
            url,
            storage_dir=saved.get("storage_dir"),
            container=str(saved.get("container", "wikipedia-qdrant")),
            image=str(saved.get("image", "qdrant/qdrant:latest")),
        )
    except (RuntimeError, OSError, ValueError) as exc:
        raise ExperimentError(f"Qdrant preparation failed: {exc}") from exc
    print(f"setup: Qdrant {runtime} at {url}", flush=True)
    config = QdrantConfig(
        url=url,
        api_key=os.environ.get("QDRANT_API_KEY"),
        collection_name=str(saved.get("collection", "")),
        float16=bool(saved.get("float16", False)),
        prefer_grpc=bool(saved.get("prefer_grpc", True)),
        grpc_port=int(saved.get("grpc_port", 6334)),
        timeout=float(saved.get("timeout", 60)),
    )
    database = QdrantVectorDB(config)
    try:
        if not database.client.collection_exists(config.collection_name):
            raise ExperimentError(
                f"Qdrant collection is absent: {config.collection_name}"
            )
        first_count = _collection_points(database)
        active = _local_ingestion_processes() if require_idle else []
        if require_idle and not active and stability_seconds > 0:
            time.sleep(stability_seconds)
            second_count = _collection_points(database)
            if second_count != first_count:
                active = [
                    f"point count changed from {first_count:,} to {second_count:,}"
                ]
            first_count = second_count
        if active:
            raise ExperimentError(
                "Wikipedia ingestion appears active; pause it before measuring. "
                f"Detected: {active[0]}"
            )
    except BaseException:
        database.close()
        raise
    incomplete = manifest.get("status") != "complete"
    if incomplete:
        completed = "unknown"
        checkpoint_path = saved.get("checkpoint")
        if checkpoint_path and Path(str(checkpoint_path)).is_file():
            try:
                checkpoint = json.loads(
                    Path(str(checkpoint_path)).read_text(encoding="utf-8")
                )
                completed = str(len(checkpoint.get("completed_shards", [])))
            except (OSError, ValueError, TypeError):
                pass
        print(
            "warning: Wikipedia bundle is incomplete; running against "
            f"{first_count:,} currently ingested points ({completed} completed shards)",
            file=sys.stderr,
            flush=True,
        )
    return WikipediaEnvironment(paths, manifest, database, first_count, incomplete)


def run_main(action: Callable[[], None]) -> None:
    try:
        action()
    except KeyboardInterrupt:
        raise SystemExit(
            "experiment interrupted; temporary resources were cleaned up"
        ) from None
    except ExperimentError as exc:
        raise SystemExit(f"experiment error: {exc}") from None
    except Exception as exc:
        raise SystemExit(f"experiment error: {type(exc).__name__}: {exc}") from None
