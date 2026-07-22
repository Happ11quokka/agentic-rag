from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import uuid
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import hf_hub_download

from agent.retrieval import TimedRetriever
from agent.runner import AgentRunner, LlamaCppClient, initial_messages, percentile, prompt_hash
from fanoutqa.dataset import find_question, load_dev, select_questions
from wikipedia.bundle import BundleError, BundlePaths, atomic_json, load_manifest
from wikipedia.encoder import Encoder
from wikipedia.qdrant import QdrantConfig, QdrantVectorDB
from wikipedia.qdrant_runtime import DEFAULT_QDRANT_URL, ensure_qdrant

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("parallel.toml")
SCHEMA_VERSION = 1
FIXED_RETRIEVAL_WARMUP = "English Wikipedia"
FIXED_LLM_WARMUP = "Reply with only: warm"


class SetupError(RuntimeError):
    pass


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SetupError(f"Unsupported experiment config schema in {path}")
    return config


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(path: Path, model: dict[str, Any]) -> None:
    if not path.is_file():
        raise SetupError(f"Model file is missing after download: {path}")
    actual_size = path.stat().st_size
    if actual_size != model["bytes"]:
        raise SetupError(
            f"Model size mismatch for {path.name}: got {actual_size}; expected {model['bytes']}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != model["sha256"]:
        raise SetupError(
            f"Model SHA-256 mismatch for {path.name}: got {actual_sha}; "
            f"expected {model['sha256']}"
        )


def resolve_model(
    role: str,
    model: dict[str, Any],
    models_dir: Path,
    *,
    downloader: Callable[..., str] = hf_hub_download,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[Path, dict[str, Any]]:
    role_dir = models_dir / role
    role_dir.mkdir(parents=True, exist_ok=True)
    started = clock_ns()
    downloaded = downloader(
        repo_id=model["repo"],
        filename=model["file"],
        revision=model["revision"],
        local_dir=role_dir,
    )
    path = Path(downloaded).resolve()
    validate_model(path, model)
    ended = clock_ns()
    return path, {
        "start_ns": started,
        "end_ns": ended,
        "duration_ms": (ended - started) / 1_000_000,
        "path": str(path),
    }


def llama_version(binary: str, minimum_build: int) -> dict[str, Any]:
    result = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=False
    )
    output = (result.stdout + result.stderr).strip()
    match = re.search(r"version:\s*(\d+)", output)
    if result.returncode != 0 or match is None:
        raise SetupError(f"Could not determine llama.cpp build from `{binary} --version`")
    build = int(match.group(1))
    if build < minimum_build:
        raise SetupError(
            f"llama.cpp build {build} is too old; build b{minimum_build} or newer is required"
        )
    return {"build": build, "raw": output}


def server_arguments(
    binary: str,
    model_path: Path,
    config: dict[str, Any],
) -> list[str]:
    runtime = config["runtime"]
    generation = config["generation"]
    return [
        binary,
        "--model",
        str(model_path),
        "--host",
        runtime["host"],
        "--port",
        str(runtime["port"]),
        "--jinja",
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
        "--ctx-size",
        str(generation["context_size"]),
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
        str(generation["seed"]),
    ]


def process_rss_bytes(pid: int) -> int | None:
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    try:
        return int(result.stdout.strip()) * 1024
    except ValueError:
        return None


class ModelServer(AbstractContextManager["ModelServer"]):
    def __init__(
        self,
        arguments: list[str],
        log_path: Path,
        base_url: str,
        *,
        startup_timeout_seconds: float,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.arguments = arguments
        self.log_path = log_path
        self.base_url = base_url
        self.startup_timeout_seconds = startup_timeout_seconds
        self.clock_ns = clock_ns
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None
        self.startup: dict[str, Any] = {}

    def __enter__(self) -> ModelServer:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=0.5)
        except httpx.HTTPError:
            pass
        else:
            raise SetupError(
                f"Refusing to start llama-server because {self.base_url} is already "
                f"responding (HTTP {response.status_code})"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("ab", buffering=0)
        started = self.clock_ns()
        self.process = subprocess.Popen(
            self.arguments,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        error: str | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                error = f"llama-server exited with code {self.process.returncode}"
                break
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    ended = self.clock_ns()
                    self.startup = {
                        "start_ns": started,
                        "end_ns": ended,
                        "duration_ms": (ended - started) / 1_000_000,
                        "rss_bytes": process_rss_bytes(self.process.pid),
                    }
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.stop()
        raise SetupError(error or "llama-server did not become healthy before timeout")

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

    def __exit__(self, *args: object) -> None:
        self.stop()


def prepare_wikipedia() -> tuple[BundlePaths, dict[str, Any], QdrantVectorDB]:
    try:
        paths = BundlePaths.resolve()
        manifest = load_manifest(paths, require_complete=True)
    except BundleError as exc:
        detail = str(exc)
        if "/Volumes/Volume" in detail and not Path("/Volumes/Volume").is_dir():
            raise SetupError("Wikipedia bundle volume is unmounted: /Volumes/Volume") from exc
        raise SetupError(str(exc)) from exc
    if str(paths.bundle_dir).startswith("/Volumes/Volume") and not Path(
        "/Volumes/Volume"
    ).is_dir():
        raise SetupError("Wikipedia bundle volume is unmounted: /Volumes/Volume")
    if not paths.model_dir.is_dir():
        raise SetupError(f"BGE-M3 model directory is missing: {paths.model_dir}")
    saved = manifest.get("qdrant")
    if not isinstance(saved, dict):
        raise SetupError("Wikipedia manifest has no Qdrant configuration")
    url = str(saved.get("url") or DEFAULT_QDRANT_URL)
    storage = saved.get("storage_dir")
    ensure_qdrant(
        url,
        storage_dir=storage,
        container=str(saved.get("container", "wikipedia-qdrant")),
        image=str(saved.get("image", "qdrant/qdrant:latest")),
    )
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
    if not database.client.collection_exists(config.collection_name):
        database.close()
        raise SetupError(f"Qdrant collection is absent: {config.collection_name}")
    info = database.client.get_collection(config.collection_name)
    count = getattr(info, "points_count", None)
    if not isinstance(count, int) or count < 1:
        database.close()
        raise SetupError(f"Qdrant collection is empty: {config.collection_name}")
    return paths, manifest, database


def generation_parameters(config: dict[str, Any]) -> dict[str, Any]:
    source = config["generation"]
    return {
        "max_tokens": source["max_tokens_per_turn"],
        "temperature": source["temperature"],
        "top_p": source["top_p"],
        "top_k": source["top_k"],
        "min_p": source["min_p"],
        "presence_penalty": source["presence_penalty"],
        "seed": source["seed"],
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_trace_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty[-1] if nonempty else -1
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == last_nonempty:
                break
            raise SetupError(f"Malformed non-trailing JSONL row {index + 1} in {path}")
        if not isinstance(value, dict):
            raise SetupError(f"Non-object JSONL row {index + 1} in {path}")
        rows.append(value)
    return rows


def repair_trailing_jsonl(path: Path) -> bool:
    """Discard only an unrecoverable partial final row left by an interrupted write."""
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty:
        return False
    last_nonempty = nonempty[-1]
    try:
        json.loads(lines[last_nonempty])
        return False
    except json.JSONDecodeError:
        pass
    for index in nonempty[:-1]:
        try:
            json.loads(lines[index])
        except json.JSONDecodeError as exc:
            raise SetupError(f"Malformed non-trailing JSONL row {index + 1} in {path}") from exc
    prefix = "".join(lines[:last_nonempty])
    with path.open("w", encoding="utf-8") as handle:
        handle.write(prefix)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def completed_keys(path: Path) -> set[tuple[str, str]]:
    return {
        (str(row.get("model_role")), str(row.get("question", {}).get("id")))
        for row in read_trace_rows(path)
    }


def run_phases(
    roles: Iterable[str],
    server_factory: Callable[[str], AbstractContextManager[Any]],
    run_role: Callable[[str, Any], None],
) -> None:
    """Testable sequential phase boundary: prior server exits before next starts."""
    for role in roles:
        with server_factory(role) as server:
            run_role(role, server)


def build_timing_summary(rows: list[dict[str, Any]], phases: dict[str, Any]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ("main", "draft"):
        selected = [row for row in rows if row.get("model_role") == role]
        timing_rows = [row["outcome"]["timing"] for row in selected]
        values = [timing["end_to_end_ms"] for timing in timing_rows]
        aggregate_fields = (
            "end_to_end_ms",
            "total_llm_wall_ms",
            "total_prefill_ms",
            "total_decode_ms",
            "total_retrieval_encode_ms",
            "total_qdrant_search_ms",
            "agent_overhead_ms",
        )
        roles[role] = {
            "questions": [
                {
                    "question_id": row["question"]["id"],
                    "terminal_status": row["outcome"]["terminal_status"],
                    **row["outcome"]["timing"],
                }
                for row in selected
            ],
            "phase": phases.get(role, {}),
            "question_count": len(selected),
            "total_end_to_end_ms": sum(values),
            "median_end_to_end_ms": percentile(values, 0.5),
            "p95_end_to_end_ms": percentile(values, 0.95),
            "timing_aggregates": {
                field: {
                    "total": sum(float(item[field]) for item in timing_rows),
                    "median": percentile(
                        [float(item[field]) for item in timing_rows], 0.5
                    ),
                    "p95": percentile(
                        [float(item[field]) for item in timing_rows], 0.95
                    ),
                }
                for field in aggregate_fields
            },
        }
    return {"schema_version": SCHEMA_VERSION, "roles": roles}


def _new_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "experiment" / "results" / f"{stamp}-{uuid.uuid4().hex[:8]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent local Qwen FanOutQA phases")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id")
    parser.add_argument("--role", choices=("main", "draft"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        _main(args)
    except (SetupError, OSError, ValueError) as exc:
        raise SystemExit(f"setup error: {exc}") from None


def _main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.resume and args.run_dir is None:
        raise SetupError("--resume requires --run-dir")
    run_dir = (args.run_dir or _new_run_dir()).resolve()
    manifest_path = run_dir / "manifest.json"
    traces_path = run_dir / "traces.jsonl"
    summary_path = run_dir / "timing_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    questions = load_dev()
    if args.resume:
        if not manifest_path.is_file():
            raise SetupError(f"Cannot resume without manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != config:
            raise SetupError(
                "Current experiment config differs from the resumed run manifest"
            )
        selected = [find_question(questions, item) for item in manifest["selected_question_ids"]]
        run_id = manifest["run_id"]
        repair_trailing_jsonl(traces_path)
    else:
        if manifest_path.exists() or traces_path.exists():
            raise SetupError(f"Run directory already contains results: {run_dir}")
        if args.question_id:
            selected = [find_question(questions, args.question_id)]
        else:
            selected = select_questions(
                questions,
                limit=config["default_limit"] if args.limit is None else args.limit,
                seed=config["seed"],
            )
        run_id = run_dir.name
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "config": config,
            "selected_question_ids": [item.id for item in selected],
            "prompt_hashes": {item.id: prompt_hash(initial_messages(item)) for item in selected},
            "phases": {},
        }
        atomic_json(manifest_path, manifest)

    roles = [args.role] if args.role else ["main", "draft"]
    done = completed_keys(traces_path)
    roles = [
        role
        for role in roles
        if any((role, question.id) not in done for question in selected)
    ]
    if not roles:
        atomic_json(
            summary_path,
            build_timing_summary(read_trace_rows(traces_path), manifest.get("phases", {})),
        )
        print(run_dir)
        return
    binary = os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
    if binary is None:
        raise SetupError("llama-server is not installed or on PATH")
    version = llama_version(binary, config["runtime"]["minimum_llama_build"])
    wiki_paths, wiki_manifest, database = prepare_wikipedia()
    try:
        encoder_load_start = time.perf_counter_ns()
        try:
            encoder = Encoder(wiki_paths.bundle_dir)
        except Exception as exc:
            raise SetupError(f"BGE-M3 could not be loaded from {wiki_paths.model_dir}: {exc}") from exc
        encoder_load_end = time.perf_counter_ns()
        retriever = TimedRetriever(
            encoder,
            database,
            top_k=config["retrieval"]["top_k"],
            max_chars_per_result=config["retrieval"]["max_chars_per_result"],
        )
        retrieval_warmup = retriever.search(FIXED_RETRIEVAL_WARMUP)
        retrieval_warmup["excluded_from_measurement"] = True
        manifest["wikipedia"] = {
            "bundle_dir": str(wiki_paths.bundle_dir),
            "manifest": wiki_manifest,
            "collection": database.config.collection_name,
            "endpoint": database.config.endpoint,
            "encoder_load": {
                "start_ns": encoder_load_start,
                "end_ns": encoder_load_end,
                "duration_ms": (encoder_load_end - encoder_load_start) / 1_000_000,
            },
            "retrieval_warmup": retrieval_warmup,
        }
        manifest["llama_cpp"] = version
        atomic_json(manifest_path, manifest)

        generation = generation_parameters(config)
        model_paths: dict[str, Path] = {}
        for role in roles:
            model_paths[role], download = resolve_model(
                role, config["models"][role], ROOT / "experiment" / "models"
            )
            manifest["phases"].setdefault(role, {})["model_download"] = download
            atomic_json(manifest_path, manifest)

        base_url = f"http://{config['runtime']['host']}:{config['runtime']['port']}"

        def server_factory(role: str) -> ModelServer:
            arguments = server_arguments(binary, model_paths[role], config)
            manifest["phases"].setdefault(role, {})["launch_arguments"] = arguments
            atomic_json(manifest_path, manifest)
            return ModelServer(
                arguments,
                run_dir / "server" / f"{role}.log",
                base_url,
                startup_timeout_seconds=config["runtime"]["server_startup_timeout_seconds"],
            )

        def run_role(role: str, server: ModelServer) -> None:
            phase_start = time.perf_counter_ns()
            phase = manifest["phases"].setdefault(role, {})
            phase["server_startup"] = server.startup
            client = LlamaCppClient(
                base_url,
                timeout_seconds=config["agent"]["request_timeout_seconds"],
            )
            try:
                warm_start = time.perf_counter_ns()
                warm_call = client.stream_completion(
                    [{"role": "user", "content": FIXED_LLM_WARMUP}],
                    {**generation, "max_tokens": 32, "temperature": 0},
                )
                warm_end = time.perf_counter_ns()
                phase["llm_warmup"] = {
                    "start_ns": warm_start,
                    "end_ns": warm_end,
                    "duration_ms": (warm_end - warm_start) / 1_000_000,
                    "llm_call": warm_call,
                    "excluded_from_measurement": True,
                }
                phase["boundary_rss_bytes"] = (
                    process_rss_bytes(server.process.pid) if server.process is not None else None
                )
                atomic_json(manifest_path, manifest)
                runner = AgentRunner(
                    client,
                    retriever,
                    generation=generation,
                    max_searches=config["agent"]["max_searches"],
                )
                for question in selected:
                    if (role, question.id) in done:
                        continue
                    outcome = runner.run(question)
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": run_id,
                        "model_role": role,
                        "model": {**config["models"][role], "path": str(model_paths[role])},
                        "question": question.agent_value(),
                        "prompt_hash": prompt_hash(initial_messages(question)),
                        "generation_parameters": generation,
                        "wikipedia": {
                            "dataset_revision": wiki_manifest.get("dataset", {}).get(
                                "resolved_revision"
                            ),
                            "model_revision": wiki_manifest.get("model", {}).get(
                                "resolved_revision"
                            ),
                            "collection": database.config.collection_name,
                            "endpoint": database.config.endpoint,
                            "manifest_sha256": hashlib.sha256(
                                json.dumps(
                                    wiki_manifest,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest(),
                        },
                        "llama_cpp": {
                            **version,
                            "launch_arguments": phase["launch_arguments"],
                        },
                        "outcome": outcome,
                    }
                    append_jsonl(traces_path, row)
                    done.add((role, question.id))
            finally:
                client.close()
                phase_end = time.perf_counter_ns()
                phase["phase_start_ns"] = phase_start
                phase["phase_end_ns"] = phase_end
                phase["phase_duration_ms"] = (phase_end - phase_start) / 1_000_000
                atomic_json(manifest_path, manifest)

        run_phases(roles, server_factory, run_role)
    finally:
        database.close()

    atomic_json(summary_path, build_timing_summary(read_trace_rows(traces_path), manifest["phases"]))
    print(run_dir)


if __name__ == "__main__":
    main()
