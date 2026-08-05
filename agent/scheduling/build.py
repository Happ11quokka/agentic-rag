from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import PROTOCOL_VERSION

MINIMUM_LLAMA_BUILD = 8360
PACKAGE_CONFIG = Path("lib/cmake/llama/llama-config.cmake")
NATIVE_DIR = Path(__file__).resolve().parent / "native"
BUILD_ROOT = Path(__file__).resolve().parent / ".build"


class NativeBuildError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LlamaPackage:
    prefix: Path
    build: int
    commit: str


@dataclass(frozen=True, slots=True)
class NativeEngine:
    path: Path
    version: dict[str, object]


def _run(arguments: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments, cwd=cwd, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise NativeBuildError(f"could not run {arguments[0]}: {exc}") from exc


def _prefix_from_pkg_config() -> Path | None:
    if shutil.which("pkg-config") is None:
        return None
    result = _run(["pkg-config", "--variable=prefix", "llama"])
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).expanduser().resolve()


def discover_llama_prefix(environment: dict[str, str] | None = None) -> Path:
    selected = environment if environment is not None else os.environ
    configured = selected.get("LLAMA_CPP_PREFIX")
    if configured:
        prefix = Path(configured).expanduser()
        if not prefix.is_absolute():
            raise NativeBuildError("LLAMA_CPP_PREFIX must be an absolute path")
        return prefix.resolve()
    prefix = _prefix_from_pkg_config()
    if prefix is None:
        raise NativeBuildError(
            "llama.cpp CMake package was not found. Set LLAMA_CPP_PREFIX to its "
            "install prefix (containing lib/cmake/llama/llama-config.cmake)."
        )
    return prefix


def read_llama_package(prefix: Path) -> LlamaPackage:
    config = prefix / PACKAGE_CONFIG
    try:
        text = config.read_text(encoding="utf-8")
    except OSError as exc:
        raise NativeBuildError(f"missing llama CMake package: {config}") from exc
    build_match = re.search(r"set\(LLAMA_BUILD_NUMBER\s+([^\s\)]+)\)", text)
    commit_match = re.search(r"set\(LLAMA_BUILD_COMMIT\s+([^\s\)]+)\)", text)
    if build_match is None or commit_match is None:
        raise NativeBuildError(f"could not read llama build metadata from {config}")
    try:
        build = int(build_match.group(1).strip('"'))
    except ValueError as exc:
        raise NativeBuildError(f"invalid llama build number in {config}") from exc
    commit = commit_match.group(1).strip('"')
    if build < MINIMUM_LLAMA_BUILD:
        raise NativeBuildError(
            f"llama.cpp build {build} is too old; build {MINIMUM_LLAMA_BUILD} or newer is required"
        )
    return LlamaPackage(prefix.resolve(), build, commit)


def native_source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(NATIVE_DIR.glob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_key(package: LlamaPackage, source_hash: str) -> str:
    raw = "|".join(
        (
            platform.system(),
            platform.machine(),
            str(package.build),
            package.commit,
            source_hash,
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def parse_engine_version(output: str) -> dict[str, object]:
    pattern = (
        r"parallel-scheduled-engine protocol=(\d+) llama_build=(\d+) "
        r"llama_commit=([^\s]+) source_sha256=([0-9a-f]{64})"
    )
    match = re.search(pattern, output)
    if match is None:
        raise NativeBuildError("native engine returned an invalid --version response")
    protocol, build, commit, source_hash = match.groups()
    if int(protocol) != PROTOCOL_VERSION:
        raise NativeBuildError(
            f"native engine protocol {protocol} does not match Python protocol {PROTOCOL_VERSION}"
        )
    if int(build) < MINIMUM_LLAMA_BUILD:
        raise NativeBuildError(
            f"native engine llama.cpp build {build} is too old; {MINIMUM_LLAMA_BUILD}+ required"
        )
    return {
        "protocol": int(protocol),
        "build": int(build),
        "commit": commit,
        "source_sha256": source_hash,
        "raw": output.strip(),
    }


def verify_engine(path: Path) -> NativeEngine:
    if not path.is_file():
        raise NativeBuildError(f"native engine does not exist: {path}")
    result = _run([str(path), "--version"])
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise NativeBuildError(f"native engine --version failed: {output}")
    return NativeEngine(path.resolve(), parse_engine_version(output))


@contextmanager
def _build_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def resolve_native_engine(environment: dict[str, str] | None = None) -> NativeEngine:
    selected = environment if environment is not None else os.environ
    override = selected.get("LLAMA_SCHEDULED_ENGINE")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise NativeBuildError("LLAMA_SCHEDULED_ENGINE must be an absolute path")
        return verify_engine(path.resolve())

    package = read_llama_package(discover_llama_prefix(selected))
    source_hash = native_source_hash()
    directory = BUILD_ROOT / build_key(package, source_hash)
    binary = directory / "parallel-scheduled-engine"
    with _build_lock(BUILD_ROOT / ".build.lock"):
        if binary.is_file():
            return verify_engine(binary)
        directory.mkdir(parents=True, exist_ok=True)
        configure = _run(
            [
                "cmake",
                "-S",
                str(NATIVE_DIR),
                "-B",
                str(directory),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_PREFIX_PATH={package.prefix}",
                f"-DPS_LLAMA_BUILD={package.build}",
                f"-DPS_LLAMA_COMMIT={package.commit}",
                f"-DPS_SOURCE_SHA256={source_hash}",
            ]
        )
        if configure.returncode != 0:
            raise NativeBuildError(
                "native engine CMake configure failed:\n"
                + (configure.stderr or configure.stdout).strip()
            )
        build = _run(["cmake", "--build", str(directory), "--config", "Release", "-j"])
        if build.returncode != 0:
            raise NativeBuildError(
                "native engine build failed:\n" + (build.stderr or build.stdout).strip()
            )
        return verify_engine(binary)
