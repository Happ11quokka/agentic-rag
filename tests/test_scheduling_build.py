from __future__ import annotations

from pathlib import Path

import pytest

from agent.scheduling import build


def _package(prefix: Path, number: int = 8360) -> Path:
    config = prefix / build.PACKAGE_CONFIG
    config.parent.mkdir(parents=True)
    config.write_text(
        f"set(LLAMA_BUILD_NUMBER {number})\nset(LLAMA_BUILD_COMMIT abc123)\n"
    )
    return prefix


def test_build_prefix_prefers_environment_override(tmp_path: Path) -> None:
    prefix = tmp_path.resolve()
    assert build.discover_llama_prefix({"LLAMA_CPP_PREFIX": str(prefix)}) == prefix


def test_build_prefix_falls_back_to_pkg_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build, "_prefix_from_pkg_config", lambda: tmp_path.resolve())
    assert build.discover_llama_prefix({}) == tmp_path.resolve()


def test_build_reports_missing_package_and_old_version(tmp_path: Path) -> None:
    with pytest.raises(build.NativeBuildError, match="missing llama CMake package"):
        build.read_llama_package(tmp_path)
    with pytest.raises(build.NativeBuildError, match="too old"):
        build.read_llama_package(_package(tmp_path, 8359))


def test_engine_override_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "engine"
    binary.write_text("placeholder")
    expected = build.NativeEngine(binary, {"build": 8360})
    monkeypatch.setattr(build, "verify_engine", lambda path: expected)
    assert build.resolve_native_engine(
        {"LLAMA_SCHEDULED_ENGINE": str(binary.resolve())}
    ) is expected


def test_engine_version_mismatch_is_rejected() -> None:
    with pytest.raises(build.NativeBuildError, match="protocol 2"):
        build.parse_engine_version(
            "parallel-scheduled-engine protocol=2 llama_build=8360 "
            "llama_commit=abc source_sha256=" + "a" * 64
        )


def test_resolve_reuses_cached_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = build.LlamaPackage(tmp_path, 8360, "abc")
    source_hash = "a" * 64
    monkeypatch.setattr(build, "BUILD_ROOT", tmp_path / "cache")
    monkeypatch.setattr(build, "discover_llama_prefix", lambda environment: tmp_path)
    monkeypatch.setattr(build, "read_llama_package", lambda prefix: package)
    monkeypatch.setattr(build, "native_source_hash", lambda: source_hash)
    directory = build.BUILD_ROOT / build.build_key(package, source_hash)
    directory.mkdir(parents=True)
    binary = directory / "parallel-scheduled-engine"
    binary.write_text("cached")
    expected = build.NativeEngine(binary, {"build": 8360})
    monkeypatch.setattr(build, "verify_engine", lambda path: expected)
    monkeypatch.setattr(
        build, "_run", lambda *args, **kwargs: pytest.fail("cache hit rebuilt engine")
    )
    assert build.resolve_native_engine({}) is expected
