from pathlib import Path
from types import SimpleNamespace

import pytest

from wikipedia import docker_runtime, milvus_runtime


def _ok(stdout: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_compose_file_enables_disk_and_keeps_volumes_relative() -> None:
    rendered = milvus_runtime.compose_file()

    assert 'QUERYNODE_ENABLEDISK: "true"' in rendered
    assert "./volumes/milvus:/var/lib/milvus" in rendered
    assert "./volumes/etcd:/etcd" in rendered
    assert "./volumes/minio:/minio_data" in rendered


def test_health_url_uses_health_port_not_grpc_port() -> None:
    assert (
        milvus_runtime.health_url("http://localhost:19530")
        == "http://localhost:9091/healthz"
    )


def test_ensure_milvus_rejects_non_posix_storage_before_starting_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docker_runtime, "filesystem_type", lambda path: "exfat")
    monkeypatch.setattr(
        docker_runtime,
        "ensure_docker",
        lambda engine: pytest.fail("Docker should not start for unsupported storage"),
    )

    with pytest.raises(RuntimeError, match="POSIX-compatible filesystem"):
        milvus_runtime.ensure_milvus(
            milvus_runtime.DEFAULT_MILVUS_URI, storage_dir=tmp_path / "milvus"
        )


def test_ensure_milvus_reuses_stack_already_bound_to_the_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "milvus"
    expected = storage / "volumes" / "milvus"
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(docker_runtime, "filesystem_type", lambda path: "apfs")
    monkeypatch.setattr(milvus_runtime, "ensure_docker", lambda engine: None)
    monkeypatch.setattr(milvus_runtime, "healthz_ok", lambda url, timeout=2.0: True)
    monkeypatch.setattr(milvus_runtime, "mounted_storage", lambda *args, **kwargs: expected.resolve())
    monkeypatch.setattr(
        milvus_runtime,
        "docker",
        lambda *args: calls.append(args) or _ok(),
    )

    # Pre-render the compose file so ensure_milvus sees unchanged content.
    storage.mkdir(parents=True, exist_ok=True)
    (storage / milvus_runtime.COMPOSE_FILENAME).write_text(
        milvus_runtime.compose_file(), encoding="utf-8"
    )

    result = milvus_runtime.ensure_milvus(
        milvus_runtime.DEFAULT_MILVUS_URI, storage_dir=storage
    )

    assert result == "already running"
    assert not any("up" in call for call in calls)


def test_ensure_milvus_replaces_stack_pointing_at_another_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "milvus"
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(docker_runtime, "filesystem_type", lambda path: "apfs")
    monkeypatch.setattr(milvus_runtime, "ensure_docker", lambda engine: None)
    monkeypatch.setattr(milvus_runtime, "healthz_ok", lambda url, timeout=2.0: True)
    monkeypatch.setattr(
        milvus_runtime, "mounted_storage", lambda *args, **kwargs: tmp_path / "elsewhere"
    )
    monkeypatch.setattr(
        milvus_runtime,
        "docker",
        lambda *args: calls.append(args) or _ok(),
    )

    result = milvus_runtime.ensure_milvus(
        milvus_runtime.DEFAULT_MILVUS_URI, storage_dir=storage
    )

    assert result == "started"
    flattened = [" ".join(call) for call in calls]
    assert any("down" in call for call in flattened)
    assert any("up -d" in call for call in flattened)


def test_ensure_milvus_requires_absolute_storage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        milvus_runtime.ensure_milvus(
            milvus_runtime.DEFAULT_MILVUS_URI, storage_dir=Path("relative/milvus")
        )
