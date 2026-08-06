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


def test_compose_file_mmaps_field_data_instead_of_loading_it_into_memory() -> None:
    """Field data must come off the storage medium, not out of RAM.

    The image disables mmap for every field type, so loading the 10M collection
    committed 22.7 GB of field data to memory and Milvus refused the remaining
    segments with "OOM if load" at 27.5 GB of the 28 GB VM — it logged
    mmapFieldCount=0. Leaving field data resident would also erase the storage
    treatment this experiment measures.
    """
    rendered = milvus_runtime.compose_file()

    assert 'QUERYNODE_MMAP_SCALARFIELD: "true"' in rendered
    assert 'QUERYNODE_MMAP_SCALARINDEX: "true"' in rendered
    assert 'QUERYNODE_MMAP_VECTORFIELD: "true"' in rendered
    assert 'QUERYNODE_MMAP_VECTORINDEX: "true"' in rendered


def test_compose_file_pins_the_diskann_search_cache_as_a_controlled_parameter() -> None:
    """The node cache decides how much of a search reaches the disk under test.

    It is pinned rather than left implicit because any comparison arm has to use
    the same value; a baseline with a different cache is measuring a different
    system. It sits at the Milvus default: an aggressively small cache forces
    reads a real deployment would not make, which is the same objection that
    ruled out Qdrant-HNSW here, and in practice it kept the USB HDD saturated
    until Docker Desktop's file-sharing watchdog killed the engine.
    """
    rendered = milvus_runtime.compose_file()

    assert (
        f'COMMON_DISKINDEX_SEARCHCACHEBUDGETGBRATIO: "{milvus_runtime.DEFAULT_SEARCH_CACHE_RATIO}"'
        in rendered
    )
    assert milvus_runtime.DEFAULT_SEARCH_CACHE_RATIO == 0.10


def test_compose_file_probes_minio_with_a_binary_the_image_ships() -> None:
    """The minio image has no curl, so a curl probe never passes.

    It reports unhealthy for the whole run — days, for a large ingest — which
    hides a genuine failure behind a permanently red container.
    """
    minio_service = milvus_runtime.compose_file().split("  minio:")[1].split("\n  standalone:")[0]
    probe = next(
        line for line in minio_service.splitlines() if line.strip().startswith("test:")
    )

    assert "curl" not in probe
    assert '"mc", "ready", "local"' in probe


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


def test_compose_file_caps_how_many_segments_load_at_once() -> None:
    rendered = milvus_runtime.compose_file()

    assert (
        f'QUERYCOORD_TASKEXECUTIONCAP: "{milvus_runtime.DEFAULT_TASK_EXECUTION_CAP}"'
        in rendered
    )


def test_compose_file_gives_the_load_longer_than_the_600_second_default() -> None:
    rendered = milvus_runtime.compose_file()

    assert (
        f'QUERYCOORD_LOADTIMEOUTSECONDS: "{milvus_runtime.DEFAULT_LOAD_TIMEOUT_SECONDS}"'
        in rendered
    )
    assert milvus_runtime.DEFAULT_LOAD_TIMEOUT_SECONDS > 600


def test_compose_file_keeps_etcd_on_the_bundle_volume_by_default() -> None:
    assert "- ./volumes/etcd:/etcd" in milvus_runtime.compose_file()


def test_compose_file_can_put_etcd_on_a_different_disk() -> None:
    rendered = milvus_runtime.compose_file(etcd_dir="/ssd/etcd")

    assert "- /ssd/etcd:/etcd" in rendered
    assert "- ./volumes/milvus:/var/lib/milvus" in rendered, "index must stay put"


def test_compose_file_can_put_etcd_on_a_named_volume() -> None:
    """A named volume takes etcd off host file sharing entirely.

    Both file-sharing implementations starve etcd under a sustained index
    write: virtiofs blocks its service, grpcfuse serialises every share
    through one host fileserver. Either way etcd misses its lease renewal and
    every Milvus role logs "connection lost detected, shuting down" and exits.
    A named volume lives in the VM's own disk image, so etcd's fsync never
    crosses the host boundary the index write is saturating.
    """
    rendered = milvus_runtime.compose_file(etcd_dir="volume:wikipedia-etcd")

    assert "- wikipedia-etcd:/etcd" in rendered
    # external, or Compose prefixes the project name and silently creates an
    # empty volume beside the populated one -- which reads as "the collection
    # is gone" and, left running, invites the GC to treat every binlog in
    # object storage as an orphan.
    assert "\nvolumes:\n  wikipedia-etcd:\n    external: true\n" in rendered
    assert "- ./volumes/milvus:/var/lib/milvus" in rendered, "index must stay put"


def test_compose_file_declares_no_volumes_section_for_a_bind_mount() -> None:
    assert "\nvolumes:\n" not in milvus_runtime.compose_file(etcd_dir="/ssd/etcd")


def test_ensure_milvus_does_not_make_a_directory_for_a_named_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "bundle"
    storage.mkdir()
    monkeypatch.setattr(milvus_runtime, "ensure_compatible_storage", lambda *a, **k: None)
    monkeypatch.setattr(milvus_runtime, "ensure_docker", lambda *a, **k: None)
    monkeypatch.setattr(milvus_runtime, "mounted_storage", lambda *a, **k: None)
    monkeypatch.setattr(milvus_runtime, "healthz_ok", lambda *a, **k: True)
    monkeypatch.setattr(
        milvus_runtime,
        "docker",
        lambda *args: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    milvus_runtime.ensure_milvus(
        "http://localhost:19530",
        storage_dir=storage,
        etcd_dir="volume:wikipedia-etcd",
    )

    written = (storage / milvus_runtime.COMPOSE_FILENAME).read_text(encoding="utf-8")
    assert "- wikipedia-etcd:/etcd" in written
    assert not (storage / "volume:wikipedia-etcd").exists(), (
        "a volume name is not a path and must not be created as one"
    )


def test_ensure_milvus_renders_the_etcd_directory_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "bundle"
    storage.mkdir()
    monkeypatch.setattr(milvus_runtime, "ensure_compatible_storage", lambda *a, **k: None)
    monkeypatch.setattr(milvus_runtime, "ensure_docker", lambda *a, **k: None)
    monkeypatch.setattr(milvus_runtime, "mounted_storage", lambda *a, **k: None)
    monkeypatch.setattr(milvus_runtime, "healthz_ok", lambda *a, **k: True)
    monkeypatch.setattr(
        milvus_runtime,
        "docker",
        lambda *args: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    elsewhere = tmp_path / "fast-disk" / "etcd"

    milvus_runtime.ensure_milvus(
        "http://localhost:19530", storage_dir=storage, etcd_dir=elsewhere
    )

    written = (storage / milvus_runtime.COMPOSE_FILENAME).read_text(encoding="utf-8")
    assert f"- {elsewhere}:/etcd" in written
    assert elsewhere.is_dir(), "the directory must exist before compose binds it"


def test_compose_file_gives_a_single_segment_load_longer_than_two_minutes() -> None:
    rendered = milvus_runtime.compose_file()

    assert (
        f'QUERYCOORD_SEGMENTTASKTIMEOUT: "{milvus_runtime.DEFAULT_SEGMENT_TASK_TIMEOUT}"'
        in rendered
    )
    assert milvus_runtime.DEFAULT_SEGMENT_TASK_TIMEOUT > 120_000


def test_compose_file_gives_a_channel_subscription_longer_than_one_minute() -> None:
    rendered = milvus_runtime.compose_file()

    assert (
        f'QUERYCOORD_CHANNELTASKTIMEOUT: "{milvus_runtime.DEFAULT_CHANNEL_TASK_TIMEOUT}"'
        in rendered
    )
    assert milvus_runtime.DEFAULT_CHANNEL_TASK_TIMEOUT > 60_000
