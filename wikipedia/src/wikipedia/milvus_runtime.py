"""Local Milvus standalone stack (etcd + MinIO + Milvus) for DiskANN experiments.

Milvus standalone is three containers, so unlike ``qdrant_runtime`` this module
renders a Compose project instead of issuing ``docker run``.

DiskANN is gated behind ``queryNode.enableDisk``, which is ``false`` by default.
It is enabled here through the ``QUERYNODE_ENABLEDISK`` environment variable —
Milvus maps an environment name onto a config key by lowercasing it and dropping
separators, the same mechanism the upstream Compose file uses for
``ETCD_ENDPOINTS`` and ``MINIO_ADDRESS``. That avoids replacing the image's full
``milvus.yaml``, where every key omitted from a hand-written file would silently
fall back to a compiled-in default.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from .docker_runtime import docker, ensure_compatible_storage, ensure_docker

DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_MILVUS_PROJECT = "wikipedia-milvus"
DEFAULT_MILVUS_IMAGE = "milvusdb/milvus:v2.5.27"
DEFAULT_ETCD_IMAGE = "quay.io/coreos/etcd:v3.5.18"
DEFAULT_MINIO_IMAGE = "minio/minio:RELEASE.2024-05-28T17-19-04Z"
DEFAULT_HEALTH_PORT = 9091
DEFAULT_STARTUP_TIMEOUT = 300.0

# QueryCoord dispatches every sealed segment at once, and each admitted load
# reserves ~128 MiB against the memory guard until it *finishes*. Measured on
# the 10M collection: 176 loads were admitted in a 0.31 s burst, none completed
# in the next 601 s, and the reservations alone reached 23,641 MB of the
# 25,218 MB ceiling. The guard then refused the next 129 MB segment.
#
# The reservation is a fixed download-buffer floor, not real DiskANN memory --
# actual process memory throughout was ~2.5 GB. So the fix is to stop admitting
# 176 segments at once. On a 5400rpm disk a narrow window is also faster: 176
# concurrent readers turn a sequential copy into seek thrash.
DEFAULT_TASK_EXECUTION_CAP = 4

# The load that "failed with OOM" was in fact cancelled by this timeout: the
# QueryCoord observer gives a collection 600 s to make progress, measured from
# the last time a segment finished. On this HDD no segment finished in 601 s.
DEFAULT_LOAD_TIMEOUT_SECONDS = 36000

# QueryCoord cancels an individual load task after segmentTaskTimeout (default
# 120,000 ms) and a channel subscription after channelTaskTimeout (60,000 ms).
# Both are sized for an SSD. Here a single segment load has to pull its binlogs,
# statslogs and DiskANN index off a 5400rpm disk, and one was measured at
# 14m26s. When the deadline passes, the read dies with "context canceled", the
# segment is released, and the whole collection load unwinds with
# "release collection due to ref count to 0" -- a failure that reads like a
# storage error rather than a timeout.
DEFAULT_SEGMENT_TASK_TIMEOUT = 3_600_000
DEFAULT_CHANNEL_TASK_TIMEOUT = 600_000

# Headroom above the default 90 for the guard the reservations were tripping.
# On its own this buys ~2.5 GB and does not fix anything; it is here so a slow
# segment cannot squeeze the run out on the margin.
DEFAULT_OVERLOADED_MEMORY_PERCENT = 95

# DiskANN caches hot graph nodes in RAM, defaulting to 10% of the raw vector
# bytes -- 4.1 GB for 10M 1024-d float32 rows.
#
# This was 0.01 to force nearly every read to the disk under test. That is not
# a configuration anyone runs, and it is what this project rejected Qdrant-HNSW
# for: measuring a setup no real system uses. It also had a concrete cost --
# with almost nothing cached, sustained random reads kept the USB HDD queue
# saturated until Docker Desktop's file-sharing watchdog fired
# ("service fs failed: injecting event blocked for 60s") and took the engine
# down mid-experiment, six times.
#
# Back to the Milvus default. The medium under test is unchanged; what changes
# is that the index is allowed the cache a real deployment would give it.
# Latency measured at 0.01 (10M: cold 386 s, median 295 s) is a different
# configuration and is not comparable to numbers taken from here.
DEFAULT_SEARCH_CACHE_RATIO = 0.10

# No container memory limit is set, because none helps.
#
# Milvus refuses to load a collection when it predicts memory exhaustion, and in
# Docker Desktop it predicts against numbers that do not describe this container:
# the usage comes from the VM, where the page cache mmap fills counts as used.
# Measured at 10M with every field mmap'd — VM: 28,020 MB total, 17,505 cached,
# 19,899 *available*; the container's own cgroup held 4,274 MB, 2,928 of it anon.
# Milvus reported 20,762 MB used and refused to load with 20 GB genuinely free.
#
# Setting mem_limit does not move that reading. A limit below the VM's memory
# only lowers the ceiling Milvus checks against (22g made it refuse sooner), and
# a limit above it is ignored — Milvus takes min(cgroup, physical), so 48g read
# back as the VM's own 28,020 MB. Dropping the VM's page cache is what clears the
# refusal; the check passes and the load proceeds.

COMPOSE_FILENAME = "docker-compose.yml"


def compose_file(
    *,
    project: str = DEFAULT_MILVUS_PROJECT,
    milvus_image: str = DEFAULT_MILVUS_IMAGE,
    etcd_image: str = DEFAULT_ETCD_IMAGE,
    minio_image: str = DEFAULT_MINIO_IMAGE,
    health_port: int = DEFAULT_HEALTH_PORT,
    search_cache_ratio: float = DEFAULT_SEARCH_CACHE_RATIO,
    task_execution_cap: int = DEFAULT_TASK_EXECUTION_CAP,
    load_timeout_seconds: int = DEFAULT_LOAD_TIMEOUT_SECONDS,
    segment_task_timeout: int = DEFAULT_SEGMENT_TASK_TIMEOUT,
    channel_task_timeout: int = DEFAULT_CHANNEL_TASK_TIMEOUT,
    overloaded_memory_percent: int = DEFAULT_OVERLOADED_MEMORY_PERCENT,
    etcd_dir: str = "./volumes/etcd",
) -> str:
    """Render the Compose project.

    Volume paths are relative to the file, so the file must be written into the
    storage directory it describes -- except `etcd_dir`, which is separable on
    purpose (see below).
    """
    return f"""\
# Generated by wikipedia.milvus_runtime — edits are overwritten.
services:
  etcd:
    container_name: {project}-etcd
    image: {etcd_image}
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - {etcd_dir}:/etcd
    command: >
      etcd -advertise-client-urls=http://etcd:2379
      -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: {project}-minio
    image: {minio_image}
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ./volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      # The image ships mc but not curl, so a curl probe can never pass and the
      # container stays unhealthy for the whole run.
      test: ["CMD", "mc", "ready", "local"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: {project}
    image: {milvus_image}
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      QUERYNODE_ENABLEDISK: "true"
      # The image loads every field into RAM (mmapFieldCount=0). Measured on the
      # 10M collection: 22.7 GB of field data committed to memory and Milvus
      # refused the rest with "OOM if load" at 27.5 GB of the 28 GB VM. Resident
      # field data would also erase the storage treatment — the point is that
      # reads come off the medium under test.
      QUERYNODE_MMAP_SCALARFIELD: "true"
      QUERYNODE_MMAP_SCALARINDEX: "true"
      QUERYNODE_MMAP_VECTORFIELD: "true"
      QUERYNODE_MMAP_VECTORINDEX: "true"
      COMMON_DISKINDEX_SEARCHCACHEBUDGETGBRATIO: "{search_cache_ratio}"
      # Reservations from concurrently loading segments, not real memory, are
      # what trip the load guard. Capping concurrency caps the reservations.
      QUERYCOORD_TASKEXECUTIONCAP: "{task_execution_cap}"
      # A slow medium makes no segment-level progress inside the 600 s default,
      # and the observer cancels the whole load when progress stalls.
      QUERYCOORD_LOADTIMEOUTSECONDS: "{load_timeout_seconds}"
      # Per-task deadlines. The defaults (120 s / 60 s) are shorter than a
      # single segment load off this disk, so every task was cancelled mid-read.
      QUERYCOORD_SEGMENTTASKTIMEOUT: "{segment_task_timeout}"
      QUERYCOORD_CHANNELTASKTIMEOUT: "{channel_task_timeout}"
      QUERYCOORD_OVERLOADEDMEMORYTHRESHOLDPERCENTAGE: "{overloaded_memory_percent}"
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{health_port}/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "{health_port}:{health_port}"
    depends_on:
      - "etcd"
      - "minio"
"""


def health_url(uri: str, health_port: int = DEFAULT_HEALTH_PORT) -> str:
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    return f"http://{host}:{health_port}/healthz"


def healthz_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _write_if_changed(path: Path, content: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _compose(storage: Path, project: str, *args: str):
    return docker(
        "compose",
        "-f",
        str(storage / COMPOSE_FILENAME),
        "-p",
        project,
        *args,
    )


def mounted_storage(container: str, destination: str = "/var/lib/milvus") -> Path | None:
    """Where a running container actually keeps its data, or None if absent."""
    result = docker(
        "container",
        "inspect",
        "--format",
        '{{range .Mounts}}{{if eq .Destination "'
        + destination
        + '"}}{{.Source}}{{end}}{{end}}',
        container,
    )
    if result.returncode != 0:
        return None
    source = result.stdout.strip()
    return Path(source).resolve() if source else None


# Restarting the standalone container is expensive in a way nothing warns about:
# Milvus treats local storage as a cache it owns and empties it at startup
# ("Clean local data cache" in roles.go). A loaded 10M collection left 93 GB
# under volumes/milvus/data and had 6.2 GB left seconds after a restart, so the
# DiskANN index is re-fetched from MinIO -- which lives on the same spindle,
# making a restart cost ~93 GB read plus ~93 GB written, measured at ~5 hours.
#
# So batch anything that recreates the container, and prefer etcd for settings
# that are refreshable (queryCoord.taskExecutionCap is).
def ensure_milvus(
    uri: str,
    *,
    storage_dir: str | Path | None = None,
    project: str = DEFAULT_MILVUS_PROJECT,
    image: str = DEFAULT_MILVUS_IMAGE,
    health_port: int = DEFAULT_HEALTH_PORT,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    etcd_dir: str | Path | None = None,
) -> str:
    """Bring up the local stack and return ``"started"`` or ``"already running"``.

    ``etcd_dir`` overrides where etcd keeps its data. etcd fsyncs on every write,
    so on a saturated disk it loses its lease and every Milvus role logs
    "connection lost detected, shuting down" and exits 1 — observed once here
    mid-load, with ``OOMKilled=false``. Pointing etcd at a quiet disk removes
    that failure mode without moving the index, which must stay on the medium
    under test.
    """
    health = health_url(uri, health_port)
    local = urlparse(uri).hostname in {"localhost", "127.0.0.1", "::1"}
    if not local:
        if healthz_ok(health):
            return "already running"
        raise RuntimeError(f"Milvus is not reachable: {uri}")
    if storage_dir is None:
        raise RuntimeError(
            "Local Milvus storage is not configured; run "
            "`uv run wikipedia-ingest milvus --bundle-dir /absolute/path` first"
        )

    storage = Path(storage_dir).expanduser()
    if not storage.is_absolute():
        raise ValueError("Milvus storage directory must be an absolute path")
    storage = storage.resolve()
    ensure_compatible_storage(storage, engine="Milvus")
    for name in ("etcd", "minio", "milvus"):
        (storage / "volumes" / name).mkdir(parents=True, exist_ok=True)
    ensure_docker("Milvus")

    if docker("compose", "version").returncode != 0:
        raise RuntimeError(
            "`docker compose` is unavailable; Milvus standalone needs the Compose plugin"
        )

    if etcd_dir is None:
        etcd_mount = "./volumes/etcd"
    else:
        resolved = Path(etcd_dir).expanduser()
        if not resolved.is_absolute():
            raise ValueError("etcd directory must be an absolute path")
        resolved.mkdir(parents=True, exist_ok=True)
        etcd_mount = str(resolved)
    rendered = _write_if_changed(
        storage / COMPOSE_FILENAME,
        compose_file(
            project=project,
            milvus_image=image,
            health_port=health_port,
            etcd_dir=etcd_mount,
        ),
    )

    # A healthy stack on this port is not necessarily *this* stack: the Compose
    # project name is fixed, so an earlier run against a different bundle leaves
    # containers that answer healthz while serving another directory's data.
    expected = (storage / "volumes" / "milvus").resolve()
    actual = mounted_storage(project)
    if actual is not None and actual != expected:
        print(
            f"milvus: reconfiguring {project} storage {actual} -> {expected}",
            flush=True,
        )
        removed = _compose(storage, project, "down")
        if removed.returncode != 0:
            detail = removed.stderr.strip() or removed.stdout.strip()
            raise RuntimeError(f"Could not replace Milvus stack: {detail}")
    elif not rendered and actual is not None and healthz_ok(health):
        return "already running"

    print(f"milvus: starting {project} with storage {storage}", flush=True)
    started = _compose(storage, project, "up", "-d")
    if started.returncode != 0:
        detail = started.stderr.strip() or started.stdout.strip()
        raise RuntimeError(f"Could not start Milvus stack: {detail}")

    deadline = time.monotonic() + startup_timeout
    announced = 0.0
    while time.monotonic() < deadline:
        if healthz_ok(health):
            return "started"
        waited = startup_timeout - (deadline - time.monotonic())
        if waited - announced >= 30:
            announced = waited
            print(f"milvus: waiting for healthz ({waited:.0f} s)", flush=True)
        time.sleep(2)
    logs = _compose(storage, project, "logs", "--tail", "40", "standalone")
    raise RuntimeError(
        f"Milvus did not become healthy within {startup_timeout:.0f} seconds.\n"
        f"{logs.stdout.strip() or logs.stderr.strip()}"
    )


# Dropping the VM page cache does NOT help a load, and the reason is worth
# recording because the opposite is intuitive. Milvus's load guard compares
# `memUsage = hardware.GetUsedMemoryCount() + committedResource.MemorySize`
# against a fraction of total memory. The first term is `RSS - Shared` from
# /proc/<pid>/statm, which subtracts file-backed pages by construction; the
# second is an in-process counter that never consults the kernel. Measured: a
# drop that freed 6.5 GB of buff/cache moved the process's anonymous RSS by
# 0 kB. A cache dropper cannot move that number by design.
#
# What these two helpers are for is the opposite job — emptying the cache
# *before* a benchmark, so a cold measurement is actually cold.
CACHE_HELPER_IMAGE = "alpine"


def vm_memory_mb() -> tuple[int, int] | None:
    """(used, total) megabytes of the Docker VM, or None if unreadable."""
    result = docker(
        "run", "--rm", "--privileged", CACHE_HELPER_IMAGE, "sh", "-c", "free -m"
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0].rstrip(":").lower() == "mem":
            try:
                return int(fields[2]), int(fields[1])
            except (IndexError, ValueError):
                return None
    return None


def drop_page_cache() -> bool:
    """Ask the VM kernel to drop clean page cache. True if it worked."""
    result = docker(
        "run",
        "--rm",
        "--privileged",
        CACHE_HELPER_IMAGE,
        "sh",
        "-c",
        "sync; echo 3 > /proc/sys/vm/drop_caches",
    )
    return result.returncode == 0
