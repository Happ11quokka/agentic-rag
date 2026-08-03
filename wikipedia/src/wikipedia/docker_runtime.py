"""Docker helpers shared by the Qdrant and Milvus runtimes."""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

UNSUPPORTED_FILESYSTEMS = {
    "cifs",
    "exfat",
    "msdos",
    "nfs",
    "ntfs",
    "smbfs",
    "vfat",
}


def filesystem_type(path: Path) -> str | None:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    result = subprocess.run(
        ["mount"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    resolved = target.resolve()
    matches: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        match = re.match(r".+ on (.+?) \(([^, )]+)", line)
        if match is None:
            match = re.match(r".+ on (.+?) type ([^ ]+)", line)
        if match is None:
            continue
        mount_point = Path(match.group(1))
        if resolved == mount_point or mount_point in resolved.parents:
            matches.append((len(mount_point.parts), match.group(2).lower()))
    return max(matches)[1] if matches else None


def ensure_compatible_storage(storage: Path, *, engine: str) -> None:
    filesystem = filesystem_type(storage)
    if filesystem not in UNSUPPORTED_FILESYSTEMS:
        return
    raise RuntimeError(
        f"{engine} storage {storage} is on {filesystem}, but {engine} requires "
        "block-level access on a POSIX-compatible filesystem. Use APFS or ext4 "
        "storage instead of ExFAT, NTFS, or a network filesystem."
    )


def url_ready(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=1):
            return True
    except OSError:
        return False


def wait_until_ready(url: str, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if url_ready(url):
            return True
        time.sleep(1)
    return False


def docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_docker(engine: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError(f"Docker is required to start local {engine}")
    if docker("info").returncode == 0:
        return
    if sys.platform != "darwin":
        raise RuntimeError("Docker daemon is not running")
    print(f"{engine.lower()}: starting Docker Desktop", flush=True)
    subprocess.run(
        ["open", "-a", "Docker"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if docker("info").returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError("Docker Desktop did not become ready within 120 seconds")
