import threading
from types import SimpleNamespace

import pytest

from wikipedia import milvus_runtime

FREE_OUTPUT = """\
              total        used        free      shared  buff/cache   available
Mem:          28020        3075       21686           7        3259       24643
Swap:          1024          55         969
"""


def _fake_docker(calls: list[tuple[str, ...]], *, stdout: str = "", returncode: int = 0):
    def run(*args: str) -> SimpleNamespace:
        calls.append(args)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    return run


def test_vm_memory_reads_used_and_total_from_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        milvus_runtime, "docker", _fake_docker([], stdout=FREE_OUTPUT)
    )

    assert milvus_runtime.vm_memory_mb() == (3075, 28020)


def test_vm_memory_returns_none_when_docker_cannot_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        milvus_runtime, "docker", _fake_docker([], stdout="", returncode=1)
    )

    assert milvus_runtime.vm_memory_mb() is None


def test_drop_page_cache_asks_the_vm_kernel_to_drop_and_reports_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(milvus_runtime, "docker", _fake_docker(calls))

    assert milvus_runtime.drop_page_cache() is True

    command = " ".join(calls[0])
    assert "--privileged" in command
    assert "drop_caches" in command


def test_drop_page_cache_reports_failure_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        milvus_runtime, "docker", _fake_docker([], returncode=1)
    )

    assert milvus_runtime.drop_page_cache() is False
