from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

TIME_QUANTUM_NS = 200_000_000
PREFILL_CHUNK_TOKENS = 256
PREFILL_ORDER = ("draft", "main")
DECODE_ORDER = ("draft", "main")
CONTEXT_SIZE = 16_384
PROTOCOL_VERSION = 1


def physical_core_count() -> tuple[int, str]:
    system = platform.system()
    if system == "Darwin":
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            try:
                result = subprocess.run(
                    ["sysctl", "-n", key],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError:
                break
            try:
                count = int(result.stdout.strip())
            except ValueError:
                continue
            if result.returncode == 0 and count > 0:
                return count, f"sysctl:{key}"
    elif system == "Linux":
        try:
            physical_id: str | None = None
            core_id: str | None = None
            cores: set[tuple[str, str]] = set()
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in (*handle, "\n"):
                    if not line.strip():
                        if physical_id is not None and core_id is not None:
                            cores.add((physical_id, core_id))
                        physical_id = core_id = None
                    elif line.startswith("physical id"):
                        physical_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":", 1)[1].strip()
            if cores:
                return len(cores), "/proc/cpuinfo:physical-cores"
        except OSError:
            pass
    return os.cpu_count() or 1, "os.cpu_count"


@dataclass(frozen=True, slots=True)
class LlamaConfig:
    n_ctx: int = CONTEXT_SIZE
    n_batch: int = 2_048
    n_ubatch: int = 512
    n_seq_max: int = 1
    type_k: str = "q8_0"
    type_v: str = "q8_0"
    offload_kqv: bool = True
    n_gpu_layers: int = 2_147_483_647
    n_threads: int | None = None
    thread_source: str | None = None

    def resolved(self) -> LlamaConfig:
        if self.n_threads is not None:
            return self
        count, source = physical_core_count()
        return LlamaConfig(
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_ubatch=self.n_ubatch,
            n_seq_max=self.n_seq_max,
            type_k=self.type_k,
            type_v=self.type_v,
            offload_kqv=self.offload_kqv,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=count,
            thread_source=source,
        )
