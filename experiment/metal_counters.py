"""Owned Instruments recording and streaming counter export for Apple M3 GPUs."""

from __future__ import annotations

import ctypes
import gzip
import hashlib
import json
import math
import os
import platform
import signal
import shutil
import sqlite3
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .common import ExperimentError

TEMPLATE = Path(__file__).parent / "resources" / "metal-counters.tracetemplate"
PROFILE = 13
CAPTURE_SECONDS = 2
MAX_CAPTURE_SECONDS = 5
MIN_FREE_BYTES = 8 * 1024**3
START_TIMEOUT = 30
STOP_TIMEOUT = 300
EXPORT_TIMEOUT = 600
MAX_ALIGNMENT_ERROR_NS = 5_000_000
BANDWIDTH_COUNTERS = {"GPU Read Bandwidth", "GPU Write Bandwidth", "GPU Bandwidth"}
REQUIRED_COUNTERS = {
    "GPU Read Bandwidth",
    "GPU Write Bandwidth",
    "ALU Utilization",
    "F16 Limiter",
    "F32 Limiter",
    "Integer and Complex Limiter",
    "Integer and Conditional Limiter",
}


def collector_metadata() -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ExperimentError(
            "contention-measure requires Apple silicon macOS and Xcode"
        )
    try:
        version = subprocess.run(
            ["xcodebuild", "-version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        templates = subprocess.run(
            ["xcrun", "xctrace", "list", "templates"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExperimentError(
            "contention-measure requires a working full Xcode installation"
        ) from exc
    if "Metal System Trace" not in templates or not TEMPLATE.is_file():
        raise ExperimentError(
            "Metal System Trace or the bundled counter template is missing"
        )
    return {
        "backend": "xctrace",
        "xcode": version,
        "macos": platform.mac_ver()[0],
        "profile": PROFILE,
        "capture_seconds": CAPTURE_SECONDS,
        "template": TEMPLATE.name,
        "template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        "scope": "GPU-wide, including other processes",
        "bandwidth_semantics": "GPU memory traffic, not physical DRAM traffic",
        "measurement_scope": "fixed-duration inference; servers stopped before finalization",
    }


@dataclass(frozen=True)
class ClockAnchor:
    offset_ns: int
    uncertainty_ns: int


def clock_anchor() -> ClockAnchor:
    before = time.perf_counter_ns()
    wall = time.time_ns()
    after = time.perf_counter_ns()
    return ClockAnchor(wall - (before + after) // 2, (after - before + 1) // 2)


def clock_mapping(toc: Path, before: ClockAnchor, after: ClockAnchor) -> dict[str, int]:
    start = ET.parse(toc).findtext("./run/info/summary/start-date")
    if not start:
        raise ExperimentError("counter trace has no start-date for clock alignment")
    parsed = datetime.fromisoformat(start)
    # Trace start dates are formatted to milliseconds. Account for their rounding.
    epoch_ns = int(parsed.timestamp()) * 1_000_000_000 + parsed.microsecond * 1000
    drift = abs(before.offset_ns - after.offset_ns)
    uncertainty = max(before.uncertainty_ns, after.uncertainty_ns) + drift + 1_000_000
    if uncertainty > MAX_ALIGNMENT_ERROR_NS:
        raise ExperimentError("counter/client clock alignment uncertainty exceeds 5 ms")
    return {
        "trace_start_client_ns": epoch_ns - (before.offset_ns + after.offset_ns) // 2,
        "uncertainty_ns": uncertainty,
        "clock_drift_ns": drift,
    }


class _RecordingNotification:
    """Darwin notification makes readiness independent of xctrace log buffering."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL(None)
        self.lib.notify_register_check.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.notify_check.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.notify_cancel.argtypes = [ctypes.c_int]
        self.name = f"contention-measure.{uuid.uuid4().hex}"
        self.token = ctypes.c_int()
        if self.lib.notify_register_check(self.name.encode(), ctypes.byref(self.token)):
            raise ExperimentError("could not register recorder readiness notification")
        self.ready()  # A new registration starts with its change flag set.

    def ready(self) -> bool:
        changed = ctypes.c_int()
        if self.lib.notify_check(self.token.value, ctypes.byref(changed)):
            raise ExperimentError("could not check recorder readiness notification")
        return bool(changed.value)

    def close(self) -> None:
        self.lib.notify_cancel(self.token.value)


class MetalRecording(AbstractContextManager["MetalRecording"]):
    def __init__(
        self, directory: Path, *, duration_seconds: int = CAPTURE_SECONDS
    ) -> None:
        if (
            not isinstance(duration_seconds, int)
            or not 1 <= duration_seconds <= MAX_CAPTURE_SECONDS
        ):
            raise ExperimentError(
                f"recording duration must be 1–{MAX_CAPTURE_SECONDS} seconds"
            )
        self.duration_seconds = duration_seconds
        self.directory = directory
        self.trace = directory / "recording.trace"
        self.log = directory / "recorder.log"
        self.process: subprocess.Popen[bytes] | None = None
        self.before: ClockAnchor | None = None
        self.after: ClockAnchor | None = None

    def _detail(self) -> str:
        if not self.log.exists():
            return ""
        return self.log.read_text(errors="replace")[-4000:]

    def _exit_description(self) -> str:
        code = self.process.returncode if self.process is not None else None
        if code is not None and code < 0:
            return f"terminated by {signal.Signals(-code).name} (returncode={code})"
        return f"exit code {code}"

    def _save_status(self) -> None:
        try:
            (self.directory / "recorder-status.json").write_text(
                json.dumps(
                    {
                        "pid": self.process.pid if self.process else None,
                        "returncode": self.process.returncode if self.process else None,
                        "capture_seconds": self.duration_seconds,
                    },
                    indent=2,
                )
                + "\n"
            )
        except OSError:
            pass  # Preserve the original error if the filesystem itself failed.

    def __enter__(self) -> MetalRecording:
        self.directory.mkdir(parents=True, exist_ok=False)
        free = shutil.disk_usage(self.directory).free
        if free < MIN_FREE_BYTES:
            raise ExperimentError(
                f"GPU profiling requires at least 8 GiB free for temporary files; "
                f"only {free / 1024**3:.1f} GiB is available. Saved results were retained."
            )
        notification = _RecordingNotification()
        try:
            self.before = clock_anchor()
            with self.log.open("wb") as output:
                self.process = subprocess.Popen(
                    [
                        "xcrun",
                        "xctrace",
                        "record",
                        "--template",
                        str(TEMPLATE),
                        "--all-processes",
                        "--output",
                        str(self.trace),
                        "--no-prompt",
                        "--notify-tracing-started",
                        notification.name,
                        "--time-limit",
                        f"{self.duration_seconds}s",
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            deadline = time.monotonic() + START_TIMEOUT
            while time.monotonic() < deadline:
                if notification.ready():
                    return self
                if self.process.poll() is not None:
                    raise ExperimentError(
                        f"counter recorder exited during startup ({self._exit_description()}):\n{self._detail()}"
                    )
                time.sleep(0.1)
            raise ExperimentError(
                f"counter recorder did not become ready in {START_TIMEOUT}s:\n{self._detail()}"
            )
        except BaseException:
            self._terminate()
            self._save_status()
            raise
        finally:
            notification.close()

    def _terminate(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                continue

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.process is None:
                raise ExperimentError("counter recorder never started")
            # Successful captures let xctrace stop on its own fixed time limit.
            # Sending SIGINT again while it is already finalizing can abort it.
            if exc_type is not None and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired as error:
                raise ExperimentError(
                    "counter recorder timed out finalizing the trace"
                ) from error
            self.after = clock_anchor()
            if (
                self.process.returncode != 0
                or "GPU Service reported error" in self._detail()
            ):
                raise ExperimentError(
                    f"GPU counter recording failed ({self._exit_description()}):\n{self._detail()}"
                )
            if (
                exc_type is None
                and "Reached specified time limit" not in self._detail()
            ):
                raise ExperimentError(
                    "counter recorder exited before its capture limit"
                )
        except (ExperimentError, OSError) as error:
            if exc_type is None:
                raise
            with self.log.open("a") as output:
                output.write(f"\nCleanup error: {error}\n")
        finally:
            self._terminate()
            self._save_status()

    def export(self, windows: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
        if self.before is None or self.after is None:
            raise ExperimentError("counter recording did not finish successfully")
        toc = self.directory / "toc.xml"
        xml = self.directory / "counters.xml"
        for args in (
            ["--toc", "--output", str(toc)],
            [
                "--xpath",
                '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-counter-intervals"]',
                "--output",
                str(xml),
            ],
        ):
            with (self.directory / "export.log").open("ab") as output:
                try:
                    subprocess.run(
                        [
                            "xcrun",
                            "xctrace",
                            "export",
                            "--input",
                            str(self.trace),
                            *args,
                        ],
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=EXPORT_TIMEOUT,
                    )
                except (OSError, subprocess.SubprocessError) as error:
                    raise ExperimentError(
                        f"GPU counter export failed; see {self.directory / 'export.log'}"
                    ) from error
        mapping = clock_mapping(toc, self.before, self.after)
        result = summarize_counters(
            xml, self.directory / "samples.jsonl.gz", windows, mapping
        )
        xml.unlink()  # Raw trace and normalized samples retain the measurements.
        result["clock_alignment"] = mapping
        (self.directory / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result


def counter_samples(xml: Path) -> Iterator[dict[str, Any]]:
    """Resolve Instruments XML references without holding the full export in RAM."""
    index = xml.with_suffix(".references.sqlite")
    db = sqlite3.connect(index)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE refs (id TEXT PRIMARY KEY, value TEXT, formatted TEXT)")

    @lru_cache(maxsize=8192)
    def reference(identifier: str) -> tuple[str | None, str | None]:
        row = db.execute(
            "SELECT value, formatted FROM refs WHERE id=?", (identifier,)
        ).fetchone()
        if row is None:
            raise ExperimentError(f"unresolved counter XML reference {identifier}")
        return row

    def resolve(element: ET.Element) -> tuple[str | None, str | None]:
        if "ref" in element.attrib:
            return reference(element.attrib["ref"])
        return element.text, element.get("fmt")

    keep = {
        "start-time",
        "duration",
        "uint32",
        "gpu-counter-name",
        "formatted-label",
        "fixed-decimal",
        "metal-device-name",
    }
    columns: list[str] = []
    stack: list[ET.Element] = []
    try:
        for event, element in ET.iterparse(xml, events=("start", "end")):
            if event == "start":
                stack.append(element)
                continue
            if element.get("id") and element.tag in keep:
                db.execute(
                    "INSERT INTO refs VALUES (?, ?, ?)",
                    (element.get("id"), element.text, element.get("fmt")),
                )
            if element.tag == "schema":
                columns = [
                    col.findtext("mnemonic", "") for col in element.findall("col")
                ]
            if element.tag == "row":
                fields = dict(zip(columns, element, strict=True))
                name = resolve(fields["name"])[1]
                if (
                    name
                    and name != "Top Performance Limiter"
                    and (
                        name in BANDWIDTH_COUNTERS
                        or name.endswith((" Limiter", " Utilization"))
                    )
                ):
                    label = resolve(fields["label"])[1] or ""
                    unit = "GB/s" if "GB/s" in label else "%" if "%" in label else None
                    expected = "GB/s" if name in BANDWIDTH_COUNTERS else "%"
                    if unit != expected:
                        raise ExperimentError(
                            f"unsupported unit for counter {name}: {label}"
                        )
                    value = float(resolve(fields["value"])[0])
                    duration = int(resolve(fields["duration"])[0])
                    yield {
                        "trace_start_ns": int(resolve(fields["start"])[0]),
                        "duration_ns": duration,
                        "name": name,
                        "unit": unit,
                        "value": value if math.isfinite(value) else None,
                        "valid": math.isfinite(value) and value >= 0 and duration > 0,
                        "counter_id": int(resolve(fields["counter-id"])[0]),
                        "gpu": resolve(fields["gpu"])[1],
                        "group_index": int(resolve(fields["group-index"])[0]),
                        "ring_buffer_index": int(
                            resolve(fields["ring-buffer-index"])[0]
                        ),
                    }
                stack[-2].remove(element)
                element.clear()
            stack.pop()
    except (ET.ParseError, ValueError, KeyError, TypeError) as error:
        raise ExperimentError(f"invalid GPU counter export: {error}") from error
    finally:
        reference.cache_clear()
        db.close()
        index.unlink(missing_ok=True)


def summarize_counters(
    xml: Path,
    samples_path: Path,
    windows: Mapping[str, tuple[int, int]],
    mapping: Mapping[str, int],
) -> dict[str, Any]:
    aggregates: dict[str, dict[str, Any]] = {name: {} for name in windows}
    observed: set[str] = set()
    sources: set[tuple[Any, ...]] = set()
    uncertainty = mapping["uncertainty_ns"]
    count = 0
    invalid: dict[str, int] = {}
    with gzip.open(samples_path, "wt", encoding="utf-8", compresslevel=1) as output:
        for sample in counter_samples(xml):
            start = mapping["trace_start_client_ns"] + sample["trace_start_ns"]
            end = start + sample["duration_ns"]
            included = [
                key
                for key, (left, right) in windows.items()
                if start >= left + uncertainty and end <= right - uncertainty
            ]
            if not included:
                continue
            sample["start_ns"] = start
            output.write(json.dumps(sample, separators=(",", ":")) + "\n")
            if not sample["valid"]:
                invalid[sample["name"]] = invalid.get(sample["name"], 0) + 1
                continue
            observed.add(sample["name"])
            sources.add(
                (sample["gpu"], sample["group_index"], sample["ring_buffer_index"])
            )
            count += 1
            for key in included:
                stats = aggregates[key].setdefault(
                    sample["name"],
                    {
                        "unit": sample["unit"],
                        "sample_count": 0,
                        "sampled_ns": 0,
                        "weighted_sum": 0.0,
                        "max": 0.0,
                    },
                )
                stats["sample_count"] += 1
                stats["sampled_ns"] += sample["duration_ns"]
                stats["weighted_sum"] += sample["value"] * sample["duration_ns"]
                stats["max"] = max(stats["max"], sample["value"])
    missing = REQUIRED_COUNTERS - observed
    if missing:
        raise ExperimentError(
            f"required GPU counter samples unavailable: {', '.join(sorted(missing))}"
        )
    if len(sources) != 1:
        raise ExperimentError(
            "counter export contains multiple GPU sources; cannot combine them safely"
        )
    result: dict[str, Any] = {}
    for key, counters in aggregates.items():
        for stats in counters.values():
            stats["mean"] = stats.pop("weighted_sum") / stats["sampled_ns"]
            stats["coverage_fraction"] = stats["sampled_ns"] / (
                windows[key][1] - windows[key][0]
            )
        result[key] = {
            "start_ns": windows[key][0],
            "end_ns": windows[key][1],
            "available": bool(counters),
            "counters": counters,
            "ranked_limiters": sorted(
                (name for name in counters if name.endswith(" Limiter")),
                key=lambda name: counters[name]["mean"],
                reverse=True,
            ),
        }
    return {
        "sample_count": count,
        "invalid_samples_by_counter": invalid,
        "source": list(next(iter(sources))),
        "windows": result,
    }
