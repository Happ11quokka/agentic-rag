import gzip
import json
import signal
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from experiment import metal_counters as mc
from experiment.common import ExperimentError


COLUMNS = (
    "start",
    "duration",
    "counter-id",
    "name",
    "label",
    "value",
    "percent-value",
    "color",
    "gpu",
    "group-index",
    "is-percentage",
    "ring-buffer-index",
)


def write_export(path: Path, samples: list[tuple[str, int, int, float]]) -> None:
    root = ET.Element("trace-query-result")
    node = ET.SubElement(root, "node")
    schema = ET.SubElement(node, "schema")
    for name in COLUMNS:
        ET.SubElement(ET.SubElement(schema, "col"), "mnemonic").text = name
    for i, (name, start, duration, value) in enumerate(samples):
        row = ET.SubElement(node, "row")
        unit = "GB/s" if name in mc.BANDWIDTH_COUNTERS else "%"
        ET.SubElement(row, "start-time").text = str(start)
        ET.SubElement(row, "duration").text = str(duration)
        ET.SubElement(row, "uint32").text = str(i)
        ET.SubElement(row, "gpu-counter-name", fmt=name).text = f"{i}.{name}"
        label = ET.SubElement(
            row, "formatted-label", fmt=f"{value} {unit} (description)"
        )
        ET.SubElement(label, "fixed-decimal", id=f"value-{i}").text = str(value)
        ET.SubElement(row, "fixed-decimal", ref=f"value-{i}")
        ET.SubElement(
            row, "percent"
        ).text = "999"  # Bandwidth must use value, not percent.
        ET.SubElement(row, "uint32").text = "0"
        if i == 0:
            ET.SubElement(
                row, "metal-device-name", id="gpu", fmt="M3 Pro"
            ).text = "M3 Pro"
        else:
            ET.SubElement(row, "metal-device-name", ref="gpu")
        ET.SubElement(row, "uint32").text = "1"
        ET.SubElement(row, "boolean").text = "1"
        ET.SubElement(row, "uint32").text = "2"
    ET.ElementTree(root).write(path, encoding="utf-8")


def test_export_references_units_and_duration_weighting(tmp_path):
    xml = tmp_path / "export.xml"
    samples = [(name, 10, 10, 20) for name in sorted(mc.REQUIRED_COUNTERS)]
    samples += [("GPU Read Bandwidth", 20, 30, 60), ("GPU Read Bandwidth", 0, 10, 999)]
    write_export(xml, samples)
    result = mc.summarize_counters(
        xml,
        tmp_path / "samples.jsonl.gz",
        {"request": (1000, 1100), "empty": (2000, 2100)},
        {"trace_start_client_ns": 1000, "uncertainty_ns": 5},
    )
    stats = result["windows"]["request"]["counters"]["GPU Read Bandwidth"]
    assert stats["mean"] == 50
    assert stats["sampled_ns"] == 40
    assert stats["unit"] == "GB/s"
    assert result["windows"]["empty"]["available"] is False
    with gzip.open(tmp_path / "samples.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert all(row["value"] != 999 for row in rows)
    assert result["source"] == ["M3 Pro", 1, 2]
    assert not xml.with_suffix(".references.sqlite").exists()


def test_empty_or_partial_counters_fail(tmp_path):
    xml = tmp_path / "export.xml"
    write_export(xml, [("GPU Read Bandwidth", 10, 10, 2)])
    with pytest.raises(
        ExperimentError, match="required GPU counter samples unavailable"
    ):
        mc.summarize_counters(
            xml,
            tmp_path / "samples.gz",
            {"request": (0, 100)},
            {"trace_start_client_ns": 0, "uncertainty_ns": 0},
        )


@pytest.mark.parametrize(
    "replacement, message",
    [
        ('ref="missing"', "unresolved"),
        ('ref="value-0"', None),
    ],
)
def test_reference_resolution(tmp_path, replacement, message):
    xml = tmp_path / "export.xml"
    write_export(xml, [("GPU Read Bandwidth", 10, 10, 12.5)])
    xml.write_text(xml.read_text().replace('ref="value-0"', replacement))
    if message:
        with pytest.raises(ExperimentError, match=message):
            list(mc.counter_samples(xml))
    else:
        assert list(mc.counter_samples(xml))[0]["value"] == 12.5


def test_unknown_units_fail(tmp_path):
    xml = tmp_path / "export.xml"
    write_export(xml, [("GPU Read Bandwidth", 10, 10, 12.5)])
    xml.write_text(xml.read_text().replace("GB/s", "bytes/cycle"))
    with pytest.raises(ExperimentError, match="unsupported unit"):
        list(mc.counter_samples(xml))


def test_categorical_top_limiter_is_not_a_percentage_metric(tmp_path):
    xml = tmp_path / "export.xml"
    write_export(xml, [("Top Performance Limiter", 10, 10, 12)])
    assert list(mc.counter_samples(xml)) == []


def test_negative_samples_are_retained_but_excluded(tmp_path):
    xml = tmp_path / "export.xml"
    samples = [(name, 10, 10, 20) for name in sorted(mc.REQUIRED_COUNTERS)]
    samples.append(("ALU Utilization", 20, 10, -0.001))
    write_export(xml, samples)
    output = tmp_path / "samples.gz"
    result = mc.summarize_counters(
        xml,
        output,
        {"request": (0, 100)},
        {"trace_start_client_ns": 0, "uncertainty_ns": 0},
    )
    assert result["invalid_samples_by_counter"] == {"ALU Utilization": 1}
    assert result["windows"]["request"]["counters"]["ALU Utilization"]["mean"] == 20
    with gzip.open(output, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[-1]["valid"] is False
    assert rows[-1]["value"] == -0.001


def test_different_gpu_sources_are_not_summed(tmp_path):
    xml = tmp_path / "export.xml"
    samples = [(name, 10, 10, 20) for name in sorted(mc.REQUIRED_COUNTERS)]
    write_export(xml, samples)
    root = ET.parse(xml)
    gpu = root.findall("./node/row")[-1].find("metal-device-name")
    gpu.attrib.clear()
    gpu.set("fmt", "Other GPU")
    gpu.text = "Other GPU"
    root.write(xml)
    with pytest.raises(ExperimentError, match="multiple GPU sources"):
        mc.summarize_counters(
            xml,
            tmp_path / "samples.gz",
            {"request": (0, 100)},
            {"trace_start_client_ns": 0, "uncertainty_ns": 0},
        )


def test_stop_timeout_terminates_owned_process(tmp_path, monkeypatch):
    process, _, calls = patch_recorder(monkeypatch)
    original = process.wait

    def wait(timeout):
        if timeout == mc.STOP_TIMEOUT:
            raise mc.subprocess.TimeoutExpired("xctrace", timeout)
        return original(timeout)

    monkeypatch.setattr(process, "wait", wait)
    with pytest.raises(ExperimentError, match="timed out finalizing"):
        with mc.MetalRecording(tmp_path / "recording"):
            pass
    assert calls == [(process.pid, signal.SIGTERM)]


def test_clock_alignment_and_wall_clock_jump(tmp_path):
    toc = tmp_path / "toc.xml"
    toc.write_text(
        "<trace-toc><run><info><summary><start-date>1970-01-01T00:00:10.123+00:00</start-date>"
        "</summary></info></run></trace-toc>"
    )
    mapping = mc.clock_mapping(
        toc, mc.ClockAnchor(2_000_000_000, 100), mc.ClockAnchor(2_000_000_000, 100)
    )
    assert mapping["trace_start_client_ns"] == 8_123_000_000
    assert mapping["uncertainty_ns"] == 1_000_100
    with pytest.raises(ExperimentError, match="exceeds 5 ms"):
        mc.clock_mapping(toc, mc.ClockAnchor(0, 0), mc.ClockAnchor(10_000_000, 0))


class FakeNotification:
    name = "test-notification"
    closed = False

    def ready(self):
        return True

    def close(self):
        self.closed = True


class FakeProcess:
    pid = 123456
    returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.returncode = 0
        return 0


def patch_recorder(monkeypatch):
    process = FakeProcess()
    notification = FakeNotification()
    calls = []
    monkeypatch.setattr(mc, "_RecordingNotification", lambda: notification)
    monkeypatch.setattr(
        mc.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * 1024**3)
    )

    def popen(command, **kwargs):
        assert command[command.index("--time-limit") + 1] == "2s"
        process.log = Path(kwargs["stdout"].name)
        return process

    original_wait = process.wait

    def wait(timeout):
        with process.log.open("a") as stream:
            stream.write("Reached specified time limit, ending recording...\n")
        return original_wait(timeout)

    monkeypatch.setattr(process, "wait", wait)
    monkeypatch.setattr(mc.subprocess, "Popen", popen)
    monkeypatch.setattr(mc.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    return process, notification, calls


def test_natural_capture_end_does_not_send_second_interrupt(tmp_path, monkeypatch):
    process, _, signals = patch_recorder(monkeypatch)
    with mc.MetalRecording(tmp_path / "recording"):
        process.returncode = 0  # Already finalizing/completed when inference stops.
    assert signals == []
    status = json.loads((tmp_path / "recording" / "recorder-status.json").read_text())
    assert status["returncode"] == 0
    assert status["capture_seconds"] == 2


def test_signal_failure_is_reported_and_persisted(tmp_path, monkeypatch):
    process, _, _ = patch_recorder(monkeypatch)

    def killed(timeout):
        process.returncode = -signal.SIGKILL
        return process.returncode

    monkeypatch.setattr(process, "wait", killed)
    with pytest.raises(ExperimentError, match="SIGKILL.*returncode=-9"):
        with mc.MetalRecording(tmp_path / "recording"):
            pass
    status = json.loads((tmp_path / "recording" / "recorder-status.json").read_text())
    assert status["returncode"] == -9


def test_low_disk_fails_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mc.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**3)
    )
    monkeypatch.setattr(
        mc.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch")
    )
    with pytest.raises(ExperimentError, match="at least 8 GiB"):
        with mc.MetalRecording(tmp_path / "recording"):
            pass


def test_capture_duration_is_bounded():
    for seconds in (0, 6, 2.5):
        with pytest.raises(ExperimentError, match="duration must be"):
            mc.MetalRecording(Path("unused"), duration_seconds=seconds)


def test_recorder_finalizes_on_interruption(tmp_path, monkeypatch):
    process, notification, calls = patch_recorder(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        with mc.MetalRecording(tmp_path / "recording"):
            raise KeyboardInterrupt()
    assert calls == [(process.pid, signal.SIGINT)]
    assert notification.closed
    assert process.returncode == 0


def test_startup_failure_reaps_recorder(tmp_path, monkeypatch):
    process, notification, calls = patch_recorder(monkeypatch)
    monkeypatch.setattr(notification, "ready", lambda: False)
    times = iter([0, mc.START_TIMEOUT + 1])
    monkeypatch.setattr(mc.time, "monotonic", lambda: next(times))
    with pytest.raises(ExperimentError, match="did not become ready"):
        with mc.MetalRecording(tmp_path / "recording"):
            pytest.fail("must not execute inference")
    assert calls == [(process.pid, signal.SIGTERM)]
    assert notification.closed


def test_gpu_warning_is_failure_even_with_zero_exit(tmp_path, monkeypatch):
    patch_recorder(monkeypatch)
    with pytest.raises(ExperimentError, match="GPU counter recording failed"):
        with mc.MetalRecording(tmp_path / "recording") as recording:
            recording.log.write_text(
                "GPU Service reported error: unsupported counter profile"
            )


def test_export_timeout_reports_artifact_location(tmp_path, monkeypatch):
    recording = mc.MetalRecording(tmp_path)
    recording.before = recording.after = mc.ClockAnchor(0, 0)

    def fail(*args, **kwargs):
        raise mc.subprocess.TimeoutExpired("xctrace", 300)

    monkeypatch.setattr(mc.subprocess, "run", fail)
    with pytest.raises(ExperimentError, match="export.log"):
        recording.export({})


def test_non_apple_platform_rejected(monkeypatch):
    monkeypatch.setattr(mc.platform, "system", lambda: "Linux")
    with pytest.raises(ExperimentError, match="Apple silicon"):
        mc.collector_metadata()
