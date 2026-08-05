from __future__ import annotations

import collections
import os
import selectors
import signal
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO, Mapping

from .build import NativeEngine, resolve_native_engine
from .config import LlamaConfig
from .protocol import (
    CommandResult,
    Frame,
    FrameType,
    ROLE_BY_NAME,
    Role,
    decode_command_result,
    decode_string,
    encode_start,
    read_frame,
    validate_response,
    write_frame,
)


class ScheduledEngineError(RuntimeError):
    pass


class _DeadlineReader:
    def __init__(self, stream: BinaryIO, timeout_seconds: float) -> None:
        self.stream = stream
        self.timeout_seconds = timeout_seconds
        self.deadline = time.monotonic() + timeout_seconds
        self.selector = selectors.DefaultSelector()
        self.selector.register(stream, selectors.EVENT_READ)

    def read(self, size: int) -> bytes:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not self.selector.select(remaining):
            raise TimeoutError(
                f"native engine response timed out after {self.timeout_seconds:g}s"
            )
        return os.read(self.stream.fileno(), size)

    def close(self) -> None:
        self.selector.close()


class ScheduledNativeEngine:
    def __init__(
        self,
        model_paths: Mapping[str, Path | str],
        llama_config: LlamaConfig | None = None,
        *,
        native_engine: NativeEngine | None = None,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.model_paths = {role: Path(path).resolve() for role, path in model_paths.items()}
        if set(self.model_paths) != {"main", "draft"}:
            raise ValueError("model_paths must contain exactly main and draft")
        self.config = (llama_config or LlamaConfig()).resolved()
        if (
            self.config.n_seq_max != 1
            or self.config.type_k != "q8_0"
            or self.config.type_v != "q8_0"
            or not self.config.offload_kqv
        ):
            raise ValueError(
                "native scheduled engine requires n_seq_max=1, Q8_0 K/V, and KQV offload"
            )
        self.native_engine = native_engine or resolve_native_engine()
        self.version = self.native_engine.version
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[bytes] | None = None
        self.request_id = 0
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=80)
        descriptor, log_name = tempfile.mkstemp(prefix="parallel-scheduled-", suffix=".log")
        os.close(descriptor)
        self.log_path = Path(log_name)
        self._protocol_healthy = True

    def __enter__(self) -> ScheduledNativeEngine:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if self.process is not None:
            return
        arguments = [
            str(self.native_engine.path),
            "--main",
            str(self.model_paths["main"]),
            "--draft",
            str(self.model_paths["draft"]),
            "--threads",
            str(self.config.n_threads),
            "--n-ctx",
            str(self.config.n_ctx),
            "--n-batch",
            str(self.config.n_batch),
            "--n-ubatch",
            str(self.config.n_ubatch),
            "--n-gpu-layers",
            str(self.config.n_gpu_layers),
        ]
        try:
            self.process = subprocess.Popen(
                arguments,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise ScheduledEngineError(f"could not start native engine: {exc}") from exc
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="scheduled-engine-stderr", daemon=True
        )
        self._stderr_thread.start()
        try:
            frame = self._receive()
            validate_response(
                frame,
                expected_type=FrameType.READY,
                request_id=0,
                role=Role.PAIR,
            )
            self._raise_engine_error(frame)
            if len(frame.payload) < 8:
                raise ScheduledEngineError("native READY payload is truncated")
            name_length = struct.unpack_from("<I", frame.payload)[0]
            expected = 4 + name_length + 4
            if len(frame.payload) != expected:
                raise ScheduledEngineError("native READY payload length is invalid")
            name = frame.payload[4 : 4 + name_length].decode("utf-8")
            threads = struct.unpack_from("<I", frame.payload, 4 + name_length)[0]
            if name != "parallel-scheduled-engine" or threads != self.config.n_threads:
                raise ScheduledEngineError("native READY configuration mismatch")
        except BaseException as exc:
            self._protocol_healthy = False
            self._terminate()
            if isinstance(exc, (KeyboardInterrupt, SystemExit, ScheduledEngineError)):
                raise
            raise self._failure(str(exc)) from exc

    def start_pair(
        self,
        messages: list[Mapping[str, str]],
        generation: Mapping[str, object],
    ) -> dict[str, int]:
        frame = self._request(
            FrameType.START,
            Role.PAIR,
            encode_start(messages, generation),
            FrameType.STARTED,
        )
        if len(frame.payload) != 8:
            raise self._failure("native STARTED payload length is invalid")
        main, draft = struct.unpack("<II", frame.payload)
        return {"main": main, "draft": draft}

    def prefill(self, role: str, maximum_tokens: int) -> CommandResult:
        frame = self._request(
            FrameType.PREFILL,
            ROLE_BY_NAME[role],
            struct.pack("<I", maximum_tokens),
            FrameType.PREFILL_RESULT,
        )
        return decode_command_result(frame.payload)

    def decode_slice(self, role: str, budget_ns: int) -> CommandResult:
        frame = self._request(
            FrameType.DECODE_SLICE,
            ROLE_BY_NAME[role],
            struct.pack("<Q", budget_ns),
            FrameType.SLICE_RESULT,
        )
        return decode_command_result(frame.payload)

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None and self._protocol_healthy:
            try:
                self._request(
                    FrameType.SHUTDOWN, Role.PAIR, b"", FrameType.BYE
                )
            except BaseException:
                self._protocol_healthy = False
        self._terminate()

    def _request(
        self,
        frame_type: FrameType,
        role: Role,
        payload: bytes,
        response_type: FrameType,
    ) -> Frame:
        process = self._require_process()
        self.request_id += 1
        try:
            assert process.stdin is not None
            write_frame(
                process.stdin,
                Frame(frame_type, role, self.request_id, payload),
            )
            frame = self._receive()
            validate_response(
                frame,
                expected_type=response_type,
                request_id=self.request_id,
                role=role,
            )
            self._raise_engine_error(frame)
            return frame
        except (BrokenPipeError, EOFError, OSError, TimeoutError, ValueError) as exc:
            self._protocol_healthy = False
            raise self._failure(str(exc)) from exc
        except RuntimeError as exc:
            self._protocol_healthy = False
            if isinstance(exc, ScheduledEngineError):
                raise
            raise self._failure(str(exc)) from exc

    def _receive(self) -> Frame:
        process = self._require_process()
        assert process.stdout is not None
        reader = _DeadlineReader(process.stdout, self.timeout_seconds)
        try:
            return read_frame(reader)  # type: ignore[arg-type]
        finally:
            reader.close()

    def _raise_engine_error(self, frame: Frame) -> None:
        if frame.type == FrameType.ERROR:
            message = decode_string(frame.payload)
            self._protocol_healthy = False
            raise self._failure(f"native engine error: {message}")

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self.process is None:
            raise ScheduledEngineError("native engine is not open")
        return self.process

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        with self.log_path.open("ab", buffering=0) as log:
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                log.write(line)
                self._stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())

    def _failure(self, message: str) -> ScheduledEngineError:
        tail = "\n".join(self._stderr_tail)
        suffix = f"; stderr log: {self.log_path}"
        if tail:
            suffix += f"; stderr tail:\n{tail}"
        return ScheduledEngineError(message + suffix)

    def _terminate(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        self.process = None
