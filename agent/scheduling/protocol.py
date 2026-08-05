from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import BinaryIO, Iterable, Mapping

from .config import PROTOCOL_VERSION

MAGIC = b"PSQ1"
MAX_FRAME_BYTES = 16 * 1024 * 1024
HEADER = struct.Struct("<4sBBBBIQ")
GENERATION = struct.Struct("<IfffifI")
RESULT_PREFIX = struct.Struct("<IIIB3xQQQI")
TOKEN_PREFIX = struct.Struct("<QiI")


class ProtocolError(RuntimeError):
    pass


class FrameType(IntEnum):
    START = 1
    PREFILL = 2
    DECODE_SLICE = 3
    SHUTDOWN = 4
    READY = 16
    STARTED = 17
    PREFILL_RESULT = 18
    SLICE_RESULT = 19
    ERROR = 20
    BYE = 21


class Role(IntEnum):
    PAIR = 0
    MAIN = 1
    DRAFT = 2


ROLE_BY_NAME = {"main": Role.MAIN, "draft": Role.DRAFT}
NAME_BY_ROLE = {value: key for key, value in ROLE_BY_NAME.items()}


@dataclass(frozen=True, slots=True)
class Frame:
    type: FrameType
    role: Role
    request_id: int
    payload: bytes = b""
    flags: int = 0


@dataclass(frozen=True, slots=True)
class TokenEvent:
    offset_ns: int
    token_id: int
    piece: bytes


@dataclass(frozen=True, slots=True)
class CommandResult:
    processed_tokens: int
    remaining_tokens: int
    output_tokens: int
    finish_code: int
    active_compute_ns: int
    start_offset_ns: int
    end_offset_ns: int
    tokens: tuple[TokenEvent, ...]

    @property
    def finished(self) -> bool:
        return self.finish_code != 0


def encode_frame(frame: Frame) -> bytes:
    if len(frame.payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame payload exceeds {MAX_FRAME_BYTES} bytes")
    return HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        int(frame.type),
        int(frame.role),
        frame.flags,
        frame.request_id,
        len(frame.payload),
    ) + frame.payload


def write_frame(stream: BinaryIO, frame: Frame) -> None:
    stream.write(encode_frame(frame))
    stream.flush()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"protocol pipe closed with {remaining} bytes unread")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO, *, max_payload: int = MAX_FRAME_BYTES) -> Frame:
    raw = _read_exact(stream, HEADER.size)
    magic, version, frame_type, role, flags, request_id, length = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ProtocolError(f"invalid protocol magic: {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    if length > max_payload:
        raise ProtocolError(f"invalid frame length: {length}")
    try:
        selected_type = FrameType(frame_type)
        selected_role = Role(role)
    except ValueError as exc:
        raise ProtocolError(f"invalid frame type or role: {frame_type}/{role}") from exc
    return Frame(
        selected_type,
        selected_role,
        request_id,
        _read_exact(stream, length),
        flags,
    )


def validate_response(
    frame: Frame,
    *,
    expected_type: FrameType | Iterable[FrameType],
    request_id: int,
    role: Role,
) -> None:
    expected = (
        {expected_type}
        if isinstance(expected_type, FrameType)
        else set(expected_type)
    )
    if frame.request_id != request_id:
        raise ProtocolError(
            f"stale request id: expected {request_id}, received {frame.request_id}"
        )
    if frame.role != role:
        raise ProtocolError(f"role mismatch: expected {role.name}, received {frame.role.name}")
    if frame.flags != 0:
        raise ProtocolError(f"unsupported response flags: {frame.flags}")
    if frame.type not in expected and frame.type != FrameType.ERROR:
        names = ", ".join(item.name for item in expected)
        raise ProtocolError(f"frame mismatch: expected {names}, received {frame.type.name}")


def _pack_bytes(value: bytes) -> bytes:
    return struct.pack("<I", len(value)) + value


def _unpack_bytes(payload: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 4 > len(payload):
        raise ProtocolError("truncated byte-string length")
    length = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    if offset + length > len(payload):
        raise ProtocolError("truncated byte-string payload")
    return payload[offset : offset + length], offset + length


def encode_start(messages: list[Mapping[str, str]], generation: Mapping[str, object]) -> bytes:
    payload = bytearray(
        GENERATION.pack(
            int(generation["max_tokens"]),
            float(generation["temperature"]),
            float(generation["top_p"]),
            float(generation["min_p"]),
            int(generation["top_k"]),
            float(generation["presence_penalty"]),
            int(generation["seed"]),
        )
    )
    payload.extend(struct.pack("<I", len(messages)))
    for message in messages:
        payload.extend(_pack_bytes(message["role"].encode("utf-8")))
        payload.extend(_pack_bytes(message["content"].encode("utf-8")))
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError("START payload is too large")
    return bytes(payload)


def decode_start(payload: bytes) -> tuple[dict[str, object], list[dict[str, str]]]:
    if len(payload) < GENERATION.size + 4:
        raise ProtocolError("truncated START payload")
    max_tokens, temperature, top_p, min_p, top_k, presence, seed = GENERATION.unpack_from(payload)
    offset = GENERATION.size
    count = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    messages = []
    for _ in range(count):
        role, offset = _unpack_bytes(payload, offset)
        content, offset = _unpack_bytes(payload, offset)
        try:
            messages.append(
                {"role": role.decode("utf-8"), "content": content.decode("utf-8")}
            )
        except UnicodeDecodeError as exc:
            raise ProtocolError("START contains invalid UTF-8") from exc
    if offset != len(payload):
        raise ProtocolError("START payload has trailing bytes")
    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "top_k": top_k,
        "presence_penalty": presence,
        "seed": seed,
    }, messages


def encode_command_result(result: CommandResult) -> bytes:
    payload = bytearray(
        RESULT_PREFIX.pack(
            result.processed_tokens,
            result.remaining_tokens,
            result.output_tokens,
            result.finish_code,
            result.active_compute_ns,
            result.start_offset_ns,
            result.end_offset_ns,
            len(result.tokens),
        )
    )
    for token in result.tokens:
        payload.extend(TOKEN_PREFIX.pack(token.offset_ns, token.token_id, len(token.piece)))
        payload.extend(token.piece)
    return bytes(payload)


def decode_command_result(payload: bytes) -> CommandResult:
    if len(payload) < RESULT_PREFIX.size:
        raise ProtocolError("truncated result payload")
    values = RESULT_PREFIX.unpack_from(payload)
    offset = RESULT_PREFIX.size
    events = []
    for _ in range(values[-1]):
        if offset + TOKEN_PREFIX.size > len(payload):
            raise ProtocolError("truncated token event")
        token_offset, token_id, length = TOKEN_PREFIX.unpack_from(payload, offset)
        offset += TOKEN_PREFIX.size
        if offset + length > len(payload):
            raise ProtocolError("truncated token piece")
        events.append(TokenEvent(token_offset, token_id, payload[offset : offset + length]))
        offset += length
    if offset != len(payload):
        raise ProtocolError("result payload has trailing bytes")
    return CommandResult(*values[:-1], tuple(events))


def encode_string(value: str) -> bytes:
    return _pack_bytes(value.encode("utf-8"))


def decode_string(payload: bytes) -> str:
    value, offset = _unpack_bytes(payload, 0)
    if offset != len(payload):
        raise ProtocolError("string payload has trailing bytes")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("invalid UTF-8 string payload") from exc


def frame_from_bytes(value: bytes) -> Frame:
    stream = io.BytesIO(value)
    frame = read_frame(stream)
    if stream.read(1):
        raise ProtocolError("multiple frames supplied")
    return frame
