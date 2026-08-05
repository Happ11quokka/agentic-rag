from __future__ import annotations

import io

import pytest

from agent.scheduling import protocol
from agent.scheduling.protocol import Frame, FrameType, ProtocolError, Role


class Fragmented(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        return super().read(min(size, 3))


def test_protocol_reads_fragmented_frame() -> None:
    expected = Frame(FrameType.START, Role.PAIR, 7, "값".encode())
    assert protocol.read_frame(Fragmented(protocol.encode_frame(expected))) == expected


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (0, ord("X"), "magic"),
        (4, 9, "version"),
    ],
)
def test_protocol_rejects_invalid_header(
    offset: int, value: int, message: str
) -> None:
    encoded = bytearray(protocol.encode_frame(Frame(FrameType.START, Role.PAIR, 1)))
    encoded[offset] = value
    with pytest.raises(ProtocolError, match=message):
        protocol.read_frame(io.BytesIO(encoded))


def test_protocol_rejects_oversize_length_without_reading_payload() -> None:
    encoded = bytearray(protocol.encode_frame(Frame(FrameType.START, Role.PAIR, 1)))
    encoded[12:20] = (protocol.MAX_FRAME_BYTES + 1).to_bytes(8, "little")
    with pytest.raises(ProtocolError, match="length"):
        protocol.read_frame(io.BytesIO(encoded))


def test_protocol_rejects_stale_request_id() -> None:
    frame = Frame(FrameType.STARTED, Role.PAIR, 6)
    with pytest.raises(ProtocolError, match="stale request id"):
        protocol.validate_response(
            frame,
            expected_type=FrameType.STARTED,
            request_id=7,
            role=Role.PAIR,
        )


def test_protocol_rejects_response_flags() -> None:
    frame = Frame(FrameType.STARTED, Role.PAIR, 7, flags=1)
    with pytest.raises(ProtocolError, match="flags"):
        protocol.validate_response(
            frame,
            expected_type=FrameType.STARTED,
            request_id=7,
            role=Role.PAIR,
        )


def test_start_payload_round_trips_unicode() -> None:
    generation = {
        "max_tokens": 12,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "presence_penalty": 1.5,
        "seed": 42,
    }
    messages = [{"role": "user", "content": "서울의 값 🧪"}]
    decoded_generation, decoded_messages = protocol.decode_start(
        protocol.encode_start(messages, generation)
    )
    assert decoded_messages == messages
    assert decoded_generation["max_tokens"] == 12
    assert decoded_generation["seed"] == 42
