"""Encode/decode lightx2v.v1 websocket protobuf messages.

The schema lives in proto/lightx2v_v1.proto. This module implements proto3
wire encoding so the service can run without a generated *_pb2.py at import
time. When protobuf/grpcio-tools are available:

    python -m grpc_tools.protoc -I lightx2v/server/ws/proto \\
        --python_out=lightx2v/server/ws/pb \\
        lightx2v/server/ws/proto/lightx2v_v1.proto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

WT_VARINT = 0
WT_LEN = 2


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint must be non-negative")
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def _decode_varint(buf: bytes, idx: int) -> tuple[int, int]:
    shift = 0
    result = 0
    while idx < len(buf):
        b = buf[idx]
        idx += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, idx
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")
    raise ValueError("truncated varint")


def _key(field: int, wire_type: int) -> bytes:
    return _encode_varint((field << 3) | wire_type)


def _enc_bytes(field: int, data: bytes) -> bytes:
    if not data:
        return b""
    return _key(field, WT_LEN) + _encode_varint(len(data)) + data


def _enc_str(field: int, value: str) -> bytes:
    if not value:
        return b""
    return _enc_bytes(field, value.encode("utf-8"))


def _enc_varint_field(field: int, value: int) -> bytes:
    if value == 0:
        return b""
    return _key(field, WT_VARINT) + _encode_varint(value)


def _enc_bool(field: int, value: bool) -> bytes:
    if not value:
        return b""
    return _key(field, WT_VARINT) + _encode_varint(1)


def _enc_msg(field: int, payload: bytes) -> bytes:
    return _key(field, WT_LEN) + _encode_varint(len(payload)) + payload


def _iter_fields(buf: bytes):
    idx = 0
    n = len(buf)
    while idx < n:
        key, idx = _decode_varint(buf, idx)
        field = key >> 3
        wire_type = key & 0x7
        if wire_type == WT_VARINT:
            value, idx = _decode_varint(buf, idx)
            yield field, wire_type, value
        elif wire_type == WT_LEN:
            length, idx = _decode_varint(buf, idx)
            end = idx + length
            if end > n:
                raise ValueError("truncated length-delimited field")
            yield field, wire_type, buf[idx:end]
            idx = end
        else:
            raise ValueError(f"unsupported wire type {wire_type}")


@dataclass
class Empty:
    def serialize(self) -> bytes:
        return b""

    @classmethod
    def parse(cls, buf: bytes) -> "Empty":
        return cls()


@dataclass
class Start:
    request_id: str = ""
    audio_format: str = ""
    image_fromat: str = ""
    image_data: list[bytes] = field(default_factory=list)
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 0
    aspect_ratio: str = ""
    chunk_frames: int = 0

    def serialize(self) -> bytes:
        out = bytearray()
        out += _enc_str(1, self.request_id)
        out += _enc_str(2, self.audio_format)
        out += _enc_str(4, self.image_fromat)
        for item in self.image_data:
            if item:
                out += _enc_bytes(5, item)
        out += _enc_str(6, self.prompt)
        out += _enc_str(7, self.negative_prompt)
        out += _enc_varint_field(8, self.seed)
        out += _enc_str(9, self.aspect_ratio)
        out += _enc_varint_field(10, self.chunk_frames)
        return bytes(out)

    @classmethod
    def parse(cls, buf: bytes) -> "Start":
        msg = cls()
        for field, wt, value in _iter_fields(buf):
            if field == 1 and wt == WT_LEN:
                msg.request_id = value.decode("utf-8")
            elif field == 2 and wt == WT_LEN:
                msg.audio_format = value.decode("utf-8")
            elif field == 4 and wt == WT_LEN:
                msg.image_fromat = value.decode("utf-8")
            elif field == 5 and wt == WT_LEN:
                msg.image_data.append(value)
            elif field == 6 and wt == WT_LEN:
                msg.prompt = value.decode("utf-8")
            elif field == 7 and wt == WT_LEN:
                msg.negative_prompt = value.decode("utf-8")
            elif field == 8 and wt == WT_VARINT:
                msg.seed = value
            elif field == 9 and wt == WT_LEN:
                msg.aspect_ratio = value.decode("utf-8")
            elif field == 10 and wt == WT_VARINT:
                msg.chunk_frames = value
        return msg


@dataclass
class AudioChunk:
    sequence: int = 0
    data: bytes = b""

    def serialize(self) -> bytes:
        return _enc_varint_field(1, self.sequence) + _enc_bytes(3, self.data)

    @classmethod
    def parse(cls, buf: bytes) -> "AudioChunk":
        msg = cls()
        for field, wt, value in _iter_fields(buf):
            if field == 1 and wt == WT_VARINT:
                msg.sequence = value
            elif field == 3 and wt == WT_LEN:
                msg.data = value
        return msg


@dataclass
class Interrupt:
    reason: str = ""

    def serialize(self) -> bytes:
        return _enc_str(1, self.reason)

    @classmethod
    def parse(cls, buf: bytes) -> "Interrupt":
        msg = cls()
        for field, wt, value in _iter_fields(buf):
            if field == 1 and wt == WT_LEN:
                msg.reason = value.decode("utf-8")
        return msg


@dataclass
class SetPrompt:
    prompt: str = ""

    def serialize(self) -> bytes:
        return _enc_str(1, self.prompt)

    @classmethod
    def parse(cls, buf: bytes) -> "SetPrompt":
        msg = cls()
        for field, wt, value in _iter_fields(buf):
            if field == 1 and wt == WT_LEN:
                msg.prompt = value.decode("utf-8")
        return msg


@dataclass
class ServerEvent:
    START_SPEAKING = 0
    STOP_SPEAKING = 1
    PROMPT_UPDATE = 2

    type: int = 0
    message: str = ""

    def serialize(self) -> bytes:
        return _enc_varint_field(1, self.type) + _enc_str(2, self.message)

    @classmethod
    def parse(cls, buf: bytes) -> "ServerEvent":
        msg = cls()
        for field, wt, value in _iter_fields(buf):
            if field == 1 and wt == WT_VARINT:
                msg.type = value
            elif field == 2 and wt == WT_LEN:
                msg.message = value.decode("utf-8")
        return msg


@dataclass
class VideoStart:
    video_width: int = 0
    video_height: int = 0

    def serialize(self) -> bytes:
        return _enc_varint_field(1, self.video_width) + _enc_varint_field(2, self.video_height)

    @classmethod
    def parse(cls, buf: bytes) -> "VideoStart":
        msg = cls()
        for field, wt, value in _iter_fields(buf):
            if field == 1 and wt == WT_VARINT:
                msg.video_width = value
            elif field == 2 and wt == WT_VARINT:
                msg.video_height = value
        return msg


@dataclass
class VideoChunk:
    sequence: int = 0
    audio_data: bytes = b""
    video_data: bytes = b""

    def serialize(self) -> bytes:
        return _enc_varint_field(1, self.sequence) + _enc_bytes(2, self.audio_data) + _enc_bytes(3, self.video_data)

    @classmethod
    def parse(cls, buf: bytes) -> "VideoChunk":
        msg = cls()
        for field, wt, value in _iter_fields(buf):
            if field == 1 and wt == WT_VARINT:
                msg.sequence = value
            elif field == 2 and wt == WT_LEN:
                msg.audio_data = value
            elif field == 3 and wt == WT_LEN:
                msg.video_data = value
        return msg


@dataclass
class Failure:
    code: str = ""
    message: str = ""
    retryable: bool = False

    def serialize(self) -> bytes:
        return _enc_str(1, self.code) + _enc_str(2, self.message) + _enc_bool(3, self.retryable)

    @classmethod
    def parse(cls, buf: bytes) -> "Failure":
        msg = cls()
        for field, wt, value in _iter_fields(buf):
            if field == 1 and wt == WT_LEN:
                msg.code = value.decode("utf-8")
            elif field == 2 and wt == WT_LEN:
                msg.message = value.decode("utf-8")
            elif field == 3 and wt == WT_VARINT:
                msg.retryable = bool(value)
        return msg


@dataclass
class ClientMessage:
    start: Optional[Start] = None
    audio_start: Optional[Empty] = None
    audio: Optional[AudioChunk] = None
    audio_end: Optional[Empty] = None
    interrupt: Optional[Interrupt] = None
    set_prompt: Optional[SetPrompt] = None

    def which(self) -> Optional[str]:
        for name in ("start", "audio_start", "audio", "audio_end", "interrupt", "set_prompt"):
            if getattr(self, name) is not None:
                return name
        return None

    def serialize(self) -> bytes:
        mapping = (
            (1, self.start),
            (2, self.audio_start),
            (3, self.audio),
            (4, self.audio_end),
            (5, self.interrupt),
            (6, self.set_prompt),
        )
        for field, body in mapping:
            if body is not None:
                return _enc_msg(field, body.serialize())
        return b""

    @classmethod
    def parse(cls, buf: bytes) -> "ClientMessage":
        msg = cls()
        parsers = {
            1: ("start", Start),
            2: ("audio_start", Empty),
            3: ("audio", AudioChunk),
            4: ("audio_end", Empty),
            5: ("interrupt", Interrupt),
            6: ("set_prompt", SetPrompt),
        }
        for field, wt, value in _iter_fields(buf):
            if field in parsers and wt == WT_LEN:
                attr, typ = parsers[field]
                setattr(msg, attr, typ.parse(value))
        return msg


@dataclass
class ServerMessage:
    event: Optional[ServerEvent] = None
    video_start: Optional[VideoStart] = None
    video: Optional[VideoChunk] = None
    error: Optional[Failure] = None

    def which(self) -> Optional[str]:
        for name in ("event", "video_start", "video", "error"):
            if getattr(self, name) is not None:
                return name
        return None

    def serialize(self) -> bytes:
        mapping = (
            (1, self.event),
            (2, self.video_start),
            (3, self.video),
            (4, self.error),
        )
        for field, body in mapping:
            if body is not None:
                return _enc_msg(field, body.serialize())
        return b""

    @classmethod
    def parse(cls, buf: bytes) -> "ServerMessage":
        msg = cls()
        parsers = {
            1: ("event", ServerEvent),
            2: ("video_start", VideoStart),
            3: ("video", VideoChunk),
            4: ("error", Failure),
        }
        for field, wt, value in _iter_fields(buf):
            if field in parsers and wt == WT_LEN:
                attr, typ = parsers[field]
                setattr(msg, attr, typ.parse(value))
        return msg


def event_message(event_type: int, message: str = "") -> bytes:
    return ServerMessage(event=ServerEvent(type=event_type, message=message)).serialize()


def video_start_message(width: int, height: int) -> bytes:
    return ServerMessage(video_start=VideoStart(video_width=width, video_height=height)).serialize()


def video_message(sequence: int, audio_data: bytes, video_data: bytes) -> bytes:
    return ServerMessage(video=VideoChunk(sequence=sequence, audio_data=audio_data, video_data=video_data)).serialize()


def error_message(code: str, message: str, retryable: bool = False) -> bytes:
    return ServerMessage(error=Failure(code=code, message=message, retryable=retryable)).serialize()
