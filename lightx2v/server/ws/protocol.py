"""Encode/decode lightx2v.v1 websocket protobuf messages.

Schema: proto/lightx2v_v1.proto
Generated stubs: pb/lightx2v_v1_pb2.py (regenerate via proto/generate_pb2.sh)
"""

from __future__ import annotations

from lightx2v.server.ws.pb.lightx2v_v1_pb2 import (
    AudioChunk,
    ClientMessage,
    Empty,
    Failure,
    Interrupt,
    ServerEvent,
    ServerMessage,
    SetPrompt,
    Start,
    VideoChunk,
    VideoStart,
)

__all__ = [
    "AudioChunk",
    "ClientMessage",
    "Empty",
    "Failure",
    "Interrupt",
    "ServerEvent",
    "ServerMessage",
    "SetPrompt",
    "Start",
    "VideoChunk",
    "VideoStart",
    "error_message",
    "event_message",
    "video_message",
    "video_start_message",
]


def _add_codec(cls):
    def serialize(self) -> bytes:
        return self.SerializeToString()

    def parse(buf: bytes):
        msg = cls()
        msg.ParseFromString(buf)
        return msg

    cls.serialize = serialize
    cls.parse = staticmethod(parse)


def _add_oneof(cls):
    def which(self):
        return self.WhichOneof("body")

    cls.which = which


for _cls in (
    Empty,
    Start,
    AudioChunk,
    Interrupt,
    SetPrompt,
    ServerEvent,
    VideoStart,
    VideoChunk,
    Failure,
    ClientMessage,
    ServerMessage,
):
    _add_codec(_cls)

_add_oneof(ClientMessage)
_add_oneof(ServerMessage)


def event_message(event_type: int, message: str = "") -> bytes:
    return ServerMessage(event=ServerEvent(type=event_type, message=message)).serialize()


def video_start_message(width: int, height: int) -> bytes:
    return ServerMessage(video_start=VideoStart(video_width=width, video_height=height)).serialize()


def video_message(sequence: int, audio_data: bytes, video_data: bytes) -> bytes:
    return ServerMessage(
        video=VideoChunk(sequence=sequence, audio_data=audio_data, video_data=video_data)
    ).serialize()


def error_message(code: str, message: str, retryable: bool = False) -> bytes:
    return ServerMessage(error=Failure(code=code, message=message, retryable=retryable)).serialize()
