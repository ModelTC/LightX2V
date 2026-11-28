"""Volcengine V3 bidirectional TTS; streams text in and mono PCM S16LE out."""

import asyncio
import gzip
import json
import struct
import time
import uuid
from enum import IntEnum
from typing import AsyncIterator

import aiohttp
from loguru import logger

from .config import BilibiliConfig
from .logging_utils import log_preview


class _Event(IntEnum):
    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52
    START_SESSION = 100
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_CANCELED = 151
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    USAGE_RESPONSE = 154
    TASK_REQUEST = 200
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352


def _request(event: int, payload: dict, session_id: str = "") -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode()
    frame = b"\x11\x14\x10\x00" + struct.pack(">i", event)
    if session_id:
        identifier = session_id.encode()
        frame += struct.pack(">I", len(identifier)) + identifier
    return frame + struct.pack(">I", len(data)) + data


def _response(data: bytes) -> tuple[int, str, bytes | dict]:
    if len(data) < 4 or data[0] >> 4 != 1:
        raise RuntimeError("Invalid Volc TTS frame header")
    offset = (data[0] & 15) * 4
    kind, flags = data[1] >> 4, data[1] & 15
    serialization, compression = data[2] >> 4, data[2] & 15
    if offset < 4 or offset > len(data) or compression not in (0, 1) or serialization not in (0, 1):
        raise RuntimeError("Unsupported Volc TTS frame header")

    def take(size):
        nonlocal offset
        if offset + size > len(data):
            raise RuntimeError("Truncated Volc TTS frame")
        value = data[offset : offset + size]
        offset += size
        return value

    def uint32():
        return struct.unpack(">I", take(4))[0]

    if kind == 15 and flags == 0:
        code = uint32()
        take(uint32())
        if offset != len(data):
            raise RuntimeError("Unexpected trailing Volc TTS error frame data")
        raise RuntimeError("Volc TTS error frame (code %s)" % code)
    if kind not in (9, 11) or flags != 4:
        raise RuntimeError("Unsupported Volc TTS response type")
    event = uint32()
    identifier = take(uint32()).decode("utf-8")
    payload = take(uint32())
    if offset != len(data):
        raise RuntimeError("Unexpected trailing Volc TTS frame data")
    if compression:
        payload = gzip.decompress(payload)
    if event == _Event.TTS_RESPONSE:
        if kind != 11 or serialization != 0:
            raise RuntimeError("Volc TTS audio response must contain raw PCM")
    else:
        if kind != 9 or serialization != 1:
            raise RuntimeError("Volc TTS control response must contain JSON")
        payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid Volc TTS response payload")
        if event in (_Event.CONNECTION_FAILED, _Event.SESSION_CANCELED, _Event.SESSION_FAILED) or payload.get("status_code", 0) not in (0, 20000000):
            raise RuntimeError("Volc TTS session failed (event %s, code %s)" % (event, payload.get("status_code")))
    return event, identifier, payload


class VolcTTSClient:
    def __init__(self, session: aiohttp.ClientSession, config: BilibiliConfig):
        self.session = session
        self.config = config

    async def synthesize(self, texts: AsyncIterator[str | None]) -> AsyncIterator[bytes | None]:
        """None closes one text segment; yield None after its last audio arrives."""
        started = time.monotonic()
        audio_bytes = 0
        completed = False
        first_request_at = None
        headers = {"X-Api-Key": self.config.tts_api_key, "X-Api-Resource-Id": self.config.tts_resource_id, "X-Api-Connect-Id": str(uuid.uuid4())}
        parameters = {
            "user": {"uid": "lightx2v-bilibili"},
            "namespace": "BidirectionalTTS",
            "req_params": {"speaker": self.config.tts_voice, "audio_params": {"format": "pcm", "sample_rate": self.config.tts_sample_rate}},
        }
        logger.info(
            "[TTS] request voice={} resource={} sample_rate={}", log_preview(self.config.tts_voice, self.config), log_preview(self.config.tts_resource_id, self.config), self.config.tts_sample_rate
        )
        try:
            async with self.session.ws_connect(self.config.tts_url, headers=headers, heartbeat=20) as websocket:
                controls: asyncio.Queue[int] = asyncio.Queue()
                session_id = ""
                sender_error = None
                closing = False

                async def wait_event(expected):
                    event = await asyncio.wait_for(controls.get(), timeout=30)
                    if event != expected:
                        raise RuntimeError("Unexpected Volc TTS event %s (expected %s)" % (event, expected))

                async def send_text():
                    nonlocal session_id, sender_error, first_request_at
                    finishing = False
                    try:
                        await websocket.send_bytes(_request(_Event.START_CONNECTION, {}))
                        await wait_event(_Event.CONNECTION_STARTED)
                        async for text in texts:
                            if text is None:
                                if session_id and not finishing:
                                    await websocket.send_bytes(_request(_Event.FINISH_SESSION, {}, session_id))
                                    finishing = True
                                # Resume the LLM immediately so its tool can run during synthesis.
                                continue
                            if not text or ((not session_id or finishing) and not text.strip()):
                                continue
                            if finishing:
                                await wait_event(_Event.SESSION_FINISHED)
                                session_id = ""
                                finishing = False
                            if not session_id:
                                session_id = str(uuid.uuid4())
                                await websocket.send_bytes(_request(_Event.START_SESSION, {**parameters, "event": _Event.START_SESSION}, session_id))
                                await wait_event(_Event.SESSION_STARTED)
                            payload = {**parameters, "event": _Event.TASK_REQUEST, "req_params": {**parameters["req_params"], "text": text}}
                            if first_request_at is None:
                                first_request_at = time.monotonic()
                            await websocket.send_bytes(_request(_Event.TASK_REQUEST, payload, session_id))
                        if session_id:
                            if not finishing:
                                await websocket.send_bytes(_request(_Event.FINISH_SESSION, {}, session_id))
                            await wait_event(_Event.SESSION_FINISHED)
                        await websocket.send_bytes(_request(_Event.FINISH_CONNECTION, {}))
                        await wait_event(_Event.CONNECTION_FINISHED)
                    except (Exception, asyncio.CancelledError) as error:
                        sender_error = error
                        if not closing:
                            await websocket.close()
                        raise

                sender = asyncio.create_task(send_text())
                pending = b""
                receiving_audio = False
                session_bytes = 0
                try:
                    while True:
                        message = await websocket.receive()
                        if sender_error is not None:
                            raise sender_error
                        if message.type != aiohttp.WSMsgType.BINARY:
                            raise RuntimeError("Volc TTS websocket ended without complete audio")
                        event, identifier, payload = _response(message.data)
                        if event not in (_Event.CONNECTION_STARTED, _Event.CONNECTION_FAILED, _Event.CONNECTION_FINISHED) and identifier != session_id:
                            raise RuntimeError("Volc TTS returned a different session ID")
                        if event == _Event.SESSION_STARTED:
                            receiving_audio = True
                            session_bytes = 0
                        elif event == _Event.TTS_RESPONSE:
                            if not receiving_audio:
                                raise RuntimeError("Volc TTS audio arrived outside an active session")
                            pending += payload
                            length = len(pending) // 2 * 2
                            if length:
                                if not audio_bytes:
                                    logger.info("[TTS] first PCM latency_ms={:.0f}", (time.monotonic() - (first_request_at or started)) * 1000)
                                audio_bytes += length
                                session_bytes += length
                                yield pending[:length]
                                pending = pending[length:]
                        elif event == _Event.SESSION_FINISHED:
                            if not receiving_audio or pending or not session_bytes:
                                raise RuntimeError("Volc TTS session ended without complete PCM samples")
                            receiving_audio = False
                        elif event not in (_Event.CONNECTION_STARTED, _Event.CONNECTION_FINISHED, _Event.USAGE_RESPONSE, _Event.TTS_SENTENCE_START, _Event.TTS_SENTENCE_END):
                            raise RuntimeError("Unexpected Volc TTS event %s" % event)
                        if event in (_Event.CONNECTION_STARTED, _Event.SESSION_STARTED, _Event.SESSION_FINISHED, _Event.CONNECTION_FINISHED):
                            controls.put_nowait(event)
                        if event == _Event.SESSION_FINISHED:
                            yield None
                        if event == _Event.CONNECTION_FINISHED:
                            break
                    await sender
                    completed = True
                finally:
                    closing = True
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
        finally:
            logger.info(
                "[TTS] end completed={} bytes={} audio_seconds={:.3f} elapsed_ms={:.0f}", completed, audio_bytes, audio_bytes / (2 * self.config.tts_sample_rate), (time.monotonic() - started) * 1000
            )
