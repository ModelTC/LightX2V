"""Volcengine HTTP V3 streaming TTS; emits mono PCM S16LE without SDKs."""

import base64
import json
import time
import uuid
from typing import AsyncIterator

import aiohttp
from loguru import logger

from .config import BilibiliConfig
from .logging_utils import log_preview


class VolcTTSClient:
    def __init__(self, session: aiohttp.ClientSession, config: BilibiliConfig):
        self.session = session
        self.config = config

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        started = time.monotonic()
        chunks = audio_bytes = 0
        completed = False
        logger.info(
            "[TTS] request voice={} resource={} sample_rate={} text_chars={} text={}",
            log_preview(self.config.tts_voice, self.config),
            log_preview(self.config.tts_resource_id, self.config),
            log_preview(self.config.tts_sample_rate, self.config),
            len(text),
            log_preview(text, self.config),
        )
        headers = {"X-Api-Key": self.config.tts_api_key, "X-Api-Resource-Id": self.config.tts_resource_id, "X-Api-Request-Id": str(uuid.uuid4())}
        payload = {
            "user": {"uid": "lightx2v-bilibili"},
            "req_params": {"text": text, "speaker": self.config.tts_voice, "audio_params": {"format": "pcm", "sample_rate": self.config.tts_sample_rate}},
        }
        pending = b""
        received = False
        finished = False
        try:
            async with self.session.post(self.config.tts_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=180, sock_read=30)) as response:
                if response.status != 200:
                    raise RuntimeError("Volc TTS request failed (HTTP %s)" % response.status)
                async for raw in response.content:
                    line = raw.strip()
                    if line.startswith(b"data:"):
                        line = line[5:].strip()
                    if not line or line.startswith((b":", b"event:", b"id:")):
                        continue
                    data = json.loads(line)
                    code = int(data.get("code", 0))
                    if code not in (0, 20000000):
                        raise RuntimeError("Volc TTS failed (code %s)" % code)
                    if data.get("data"):
                        pending += base64.b64decode(data["data"], validate=True)
                        length = len(pending) // 2 * 2
                        if length:
                            if not received:
                                logger.info("[TTS] first PCM latency_ms={:.0f}", (time.monotonic() - started) * 1000)
                            received = True
                            chunks += 1
                            audio_bytes += length
                            yield pending[:length]
                            pending = pending[length:]
                    if code == 20000000:
                        finished = True
                        break
            if pending:
                raise RuntimeError("Volc TTS returned a partial PCM sample")
            if not finished or not received:
                raise RuntimeError("Volc TTS stream ended without complete audio")
            completed = True
        finally:
            sample_rate = self.config.tts_sample_rate
            audio_seconds = audio_bytes / (sample_rate * 2) if isinstance(sample_rate, (int, float)) and sample_rate > 0 else 0.0
            logger.info(
                "[TTS] end completed={} chunks={} bytes={} audio_seconds={:.3f} elapsed_ms={:.0f}",
                completed,
                chunks,
                audio_bytes,
                audio_seconds,
                (time.monotonic() - started) * 1000,
            )
