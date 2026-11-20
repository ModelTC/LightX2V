"""Bilibili Open Live HTTP signing, websocket events, and both heartbeats."""

import asyncio
import hashlib
import hmac
import json
import struct
import time
import uuid
import zlib
from dataclasses import dataclass, field
from typing import Awaitable, Callable
from urllib.parse import urlsplit

import aiohttp
from loguru import logger

from .config import BilibiliConfig
from .logging_utils import log_preview


@dataclass
class BiliEvent:
    kind: str
    data: dict
    received_at: float = field(default_factory=time.time)


def pack_packet(body: str = "", operation: int = 2) -> bytes:
    payload = body.encode("utf-8")
    return struct.pack(">IHHII", 16 + len(payload), 16, 0, operation, 1) + payload


def unpack_packets(data: bytes) -> list[tuple[int, bytes]]:
    packets = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < 16:
            raise ValueError("Incomplete Bilibili packet header")
        length, header, version, operation, _ = struct.unpack_from(">IHHII", data, offset)
        if header < 16 or length < header or offset + length > len(data):
            raise ValueError("Invalid Bilibili packet length")
        payload = data[offset + header : offset + length]
        if version == 2:
            packets.extend(unpack_packets(zlib.decompress(payload)))
        elif version == 3:
            import brotli

            packets.extend(unpack_packets(brotli.decompress(payload)))
        else:
            packets.append((operation, payload))
        offset += length
    return packets


def parse_event(payload: bytes) -> BiliEvent | None:
    message = json.loads(payload)
    command = message.get("cmd", "")
    data = message.get("data") or {}
    kinds = {
        "LIVE_OPEN_PLATFORM_DM": "danmaku",
        "LIVE_OPEN_PLATFORM_LIVE_ROOM_ENTER": "room_enter",
        "LIVE_OPEN_PLATFORM_SEND_GIFT": "gift",
        "LIVE_OPEN_PLATFORM_SUPER_CHAT": "super_chat",
        "LIVE_OPEN_PLATFORM_INTERACTION_END": "interaction_end",
    }
    kind = kinds.get(command)
    if not kind or (kind == "danmaku" and data.get("dm_type", 0) != 0):
        return None
    return BiliEvent(kind, data)


class BilibiliClient:
    def __init__(self, session: aiohttp.ClientSession, config: BilibiliConfig, on_event: Callable[[BiliEvent], Awaitable[None]]):
        self.session = session
        self.config = config
        self.on_event = on_event
        self.game_id = ""
        self.running = True
        self.connected = asyncio.Event()

    def signed_headers(self, body: str) -> dict[str, str]:
        headers = {
            "x-bili-accesskeyid": self.config.access_key,
            "x-bili-content-md5": hashlib.md5(body.encode("utf-8")).hexdigest(),
            "x-bili-signature-method": "HMAC-SHA256",
            "x-bili-signature-nonce": uuid.uuid4().hex,
            "x-bili-signature-version": "1.0",
            "x-bili-timestamp": str(int(time.time())),
        }
        canonical = "\n".join(key + ":" + headers[key] for key in sorted(headers))
        headers["Authorization"] = hmac.new(self.config.access_secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        return headers

    async def _post(self, path: str, data: dict) -> dict:
        started = time.monotonic()
        logger.debug("[Bili] HTTP 请求 | path={}", path)
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        async with self.session.post(
            self.config.host + path,
            data=body.encode("utf-8"),
            headers=self.signed_headers(body),
            timeout=aiohttp.ClientTimeout(total=20),
        ) as response:
            if response.status != 200:
                raise RuntimeError("Bilibili API %s failed (HTTP %s)" % (path, response.status))
            result = await response.json()
        if result.get("code") != 0:
            code = result.get("code")
            detail = "Accept must be application/json" if code == 4013 else str(result.get("message") or result.get("msg") or "unknown error")
            for secret in (self.config.id_code, self.config.access_key, self.config.access_secret):
                if secret:
                    detail = detail.replace(secret, "<redacted>")
            raise RuntimeError("Bilibili API %s failed (code %s: %s)" % (path, code, detail.replace("\n", " ")[:200]))
        logger.debug("[Bili] HTTP 成功 | path={} | elapsed={:.3f}s", path, time.monotonic() - started)
        return result.get("data") or {}

    async def _end_app(self) -> None:
        game_id, self.game_id = self.game_id, ""
        if game_id:
            try:
                await asyncio.wait_for(self._post("/v2/app/end", {"game_id": game_id, "app_id": self.config.app_id}), timeout=3)
                logger.info("[Bili] 互动场次已关闭")
            except Exception as error:
                logger.warning("[Bili] 关闭互动场次失败: {}", log_preview(error, self.config, 300))

    async def _heartbeat(self, websocket: aiohttp.ClientWebSocketResponse, application: bool) -> None:
        while self.running:
            await asyncio.sleep(20)
            if application:
                await self._post("/v2/app/heartbeat", {"game_id": self.game_id})
                logger.info("[Bili] 应用心跳成功 | websocket_connected={}", self.connected.is_set())
            else:
                await websocket.send_bytes(pack_packet())

    async def _receive(self, websocket: aiohttp.ClientWebSocketResponse) -> None:
        async for message in websocket:
            if message.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError("Bilibili websocket failed")
            if message.type != aiohttp.WSMsgType.BINARY:
                continue
            for operation, payload in unpack_packets(message.data):
                if operation != 5:
                    continue
                event = parse_event(payload)
                if event is not None:
                    await self.on_event(event)
                    if event.kind == "interaction_end":
                        self.running = False
                        return

    async def _connect(self, link: str, auth_body: str) -> None:
        logger.info("[Bili] 连接 WebSocket | host={}", log_preview(urlsplit(link).hostname, self.config))
        async with self.session.ws_connect(link, max_msg_size=10 * 1024 * 1024) as websocket:
            # The platform supplies the complete authentication body; forward it verbatim.
            await websocket.send_bytes(pack_packet(auth_body, operation=7))
            auth = await asyncio.wait_for(websocket.receive(), timeout=15)
            if auth.type != aiohttp.WSMsgType.BINARY:
                raise RuntimeError("Bilibili authentication response was not binary")
            authenticated = False
            for operation, payload in unpack_packets(auth.data):
                if operation == 8:
                    if json.loads(payload).get("code") != 0:
                        raise RuntimeError("Bilibili websocket authentication failed")
                    authenticated = True
                elif operation == 5:
                    event = parse_event(payload)
                    if event is not None:
                        await self.on_event(event)
            if not authenticated:
                raise RuntimeError("Bilibili websocket authentication acknowledgement missing")
            logger.info("[Bili] WebSocket 认证成功，开始接收直播事件")
            self.connected.set()
            tasks = [asyncio.create_task(self._receive(websocket)), asyncio.create_task(self._heartbeat(websocket, False)), asyncio.create_task(self._heartbeat(websocket, True))]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                self.connected.clear()
                logger.info("[Bili] WebSocket 连接结束")
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self) -> None:
        try:
            while self.running:
                try:
                    logger.info("[Bili] 请求开启互动场次 | host={}", log_preview(urlsplit(self.config.host).hostname, self.config))
                    info = await self._post("/v2/app/start", {"code": self.config.id_code, "app_id": self.config.app_id})
                    self.game_id = str(info["game_info"]["game_id"])
                    websocket_info = info["websocket_info"]
                    links = websocket_info["wss_link"]
                    auth_body = websocket_info["auth_body"]
                    logger.info(
                        "[Bili] HTTP 开场成功 | room_id={} | websocket_endpoints={}",
                        log_preview((info.get("anchor_info") or {}).get("room_id", "unknown"), self.config),
                        len(links or []),
                    )
                    if not links:
                        raise RuntimeError("Bilibili returned no websocket endpoints")
                    for link in links:
                        try:
                            await self._connect(link, auth_body)
                            break
                        except asyncio.CancelledError:
                            raise
                        except Exception as error:
                            logger.warning("[Bili] WebSocket 断开: {}", type(error).__name__)
                    if self.running:
                        await self._end_app()
                        await asyncio.sleep(5)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning("[Bili] 连接失败: {} | 5 秒后重试", log_preview(error, self.config, 300))
                    await self._end_app()
                    await asyncio.sleep(5)
        finally:
            self.running = False
            await self._end_app()
            logger.info("[Bili] 连接任务已停止")
