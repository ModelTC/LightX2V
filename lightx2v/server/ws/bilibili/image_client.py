"""Client for the existing LightX2V task-based image editing service."""

import asyncio
import base64
import io
import time
import uuid
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp
from loguru import logger

from .config import BilibiliConfig
from .logging_utils import log_preview


class ImageTaskClient:
    REQUEST_INTERVAL = 0.1
    MAX_RATE_LIMIT_RETRIES = 3

    def __init__(self, session: aiohttp.ClientSession, config: BilibiliConfig, output_dir: str):
        self.session = session
        self.config = config
        self.output_dir = Path(output_dir)
        self._request_lock = asyncio.Lock()
        self._next_request_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.config.image_base_url and self.config.image_api_key)

    async def _json(self, method: str, path: str, **kwargs) -> dict:
        return await self._request(
            method,
            self.config.image_base_url + path,
            headers={"Authorization": "Bearer " + self.config.image_api_key},
            as_json=True,
            **kwargs,
        )

    async def _request(self, method: str, url: str, *, headers: dict, as_json=False, timeout=30, error_name="Image task service", **kwargs):
        # The service shares one user-level limit across submit/query/result
        # and authenticated downloads, so polling sleeps alone are insufficient.
        for attempt in range(self.MAX_RATE_LIMIT_RETRIES + 1):
            async with self._request_lock:
                wait = self._next_request_at - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                delay = self.REQUEST_INTERVAL
                try:
                    async with self.session.request(method, url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout), **kwargs) as response:
                        if response.status == 429 and attempt < self.MAX_RATE_LIMIT_RETRIES:
                            delay = self._retry_delay(response.headers.get("Retry-After"), attempt)
                            logger.warning(
                                "[Bili] 图片请求限流 | path={} | retry={}/{} | wait={:.2f}s",
                                log_preview(urlsplit(url).path, self.config, 160),
                                attempt + 1,
                                self.MAX_RATE_LIMIT_RETRIES,
                                delay,
                            )
                            continue
                        if response.status != 200:
                            raise RuntimeError("%s failed (HTTP %s)" % (error_name, response.status))
                        return await response.json() if as_json else await response.read()
                finally:
                    # Also preserve spacing after errors. Retries and other
                    # concurrent calls share the same backoff deadline.
                    self._next_request_at = time.monotonic() + delay

    @classmethod
    def _retry_delay(cls, retry_after: str | None, attempt: int) -> float:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                delay = 0.5 * 2**attempt
        return max(cls.REQUEST_INTERVAL, delay)

    async def edit(self, prompt: str, image_path: str) -> str:
        if not self.enabled:
            raise ValueError("Set X2V_API_BASE_URL and X2V_API_KEY to enable image editing")
        # Bound the entire operation, including a successful-but-never-finishing poll.
        return await asyncio.wait_for(self._edit(prompt, image_path), timeout=self.config.image_timeout)

    async def _edit(self, prompt: str, image_path: str) -> str:
        from PIL import Image

        image = await asyncio.to_thread(Path(image_path).read_bytes)
        with Image.open(io.BytesIO(image)) as source:
            custom_shape = [source.height, source.width]
        payload = {
            "task": "i2i",
            "model_cls": self.config.image_model,
            "stage": "single_stage",
            "prompt": "仅需要改变：%s，保留人物身份和其余画面内容" % prompt,
            "seed": time.time_ns() % (2**30),
            "custom_shape": custom_shape,
            "input_image": {"type": "base64", "data": base64.b64encode(image).decode("ascii")},
        }
        submitted = await self._json("POST", "/api/v1/task/submit", json=payload)
        task_id = submitted["task_id"]
        while True:
            result = await self._json("GET", "/api/v1/task/query", params={"task_id": task_id})
            status = str(result.get("status", "")).upper()
            if status == "SUCCEED":
                outputs = result.get("outputs") or {}
                if not outputs:
                    raise RuntimeError("Image task succeeded without an output")
                name = next(iter(outputs))
                break
            if status not in {"CREATED", "PENDING", "RUNNING"}:
                raise RuntimeError("Image task ended with status %s" % status)
            await asyncio.sleep(self.config.image_poll_interval)
        result = await self._json("GET", "/api/v1/task/result_url", params={"task_id": task_id, "name": name})
        url = urljoin(self.config.image_base_url + "/", result["url"])
        headers = {}
        if urlsplit(url)[:2] == urlsplit(self.config.image_base_url)[:2]:
            headers["Authorization"] = "Bearer " + self.config.image_api_key
        content = await self._request("GET", url, headers=headers, timeout=60, error_name="Image download")
        path = self.output_dir / ("avatar_" + uuid.uuid4().hex + ".png")
        await asyncio.to_thread(self._save_image, content, path)
        return str(path)

    @staticmethod
    def _save_image(content: bytes, path: Path) -> None:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(io.BytesIO(content)) as image:
            image.convert("RGB").save(path, format="PNG")
