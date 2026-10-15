"""Bounded async reward HTTP transport; sessions are owned by each scoring call."""

import asyncio
import json
import math
from weakref import WeakKeyDictionary, WeakValueDictionary

import aiohttp

from lightx2v_rl.rewards.client import RemoteRewardError
from lightx2v_rl.rewards.endpoint_pool import normalize_endpoints

# Weak values avoid retaining a closed event loop through a contended semaphore.
# Every active scoring call holds its semaphore until it releases its slot.
_limiters = WeakKeyDictionary()
MAX_RESPONSE_BYTES = 1024 * 1024


def scoring_limiter(limit):
    """Limit samples per worker/event loop, including CPU image preparation."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_concurrent_samples must be a positive integer")
    loop = asyncio.get_running_loop()
    limits = _limiters.setdefault(loop, WeakValueDictionary())
    limiter = limits.get(limit)
    if limiter is None:
        limiter = asyncio.Semaphore(limit)
        limits[limit] = limiter
    return limiter


def validate_timeout(timeout_seconds):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")


async def post_json(session, endpoint, route, *, payload, content_type, timeout_seconds):
    """Read bounded JSON with explicit HTTP errors and a per-request deadline."""
    normalize_endpoints(endpoint)
    validate_timeout(timeout_seconds)
    url = endpoint.rstrip("/") + route
    try:
        async with session.post(url, data=payload, headers={"Content-Type": content_type},
                                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
                                allow_redirects=False) as response:
            if not 200 <= response.status < 300:
                detail = (await response.content.read(4096)).decode("utf-8", errors="replace")
                raise RemoteRewardError(f"Reward service {endpoint} returned HTTP {response.status}: {detail}")
            body = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise RemoteRewardError(f"Reward service {endpoint} response exceeds 1 MiB")
        result = json.loads(body)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError) as exc:
        raise RemoteRewardError(f"Reward service {endpoint} failed: {exc}") from exc
    if not isinstance(result, dict):
        raise RemoteRewardError(f"Reward service {endpoint} returned a non-object JSON response")
    return result


async def gather_scores(*coroutines):
    """Fail together: drain cancelled siblings before the HTTP session closes.

    Cancelling HTTP requests does not guarantee cancellation of server-side GPU
    work. No retries are issued, and partial rewards are never returned.
    """
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
