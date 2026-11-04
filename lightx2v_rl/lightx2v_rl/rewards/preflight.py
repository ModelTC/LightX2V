"""Run real generation and reference-guided editing UnifiedReward probes."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time

import aiohttp
from PIL import Image, ImageDraw

from lightx2v_rl.rewards import unifiedreward
from lightx2v_rl.rewards.client import RemoteRewardError
from lightx2v_rl.rewards.endpoint_pool import normalize_endpoints

DEFAULT_TIMEOUT_SECONDS = 300.0
TIMEOUT_ENV = "LIGHTX2V_REWARD_PREFLIGHT_TIMEOUT"


def _timeout_seconds(value):
    if value is None:
        value = os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS)
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{TIMEOUT_ENV}/--timeout must be positive and finite") from exc
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise ValueError(f"{TIMEOUT_ENV}/--timeout must be positive and finite")
    return result


def _probe_image(width, height, circle_color):
    """Use representative image sizes; score quality is not a readiness criterion."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (width, height)):
        raise ValueError("Reward preflight image dimensions must be positive integers")
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((width // 8, height // 4, width // 2, 3 * height // 4), fill=circle_color)
    draw.rectangle((5 * width // 8, height // 4, 7 * width // 8, 3 * height // 4), fill="green")
    return image


def _progress(endpoint, task, status, elapsed, **details):
    # Preflight also runs before logging is configured by the training entrypoint.
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    print(
        f"[reward-preflight] endpoint={endpoint} task={task} status={status} elapsed={elapsed:.2f}s {fields}".rstrip(),
        file=sys.stderr,
        flush=True,
    )


async def check_reward_scoring_async(settings, *, width=2048, height=2048, reference_size=1024, i2i_source_count=1, timeout_seconds=None):
    """Score both tasks per replica, parallel across replicas and sequential within.

    Each request has a wall-clock deadline including encoding and real completion
    parsing. No retries or score thresholds are used. Malformed completions use
    the same visible midpoint fallback as training; transport failures and
    deadlines fail readiness. A replica that fails T2I is not sent another
    request; other replicas finish their bounded probes so failures identify
    all affected endpoints.
    """
    if settings.get("unifiedreward_weight", 0) <= 0:
        return {}
    if isinstance(i2i_source_count, bool) or not isinstance(i2i_source_count, int) or i2i_source_count < 1:
        raise ValueError("i2i_source_count must be a positive integer (canvas plus donor references)")
    endpoints = tuple(dict.fromkeys(normalize_endpoints(settings["unifiedreward_endpoint"], name="unifiedreward_endpoint")))
    timeout = _timeout_seconds(timeout_seconds)
    max_tokens = settings.get("max_tokens", 4096)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    generated = _probe_image(width, height, "blue")
    reference = _probe_image(reference_size, reference_size, "red")
    edit_sources = (reference,)
    edit_prompt = "Change the red circle to blue and preserve the green square and white background."
    if i2i_source_count > 1:
        edit_sources += tuple(_probe_image(reference_size, reference_size, "blue") for _ in range(i2i_source_count - 1))
        edit_prompt = (
            "On <image1> (canvas), replace the red circle with the blue circle from <image2>. "
            "Preserve the green square and white background of <image1>. "
            "Any additional reference images supply the same blue-circle appearance."
        )

    async def probe_endpoint(endpoint, session):
        result = {}
        for task, prompt, sources in (
            ("t2i", "A blue circle on the left and a green square on the right, on a white background.", ()),
            ("i2i", edit_prompt, edit_sources),
        ):
            started = time.monotonic()
            _progress(
                endpoint,
                task,
                "start",
                0,
                images=1 + len(sources),
                generated=f"{width}x{height}",
                reference=f"{reference_size}x{reference_size}" if sources else "none",
                timeout=f"{timeout:g}s",
            )
            try:
                scores = await asyncio.wait_for(
                    unifiedreward.score_image_async(
                        task,
                        prompt,
                        generated,
                        sources,
                        session=session,
                        endpoint=endpoint,
                        timeout_seconds=timeout,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                reason = f"deadline exceeded ({timeout:g}s)" if isinstance(exc, asyncio.TimeoutError) else f"{type(exc).__name__}: {exc}"
                _progress(endpoint, task, "failure", elapsed, reason=reason)
                raise RemoteRewardError(f"UnifiedReward preflight endpoint={endpoint} task={task} failed after {elapsed:.2f}s: {reason}") from exc
            elapsed = time.monotonic() - started
            _progress(endpoint, task, "format_fallback" if scores.get("format_error") else "success", elapsed, normalized=scores["normalized"])
            result[task] = {"elapsed_seconds": elapsed, "scores": scores}
        return result

    # One active completion per replica, even when the pool contains six servers.
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=len(endpoints))) as session:
        results = await asyncio.gather(*(probe_endpoint(endpoint, session) for endpoint in endpoints), return_exceptions=True)
    failures = [str(result) for result in results if isinstance(result, BaseException)]
    if failures:
        raise RemoteRewardError("Reward scoring preflight failed:\n" + "\n".join(failures))
    return {"unifiedreward_scoring": dict(zip(endpoints, results))}


def check_reward_scoring(settings, **kwargs):
    """Synchronous entrypoint for startup checks before Ray or an event loop starts."""
    return asyncio.run(check_reward_scoring_async(settings, **kwargs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True, help="UnifiedReward base URL; repeat for every replica")
    parser.add_argument("--timeout", type=float, help=f"Seconds per task, including response parsing (default: ${TIMEOUT_ENV} or 300)")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--reference-size", type=int, default=1024)
    parser.add_argument("--i2i-source-count", type=int, default=1, help="Number of editing inputs, including the canvas (2 for canvas + one donor)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        report = check_reward_scoring(
            {"unifiedreward_weight": 1.0, "unifiedreward_endpoint": args.endpoint, "max_tokens": args.max_tokens},
            width=args.width,
            height=args.height,
            reference_size=args.reference_size,
            i2i_source_count=args.i2i_source_count,
            timeout_seconds=args.timeout,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Preflight failed: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
