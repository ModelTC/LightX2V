"""verl-omni custom scorer; model libraries live in a separate reward service."""

import argparse
import asyncio
import json
import math
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from lightx2v_rl.rewards.common import (
    DEFAULT_WEIGHTS,
    PROTOCOL_VERSION,
    encode_request,
    resolve_source_fps,
    validate_metrics,
    video_to_tchw,
)


class RemoteRewardError(RuntimeError):
    """The requested reward was not computed; never substitute a zero reward."""


def _request(endpoint, route, *, payload=None, timeout_seconds=1800.0):
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("Reward endpoint must be an http(s) base URL")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    request = Request(
        endpoint.rstrip("/") + route, data=payload,
        headers={"Content-Type": "application/x-npz"} if payload is not None else {},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read(1024 * 1024))
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RemoteRewardError(f"Reward service {endpoint} returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RemoteRewardError(f"Reward service {endpoint} failed: {exc}") from exc
    if not isinstance(result, dict) or result.get("version") != PROTOCOL_VERSION:
        raise RemoteRewardError("Reward service returned an unsupported protocol response")
    return result


def healthcheck(endpoint="http://127.0.0.1:8791", timeout_seconds=10.0):
    return _request(endpoint, "/health", timeout_seconds=timeout_seconds)


def aggregate_scores(scores, weights, aggregation="mean"):
    """FewStepRL multi_score: weighted terms averaged by count, not weight sum."""
    if aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation must be mean (FewStepRL) or sum")
    result = {}
    for name, weight in weights.items():
        try:
            raw = float(scores[name])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RemoteRewardError(f"Reward service omitted or returned invalid {name}") from exc
        if not math.isfinite(raw):
            raise RemoteRewardError(f"Reward service returned nonfinite {name}")
        result[name + "_raw"] = raw
        result[name] = raw * weight
    total = sum(result[name] for name in weights)
    result["score"] = total / len(weights) if aggregation == "mean" else total
    if any(not math.isfinite(value) for value in result.values()):
        raise RemoteRewardError("Weighted reward overflowed")
    return result


def compute_score(
    data_source,
    solution_image,
    ground_truth,
    extra_info=None,
    *,
    endpoint="http://127.0.0.1:8791",
    weights=None,
    aggregation="mean",
    source_fps=16.0,
    timeout_seconds=1800.0,
    video_layout="TCHW",
):
    """Score one uint8 video using its text prompt in reward_model.ground_truth.

    Returns upstream's scalar ``score``, plus every weighted and raw metric for
    logging. This does not implement FewStepRL's per-reward advantage estimator.
    """
    payload, weights = _prepare_score_request(solution_image, ground_truth, extra_info, weights,
                                             aggregation, source_fps, video_layout)
    result = _request(endpoint, "/score", payload=payload, timeout_seconds=timeout_seconds)
    return _parse_score_response(result, weights, aggregation)


def _prepare_score_request(solution_image, ground_truth, extra_info, weights, aggregation, source_fps, video_layout):
    weights = DEFAULT_WEIGHTS if weights is None else weights
    if not isinstance(weights, Mapping):
        raise ValueError("weights must map supported reward names to finite weights")
    metrics = validate_metrics(list(weights))
    weights = {name: float(weights[name]) for name in metrics}
    if any(not math.isfinite(weight) for weight in weights.values()) or not any(weights.values()):
        raise ValueError("Reward weights must be finite, with at least one nonzero weight")
    if aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation must be mean or sum")
    video = video_to_tchw(solution_image, video_layout)
    payload = encode_request(video, ground_truth, metrics, resolve_source_fps(extra_info, source_fps))
    return payload, weights


def _parse_score_response(result, weights, aggregation):
    if not isinstance(result, dict) or result.get("version") != PROTOCOL_VERSION:
        raise RemoteRewardError("Reward service returned an unsupported protocol response")
    scores = result.get("scores")
    if not isinstance(scores, dict):
        raise RemoteRewardError("Reward service response is missing scores")
    return aggregate_scores(scores, weights, aggregation)


async def compute_score_async(data_source, solution_image, ground_truth, extra_info=None, *, session,
                              endpoint="http://127.0.0.1:8791", weights=None, aggregation="mean",
                              source_fps=16.0, timeout_seconds=1800.0, video_layout="TCHW"):
    """Async transport with the same NPZ protocol and score semantics as Wan."""
    from lightx2v_rl.rewards.async_http import post_json

    payload, weights = await asyncio.to_thread(
        _prepare_score_request, solution_image, ground_truth, extra_info,
        weights, aggregation, source_fps, video_layout,
    )
    result = await post_json(session, endpoint, "/score", payload=payload,
                             content_type="application/x-npz", timeout_seconds=timeout_seconds)
    return _parse_score_response(result, weights, aggregation)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check that the isolated reward service is ready")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8791")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(healthcheck(args.endpoint, args.timeout_seconds), indent=2))
    except (ValueError, RemoteRewardError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
