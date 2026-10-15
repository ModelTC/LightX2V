"""Joint T2I/I2I reward: task-aware UnifiedReward-Edit + 0.5 * HPSv3."""

import asyncio
import math
from collections.abc import Mapping

import aiohttp

from lightx2v_rl.rewards import client as video_client
from lightx2v_rl.rewards import unifiedreward
from lightx2v_rl.rewards.async_http import gather_scores, scoring_limiter, validate_timeout
from lightx2v_rl.rewards.endpoint_pool import normalize_endpoints, select_endpoint


async def compute_score(data_source, solution_image, ground_truth, extra_info=None, *,
                  unifiedreward_endpoint="http://127.0.0.1:8792", hps_endpoint="http://127.0.0.1:8791",
                  unifiedreward_weight=1.0, hps_weight=0.5, hps_offset=0.0, hps_scale=1.0,
                  timeout_seconds=1800.0, image_layout="CHW", max_tokens=4096, max_concurrent_samples=8):
    """Score one image, retaining separate raw, normalized and weighted metrics.

    T2I UR = (mean(alignment, coherence, style)-1)/4.
    I2I UR = mean(edit_success, preservation)/25.
    Total = unifiedreward_weight * UR + hps_weight * (HPS_mu-hps_offset)/hps_scale.
    HPS_mu is unbounded: weight 0.5 does not imply half the effective influence.
    Reference paths are resolved by the dataset/runtime before calling this hook.
    Each endpoint accepts one URL or a nonempty list of equivalent replicas;
    lists use process-local round robin with a distributed worker starting offset.
    UR/HPS run concurrently through upstream VisualRewardManager's async hook.
    A call owns its HTTP session; at most max_concurrent_samples calls per worker
    event loop enter image preparation and HTTP scoring at once.
    """
    import numpy as np

    if not isinstance(extra_info, Mapping):
        raise ValueError("Image reward requires extra_info.task_type and extra_info.source_images")
    task_type = extra_info.get("task_type")
    sources = extra_info.get("source_images", [])
    if task_type not in {"t2i", "i2i"}:
        raise ValueError("Image reward requires task_type t2i or i2i")
    if isinstance(sources, (str, bytes)):
        raise ValueError("source_images must be a list, not a string")
    sources = list(sources)
    if (task_type == "t2i" and sources) or (task_type == "i2i" and len(sources) != 1):
        raise ValueError("The default image reward supports zero T2I / one I2I reference")
    parameters = (unifiedreward_weight, hps_weight, hps_offset, hps_scale)
    if any(not math.isfinite(value) for value in parameters) or hps_scale <= 0:
        raise ValueError("Reward weights/calibration must be finite and hps_scale positive")
    if unifiedreward_weight < 0 or hps_weight < 0 or unifiedreward_weight + hps_weight <= 0:
        raise ValueError("Reward weights must be nonnegative with at least one active reward")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError("ground_truth must contain the original generation/edit instruction")
    ur_pool = normalize_endpoints(unifiedreward_endpoint, name="unifiedreward_endpoint") if unifiedreward_weight else None
    hps_pool = normalize_endpoints(hps_endpoint, name="hps_endpoint") if hps_weight else None
    validate_timeout(timeout_seconds)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

    def prepare_images():
        image = unifiedreward.image_to_rgb(solution_image, image_layout)
        references = [unifiedreward.image_to_rgb(source) for source in sources] if ur_pool else []
        video = np.asarray(image).transpose(2, 0, 1)[None, ...] if hps_pool else None
        return image, references, video

    async def score_ur(session, image, references):
        endpoint = select_endpoint(ur_pool, name="unifiedreward_endpoint")
        try:
            scores = await unifiedreward.score_image_async(
                task_type, ground_truth, image, references, session=session,
                endpoint=endpoint, timeout_seconds=timeout_seconds, max_tokens=max_tokens,
            )
        except video_client.RemoteRewardError as exc:
            raise video_client.RemoteRewardError(f"UnifiedReward replica {endpoint} failed: {exc}") from exc
        return {**{"unifiedreward_" + key: value for key, value in scores.items()},
                "unifiedreward": unifiedreward_weight * scores["normalized"]}

    async def score_hps(session, video):
        endpoint = select_endpoint(hps_pool, name="hps_endpoint")
        try:
            scores = await video_client.compute_score_async(
                data_source, video, ground_truth, extra_info, session=session,
                endpoint=endpoint, weights={"hpsv3_general": 1.0}, aggregation="sum",
                timeout_seconds=timeout_seconds,
            )
        except video_client.RemoteRewardError as exc:
            raise video_client.RemoteRewardError(f"HPSv3 replica {endpoint} failed: {exc}") from exc
        raw = scores["hpsv3_general_raw"]
        calibrated = (raw - hps_offset) / hps_scale
        return {"hpsv3_general_raw": raw, "hpsv3_general_calibrated": calibrated,
                "hpsv3_general": hps_weight * calibrated}

    async with scoring_limiter(max_concurrent_samples):
        image, references, video = await asyncio.to_thread(prepare_images)
        # Per-call ownership avoids leaked sessions and cross-loop reuse in Ray.
        # The connector permits both independent reward requests simultaneously.
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=2)) as session:
            requests = []
            if ur_pool:
                requests.append(score_ur(session, image, references))
            if hps_pool:
                requests.append(score_hps(session, video))
            parts = await gather_scores(*requests)
    result = {key: value for part in parts for key, value in part.items()}
    result["score"] = result.get("unifiedreward", 0.0) + result.get("hpsv3_general", 0.0)
    # Observability only: label the actual scored task, including HPS-only runs.
    # The trainer consumes `score`; this flag travels as reward_extra_info.
    result["task_is_i2i"] = float(task_type == "i2i")
    if any(not math.isfinite(value) for value in result.values()):
        raise video_client.RemoteRewardError("Image reward produced nonfinite weighted scores")
    return result
