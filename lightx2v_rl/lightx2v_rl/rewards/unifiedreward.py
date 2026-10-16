"""Task-aware UnifiedReward-Edit inference through its official vLLM API."""

import asyncio
import base64
import json
import logging
import math
import re
from io import BytesIO
from os import PathLike
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from lightx2v_rl.rewards._vendor.unifiedreward.prompts import I2I_PROMPT, T2I_PROMPT
from lightx2v_rl.rewards.client import RemoteRewardError

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LOGGER = logging.getLogger(__name__)


def image_to_rgb(image, layout="CHW"):
    """Keep rollout RGBA intact; composite a separate judge image onto white.

    Paths/PIL images and uint8 arrays/tensors are accepted. Floating point images
    are rejected to avoid silently guessing [-1,1] versus [0,1] scaling.
    """
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, PathLike)):
        with Image.open(image) as source:
            result = source.copy()
    elif isinstance(image, Image.Image):
        result = image.copy()
    else:
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise ValueError("Image reward expects uint8 pixels")
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if layout == "CHW" and array.ndim == 3:
            array = array.transpose(1, 2, 0)
        elif layout != "HWC":
            raise ValueError("image_layout must be CHW or HWC")
        if array.ndim != 3 or array.shape[2] not in {3, 4} or min(array.shape[:2]) < 1:
            raise ValueError("Image must have shape CHW/HWC with RGB or RGBA channels")
        result = Image.fromarray(array)
    if result.mode in {"RGBA", "LA"} or "transparency" in result.info:
        rgba = result.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return result.convert("RGB")


def encode_image(image):
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def build_request(task_type, prompt, image, source_images=(), *, model="UnifiedReward", max_tokens=4096):
    if task_type not in {"t2i", "i2i"}:
        raise ValueError("task_type must be t2i or i2i")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Reward prompt must be a nonempty string")
    source_images = list(source_images)
    if (task_type == "t2i" and source_images) or (task_type == "i2i" and len(source_images) != 1):
        raise ValueError("Official UnifiedReward templates require zero T2I or exactly one I2I source image")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    images = [*source_images, image]
    template = T2I_PROMPT if task_type == "t2i" else I2I_PROMPT
    content = [{"type": "image_url", "image_url": {"url": encode_image(value)}} for value in images]
    content.append({"type": "text", "text": template.replace("{prompt}", prompt)})
    return {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}


def parse_response(text, task_type):
    """Bound finite judge predictions to the rubric; retain raw scores for audit.

    The generative judge can emit finite values outside its requested scale
    (e.g. T2I style=0.7684). Such values saturate at the rubric boundary before
    normalization. Missing/ambiguous scores and nonfinite values still fail.
    """
    if not isinstance(text, str):
        raise RemoteRewardError("UnifiedReward response content is not text")
    if task_type == "t2i":
        scores = {}
        for axis in ("alignment", "coherence", "style"):
            matches = re.findall(rf"{axis}\s+Score\s*\(1\s*-\s*5\)\s*:\s*({_NUMBER})", text, re.I)
            if len(matches) != 1:
                raise RemoteRewardError(f"UnifiedReward output must contain exactly one {axis} score")
            scores[axis] = float(matches[0])
        lower, upper = 1.0, 5.0
    elif task_type == "i2i":
        matches = re.findall(r'"score"\s*:\s*(\[[^\[\]]*\])', text)
        if len(matches) != 1:
            raise RemoteRewardError("UnifiedReward output must contain exactly one score array")
        try:
            values = json.loads(matches[0])
        except (ValueError, TypeError) as exc:
            raise RemoteRewardError("UnifiedReward returned an invalid score array") from exc
        if len(values) != 2 or any(isinstance(value, bool) or not isinstance(value, (float, int)) for value in values):
            raise RemoteRewardError("UnifiedReward editing score must contain two numbers")
        scores = dict(zip(("edit_success", "preservation"), map(float, values)))
        lower, upper = 0.0, 25.0
    else:
        raise ValueError("task_type must be t2i or i2i")
    if any(not math.isfinite(value) for value in scores.values()):
        raise RemoteRewardError(f"UnifiedReward returned a nonfinite {task_type} score: {scores!r}")
    raw_scores = scores
    scores = {key: min(upper, max(lower, value)) for key, value in raw_scores.items()}
    clip_max_abs = max(abs(raw_scores[key] - value) for key, value in scores.items())
    if clip_max_abs:
        LOGGER.warning("UnifiedReward %s scores clipped to [%s, %s]: raw=%s bounded=%s", task_type, lower, upper, raw_scores, scores)
    # Compute normalization over scoring axes only, before adding diagnostics.
    scores["normalized"] = (sum(scores.values()) / len(scores) - lower) / (upper - lower)
    scores.update({key + "_raw": value for key, value in raw_scores.items()})
    scores["range_clipped"] = float(clip_max_abs > 0)
    scores["range_clip_max_abs"] = clip_max_abs
    return scores


def _parse_completion(result, task_type):
    try:
        choice = result["choices"][0]
        content = choice["message"]["content"]
        try:
            if choice.get("finish_reason") != "stop":
                raise RemoteRewardError(f"UnifiedReward generation did not finish: {choice.get('finish_reason')!r}")
            return parse_response(content, task_type)
        except RemoteRewardError as exc:
            # Include a bounded, escaped excerpt, never the image request body.
            raise RemoteRewardError(f"{exc}; task={task_type}; request_id={str(result.get('id', 'unknown'))[:128]!r}; response_excerpt={str(content)[:2048]!r}") from exc
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RemoteRewardError(f"UnifiedReward invalid completion: {exc}") from exc


def score_image(task_type, prompt, image, source_images=(), *, endpoint="http://127.0.0.1:8792", model="UnifiedReward", timeout_seconds=1800.0, max_tokens=4096):
    """Synchronous diagnostic API; training uses score_image_async."""
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("UnifiedReward endpoint must be an HTTP(S) base URL")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    payload = build_request(task_type, prompt, image, source_images, model=model, max_tokens=max_tokens)
    request = Request(endpoint.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read(1024 * 1024))
        return _parse_completion(result, task_type)
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RemoteRewardError(f"UnifiedReward service returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise RemoteRewardError(f"UnifiedReward request failed: {exc}") from exc


async def score_image_async(task_type, prompt, image, source_images=(), *, session, endpoint="http://127.0.0.1:8792", model="UnifiedReward", timeout_seconds=1800.0, max_tokens=4096):
    """Use native async HTTP, keeping JPEG/base64/JSON work off the event loop."""
    from lightx2v_rl.rewards.async_http import post_json

    def encode_payload():
        return json.dumps(build_request(task_type, prompt, image, source_images, model=model, max_tokens=max_tokens)).encode("utf-8")

    payload = await asyncio.to_thread(encode_payload)
    result = await post_json(session, endpoint, "/v1/chat/completions", payload=payload, content_type="application/json", timeout_seconds=timeout_seconds)
    return _parse_completion(result, task_type)
