import json

import numpy as np

POLICY_PROMPT_FORMATS = ("official_text", "json")


def normalize_policy_prompt_format(value):
    prompt_format = str(value or "official_text").strip().lower()
    aliases = {"official": "official_text", "plain": "official_text", "text": "official_text"}
    prompt_format = aliases.get(prompt_format, prompt_format)
    if prompt_format not in POLICY_PROMPT_FORMATS:
        raise ValueError(f"unsupported Cosmos3 policy prompt format {value!r}; expected one of {POLICY_PROMPT_FORMATS}")
    return prompt_format


def _append_sentence(base, addition):
    base = str(base or "").rstrip().rstrip(".")
    return f"{base}. {addition}" if base else addition


def build_official_policy_prompt(description, framing, num_frames, fps, height, width):
    """Match the fallback ActionTransformPipeline used by NVIDIA's RoboLab server."""
    prompt = str(description or "").rstrip()
    if framing:
        prompt = _append_sentence(prompt, framing)
    duration = int(num_frames / fps) if fps > 0 and np.isfinite(fps) else 0
    prompt = _append_sentence(prompt, f"The video is {duration:.1f} seconds long and is of {fps:.0f} FPS.")
    return _append_sentence(prompt, f"This video is of {height}x{width} resolution.")


def build_json_policy_prompt(description, framing, num_frames, fps, height, width):
    duration_seconds = num_frames / fps if fps > 0 else 0.0
    duration = int(duration_seconds) if duration_seconds >= 0 and np.isfinite(duration_seconds) else 0
    action_end = round(duration_seconds) if duration_seconds >= 0 and np.isfinite(duration_seconds) else 0
    minutes, seconds = divmod(action_end, 60)
    desc = str(description or "").strip()
    if desc and not desc.endswith((".", "!", "?")):
        desc = f"{desc}."

    prompt = {}
    if framing:
        prompt["cinematography"] = {"framing": framing}
    prompt["actions"] = [{"time": f"0:00-{minutes}:{seconds:02d}", "description": desc}]
    prompt["duration"] = f"{duration}s"
    prompt["fps"] = float(fps)
    prompt["resolution"] = {"H": int(height), "W": int(width)}
    ratio = width / height if height > 0 else 1.0
    prompt["aspect_ratio"] = min(
        ("1,1", "4,3", "3,4", "16,9", "9,16"),
        key=lambda value: abs(int(value.split(",")[0]) / int(value.split(",")[1]) - ratio),
    )
    return json.dumps(prompt)


class PolicySeedSequence:
    """Generate per-plan seeds with the same RNG semantics as the official server."""

    def __init__(self, base_seed):
        self._rng = np.random.default_rng(int(base_seed))

    def next_seed(self):
        return int(self._rng.integers(0, 2**31))
