"""Piecewise H3 rollout shifts indexed by completed optimizer-role updates."""

import math


def rollout_shifts(dmd, completed_updates):
    video = float(dmd.get("official_rollout_video_shift", 1.0))
    audio = float(dmd.get("official_rollout_audio_shift", 1.0))
    for phase in dmd.get("rollout_shift_schedule", ()):
        if completed_updates < phase["start_update"]:
            break
        video, audio = float(phase["video_shift"]), float(phase["audio_shift"])
    return video, audio


def validate_rollout_schedule(config):
    dmd = config["training"]["dmd"]
    phases = dmd.get("rollout_shift_schedule")
    if phases is None:
        return
    if config["model"]["name"] not in {"minimax_h3_t2av", "minimax_h3_ref2av"}:
        raise ValueError("Rollout shift schedules require an H3 AV model")
    if not isinstance(phases, list) or not phases or phases[0]["start_update"] != 0:
        raise ValueError("rollout_shift_schedule must start at update 0")
    previous = -1
    cycle = int(dmd.get("fake_update_ratio", 5)) + 1
    for phase in phases:
        start = phase["start_update"]
        if type(start) is not int or start <= previous:
            raise ValueError("Schedule start_update values must be increasing nonnegative integers")
        if start % cycle == cycle - 1:
            raise ValueError("Cannot change rollout shifts between a cached fake trajectory and its student update")
        if any(not math.isfinite(float(phase[key])) or float(phase[key]) <= 0 for key in ("video_shift", "audio_shift")):
            raise ValueError("Rollout shifts must be finite and positive")
        previous = start
    matching = config["model"]["capabilities"]["distribution_matching"]
    initial = (float(dmd.get("official_rollout_video_shift", 1.0)), float(dmd.get("official_rollout_audio_shift", 1.0)))
    if rollout_shifts(dmd, 0) != initial or initial != (matching["video_flow_shift"], matching["audio_flow_shift"]):
        raise ValueError("Initial schedule, rollout and model shifts must agree")
