"""Global vs video action queues for H3 prompt travel.

JSON is the raw global queue. ``H3ActionManager`` compiles an executable
``glob_a_q`` (absolute cues + rest inserts) and never rewrites those times.

Scheme C: video condition stays H ≈ 5.1667 s. The **compile window**
extends ``text_lead_units`` causal chunks past ``last_frame`` so the
non-causal slot can see the next action (warmup). ``At = clip(cue −
left, 0, H)`` — do not subtract lead from At. Now maps to clip end.
Past cues older than ``history_hold_seconds`` are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from lightx2v.models.networks.minimax_h3.packing import FPS, FRAMES_PER_CHUNK

CHUNK_SECONDS = FRAMES_PER_CHUNK / float(FPS)  # 17/24
VIDEO_HORIZON_SECONDS = 124.0 / float(FPS)  # 5.1666...
REST_GAP_SECONDS = 5.0

DEFAULT_REST_PROMPT = (
    "<Subject 1> holds the original rest pose from <Picture 1>: face toward the camera, hands toward the established rest, no new gesture. This event ends with holding the original rest."
)


@dataclass(frozen=True)
class TimedAction:
    time: float
    prompt: str


def flatten_raw_queue(stacks) -> list[TimedAction]:
    """JSON stacks → one (time, prompt) per beat."""
    events: list[TimedAction] = []
    for stack in stacks:
        for beat in stack.beats:
            events.append(TimedAction(time=stack.time + beat.rel_time, prompt=beat.body))
    events.sort(key=lambda item: item.time)
    return events


class H3ActionManager:
    """Scheduler: executable glob_a_q + last_frame window with text extend."""

    def __init__(
        self,
        raw_stacks,
        *,
        no_action_prompt: str = DEFAULT_REST_PROMPT,
        rest_gap_seconds: float = REST_GAP_SECONDS,
        video_horizon_seconds: float = VIDEO_HORIZON_SECONDS,
        max_actions: int = 4,
        interval_chunks: int = 1,
        text_lead_units: int = 2,
        history_hold_seconds: float = 2.0,
    ):
        self.no_action_prompt = no_action_prompt
        self.rest_gap_seconds = rest_gap_seconds
        self.video_horizon_seconds = video_horizon_seconds
        self.max_actions = max_actions
        self.interval_chunks = interval_chunks
        self.text_lead_units = text_lead_units
        self.history_hold_seconds = history_hold_seconds
        self.lookahead_seconds = text_lead_units * interval_chunks * CHUNK_SECONDS
        self.raw_glob_a_q = flatten_raw_queue(raw_stacks)
        self.glob_a_q: list[TimedAction] = []
        self.video_a_q: list[TimedAction] = []
        self.last_frame: float = 0.0
        self.compile_global_queue()

    def compile_global_queue(self) -> list[TimedAction]:
        compiled: list[TimedAction] = []
        cursor = 0.0
        original_times = {round(item.time, 3) for item in self.raw_glob_a_q}

        def insert_rest(at: float) -> None:
            key = round(at, 3)
            if key in original_times:
                return
            compiled.append(TimedAction(time=at, prompt=self.no_action_prompt))

        for event in self.raw_glob_a_q:
            if event.time - cursor > self.rest_gap_seconds + 1e-6:
                insert_rest(cursor + self.rest_gap_seconds)
            compiled.append(event)
            cursor = event.time
        insert_rest(cursor + self.rest_gap_seconds)
        compiled.sort(key=lambda item: item.time)
        self.glob_a_q = compiled
        return compiled

    def set_last_frame(self, last_frame: float) -> None:
        self.last_frame = max(0.0, float(last_frame))

    def compile_video_queue(self, last_frame: float | None = None) -> str:
        """Absolute glob → action text. Right edge extends; At stays on [0, H]."""
        if last_frame is not None:
            self.set_last_frame(last_frame)
        horizon = self.video_horizon_seconds
        last = self.last_frame
        lead = self.lookahead_seconds
        if last <= horizon + 1e-6:
            left = 0.0
            right = horizon + lead
            window = [item for item in self.glob_a_q if -1e-6 <= item.time < right - 1e-9]
        else:
            left = last - horizon
            right = last + lead
            window = [item for item in self.glob_a_q if left + 1e-6 < item.time <= right + 1e-6]
        if self.history_hold_seconds > 1e-9 and last > horizon + 1e-6:
            cut = last - self.history_hold_seconds
            window = [item for item in window if item.time > cut + 1e-9]
        window = window[: self.max_actions]
        self.video_a_q = window
        if not window:
            return ""
        lines = []
        for index, item in enumerate(window, start=1):
            at = min(horizon, max(0.0, item.time - left))
            lines.append(f"[Action {index}]: At {format_at(at)}. {item.prompt}")
        return "\n\n".join(lines)


def format_at(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000.0))
    minutes, rem = divmod(total_ms, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"
