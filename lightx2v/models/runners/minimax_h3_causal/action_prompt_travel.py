"""Timed H3 action prompts and fixed-length Qwen presentations.

Matches Zoe's naive travel and Scheme C. Reference tokens are a separately
tokenized prefix, so their constant length can be added to each text length.
"""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from lightx2v.models.runners.minimax_h3_causal.action_manager import DEFAULT_REST_PROMPT, H3ActionManager

REST_HEADER = "[Rest Action]:"
REST_UNIT = " N/A"
REST_REMAINDER = " NA"


class ActionPromptTravelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    enabled: bool = False
    naive_travel: bool = False
    prompt_form: Literal["six_key", "second", "action_only"] = "six_key"
    naive_pad: Literal["token_id", "zero_feature"] = "token_id"
    min_slice_rolling: bool = True
    unit_transition: int = Field(default=1, ge=1, validation_alias=AliasChoices("unit_transition", "x", "interval_chunks", "min_slice_chunks"))
    action_slot_len: int = Field(default=2560, ge=8)
    # Upstream keeps this for config compatibility; rest insertion uses the
    # gap and optional rest prompt, not separate keep_last/specified branches.
    hold_mode: Literal["keep_last", "specified", "empty_rest"] = "empty_rest"
    hold_action_prompt: str | None = None
    rest_gap_seconds: float = Field(default=5.0, ge=0)
    video_horizon_seconds: float = Field(default=124 / 24, gt=0)
    max_actions: int = Field(default=4, ge=1, le=8)
    text_lead_units: int = Field(default=2, ge=0)
    history_hold_seconds: float = Field(default=2.0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_interval(cls, values):
        values = dict(values)
        values.pop("min_chunk_unit", None)
        interval = values.pop("interval_chunks", None)
        interval = values.pop("min_slice_chunks", None) if interval is None else interval
        values.pop("min_slice_chunks", None)
        if interval is not None and "unit_transition" not in values and "x" not in values:
            values["unit_transition"] = interval
        return values


@dataclass
class ActionBeat:
    body: str
    rel_time: float = 0.0


@dataclass
class ActionStack:
    time: float
    beats: list[ActionBeat]


def coerce_action_prompts(raw):
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else None
    if raw is None or raw == {} or raw == []:
        return None
    if not isinstance(raw, (dict, list)):
        raise ValueError("action_prompts must be a dict, list, or JSON string")
    return raw


def strip_action_header(text):
    line = text.strip()
    if line.startswith(("[Action", "[Shot")):
        pos = line.find(". ")
        if pos >= 0:
            return line[pos + 2 :].strip()
        colon = line.find(":")
        if colon >= 0:
            rest = line[colon + 1 :].strip()
            if rest.lower().startswith("at "):
                rest = rest.split(".", 1)[-1].strip()
            return rest
    return line


def parse_action_prompts(raw):
    if isinstance(raw, list):
        mapping = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("action_prompts list entries must be dicts")
            mapping[str(item.get("time", item.get("start", 0)))] = item.get("actions", item.get("prompt", item.get("text")))
        raw = mapping
    if not isinstance(raw, dict) or not raw:
        raise ValueError("action_prompts must be a non-empty dict time -> actions")
    stacks = []
    for key, value in raw.items():
        if isinstance(value, str):
            value = [part.strip() for part in value.split("\n\n") if part.strip()]
        if not isinstance(value, list) or not value:
            raise ValueError(f"action stack at t={key} must be a non-empty string or list")
        beats = []
        for index, item in enumerate(value):
            if isinstance(item, str):
                beats.append(ActionBeat(strip_action_header(item), float(index)))
            elif isinstance(item, dict):
                beats.append(ActionBeat(strip_action_header(str(item.get("text", item.get("body", "")))), float(item.get("rel_time", index))))
            else:
                raise ValueError("action list items must be str or dict")
        stacks.append(ActionStack(float(key), beats))
    return sorted(stacks, key=lambda stack: stack.time)


def join_annot_and_slot(annot, slot):
    annot = annot.rstrip()
    if annot.endswith("action_prompt:"):
        return f"{annot}\n\n{slot}"
    return f"{annot}\n\naction_prompt:\n\n{slot}"


def pack_action_slot(tokenizer, body, slot_len):
    prefix = f"{body}\n\n{REST_HEADER}" if body.strip() else REST_HEADER
    count = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    unit = len(tokenizer(REST_UNIT, add_special_tokens=False)["input_ids"])
    remainder = len(tokenizer(REST_REMAINDER, add_special_tokens=False)["input_ids"])
    if unit != 2 or remainder != 1:
        raise ValueError("N/A pad units must be 2+1 Qwen tokens")
    if count > slot_len:
        raise ValueError(f"action tokens {count} exceed slot_len {slot_len}")
    repeats, leftover = divmod(slot_len - count, unit)
    packed = prefix + REST_UNIT * repeats + (REST_REMAINDER if leftover else "")
    if len(tokenizer(packed, add_special_tokens=False)["input_ids"]) != slot_len:
        raise RuntimeError(f"BPE did not hit action_slot_len={slot_len}")
    return packed


class ActionPromptTravel:
    def __init__(self, prompt, actions, config, tokenizer, reference_prefix_length):
        self.config = config
        self.tokenizer = tokenizer
        self.reference_prefix_length = reference_prefix_length
        self.text_annot = prompt.split("\naction_prompt:", 1)[0].rstrip()
        self.stacks = parse_action_prompts(actions)
        self.manager = H3ActionManager(
            self.stacks,
            no_action_prompt=config.hold_action_prompt or DEFAULT_REST_PROMPT,
            rest_gap_seconds=config.rest_gap_seconds,
            video_horizon_seconds=config.video_horizon_seconds,
            max_actions=config.max_actions,
            interval_chunks=config.unit_transition,
            text_lead_units=config.text_lead_units,
            history_hold_seconds=config.history_hold_seconds,
        )
        self.last_body = self.action_body(0.0)
        self.prompt = self.assemble_prompt(self.last_body, first=True)
        if config.naive_travel:
            candidates = [self.prompt]
            for stack in self.stacks:
                candidates.append(self.assemble_prompt(self.action_body(stack.time)))
            self.text_length = max(self.presentation_length(text) for text in candidates)
        else:
            self.text_length = self.presentation_length(self.prompt)

    def presentation_length(self, prompt):
        return self.reference_prefix_length + len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def action_body(self, seconds):
        if not self.config.naive_travel:
            return self.manager.compile_video_queue(seconds)
        chosen = self.stacks[0]
        for stack in self.stacks:
            if stack.time <= seconds + 1e-6:
                chosen = stack
            else:
                break
        sentence = chosen.beats[0].body
        return f"[Action 1]: {sentence}" if self.config.prompt_form == "six_key" else sentence

    def assemble_prompt(self, body, *, first=False):
        if self.config.naive_travel:
            if self.config.prompt_form == "action_only" or (self.config.prompt_form == "second" and not first):
                return body
            if self.config.prompt_form == "second":
                body = f"[Action 1]: {body}"
            return join_annot_and_slot(self.text_annot, body)
        return join_annot_and_slot(self.text_annot, pack_action_slot(self.tokenizer, body, self.config.action_slot_len))

    def update(self, seconds):
        """Return whether the next chunk needs a new text encoding and KV fill."""
        body = self.action_body(max(0.0, seconds))
        if body == self.last_body:
            return False
        prompt = self.assemble_prompt(body)
        if not self.config.naive_travel:
            # Packing the slot alone can change BPE boundaries in the full text.
            # Match Zoe's correction using only the trailing N/A padding.
            for _ in range(64):
                length = self.presentation_length(prompt)
                if length == self.text_length:
                    break
                if length < self.text_length:
                    prompt += REST_UNIT
                elif prompt.endswith(REST_REMAINDER):
                    prompt = prompt[: -len(REST_REMAINDER)]
                elif prompt.endswith(REST_UNIT):
                    prompt = prompt[: -len(REST_UNIT)]
                else:
                    break
            if self.presentation_length(prompt) != self.text_length:
                raise RuntimeError("action presentation length changed; keep action_slot_len fixed")
        self.prompt, self.last_body = prompt, body
        return True
