"""Compatibility names for field groups defined by the shared arguments."""

from lightx2v.models.runners.request_args import COMMON, IMAGE, PROMPT, RESAMPLING, VIDEO, VIDEO_OUTPUT

COMMON_REQUEST_FIELDS = frozenset(COMMON)
PROMPT_FIELDS = frozenset(PROMPT)
VIDEO_OUTPUT_FIELDS = frozenset(VIDEO_OUTPUT)
RESAMPLING_FIELDS = frozenset(RESAMPLING)
IMAGE_REQUEST_FIELDS = frozenset(IMAGE)
VIDEO_REQUEST_FIELDS = frozenset(VIDEO)
