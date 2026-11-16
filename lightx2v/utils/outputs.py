"""Artifacts produced by an inference request."""

import mimetypes
from dataclasses import dataclass


@dataclass
class Output:
    name: str
    type: str
    path: str | None = None
    content: bytes | None = None

    @property
    def media_type(self):
        if self.type == "image" and (self.content is not None or self.path is None):
            return "image/png"
        return mimetypes.guess_type(self.path or "")[0] or "application/octet-stream"
