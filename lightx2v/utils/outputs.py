"""Artifacts produced by an inference request."""

import mimetypes
from dataclasses import dataclass

from lightx2v.utils.image_output import image_format_for_bytes


@dataclass
class Output:
    name: str
    type: str
    path: str | None = None
    content: bytes | None = None

    @property
    def media_type(self):
        if self.type == "image" and self.content is not None:
            return "image/" + image_format_for_bytes(self.content)
        if self.type == "image" and self.path is None:
            return "image/png"
        return mimetypes.guess_type(self.path or "")[0] or "application/octet-stream"
