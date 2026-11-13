"""Shared output format and quality rules for image serving."""

from pathlib import Path


def image_format_for_path(path):
    suffix = Path(path).suffix.lower() if path else ""
    formats = {"": "png", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}
    if suffix not in formats:
        raise ValueError("Image output path must end in .png, .jpg, .jpeg or .webp")
    return formats[suffix]


def image_format_for_bytes(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ValueError("Unsupported encoded image result")


def validate_output_compression(value):
    # Multipart form scalars arrive as strings. Reject booleans and floats.
    if isinstance(value, str) and value.lstrip("+-").isdigit():
        value = int(value)
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("output_compression must be an integer between 0 and 100")
    return value
