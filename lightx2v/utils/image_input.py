"""Reference image decoding and HEIF container identification."""

import os
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

HEIF_BRANDS = frozenset({b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1"})


def is_heif_header(data: bytes) -> bool:
    if len(data) < 16 or data[4:8] != b"ftyp":
        return False
    box_size = int.from_bytes(data[:4], "big")
    if box_size < 16:
        return False
    brands = [data[8:12]] + [data[i : i + 4] for i in range(16, min(box_size, len(data)) - 3, 4)]
    return any(brand in HEIF_BRANDS for brand in brands)


@contextmanager
def _reference_source(value):
    """Own path handles, but leave caller-owned streams open and seekable ones restored."""
    if isinstance(value, (str, bytes, os.PathLike)):
        with open(value, "rb") as source:
            yield source
        return
    try:
        position = value.tell()
        value.seek(0)
    except (AttributeError, OSError):
        # Pillow also buffers non-seekable input. Only this temporary copy is owned.
        with BytesIO(value.read()) as source:
            yield source
    else:
        try:
            yield value
        finally:
            value.seek(position)


def load_rgb_reference(value):
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    name = value if isinstance(value, (str, bytes, os.PathLike)) else getattr(value, "name", None)
    label = Path(os.fsdecode(name)).name if isinstance(name, (str, bytes, os.PathLike)) else "<stream>"
    try:
        with _reference_source(value) as source:
            header = source.read(128)
            source.seek(0)
            if is_heif_header(header) or Path(label).suffix.lower() in {".heic", ".heif"}:
                try:
                    from pillow_heif import register_heif_opener
                except ImportError as exc:
                    raise ValueError("HEIC/HEIF references require pillow-heif; install pillow-heif before editing") from exc
                register_heif_opener()
            with Image.open(source) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"Cannot decode reference image {label!r}: {exc}") from exc
