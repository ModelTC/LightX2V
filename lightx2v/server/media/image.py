import re

from pillow_heif import get_file_mimetype

from .base import MediaHandler


class ImageHandler(MediaHandler):
    data_url_prefix = "data:image/"
    data_url_header = re.compile(rf"{data_url_prefix}(\w+)(?:-sequence)?;base64,")
    default_extension = "png"
    signatures = {
        b"\x89PNG\r\n\x1a\n": "png",
        b"\xff\xd8\xff": "jpg",
        b"GIF87a": "gif",
        b"GIF89a": "gif",
    }

    def _extension_from_signature(self, data: bytes) -> str | None:
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "webp"
        if data[4:8] == b"ftyp":
            extension = get_file_mimetype(data).removeprefix("image/").removesuffix("-sequence")
            if extension in ("heic", "heif"):
                return extension
        return super()._extension_from_signature(data)


_handler = ImageHandler()
is_base64_image = _handler.is_base64
save_base64_image = _handler.save_base64
resolve_image_path = _handler.resolve_path
