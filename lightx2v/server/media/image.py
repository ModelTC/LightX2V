from typing import Dict

from .base import MediaHandler


class ImageHandler(MediaHandler):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_media_signatures(self) -> Dict[bytes, str]:
        return {
            b"\x89PNG\r\n\x1a\n": "png",
            b"\xff\xd8\xff": "jpg",
            b"GIF87a": "gif",
            b"GIF89a": "gif",
        }

    def get_data_url_prefix(self) -> str:
        return "data:image/"

    def get_data_url_pattern(self) -> str:
        return r"data:image/(\w+);base64,(.+)"

    def get_default_extension(self) -> str:
        return "png"

    def is_base64(self, data: str) -> bool:
        if data.startswith(self.get_data_url_prefix()):
            return True

        try:
            import base64

            if len(data) % 4 == 0:
                base64.b64decode(data, validate=True)
                decoded = base64.b64decode(data[:100])
                for signature in self.get_media_signatures().keys():
                    if decoded.startswith(signature):
                        return True
                if len(decoded) > 12 and decoded[8:12] == b"WEBP":
                    return True
        except Exception:
            return False

        return False

    def detect_extension(self, data: bytes) -> str:
        for signature, ext in self.get_media_signatures().items():
            if data.startswith(signature):
                return ext
        if len(data) > 12 and data[8:12] == b"WEBP":
            return "webp"
        return self.get_default_extension()


_handler = ImageHandler()


def is_base64_image(data: str) -> bool:
    return _handler.is_base64(data)


def save_base64_image(base64_data: str, output_dir: str) -> str:
    return _handler.save_base64(base64_data, output_dir)
