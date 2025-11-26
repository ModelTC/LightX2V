from typing import Dict

from .base import MediaHandler


class AudioHandler(MediaHandler):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_media_signatures(self) -> Dict[bytes, str]:
        return {
            b"ID3": "mp3",
            b"\xff\xfb": "mp3",
            b"\xff\xf3": "mp3",
            b"\xff\xf2": "mp3",
            b"OggS": "ogg",
            b"fLaC": "flac",
        }

    def get_data_url_prefix(self) -> str:
        return "data:audio/"

    def get_data_url_pattern(self) -> str:
        return r"data:audio/(\w+);base64,(.+)"

    def get_default_extension(self) -> str:
        return "mp3"

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
                if decoded.startswith(b"RIFF") and b"WAVE" in decoded[:12]:
                    return True
                if decoded[:4] in [b"ftyp", b"\x00\x00\x00\x20", b"\x00\x00\x00\x18"]:
                    return True
        except Exception:
            return False

        return False

    def detect_extension(self, data: bytes) -> str:
        for signature, ext in self.get_media_signatures().items():
            if data.startswith(signature):
                return ext
        if data.startswith(b"RIFF") and b"WAVE" in data[:12]:
            return "wav"
        if data[:4] in [b"ftyp", b"\x00\x00\x00\x20", b"\x00\x00\x00\x18"]:
            return "m4a"
        return self.get_default_extension()


_handler = AudioHandler()


def is_base64_audio(data: str) -> bool:
    return _handler.is_base64(data)


def save_base64_audio(base64_data: str, output_dir: str) -> str:
    return _handler.save_base64(base64_data, output_dir)
