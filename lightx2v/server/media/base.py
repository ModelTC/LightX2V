import base64
import re
import uuid
from pathlib import Path


class MediaHandler:
    data_url_prefix: str
    data_url_header: re.Pattern[str]
    signatures: dict[bytes, str]
    default_extension: str

    def _extension_from_signature(self, data: bytes) -> str | None:
        for signature, extension in self.signatures.items():
            if data.startswith(signature):
                return extension
        return None

    def detect_extension(self, data: bytes) -> str:
        return self._extension_from_signature(data) or self.default_extension

    def _decode_bare(self, value: str) -> tuple[bytes, str] | None:
        try:
            data = base64.b64decode(value, validate=True)
        except ValueError:
            return None
        extension = self._extension_from_signature(data)
        return (data, extension) if extension else None

    def is_base64(self, data: str) -> bool:
        if data.startswith("data:"):
            return self.data_url_header.match(data) is not None
        return self._decode_bare(data) is not None

    def resolve_path(self, value: str, output_dir: str | Path) -> str:
        """Cache an inline reference or return the local path."""
        if value.startswith(self.data_url_prefix):
            return self.save_base64(value, output_dir)
        media = self._decode_bare(value)
        if media is None:
            return value
        data, extension = media
        return _write_media(data, extension, output_dir)

    def save_base64(self, base64_data: str, output_dir: str | Path) -> str:
        extension = None
        if base64_data.startswith("data:"):
            header = self.data_url_header.match(base64_data)
            if header is None:
                raise ValueError("Invalid base64 media URL")
            extension = header[1]
            base64_data = base64_data[header.end() :]

        data = base64.b64decode("".join(base64_data.split()), validate=True)
        if not data:
            raise ValueError("Empty base64 media data")
        return _write_media(data, extension or self.detect_extension(data), output_dir)


def _write_media(data: bytes, extension: str, output_dir: str | Path) -> str:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4()}.{extension}"
    path.write_bytes(data)
    return str(path)
