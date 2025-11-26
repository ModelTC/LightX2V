import base64
import os
import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger


class MediaHandler(ABC):
    @abstractmethod
    def get_media_signatures(self) -> Dict[bytes, str]:
        """Return the binary signatures of this media type and their corresponding file extensions."""
        pass

    @abstractmethod
    def get_data_url_prefix(self) -> str:
        """Return the data URL prefix, e.g. 'data:image/' or 'data:audio/'."""
        pass

    @abstractmethod
    def get_data_url_pattern(self) -> str:
        """Return the regex pattern for data URL."""
        pass

    @abstractmethod
    def get_default_extension(self) -> str:
        """Return the default extension for this media type."""
        pass

    def is_base64(self, data: str) -> bool:
        if data.startswith(self.get_data_url_prefix()):
            return True

        try:
            if len(data) % 4 == 0:
                base64.b64decode(data, validate=True)
                decoded = base64.b64decode(data[:100])
                for signature in self.get_media_signatures().keys():
                    if decoded.startswith(signature):
                        return True
        except Exception as e:
            logger.warning(f"Error checking base64 {self.__class__.__name__}: {e}")
            return False

        return False

    def extract_base64_data(self, data: str) -> Tuple[str, Optional[str]]:
        if data.startswith("data:"):
            match = re.match(self.get_data_url_pattern(), data)
            if match:
                format_type = match.group(1)
                base64_data = match.group(2)
                return base64_data, format_type

        return data, None

    def detect_extension(self, data: bytes) -> str:
        for signature, ext in self.get_media_signatures().items():
            if data.startswith(signature):
                return ext
        return self.get_default_extension()

    def save_base64(self, base64_data: str, output_dir: str) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        data, format_type = self.extract_base64_data(base64_data)
        file_id = str(uuid.uuid4())

        try:
            media_data = base64.b64decode(data)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")

        if format_type:
            ext = format_type
        else:
            ext = self.detect_extension(media_data)

        file_path = os.path.join(output_dir, f"{file_id}.{ext}")
        with open(file_path, "wb") as f:
            f.write(media_data)

        return file_path
