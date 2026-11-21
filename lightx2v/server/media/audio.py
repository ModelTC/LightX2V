import re
from io import BytesIO

from .base import MediaHandler


def _is_audio_only_mp4(data: bytes) -> bool:
    import av

    try:
        with av.open(BytesIO(data), format="mov") as container:
            streams = container.streams
            return bool(streams.audio) and all(video.disposition & av.stream.Disposition.attached_pic for video in streams.video)
    except av.FFmpegError:
        return False


class AudioHandler(MediaHandler):
    data_url_prefix = "data:audio/"
    data_url_header = re.compile(rf"{data_url_prefix}(\w+);base64,")
    default_extension = "mp3"
    signatures = {
        b"ID3": "mp3",
        b"\xff\xfb": "mp3",
        b"\xff\xf3": "mp3",
        b"\xff\xf2": "mp3",
        b"OggS": "ogg",
        b"fLaC": "flac",
    }

    def _extension_from_signature(self, data: bytes) -> str | None:
        if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
            return "wav"
        if data[4:8] == b"ftyp" and _is_audio_only_mp4(data):
            return "m4a"
        return super()._extension_from_signature(data)


_handler = AudioHandler()
is_base64_audio = _handler.is_base64
save_base64_audio = _handler.save_base64
resolve_audio_path = _handler.resolve_path
