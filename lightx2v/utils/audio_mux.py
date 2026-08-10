import logging
import os
import shutil
import subprocess
from typing import Optional

MP4_AAC_BITRATE = "256k"
logger = logging.getLogger(__name__)


def _find_ffprobe(ffmpeg_exe: Optional[str] = None) -> Optional[str]:
    if ffmpeg_exe:
        sibling = os.path.join(os.path.dirname(os.path.abspath(ffmpeg_exe)), "ffprobe")
        if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
            return sibling
    return shutil.which("ffprobe")


def probe_audio_codec(source_path: str, ffmpeg_exe: Optional[str] = None) -> Optional[str]:
    """Return the codec name of the first audio stream, if it can be probed."""
    ffprobe_exe = _find_ffprobe(ffmpeg_exe)
    if ffprobe_exe is None:
        logger.warning("ffprobe was not found; audio will be transcoded to AAC for MP4 compatibility")
        return None

    cmd = [
        ffprobe_exe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        source_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except OSError as exc:
        logger.warning(f"Failed to run ffprobe for {source_path}: {exc}; audio will be transcoded to AAC")
        return None

    codec = result.stdout.strip().splitlines()[0].lower() if result.returncode == 0 and result.stdout.strip() else None
    if codec is None:
        stderr = result.stderr.strip() if result.stderr else "audio stream not found"
        logger.warning(f"Failed to probe audio codec for {source_path}: {stderr}; audio will be transcoded to AAC")
    return codec


def mp4_audio_codec_args(source_path: str, ffmpeg_exe: Optional[str] = None) -> list[str]:
    """Choose MP4-safe FFmpeg audio arguments without re-encoding AAC audio."""
    if probe_audio_codec(source_path, ffmpeg_exe) == "aac":
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", MP4_AAC_BITRATE]
