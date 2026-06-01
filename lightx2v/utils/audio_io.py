"""Audio file loading compatible with torchaudio 2.9+ (torchcodec) and legacy backends."""

from __future__ import annotations

from typing import Tuple, Union

import torch

_TORCHAUDIO_DECODE_AVAILABLE = None


def _torchaudio_decode_available() -> bool:
    global _TORCHAUDIO_DECODE_AVAILABLE
    if _TORCHAUDIO_DECODE_AVAILABLE is None:
        try:
            import torchcodec  # noqa: F401
        except Exception:
            _TORCHAUDIO_DECODE_AVAILABLE = False
        else:
            _TORCHAUDIO_DECODE_AVAILABLE = True
    return _TORCHAUDIO_DECODE_AVAILABLE


def load_audio_file(
    uri: Union[str, "os.PathLike"],
    frame_offset: int = 0,
    num_frames: int = -1,
    channels_first: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load audio as ``(tensor, sample_rate)``, matching ``torchaudio.load`` defaults."""
    if _torchaudio_decode_available():
        import torchaudio as ta

        return ta.load(
            uri,
            frame_offset=frame_offset,
            num_frames=num_frames,
            channels_first=channels_first,
        )

    import soundfile as sf

    start = frame_offset if frame_offset > 0 else None
    stop = None if num_frames < 0 else frame_offset + num_frames
    data, sample_rate = sf.read(uri, always_2d=True, start=start, stop=stop, dtype="float32")
    if channels_first:
        tensor = torch.from_numpy(data.T.copy())
    else:
        tensor = torch.from_numpy(data)
    return tensor, sample_rate
