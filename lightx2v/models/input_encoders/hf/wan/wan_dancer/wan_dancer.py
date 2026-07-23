import os
import tempfile

import numpy as np
import soundfile as sf
import torch

try:
    import librosa
except ImportError:
    librosa = None


def extract_music_features(path, fps=30):
    """Reproduce Wan-Dancer's 35-channel librosa feature, including its SR quirk."""
    hop_length = 512
    data, _ = librosa.load(path, sr=fps * hop_length)
    feature_sr = 22050
    envelope = librosa.onset.onset_strength(y=data, sr=feature_sr)
    mfcc = librosa.feature.mfcc(y=data, sr=feature_sr, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=data, sr=feature_sr, hop_length=hop_length, n_chroma=12).T
    peaks = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=feature_sr, hop_length=hop_length)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peaks] = 1.0
    start_bpm = librosa.beat.tempo(y=librosa.load(path)[0])[0]
    _, beats = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=feature_sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beats] = 1.0
    result = np.concatenate([envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]], axis=-1)
    return torch.from_numpy(result)


def split_music_features(path, segment_frames=149, fps=30):
    """Split through PCM16 WAVs exactly like the local-stage reference script."""
    audio, sample_rate = librosa.load(path, sr=None)
    total_duration = len(audio) / sample_rate
    segment_duration = segment_frames / fps
    results = []
    with tempfile.TemporaryDirectory(prefix="wan_dancer_audio_") as directory:
        start = 0.0
        index = 0
        while start + 0.2 < total_duration:
            end = min(start + segment_duration, total_duration)
            segment_path = os.path.join(directory, f"{index:03d}.wav")
            sf.write(segment_path, audio[int(start * sample_rate) : int(end * sample_rate)], sample_rate)
            results.append(extract_music_features(segment_path, fps=fps))
            start += segment_duration
            index += 1
    return results, total_duration
