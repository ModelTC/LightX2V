import math
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

WHISPER_SAMPLE_RATE = 16000


class KimVocalSeparator:
    """Waveform adapter for the Kim_Vocal_2 ONNX checkpoint in the release."""

    sample_rate = 44100
    n_fft = 7680
    hop_length = 1024
    frequency_bins = 3072
    chunk_samples = 255 * hop_length

    def __init__(self, model_path, device):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.window = torch.hann_window(self.n_fft).to(device)

    def _separate_chunks(self, chunks):
        waveform = torch.from_numpy(chunks).to(self.window.device).flatten(0, 1)
        spectrum = torch.stft(waveform, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        spectrum = torch.view_as_real(spectrum).permute(0, 3, 1, 2)
        spectrum = spectrum.reshape(len(chunks), 4, self.n_fft // 2 + 1, -1)[:, :, : self.frequency_bins].contiguous()
        spectrum[:, :, :3] = 0
        predicted = self.session.run(None, {self.input_name: spectrum.cpu().numpy()})[0]
        spectrum = torch.from_numpy(predicted).to(self.window.device)
        spectrum = F.pad(spectrum, (0, 0, 0, self.n_fft // 2 + 1 - self.frequency_bins))
        spectrum = spectrum.unflatten(1, (2, 2)).flatten(0, 1).permute(0, 2, 3, 1).contiguous()
        waveform = torch.istft(torch.view_as_complex(spectrum), self.n_fft, self.hop_length, window=self.window)
        return waveform.unflatten(0, (-1, 2)).cpu().numpy()

    def separate(self, waveform, sample_rate, duration):
        mix = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.sample_rate)
        if mix.ndim == 1:
            mix = np.stack((mix, mix))
        mix = _limit_peak(mix)
        output_samples = min(math.ceil(duration * self.sample_rate), mix.shape[-1])
        mix = mix[:, : output_samples + self.chunk_samples]
        trim = self.n_fft // 2
        stride = self.chunk_samples * 3 // 4
        usable = self.chunk_samples - 2 * trim
        padded = np.pad(mix, ((0, 0), (trim, usable + trim - mix.shape[-1] % usable)))
        result = np.zeros_like(padded)
        weights = np.zeros(padded.shape[-1], dtype=np.float32)
        starts = range(0, padded.shape[-1], stride)
        for index in range(0, len(starts), 2):
            batch_starts = starts[index : index + 2]
            pieces = [padded[:, start : start + self.chunk_samples] for start in batch_starts]
            chunks = np.stack([librosa.util.fix_length(piece, size=self.chunk_samples) for piece in pieces])
            separated = self._separate_chunks(chunks)
            for start, piece, vocals in zip(batch_starts, pieces, separated):
                length = piece.shape[-1]
                window = np.hanning(length)
                vocals = vocals[:, :length]
                vocals *= window
                result[:, start : start + length] += vocals
                weights[start : start + length] += window

        source_slice = slice(trim, trim + mix.shape[-1])
        vocals = result[:, source_slice] / weights[source_slice] * 1.009
        vocals = librosa.to_mono(_limit_peak(vocals))
        vocals = librosa.resample(vocals, orig_sr=self.sample_rate, target_sr=WHISPER_SAMPLE_RATE)
        return vocals[: math.ceil(output_samples * WHISPER_SAMPLE_RATE / self.sample_rate)]


def _limit_peak(waveform):
    peak = np.abs(waveform).max()
    return waveform * (0.9 / peak) if peak > 0.9 else waveform


def _read_audio(path):
    waveform, rate = librosa.load(path, sr=None, mono=False)
    if not waveform.size:
        raise ValueError(f"Empty driving audio: {path}")
    if waveform.ndim == 2 and waveform.shape[0] > 2:
        waveform = librosa.to_mono(waveform)
    return waveform, rate


def _mono_16k(waveform, rate, samples):
    return librosa.resample(librosa.to_mono(waveform), orig_sr=rate, target_sr=WHISPER_SAMPLE_RATE)[:samples]


class LongCatWhisperEncoder:
    def __init__(self, checkpoint, device, vocal_checkpoint=None):
        self.device = device
        self.features = WhisperFeatureExtractor.from_pretrained(checkpoint, local_files_only=True)
        with torch.device("meta"):
            self.encoder = WhisperEncoder(WhisperConfig.from_pretrained(checkpoint, local_files_only=True))
        # The release also ships FP32 shards and a decoder. Its own inference
        # loads the default safetensors file and evaluates the encoder in FP32.
        prefix = "model.encoder."
        with safe_open(Path(checkpoint) / "model.safetensors", framework="pt") as weights:
            state = {key.removeprefix(prefix): weights.get_tensor(key).to(device=device, dtype=torch.float32) for key in weights.keys() if key.startswith(prefix)}
        self.encoder.load_state_dict(state, assign=True)
        self.encoder.eval()
        self.separator = KimVocalSeparator(vocal_checkpoint, device) if vocal_checkpoint is not None else None

    def prepare_tracks(self, paths, frames, fps, audio_type, silent_background):
        # The release gives Whisper an extra second of context beyond the video.
        duration = frames / fps + 1
        samples = math.ceil(duration * WHISPER_SAMPLE_RATE)
        if len(paths) == 1:
            soundtrack, output_rate = _read_audio(paths[0])
            voice = self.separator.separate(soundtrack, output_rate, duration) if self.separator is not None else _mono_16k(soundtrack, output_rate, samples)
            vocals = [voice]
        else:
            vocals, mono_tracks = [], []
            for path in paths:
                waveform, rate = _read_audio(path)
                track = _mono_16k(waveform, rate, samples)
                mono_tracks.append(track)
                vocals.append(self.separator.separate(waveform, rate, duration) if self.separator is not None else track)
            left, right = vocals
            left_track, right_track = mono_tracks
            if audio_type == "add":
                vocals = [np.pad(left, (0, len(right))), np.pad(right, (len(left), 0))]
                mixed = np.concatenate((left_track, right_track))[:samples]
            else:
                # The parallel duet aligns both voices and both originals to the shorter track.
                length = min(len(left), len(right), len(left_track), len(right_track))
                vocals = [left[:length], right[:length]]
                mixed = left_track[:length] + right_track[:length]
            # Match the PCM16 mix written by the official two-speaker demo.
            soundtrack = np.clip(np.floor(mixed * 32768), -32768, 32767) / 32768
            output_rate = WHISPER_SAMPLE_RATE
        if silent_background:
            vocals.append(np.zeros_like(vocals[0]))
        embeddings = []
        for voice in vocals:
            if len(voice) < samples:
                voice = np.pad(voice, (0, samples - len(voice)))
            embeddings.append(self.encode(voice, fps))
        return torch.stack(embeddings), soundtrack, output_rate

    def encode(self, speech, fps):
        chunk_samples = self.features.n_samples
        encoded = []
        for start in range(0, len(speech), chunk_samples):
            mel = self.features(speech[start : start + chunk_samples], sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt").input_features.to(self.device)
            states = self.encoder(mel, output_hidden_states=True).hidden_states
            averaged_layers = [torch.stack(states[index : index + 8], dim=2).mean(dim=2) for index in (0, 8, 16, 24)]
            encoded.append(torch.stack((*averaged_layers, states[32]), dim=2))

        video_length = len(speech) * fps // WHISPER_SAMPLE_RATE
        features = torch.cat(encoded, dim=1)[:, : video_length * 2]
        shape = features.shape[-2:]
        features = F.interpolate(features.flatten(2).transpose(1, 2), size=video_length, mode="linear", align_corners=True)
        return features[0].transpose(0, 1).unflatten(-1, shape).to(torch.bfloat16)
