"""Prepared conditions to playable audio, including the learned BigVGAN decoder."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.audio_encoders.hf.seedvc.vocoder import SeedVCVocoder
from lightx2v.models.networks.seedvc.model import SeedVCModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.seedvc.scheduler import SeedVCScheduler
from lightx2v.utils.input_info import InputInfo
from lightx2v.utils.registry_factory import RUNNER_REGISTER

from .sample import load_sample


@dataclass
class SeedVCInputInfo(InputInfo):
    metadata: dict = field(init=False, repr=False)
    tensors: dict = field(init=False, repr=False)


@RUNNER_REGISTER("seedvc")
class SeedVCRunner(DefaultRunner):
    input_info_state = SeedVCInputInfo

    def set_init_device(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Seed-VC requires CUDA")
        self.init_device = torch.device("cuda", torch.cuda.current_device())

    def init_scheduler(self):
        self.scheduler = SeedVCScheduler()

    def init_modules(self):
        self.model = self.vocoder = None
        self.warmup_passes = 0
        self.warmup_seconds = 0.0
        self.compile_graphs_after_warmup = 0
        self.last_metrics = {}
        pre = self.config["architecture"]["preprocess_params"]
        self.sample_rate = pre["sr"]
        self.hop_length = pre["spect_params"]["hop_length"]
        self.model = SeedVCModel(self.config, self.init_device)
        self.model.set_scheduler(self.scheduler)
        self.vocoder = self.rank_zero(lambda: SeedVCVocoder(self.config["vocoder_path"], self.init_device, self.sample_rate, pre["spect_params"]["n_mels"], self.hop_length))
        self.config.lock()

    def rank_zero(self, fn):
        result, error = None, None
        if self.model.tp_rank == 0:
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - every rank must observe decoder/output failures
                error = f"{type(exc).__name__}: {exc}"
        if self.model.tp_size > 1:
            message = [error]
            dist.broadcast_object_list(message, src=0, group=self.model.tp_group)
            error = message[0]
        if error:
            raise RuntimeError(error)
        return result

    def create_input_info(self, request_data):
        info = super().create_input_info(request_data)
        if not info.sample_path:
            raise ValueError("sample_path is required")
        info.metadata, info.tensors = load_sample(info.sample_path, self.config)
        if self.config["model_variant"] != "svc" and (info.semitone_shift or info.auto_f0_adjust):
            raise ValueError("Pitch controls require the F0-conditioned singing model")
        if not np.isfinite([info.length_adjust, info.cfg_rate]).all():
            raise ValueError("length_adjust and cfg_rate must be finite")
        if not 0 < int(info.metadata["source_mel_frames"] * info.length_adjust):
            raise ValueError("length_adjust produces an empty output")
        if info.save_result_path and Path(info.save_result_path).suffix.lower() not in (".wav", ".flac"):
            raise ValueError("Seed-VC output must be WAV or FLAC")
        return info

    def get_output_extension(self, input_info):
        return ".wav"

    def graph_count(self):
        if not self.config["use_compile"]:
            return 0
        from torch._dynamo.utils import counters

        return counters["stats"]["unique_graphs"]

    @torch.inference_mode()
    def generate(self, info):
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
                return self._generate(info)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    def _generate(self, info):
        started = time.perf_counter()
        values = {key: value.to(self.init_device) for key, value in info.tensors.items()}
        source_f0 = values.get("source_f0")
        reference_f0 = values.get("reference_f0")
        if source_f0 is not None:
            voiced = source_f0 > 1
            log_f0 = (source_f0 + 1e-5).log()
            if info.auto_f0_adjust:
                ref_voiced = reference_f0 > 1
                if not voiced.any() or not ref_voiced.any():
                    raise ValueError("Automatic pitch alignment requires voiced source and reference audio")
                log_f0[voiced] = log_f0[voiced] - (source_f0[voiced] + 1e-5).log().median() + (reference_f0[ref_voiced] + 1e-5).log().median()
            source_f0 = log_f0.exp()
            if info.semitone_shift:
                source_f0[voiced] *= 2 ** (info.semitone_shift / 12)
        mel = values["reference_mel"]
        frames = int(info.metadata["source_mel_frames"] * info.length_adjust)
        condition = self.model.regulator.infer(values["source_content"], frames, source_f0)
        reference = self.model.regulator.infer(values["reference_content"], mel.shape[-1], reference_f0)
        generator = torch.Generator(device=self.init_device).manual_seed(info.seed)
        window = self.sample_rate // self.hop_length * 30 - mel.shape[-1]
        overlap_frames = 16
        overlap = overlap_frames * self.hop_length
        processed, chunks = 0, 0
        output, previous = [], None
        while processed < frames:
            self.check_stop()
            chunk = condition[:, processed : processed + window]
            last = processed + window >= frames
            prediction = self.scheduler.sample(self.model, torch.cat((reference, chunk), 1), mel, values["style"], info.diffusion_steps, info.cfg_rate, generator, self.check_stop)
            wave = self.rank_zero(lambda mel=prediction: self.vocoder.decode(mel))
            if self.model.tp_rank == 0:
                wave = wave.numpy()
                keep = wave if last else wave[:-overlap]
                if previous is not None:
                    fade = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
                    keep = keep.copy()
                    keep[:overlap] = previous * fade + keep[:overlap] * (np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2)
                output.append(keep)
                previous = None if last else wave[-overlap:].copy()
            chunks += 1
            if last:
                break
            processed += chunk.shape[1] - overlap_frames
        audio = torch.from_numpy(np.concatenate(output).astype(np.float32)) if self.model.tp_rank == 0 else None
        self.rank_zero(lambda: self._validate_audio(audio, frames * self.hop_length))
        torch.cuda.synchronize(self.init_device)
        return audio, {
            "generation_seconds": time.perf_counter() - started,
            "chunks": chunks,
            "audio_frames": frames * self.hop_length,
            "sample_rate": self.sample_rate,
            "cfg_branches": 2 if info.cfg_rate > 0 else 1,
        }

    @staticmethod
    def _validate_audio(audio, frames):
        if audio.numel() != frames or not torch.isfinite(audio).all() or audio.abs().max() == 0:
            raise FloatingPointError("Vocoder produced empty, non-finite or silent audio")

    def run_request(self, input_info):
        self._outputs = []
        return self.run_pipeline(input_info)

    @torch.inference_mode()
    def run_pipeline(self, input_info):
        self.input_info = input_info
        torch.cuda.reset_peak_memory_stats(self.init_device)
        before = self.model.runtime_stats()
        try:
            audio, timing = self.generate(input_info)

            def save():
                if input_info.save_result_path:
                    path = Path(input_info.save_result_path).expanduser().resolve()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, audio.numpy(), self.sample_rate, subtype="FLOAT" if path.suffix.lower() == ".wav" else "PCM_24")
                    self.record_output("audio", path)
                    logger.info("Seed-VC audio: {} ({:.3f}s, {} Hz, mono)", path, len(audio) / self.sample_rate, self.sample_rate)

            self.rank_zero(save)
            stats = self.model.runtime_stats()
            for key in ("blocks_executed", "offload_transfers", "offload_bytes"):
                stats[key] -= before[key]
            self.last_metrics = (
                timing
                | stats
                | {
                    "warmup_passes": self.warmup_passes,
                    "warmup_seconds": self.warmup_seconds,
                    "compile_graphs_after_warmup": self.compile_graphs_after_warmup,
                    "compile_graphs_total": self.graph_count(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.init_device),
                }
            )
            logger.info("Seed-VC metrics: {}", json.dumps(self.last_metrics))
            return {"audio": audio if input_info.return_result_tensor else None, "sample_rate": self.sample_rate, "metrics": self.last_metrics}
        finally:
            self.input_info = None

    def run_warmup(self):
        sample_path = self.config["warmup_sample_path"]
        started = time.perf_counter()
        self.warmup_passes += 1
        try:
            with torch.random.fork_rng(devices=[self.init_device.index]):
                for cfg in (0.0, 0.7):
                    info = self.create_input_info({"task": "a2a", "sample_path": sample_path, "diffusion_steps": 1, "cfg_rate": cfg, "seed": 0})
                    self.generate(info)
            torch.cuda.synchronize(self.init_device)
        finally:
            self.input_info = None
            self._outputs = []
        self.warmup_seconds = time.perf_counter() - started
        self.compile_graphs_after_warmup = self.graph_count()
        logger.info("Seed-VC startup warmup: {:.3f}s, {} graphs", self.warmup_seconds, self.compile_graphs_after_warmup)

    def close(self):
        if getattr(self, "model", None) is not None:
            self.model.close()
            self.model = None
        if getattr(self, "vocoder", None) is not None:
            self.vocoder.close()
            self.vocoder = None
        super().close()
