"""End-to-end YuE2 runner: prepared conditions in, playable audio out."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.audio_encoders.hf.yue2.vae import YuE2AudioVAE
from lightx2v.models.networks.yue2.model import YuE2Model
from lightx2v.models.networks.yue2.protocol import CODEC_OFFSET, GenerationConfig, Sampling, SongRequest, negative_prefix, resolve_sampling, token_prefixes
from lightx2v.models.networks.yue2.tokenizer import YuE2TextTokenizer
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.yue2.scheduler import YuE2Scheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import InputInfo
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import seed_all


@dataclass
class YuE2InputInfo(InputInfo):
    song: SongRequest = field(init=False, repr=False)
    abc_generation: Sampling = field(init=False, repr=False)
    semantic_generation: Sampling = field(init=False, repr=False)


@RUNNER_REGISTER("yue2")
class YuE2Runner(DefaultRunner):
    input_info_state = YuE2InputInfo
    _WARMUP_PROMPTS = ("Warm piano instrumental.", "An instrumental arrangement with acoustic piano, rounded bass and gentle brushed drums.")

    def set_init_device(self):
        if not torch.cuda.is_available():
            raise RuntimeError("This YuE2 integration requires CUDA")
        if GET_DTYPE() != torch.bfloat16:
            raise ValueError("YuE2 currently requires DTYPE=BF16; the VAE remains FP32")
        self.init_device = torch.device("cuda", torch.cuda.current_device())

    def init_scheduler(self):
        self.generation = GenerationConfig.from_dict(self.config.get("generation", {}))
        self.scheduler = YuE2Scheduler(self.generation.ode_steps)

    def init_modules(self):
        self.model = None
        self.vae_decoder = None
        self.warmup_passes = 0
        self.warmup_seconds = 0.0
        self.compile_graphs_after_warmup = 0
        self.last_metrics = {}
        self.tokenizer = YuE2TextTokenizer(Path(self.config["model_path"]) / "qwen.tiktoken")
        self.model = YuE2Model(self.config, self.init_device)
        self.model.set_scheduler(self.scheduler)
        if self.model.tp_rank == 0:
            self.vae_decoder = YuE2AudioVAE(self.config["audio_vae_original_ckpt"], self.init_device, self.config.get("vae_core_frames", 1024), self.config.get("vae_halo_frames", 16))
        self.config.lock()

    def create_input_info(self, request_data):
        info = super().create_input_info(request_data)
        info.song = SongRequest(style=info.style, lyrics=info.lyrics, cot=info.cot, seed=info.seed, abc=info.abc, cfg_scale=info.cfg_scale, id=info.id)
        info.abc_generation = resolve_sampling(info.abc_sampling, self.generation.abc)
        info.semantic_generation = resolve_sampling(info.semantic_sampling, self.generation.semantic)
        if info.save_result_path and Path(info.save_result_path).suffix.lower() not in (".flac", ".wav"):
            raise ValueError("YuE2 output must have a .flac or .wav extension")
        return info

    def get_output_extension(self, input_info):
        return ".flac"

    def _check_local_stop(self):
        if getattr(self, "stop_signal", False) or getattr(self, "pause_signal", False):
            raise InterruptedError("YuE2 request cancelled")

    @torch.inference_mode()
    def _generate_audio(self, info, allow_truncated=False):
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
                return self._generate_audio_impl(info, allow_truncated)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous_fp16

    def _generate_audio_impl(self, info, allow_truncated=False):
        request = info.song
        started = time.perf_counter()
        timing = {}
        abc_ids = []
        if request.cot != "off":
            if request.abc is not None:
                abc_ids = self.tokenizer.encode(request.abc)
            else:
                abc_ids, timing["abc"] = self.model.generate_tokens(token_prefixes(request, self.tokenizer), info.abc_generation, request.seed, "abc", self.check_stop)
                if not timing["abc"]["ended"] and not allow_truncated:
                    raise RuntimeError("ABC generation reached its token budget; increase abc_sampling.max_tokens")
        prefix = token_prefixes(request, self.tokenizer, abc_ids)
        negative = negative_prefix(request, self.tokenizer, abc_ids) if request.guidance != 1 else None
        ids, timing["semantic"] = self.model.generate_tokens(
            prefix,
            info.semantic_generation,
            request.seed,
            "semantic",
            self.check_stop,
            negative=negative,
            cfg_scale=request.guidance,
            legacy_off=request.cot == "off",
        )
        if not timing["semantic"]["ended"] and not allow_truncated:
            raise RuntimeError("Audio generation reached its token budget; increase semantic_sampling.max_tokens")
        codec = [v - CODEC_OFFSET for v in ids]
        if not codec:
            raise RuntimeError("YuE2 generated no audio tokens")
        stage = time.perf_counter()
        latents = self.model.synthesize(prefix, codec, request.seed, self.check_stop)
        timing["nar_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        audio, error = None, None
        if self.model.tp_rank == 0:
            try:
                audio = self.vae_decoder.decode(latents, self._check_local_stop)
            except Exception as exc:  # noqa: BLE001 - propagate rank-zero decoder failures to every rank
                error = f"{type(exc).__name__}: {exc}"
        if self.model.tp_size > 1:
            messages = [error]
            dist.broadcast_object_list(messages, src=0, group=self.model.tp_group)
            error = messages[0]
        if error:
            raise RuntimeError(error)
        timing["vae_seconds"] = time.perf_counter() - stage
        timing["generation_seconds"] = time.perf_counter() - started
        timing["semantic_tokens"] = len(codec)
        timing["cfg_branches"] = 1 if request.guidance == 1 else 2
        return audio, timing

    def run_request(self, input_info):
        self._outputs = []
        # NumPy uses 32-bit seeds; model-local generators retain the 63-bit seed.
        seed_all(input_info.seed % (2**32))
        return self.run_pipeline(input_info)

    def compile_graph_count(self):
        if not self.config.get("use_compile", False):
            return 0
        from torch._dynamo.utils import counters

        return counters["stats"]["unique_graphs"]

    @torch.inference_mode()
    def run_pipeline(self, input_info):
        self.input_info = input_info
        torch.cuda.reset_peak_memory_stats(self.init_device)
        try:
            audio, timing = self._generate_audio(input_info)
            if self.model.tp_rank == 0:
                if input_info.save_result_path:
                    path = Path(input_info.save_result_path).expanduser().resolve()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(path), audio.numpy(), 48000, subtype="PCM_24" if path.suffix.lower() == ".flac" else "FLOAT")
                    self.record_output("audio", path)
                    logger.info("YuE2 audio: {} ({:.3f} seconds, 48000 Hz, stereo)", path, len(audio) / 48000)
                timing["audio_seconds"] = len(audio) / 48000
            self.last_metrics = (
                timing
                | self.model.runtime_stats()
                | {
                    "warmup_passes": self.warmup_passes,
                    "warmup_seconds": self.warmup_seconds,
                    "compile_graphs_after_warmup": self.compile_graphs_after_warmup,
                    "compile_graphs_total": self.compile_graph_count(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.init_device),
                }
            )
            logger.info("YuE2 metrics: {}", json.dumps(self.last_metrics))
            return {"audio": audio if input_info.return_result_tensor else None, "sample_rate": 48000, "metrics": self.last_metrics}
        finally:
            self.scheduler.clear()
            self.input_info = None

    @torch.inference_mode()
    def run_warmup(self):
        requests = self.config.get("warmup_requests", None)
        if not requests:
            # Short warmup requests cover variable lengths and CFG.
            requests = [
                {"style": prompt, "lyrics": "", "cot": "off", "cfg_scale": scale, "semantic_sampling": {"min_tokens": count, "max_tokens": count + 1}}
                for prompt, scale, count in zip(self._WARMUP_PROMPTS, (1.0, 1.01), (32, 48), strict=True)
            ]
        started = time.perf_counter()
        self.warmup_passes += 1
        logger.info("YuE2 startup warmup begin: {} rounds", len(requests))
        try:
            with torch.random.fork_rng(devices=[self.init_device.index]):
                for request in requests:
                    info = self.create_input_info({"task": "t2a", **request})
                    _, timing = self._generate_audio(info, allow_truncated=True)
                    logger.info("YuE2 warmup round: {}", json.dumps(timing))
            torch.cuda.synchronize(self.init_device)
        finally:
            self.scheduler.clear()
            self.input_info = None
            self._outputs = []
        self.warmup_seconds = time.perf_counter() - started
        self.compile_graphs_after_warmup = self.compile_graph_count()
        logger.info("YuE2 startup warmup complete: {:.3f}s; graphs retained", self.warmup_seconds)

    def close(self):
        if getattr(self, "model", None) is not None:
            self.model.close()
            self.model = None
        if getattr(self, "vae_decoder", None) is not None:
            self.vae_decoder.close()
            self.vae_decoder = None
        if getattr(self, "scheduler", None) is not None:
            self.scheduler.clear()
        super().close()
