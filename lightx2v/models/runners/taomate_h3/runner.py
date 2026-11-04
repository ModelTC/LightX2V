"""Native TaoMate-H3 Base10-to-Stage3 audio-video runner."""

import json
import subprocess
import time
from contextlib import suppress
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.minimax_h3.packing import unpack_audio_tokens, unpatchify_video_tokens
from lightx2v.models.networks.minimax_h3.taomate.base10.producer import TaoMateBase10Producer
from lightx2v.models.networks.minimax_h3.taomate.model import TaoMateH3NativeModel
from lightx2v.models.networks.minimax_h3.taomate.publication import TaoMateVideoPublication
from lightx2v.models.runners.h3.autoregressive_runner import H3AutoregressiveRunnerMixin
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.schedulers.taomate_h3.geometry import request_plan
from lightx2v.models.schedulers.taomate_h3.scheduler import TaoMateH3Scheduler
from lightx2v.models.video_encoders.hf.minimax_h3_causal.streaming import MiniMaxH3StreamingVideoDecoder
from lightx2v.utils.async_vae import AsyncVAEChunkDecoder
from lightx2v.utils.profiler import no_sync_profiling
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

_VIDEO_NOISE_REQUEST_STRIDE = 1_000_003


@RUNNER_REGISTER("taomate_h3")
class TaoMateH3Runner(H3AutoregressiveRunnerMixin, MiniMaxH3Runner):
    """Generate Base10 audio guides, then Stage3 video and audio in LightX2V."""

    supported_request_fields_by_task = {"t2av": COMMON_REQUEST_FIELDS | VIDEO_OUTPUT_FIELDS | {"prompt"}}
    frames_per_request = 120
    max_continuous_requests = 12

    def __init__(self, config):
        self.segment_prompts = self.segment_seeds = None
        if config.get("prompt_json"):
            prompt_path = Path(config["prompt_json"])
            if not prompt_path.is_absolute() and not prompt_path.exists():
                prompt_path = Path(__file__).resolve().parents[4] / prompt_path
            payload = json.loads(prompt_path.read_text())
            prompts = payload.get("prompts")
            seeds = payload.get("seeds")
            if not isinstance(prompts, list) or not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
                raise ValueError("TaoMate-H3 prompt_json requires a nonempty prompts list")
            if seeds is not None and (not isinstance(seeds, list) or len(seeds) != len(prompts) or any(type(seed) is not int or seed < 0 for seed in seeds)):
                raise ValueError("TaoMate-H3 prompt_json seeds must be one nonnegative integer per prompt")
            if int(config["num_frames"]) != len(prompts) * self.frames_per_request:
                raise ValueError("TaoMate-H3 prompt_json needs one prompt per 120 output frames")
            self.segment_prompts = tuple(prompt.strip() for prompt in prompts)
            self.segment_seeds = tuple(seeds) if seeds is not None else None
        self._segment_inputs = ()
        self._base10_producer = None
        self.parallel_vae_epoch_wait_seconds = 0.0
        super().__init__(config)
        self.request_count = 1

    def get_supported_tasks(self):
        return ("t2av",)

    def init_scheduler(self):
        self.scheduler = TaoMateH3Scheduler(self.config)

    def load_transformer(self):
        return TaoMateH3NativeModel(self.config["model_path"], self.config, self.init_device)

    def _decode_on_this_rank(self):
        return self.config.get("vae_decode_parallel", False) or not dist.is_initialized() or dist.get_rank() == 0

    def load_vae(self):
        if not self._decode_on_this_rank():
            return None, None
        return super().load_vae()

    @torch.no_grad()
    def run_warmup(self):
        """Warm the complete configured request, including Base10 and codecs."""

        self.input_info = self.create_input_info(
            {
                "task": "t2av",
                "prompt": self.config.get("prompt", "A red panda plays a small guitar on a quiet forest stage at sunrise."),
                "seed": 8301,
                "size": list(self.config["size"]),
                "num_frames": self.config["num_frames"],
                "return_result_tensor": True,
            }
        )
        try:
            self.inputs = self._run_input_encoder_local_h3()
            self.run_main()
        finally:
            if self.config.get("cpu_offload", False):
                self._offload_transformer()
            self.model.reset_autoregressive_state()
            self.scheduler.clear()
            self.end_run()

    def _run_input_encoder_local_h3(self):
        if self.segment_prompts is not None:
            self.input_info.prompt = self.segment_prompts[0]
        first = super()._run_input_encoder_local_h3()
        first["text_encoder_output"] = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in first["text_encoder_output"].items()}
        if self.segment_prompts is not None and len(self.segment_prompts) != self.request_count:
            raise ValueError("TaoMate-H3 requested frame count differs from prompt_json")
        inputs = [first]
        for prompt in (self.segment_prompts or ())[1:]:
            segment_info = copy(self.input_info)
            segment_info.prompt = prompt
            features = self.run_text_encoder(segment_info)
            inputs.append({"text_encoder_output": {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in features.items()}})
        self._segment_inputs = tuple(inputs) if self.segment_prompts is not None else (first,) * self.request_count
        return first

    def _segment_seed(self, request_index):
        return int(self.input_info.seed) if self.segment_seeds is None else self.segment_seeds[request_index]

    def _resolve_request_geometry(self, geometry_image=None):
        requested_frames = int(self.input_info.num_frames or self.config["num_frames"])
        if requested_frames <= 0 or requested_frames % self.frames_per_request:
            raise ValueError("TaoMate-H3 publishes 120 frames per request; num_frames must be a positive multiple of 120")
        size = self.input_info.size or self.config["size"]
        if len(size) != 2:
            raise ValueError("TaoMate-H3 size must be [height, width]")
        height, width = map(int, size)
        if height % 32 or width % 32 or min(height, width) not in (480, 768, 1088):
            raise ValueError("TaoMate-H3 canvas needs 32 alignment and a supported short edge")
        self.request_count = requested_frames // self.frames_per_request
        if self.request_count > self.max_continuous_requests:
            raise ValueError("TaoMate-H3 currently supports at most 12 continuous 120-frame requests")
        self.request_height, self.request_width = height, width
        self.request_num_frames = requested_frames
        self.input_info.size = [height, width]

    def _base10_config(self):
        config = dict(self.config)
        config["size"] = [self.request_height, self.request_width]
        for option in ("lora_configs", "lora_dynamic_apply", "lora_path"):
            config.pop(option, None)
        # The Stage3 weights are already loaded. Keep Base10 blocks on the CPU
        # so the two full BF16 DiTs never occupy GPU memory simultaneously.
        config.update(
            cpu_offload=True,
            offload_granularity="block",
            dit_prepost_resident=False,
        )
        base10_cache_dir = config.get("base10_adaln_cache_dir")
        config["use_adaln_cache"] = bool(base10_cache_dir)
        if base10_cache_dir:
            config.update(
                infer_steps=9,
                adaln_cache_profile="taomate_base10",
                adaln_cache_dir=base10_cache_dir,
            )
        if config.get("base10_share_weights", False):
            config.update(cpu_offload=False, dit_prepost_resident=False)
        return config

    def _produce_audio_guidance(self):
        if self._base10_producer is None:
            options = {"shared_base": self.model} if self.config.get("base10_share_weights", False) else {}
            self._base10_producer = TaoMateBase10Producer(self.config["model_path"], self._base10_config(), **options)
        producer = self._base10_producer
        seed = self._segment_seed(0)
        milestones = []
        previous_clean = None
        for request_index, inputs in enumerate(self._segment_inputs):
            self.check_stop()
            text = inputs["text_encoder_output"]
            current = producer.produce(text["prompt_embeds"], text["text_token_tags"], seed, request_index, previous_clean)
            milestones.append(current)
            previous_clean = current[-1]
        return milestones

    def _cache_capacity(self):
        frame_rows = (self.request_height // 32) * (self.request_width // 32)
        phases = request_plan(0).phases
        sink = phases[0].video_latents * frame_rows
        largest_chunk = max(phase.video_latents * frame_rows + 2 * phase.audio_latents for phase in phases)
        return sink + 2 * largest_chunk

    def _prepare_stage3_request(self, milestones, *, request_index):
        seed = self._segment_seed(request_index)
        self.scheduler.prepare_request(
            request_index=request_index,
            video_noise_seed=seed + request_index * _VIDEO_NOISE_REQUEST_STRIDE,
            audio_noise_seed=self._segment_seed(0) + request_index,
            height=self.request_height,
            width=self.request_width,
            text_token_tags=self.inputs["text_encoder_output"]["text_token_tags"],
            teacher_milestones=milestones,
        )
        if self.config.get("cpu_offload", False) and not self.model.block_offload:
            self.model.to_cuda()
        if request_index == 0:
            self.model.initialize_kv_cache(self._cache_capacity())

    def _run_stage3(self, *, request_index, video_decoder=None, async_decoder=None):
        scheduler = self.scheduler
        inputs = self.inputs
        for phase_index in range(len(scheduler.plan.phases)):
            self.check_stop()
            scheduler.prepare_phase(phase_index)
            for step_index in range(scheduler.infer_steps):
                scheduler.step_pre(step_index)
                self.model.infer(inputs)
                scheduler.step_post()
            scheduler.prepare_clean_commit()
            self.model.commit_clean_chunk(inputs, request_index * len(scheduler.plan.phases) + phase_index)
            scheduler.finish_phase()
            if async_decoder is not None:
                if self.model.block_offload:
                    # Staging must observe the offload stream's Graph writes.
                    torch.cuda.current_stream().wait_stream(self.model.transformer_infer.offload_manager.compute_stream)
                latents = unpatchify_video_tokens(
                    scheduler.video_latents,
                    scheduler.phase.video_latents,
                    scheduler.latent_height,
                    scheduler.latent_width,
                )
                async_decoder.submit(video_decoder.push, latents.clone())
                if async_decoder.is_async and video_decoder.vae.decode_parallel:
                    # The decoder gathers tiles on a full-rank NCCL group while
                    # Stage3 uses TP/SP collectives on the main stream. Finish
                    # this decode epoch on every rank before the next Stage3
                    # slot upload or collective enters either communicator.
                    epoch_wait_started = time.perf_counter()
                    async_decoder.synchronize()
                    dist.barrier(group=video_decoder.vae.decode_group)
                    self.parallel_vae_epoch_wait_seconds += time.perf_counter() - epoch_wait_started
        video_rows, audio_rows = scheduler.finish_request()
        if not torch.equal(audio_rows, scheduler.teacher_milestones[-1]):
            raise RuntimeError("TaoMate Stage3 published audio differs from the Base10 clean milestone")
        return video_rows, audio_rows

    @staticmethod
    def _finish_video_decode(decoder):
        tail = decoder.finish()
        return () if tail is None else (tail,)

    @staticmethod
    def _retime_video_for_publication(video, published_frames):
        """Match TaoMate's endpoint-preserving FFmpeg fps filter on decoded frames."""

        native_frames = video.shape[2]
        output_frames = torch.arange(1, published_frames + 1, device=video.device)
        source_indices = (output_frames * native_frames - 1) // published_frames
        return video.index_select(2, source_indices)

    @staticmethod
    def _retime_audio_for_publication(waveform, sample_rate, target_samples):
        native_samples = waveform.shape[-1]
        interleaved = waveform[0].float().cpu().transpose(0, 1).contiguous().numpy()
        audio_filter = f"atempo={native_samples}/{target_samples},aresample={sample_rate},apad=whole_len={target_samples},atrim=end_sample={target_samples},asetpts=N/SR/TB"
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "f32le",
                "-ar",
                str(sample_rate),
                "-ac",
                "2",
                "-i",
                "pipe:0",
                "-af",
                audio_filter,
                "-f",
                "f32le",
                "pipe:1",
            ],
            input=interleaved.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        samples = np.frombuffer(result.stdout, dtype="<f4")
        if samples.size != target_samples * 2:
            raise RuntimeError(f"TaoMate audio filter returned {samples.size // 2} samples, expected {target_samples}")
        return torch.from_numpy(samples.copy().reshape(target_samples, 2)).T.unsqueeze(0)

    def _log_cuda_graph(self):
        stack = self.model.transformer_infer.cuda_graph_stack
        if stack is not None:
            logger.info("TaoMate Stage3 CUDA Graph: captures={}, replays={}", stack.capture_count, stack.replay_count)

    def run_main(self):
        self.parallel_vae_epoch_wait_seconds = 0.0
        transformer_offloaded = not self.config.get("cpu_offload", False)
        async_decoder = None
        publication = None
        try:
            milestones = self._produce_audio_guidance()
            if self._decode_on_this_rank():
                self._ensure_vae_loaded()
                self.set_vae_decode_tile_shape()
                native_frames = request_plan(0).native_frames + (self.request_count - 1) * request_plan(1).native_frames
                video_decoder = MiniMaxH3StreamingVideoDecoder(
                    self.video_vae,
                    native_frames,
                    output_uint8=not self.input_info.return_result_tensor,
                )
                if getattr(self, "_streaming_video_output", None) is not None:
                    publication = TaoMateVideoPublication(native_frames, self.frames_per_request * self.request_count, self.record_video_chunk)
                async_decoder = AsyncVAEChunkDecoder.from_config(
                    self.config,
                    device=AI_DEVICE,
                    dispatch_in_worker=True,
                    on_chunk=publication,
                    collect_outputs=not self.streaming_video_only,
                )
                if async_decoder.is_async:
                    async_decoder.prepare_vae(self.video_vae)
            else:
                video_decoder = None
            # Global CUDA capture cannot overlap a decoder worker allocating
            # or synchronizing GPU memory. Drain before a new capture. With
            # distributed VAE decode, _run_stage3 also drains every decode
            # epoch before the next Stage3 phase, including warmed replays.
            self.model.transformer_infer.before_graph_capture = async_decoder.synchronize if async_decoder is not None and async_decoder.is_async else None
            audio_chunks = []
            with no_sync_profiling(enabled=async_decoder is not None and async_decoder.is_async):
                for request_index, current in enumerate(milestones):
                    self.inputs = self._segment_inputs[request_index]
                    self._prepare_stage3_request(current, request_index=request_index)
                    _, audio_rows = self._run_stage3(request_index=request_index, video_decoder=video_decoder, async_decoder=async_decoder)
                    audio_chunks.append(audio_rows.reshape(2, self.scheduler.plan.audio_latents, 32))
                if async_decoder is not None:
                    async_decoder.submit(self._finish_video_decode, video_decoder)
                    video_chunks = [frames for output in async_decoder.finish() for frames in output]
                    if publication is not None:
                        publication.finish()
            async_decoder = None
            self._log_cuda_graph()
            if self.config.get("cpu_offload", False):
                self._offload_transformer()
                transformer_offloaded = True
            if not self._decode_on_this_rank():
                return {"video": None, "audio": None}
            self.gen_video = torch.cat(video_chunks, dim=2) if video_chunks else None
            self.gen_audio = None
            if not self.video_vae.decode_parallel or not dist.is_initialized() or dist.get_rank() == 0:
                audio_rows = torch.cat(audio_chunks, dim=1).reshape(-1, 32)
                self.gen_audio = self.audio_vae.decode(unpack_audio_tokens(audio_rows, sum(chunk.shape[1] for chunk in audio_chunks)))
            if self.gen_video is not None:
                self.gen_video = self._retime_video_for_publication(self.gen_video, self.frames_per_request * self.request_count)
            if self.gen_audio is not None:
                sample_rate = self.audio_vae.sampling_rate
                target_samples = round(self.frames_per_request * self.request_count * sample_rate / self.config["fps"])
                self.gen_audio = self._retime_audio_for_publication(self.gen_audio, sample_rate, target_samples)
                self.set_stream_audio(self.gen_audio, sample_rate)
            return self.process_images_after_vae_decoder()
        finally:
            self.model.transformer_infer.before_graph_capture = None
            if async_decoder is not None:
                async_decoder.finish()
            if self.config.get("cpu_offload", False) and not transformer_offloaded:
                with suppress(Exception):
                    self._offload_transformer()
            self._release_low_memory_vae()
            self.scheduler.clear()
            self.end_run()
            self.gen_video = None
            self.gen_audio = None
            self._segment_inputs = ()

    def close(self):
        if self._base10_producer is not None:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            self._base10_producer.close()
            self._base10_producer = None
        super().close()


__all__ = ["TaoMateH3Runner"]
