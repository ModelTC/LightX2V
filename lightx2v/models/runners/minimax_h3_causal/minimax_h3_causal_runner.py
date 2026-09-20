import io
import math
import wave
from contextlib import suppress

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from PIL import Image
from loguru import logger

from lightx2v.common.kvcache.manager import KVCacheManager
from lightx2v.models.networks.minimax_h3.packing import align_num_frames, audio_latent_num_frames, unpack_audio_tokens, unpatchify_video_tokens
from lightx2v.models.networks.minimax_h3.packing_ref2av import MAX_REFERENCE_IMAGES, MiniMaxH3PreparedReference, resolve_reference_image_size
from lightx2v.models.networks.minimax_h3_causal.model import MiniMaxH3CausalModel
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.schedulers.minimax_h3_causal.scheduler import MiniMaxH3CausalScheduler
from lightx2v.models.video_encoders.hf.minimax_h3_causal.streaming import MiniMaxH3StreamingVideoDecoder
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import INPUT_INFO_TYPES
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import seed_all
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("minimax_h3_causal")
class MiniMaxH3CausalRunner(MiniMaxH3Runner):
    supported_request_fields_by_task = {
        "refa2v": COMMON_REQUEST_FIELDS | VIDEO_OUTPUT_FIELDS | {"image_path", "audio_path", "prompt"},
    }

    def __init__(self, config):
        ar_config = config.get("ar_config", {})
        if ar_config.get("kv_offload", False):
            raise NotImplementedError("Causal H3 does not support KV cache offload")
        if ar_config.get("kv_cache_scheme", "fp") != "fp" or ar_config.get("step_kv_cache", True) is not True:
            raise ValueError("Causal H3 requires ar_config.kv_cache_scheme='fp' and ar_config.step_kv_cache=true")
        if ar_config.get("kv_quant"):
            raise NotImplementedError("Causal H3 does not support KV cache quantization")
        if GET_DTYPE() != torch.bfloat16:
            raise ValueError("Causal H3 requires DTYPE=BF16 to match the training checkpoint")
        if config.get("lora_configs"):
            raise ValueError("Causal H3 loads the distilled checkpoint directly; released H3 LoRA adapters are not supported")
        super().__init__(config)
        self.kv_cache_manager = None
        # Called on rank 0 with (RGB [1,3,F,H,W], start_frame, is_final).
        self.stream_callback = None

    def get_supported_tasks(self):
        return ("refa2v",)

    def init_scheduler(self):
        self.scheduler = MiniMaxH3CausalScheduler(self.config)

    def load_transformer(self):
        return MiniMaxH3CausalModel(self.config["model_path"], self.config, self.init_device)

    @ProfilingContext4DebugL1("Warmup")
    def run_warmup(self):
        height, width = self.config["size"]
        self.input_info = INPUT_INFO_TYPES["refa2v"](
            task="refa2v",
            seed=0,
            prompt="A person speaks naturally into a microphone.",
            image_path=Image.new("RGB", (width, height)),
            size=[height, width],
            num_frames=self.config["num_frames"],
        )
        transformer_offloaded = not self.config.get("cpu_offload", False)
        try:
            # Match the cuDNN settings installed by BaseRunner.run_request.
            seed_all(self.input_info.seed)
            self.scheduler.generator = None
            # Exercise the normal audio reader too. Request preparation pads
            # this short stereo PCM clip to the configured duration.
            with io.BytesIO() as audio_file:
                with wave.open(audio_file, "wb") as pcm:
                    pcm.setnchannels(2)
                    pcm.setsampwidth(2)
                    pcm.setframerate(self.audio_vae.sampling_rate)
                    pcm.writeframes(bytes(2 * 2 * self.audio_vae.hop_length))
                audio_file.seek(0)
                self.input_info.audio_path = audio_file
                self.inputs = self._run_input_encoder_local_h3()
            logger.info(f"Warmup refa2v: {self.request_height}x{self.request_width}x{self.request_num_frames}")
            self.init_run()
            video, audio_rows = self.run_segment()
            self._offload_transformer()
            transformer_offloaded = True
            self.run_vae_decoder(video, audio_rows)
            getattr(torch, AI_DEVICE).synchronize()
            del video, audio_rows
        finally:
            if not transformer_offloaded:
                with suppress(Exception):
                    self._offload_transformer()
            self.end_run()
        self._maybe_freeze_gc()

    def _resolve_request_geometry(self, geometry_image=None):
        self.request_height, self.request_width = map(int, self.input_info.size or self.config["size"])
        self.request_num_frames = align_num_frames(int(self.input_info.num_frames or self.config["num_frames"]))
        if self.request_num_frames < 22 or min(self.request_height, self.request_width) <= 0 or self.request_height % 32 or self.request_width % 32:
            raise ValueError("Causal H3 requires at least 22 frames and positive canvas dimensions divisible by 32")
        self.input_info.size = [self.request_height, self.request_width]
        self.input_info.num_frames = self.request_num_frames

    def _run_input_encoder_local_h3(self):
        if not self.input_info.audio_path or not self.input_info.image_path:
            raise ValueError("RefA2V requires a reference image and a driving audio file")
        self.clear_conditioning_state()
        self._resolve_request_geometry()
        self.prepared_references = self._prepare_references()
        text = self.run_text_encoder(self.input_info, references=self.prepared_references)
        self.condition_video_latents, self.condition_audio_latents = self._encode_references(self.prepared_references)
        waveform, sample_rate = torchaudio.load(self.input_info.audio_path)
        if waveform.shape[-1] == 0:
            raise ValueError("RefA2V driving audio must contain samples")
        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)
        else:
            waveform = waveform[:2]
        if sample_rate != self.audio_vae.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.audio_vae.sampling_rate)
        samples = audio_latent_num_frames(self.request_num_frames) * self.audio_vae.hop_length
        self.waveform = F.pad(waveform[..., :samples], (0, max(0, samples - waveform.shape[-1]))).contiguous()
        return {"text_encoder_output": text}

    def _prepare_references(self):
        paths = self._split_reference_paths(self.input_info.image_path)
        if not 1 <= len(paths) <= MAX_REFERENCE_IMAGES:
            raise ValueError("Causal H3 requires 1..9 reference images")
        references = []
        for path in paths:
            image = self._load_rgb_image(path)
            # Zoe pads extreme aspect ratios, then fits the image without
            # distortion into the 32-aligned canvas used by both encoders.
            width = max(image.width, math.ceil(image.height / 4))
            height = max(image.height, math.ceil(image.width / 4))
            canvas = Image.new("RGB", (width, height))
            canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            height, width = resolve_reference_image_size(
                width, height, target_width=self.request_width, target_height=self.request_height, mode=self.config.get("reference_image_resize_mode", "match")
            )
            height = max(height, math.ceil(width / 128) * 32)
            width = max(width, math.ceil(height / 128) * 32)
            scale = min(width / canvas.width, height / canvas.height)
            resized = canvas.resize((min(width, max(1, round(canvas.width * scale))), min(height, max(1, round(canvas.height * scale)))), Image.Resampling.LANCZOS)
            image = Image.new("RGB", (width, height))
            image.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
            references.append(MiniMaxH3PreparedReference("image", image=image))
        return references

    def init_run(self):
        super().init_run()
        heads = int(self.config["num_attention_heads"])
        head_dim = int(self.config["attention_head_dim"])
        sp_size = dist.get_world_size(self.model.seq_p_group) if self.model.seq_p_group is not None else 1
        cache_config = dict(self.config, num_heads=heads, dim=heads * head_dim, ar_config=dict(self.config["ar_config"]))
        self.kv_cache_manager = KVCacheManager(cache_config, device=AI_DEVICE, sp_group=self.model.seq_p_group)
        self.scheduler.cache = self.kv_cache_manager.create_self_attn_kv_cache(
            "media",
            self.scheduler.plan.cache_size,
            kv_cache_scheme="fp",
            step_kv_cache=True,
            num_heads=heads // self.model.tp_size // sp_size,
        )
        self.scheduler.cache._init_kv_buffer()

    def run_segment(self, segment_idx=0):
        for step in range(self.scheduler.infer_steps):
            self.scheduler.step_pre(step)
            self.model.infer(self.inputs)
        self.set_vae_decode_tile_shape()
        # Callbacks and tensor returns retain their float RGB contract.
        decoder = MiniMaxH3StreamingVideoDecoder(
            self.video_vae,
            self.request_num_frames,
            output_uint8=not self.input_info.return_result_tensor and self.stream_callback is None,
        )
        videos, left, right = [], [], []
        emitted = 0
        chunks = self.scheduler.plan.chunks
        for index, chunk in enumerate(chunks):
            self.check_stop()
            logger.info(f"MiniMax-H3 causal chunk {index + 1}/{len(chunks)}")
            # Re-encode only the available waveform prefix, and commit only new
            # latents. Full-waveform encoding would leak future audio context.
            latents = self.audio_vae.encode(self.waveform[..., : chunk.audio_frames.stop * self.audio_vae.hop_length], return_cpu=False)
            clean = latents[..., chunk.audio_frames].transpose(1, 2).reshape(-1, latents.shape[1]).contiguous()
            self.scheduler.prepare_chunk(index, clean)
            for step in range(self.scheduler.infer_steps):
                self.check_stop()
                self.scheduler.step_pre(step)
                self.model.infer(self.inputs)
                self.scheduler.step_post()
                if self.progress_callback:
                    self.progress_callback(100 * (index * self.scheduler.infer_steps + step + 1) / (len(chunks) * self.scheduler.infer_steps), 100)
            video = unpatchify_video_tokens(self.scheduler.video_latents, chunk.video_frames.stop - chunk.video_frames.start, self.scheduler.latent_height, self.scheduler.latent_width)
            for frames in decoder.push(video):
                videos.append(frames)
                if self.stream_callback is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                    self.stream_callback(frames, emitted, False)
                emitted += frames.shape[2]
            left.append(clean[: chunk.num_audio_frames].cpu())
            right.append(clean[chunk.num_audio_frames :].cpu())
        tail = decoder.finish()
        if tail is not None:
            videos.append(tail)
            if self.stream_callback is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                self.stream_callback(tail, emitted, True)
        return torch.cat(videos, dim=2) if videos else None, torch.cat((torch.cat(left), torch.cat(right)))

    def run_vae_decoder(self, video, audio_rows):
        audio = self.audio_vae.decode(unpack_audio_tokens(audio_rows, self.scheduler.num_audio_latents)) if video is not None else None
        return video, audio

    def end_run(self):
        super().end_run()
        self.kv_cache_manager = None
        self.waveform = None
