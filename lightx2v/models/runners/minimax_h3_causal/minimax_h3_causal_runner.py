import io
import math
import wave
from contextlib import suppress
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from PIL import Image
from loguru import logger

from lightx2v.models.audio_encoders.hf.minimax_h3.audio_vae import MiniMaxH3AudioEncoderPrefixCache
from lightx2v.models.input_encoders.hf.minimax_h3_causal import MiniMaxH3CausalQwen3VLTextEncoder
from lightx2v.models.networks.minimax_h3.causal.model import MiniMaxH3CausalModel
from lightx2v.models.networks.minimax_h3.packing import FPS, FRAMES_PER_CHUNK, align_num_frames, audio_latent_num_frames, unpack_audio_tokens, unpatchify_video_tokens
from lightx2v.models.networks.minimax_h3.packing_ref2av import MAX_REFERENCE_IMAGES, MiniMaxH3PreparedReference, resolve_reference_image_size
from lightx2v.models.runners.h3.autoregressive_runner import H3AutoregressiveRunnerMixin
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.runners.minimax_h3_causal.action_prompt_travel import ActionPromptTravel, ActionPromptTravelConfig, coerce_action_prompts
from lightx2v.models.runners.minimax_h3_causal.reference_cache import MiniMaxH3ReferenceCache
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.schedulers.minimax_h3_causal.scheduler import MiniMaxH3CausalScheduler
from lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops import Audio
from lightx2v.models.video_encoders.hf.minimax_h3_causal.streaming import MiniMaxH3StreamingVideoDecoder
from lightx2v.models.video_encoders.hf.minimax_h3_causal.vae_graph import MiniMaxH3VideoDecoderGraph
from lightx2v.models.video_encoders.hf.minimax_h3_causal.video_writer import MiniMaxH3StreamingVideoWriter
from lightx2v.utils.async_vae import AsyncVAEChunkDecoder
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import INPUT_INFO_TYPES
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2, no_sync_profiling
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import seed_all
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("minimax_h3_causal")
class MiniMaxH3CausalRunner(H3AutoregressiveRunnerMixin, MiniMaxH3Runner):
    _WARMUP_SIZE = (256, 256)

    supported_request_fields_by_task = {
        "refa2v": COMMON_REQUEST_FIELDS | VIDEO_OUTPUT_FIELDS | {"image_path", "audio_path", "prompt", "action_prompts"},
    }

    def __init__(self, config):
        ar_config = config.get("ar_config", {})
        scheme = ar_config.get("kv_cache_scheme", "rolling")
        if scheme not in {"rolling", "fp", "fp8"} or ar_config.get("step_kv_cache", True) is not True:
            raise ValueError("Causal H3 requires rolling or fp8 KV and step_kv_cache=true")
        if scheme == "fp8" and ar_config.get("kv_offload", False):
            raise ValueError("Causal H3 FP8 KV does not support KV offload")
        if config.get("async_vae_decode", False) and config.get("vae_decode_overlap", False):
            raise ValueError("Select either async_vae_decode or vae_decode_overlap, not both")
        if GET_DTYPE() != torch.bfloat16:
            raise ValueError("Causal H3 requires DTYPE=BF16 to match the training checkpoint")
        if config.get("lora_configs"):
            raise ValueError("Causal H3 loads the distilled checkpoint directly; released H3 LoRA adapters are not supported")
        self.action_prompt_travel_config = ActionPromptTravelConfig.model_validate(config.get("action_prompt_travel") or {})
        self.reference_cache = MiniMaxH3ReferenceCache() if config.get("reference_condition_cache", False) else None
        self.video_writer = None
        self.video_decode_graph = None
        super().__init__(config)
        # Called on rank 0 with (RGB [1,3,F,H,W], start_frame, is_final).
        self.stream_callback = None

    def get_supported_tasks(self):
        return ("refa2v",)

    def create_input_info(self, request_data):
        input_info = super().create_input_info(request_data)
        input_info.action_prompts = coerce_action_prompts(input_info.action_prompts)
        return input_info

    def clear_conditioning_state(self):
        super().clear_conditioning_state()
        self.action_travel = None
        self.text_reference_context = None

    def run_text_encoder(self, input_info, keyframes=None, references=None):
        use_action_travel = self.action_prompt_travel_config.enabled and bool(input_info.action_prompts)
        if not use_action_travel and self.reference_cache is None:
            return super().run_text_encoder(input_info, keyframes=keyframes, references=references)
        encoder = self.text_encoders[0]
        if self.text_reference_context is None:
            self.text_reference_context = encoder.prepare_reference_context(references)
        if not use_action_travel:
            if (input_info.negative_prompt or "").strip():
                logger.warning("MiniMax-H3 is guidance-distilled; negative_prompt is ignored")
            return encoder.infer(input_info.prompt, reference_context=self.text_reference_context)
        reference_length = len(self.text_reference_context["token_ids"])
        self.action_travel = ActionPromptTravel(input_info.prompt, input_info.action_prompts, self.action_prompt_travel_config, encoder.tokenizer, reference_length)
        return self.encode_action_prompt()

    def encode_action_prompt(self):
        travel = self.action_travel
        text = self.text_encoders[0].infer(
            travel.prompt,
            reference_context=self.text_reference_context,
            pad_to_len=travel.text_length if travel.config.naive_travel else None,
            pad_mode=travel.config.naive_pad,
        )
        if text["text_token_tags"].numel() != travel.text_length:
            raise RuntimeError("H3 action prompt length changed while rolling")
        return text

    def init_scheduler(self):
        self.scheduler = MiniMaxH3CausalScheduler(self.config)

    def load_transformer(self):
        return MiniMaxH3CausalModel(self.config["model_path"], self.config, self.init_device)

    def load_text_encoder(self):
        return [MiniMaxH3CausalQwen3VLTextEncoder(self.config)]

    def load_vae(self):
        video_vae, audio_vae = super().load_vae()
        video_vae.stitch_preallocate = self.config.get("vae_stitch_preallocate", False)
        if self.config.get("vae_decode_overlap", False):
            self.video_decode_graph = MiniMaxH3VideoDecoderGraph(video_vae)
            logger.info("MiniMax-H3 causal VAE: tile CUDA Graph + DiT overlap enabled")
        if self.config.get("vae_encoder_cpu_offload", False) and not video_vae.cpu_offload:
            video_vae.encoder.cpu()
            video_vae.quant_conv.cpu()
        # The decoder is unused when the output reuses the driving waveform.
        if self.config.get("reuse_input_audio", False) and not audio_vae.cpu_offload:
            audio_vae.decoder.cpu()
            audio_vae.dec_in_proj.cpu()
        return video_vae, audio_vae

    def _encode_references(self, references):
        if self.video_vae.cpu_offload or not self.config.get("vae_encoder_cpu_offload", False):
            return super()._encode_references(references)
        try:
            self.video_vae.encoder.to(self.video_vae.execution_device)
            self.video_vae.quant_conv.to(self.video_vae.execution_device)
            return super()._encode_references(references)
        finally:
            self.video_vae.encoder.cpu()
            self.video_vae.quant_conv.cpu()

    @ProfilingContext4DebugL1("Warmup")
    def run_warmup(self):
        sizes = dict.fromkeys((self._WARMUP_SIZE, tuple(self.config["size"])))
        for height, width in sizes:
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
                # Request preparation pads this short stereo PCM clip to the
                # configured duration before the normal streaming encoder.
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
                if self.config.get("cpu_offload", False):
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
        cached = self.reference_cache.get(self.prepared_references) if self.reference_cache is not None else None
        if cached is not None:
            self.text_reference_context, latents = cached
        text = self.run_text_encoder(self.input_info, references=self.prepared_references)
        if cached is None:
            latents = self._encode_references(self.prepared_references)
            if self.reference_cache is not None:
                self.reference_cache.store(self.prepared_references, self.text_reference_context, latents)
        self.condition_video_latents, self.condition_audio_latents = latents
        waveform, sample_rate = torchaudio.load(self.input_info.audio_path)
        if waveform.shape[-1] == 0:
            raise ValueError("RefA2V driving audio must contain samples")
        if self.config.get("audio_mono", False):
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)
        else:
            waveform = waveform[:2]
        if sample_rate != self.audio_vae.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.audio_vae.sampling_rate)
        samples = audio_latent_num_frames(self.request_num_frames) * self.audio_vae.hop_length
        if self.config.get("loop_audio", False) and waveform.shape[-1] < samples:
            repeats = (samples + waveform.shape[-1] - 1) // waveform.shape[-1]
            waveform = waveform.repeat(1, repeats)
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

    def _use_streaming_video_writer(self, input_info):
        return self.config.get("video_streaming_encode", False) and self.config.get("reuse_input_audio", False) and not input_info.return_result_tensor and self.stream_callback is None

    def get_stream_output_path(self, input_info):
        if self._use_streaming_video_writer(input_info):
            return None
        return super().get_stream_output_path(input_info)

    def init_run(self):
        super().init_run()
        output_path = super().get_stream_output_path(self.input_info)
        if output_path and self._use_streaming_video_writer(self.input_info):
            if Path(output_path).suffix.lower() != ".mp4":
                raise ValueError("MiniMax-H3 AV output uses H.264/AAC; save_result_path must end in .mp4")
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.video_writer = MiniMaxH3StreamingVideoWriter(
                    output_path,
                    self.request_num_frames,
                    int(self.config.get("fps", 24)),
                    Audio(waveform=self.waveform.float().cpu(), sampling_rate=self.audio_vae.sampling_rate),
                    pinned_memory=self.config.get("vae_rgb_pinned_memory", False),
                    video_codec_options=self.config.get("video_codec_options"),
                )
        self.model.initialize_kv_cache(self.scheduler.plan.cache_size)

    def fill_condition_cache(self, *, overwrite=False):
        self.scheduler.prepare_condition(overwrite=overwrite)
        for step in range(self.scheduler.infer_steps):
            self.check_stop()
            self.scheduler.step_pre(step)
            self.model.infer(self.inputs)

    def run_segment(self, segment_idx=0):
        self.fill_condition_cache()
        self.set_vae_decode_tile_shape()
        decoder = MiniMaxH3StreamingVideoDecoder(
            self.video_vae,
            self.request_num_frames,
            output_uint8=not self.input_info.return_result_tensor and self.stream_callback is None,
            # The shared file writer already bounds RGB storage by chunk.
            pinned_memory=self.config.get("vae_rgb_pinned_memory", False) and not self.streaming_video_only,
            decode_graph=self.video_decode_graph,
            output_writer=self.video_writer,
        )
        async_decoder = None
        if self.config.get("async_vae_decode", False):
            async_decoder = AsyncVAEChunkDecoder.from_config(
                self.config,
                device=AI_DEVICE,
                vae_decoder=self.video_vae,
                dispatch_in_worker=True,
                collect_outputs=not self.streaming_video_only,
            )
        # Capturing a DiT graph must wait for work on the VAE stream.
        self.model.transformer_infer.before_graph_capture = async_decoder.synchronize if async_decoder is not None else torch.cuda.synchronize
        reuse_input_audio = self.config.get("reuse_input_audio", False)
        self._decoded_video_frames = 0
        with no_sync_profiling(self.video_decode_graph is not None or async_decoder is not None):
            try:
                videos, left, right = [], [], []
                chunks = self.scheduler.plan.chunks
                waveform = self.waveform[:1] if self.config.get("audio_mono", False) else self.waveform
                audio_encoder_cache = MiniMaxH3AudioEncoderPrefixCache(self.audio_vae.encoder)
                for index, chunk in enumerate(chunks):
                    self.check_stop()
                    logger.info(f"MiniMax-H3 causal chunk {index + 1}/{len(chunks)}")
                    # Recompute the right edge from the current prefix; future audio
                    # must not change the current chunk's conditioning.
                    latents = self.audio_vae.encode(waveform[..., : chunk.audio_frames.stop * self.audio_vae.hop_length], return_cpu=False, encoder_cache=audio_encoder_cache)
                    if waveform.shape[0] == 1:
                        latents = latents.expand(2, -1, -1)
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
                    if async_decoder is None:
                        videos.extend(self.decode_video_chunk(decoder, video))
                    else:
                        async_decoder.submit(self.decode_video_chunk, decoder, video.clone())
                    travel = self.action_travel
                    if travel is not None and travel.config.min_slice_rolling and (index + 1) % travel.config.unit_transition == 0 and index + 1 < len(chunks):
                        if async_decoder is not None:
                            async_decoder.synchronize()
                        seconds = (decoder.emitted_frames or (index + 1) * FRAMES_PER_CHUNK) / FPS
                        if travel.update(seconds):
                            self.inputs["text_encoder_output"] = self.encode_action_prompt()
                            self.fill_condition_cache(overwrite=True)
                            logger.info(f"H3 action prompt updated after chunk {index + 1} at {seconds:.3f}s")
                    if not reuse_input_audio:
                        left.append(clean[: chunk.num_audio_frames].cpu())
                        right.append(clean[chunk.num_audio_frames :].cpu())
                if async_decoder is None:
                    videos.extend(self.decode_video_chunk(decoder))
                else:
                    async_decoder.submit(self.decode_video_chunk, decoder)
                    videos = [frames for outputs in async_decoder.finish() for frames in outputs]
                video = decoder.output_buffer if decoder.output_buffer is not None else (torch.cat(videos, dim=2) if videos else None)
                audio_rows = None if reuse_input_audio else torch.cat((torch.cat(left), torch.cat(right)))
                return video, audio_rows
            finally:
                try:
                    if async_decoder is not None:
                        async_decoder.finish()
                finally:
                    self.model.transformer_infer.before_graph_capture = None
                    decoder.close()

    def decode_video_chunk(self, decoder, video=None):
        if video is None:
            tail = decoder.finish()
            frames = [tail] if tail is not None else []
        else:
            frames = decoder.push(video)
        for pixels in frames:
            self.record_video_chunk(pixels, value_range=(0, 1))
            if self.stream_callback is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                self.stream_callback(pixels, self._decoded_video_frames, video is None)
            self._decoded_video_frames += pixels.shape[2]
        return frames if decoder.output_buffer is None and not self.streaming_video_only else []

    def run_vae_decoder(self, video, audio_rows):
        # FP8 caches have no persistent graph bindings and can be freed before audio decoding.
        self.model.reset_autoregressive_state()
        audio_output = video is not None or (self.streaming_video_only and (not dist.is_initialized() or dist.get_rank() == 0))
        if not audio_output:
            return video, None
        audio = self.waveform.unsqueeze(0) if self.config.get("reuse_input_audio", False) else self.audio_vae.decode(unpack_audio_tokens(audio_rows, self.scheduler.num_audio_latents))
        self.set_stream_audio(audio, self.audio_vae.sampling_rate)
        return video, audio

    def process_images_after_vae_decoder(self):
        if self.video_writer is None:
            return super().process_images_after_vae_decoder()
        # Only residual encoding remains here; earlier chunks overlapped DiT.
        with ProfilingContext4DebugL2("Finish Audio-Video Output"):
            self.video_writer.finish()
        logger.info("Video saved to {}", self.video_writer.output_path)
        return {"video": None, "audio": None}

    def end_run(self):
        try:
            if self.video_writer is not None:
                self.video_writer.close()
        finally:
            self.video_writer = None
            self.model.reset_autoregressive_state()
            super().end_run()
            self.waveform = None

    def close(self):
        try:
            if self.video_decode_graph is not None:
                self.video_decode_graph.clear()
        finally:
            super().close()
