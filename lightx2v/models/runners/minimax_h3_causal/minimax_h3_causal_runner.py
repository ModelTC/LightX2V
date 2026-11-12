import io
import math
import time
import wave
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from PIL import Image
from loguru import logger

from lightx2v.models.audio_encoders.hf.minimax_h3.audio_vae import MiniMaxH3AudioEncoderPrefixCache
from lightx2v.models.input_encoders.hf.minimax_h3_causal import MiniMaxH3CausalQwen3VLTextEncoder
from lightx2v.models.networks.minimax_h3.causal.model import MiniMaxH3CausalModel
from lightx2v.models.networks.minimax_h3.causal.streaming import build_causal_stream_chunk
from lightx2v.models.networks.minimax_h3.packing import (
    AUDIO_LATENTS_PER_SECOND,
    FPS,
    FRAMES_PER_CHUNK,
    LATENTS_PER_CHUNK,
    align_num_frames,
    audio_latent_num_frames,
    temporal_position_span,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
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
from lightx2v.utils.va_controller import VAController
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
        self.stream_waveform = None
        self.last_rest_audio = None
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
        self.stream_waveform = None
        self.last_rest_audio = None
        self.text_reference_context = None

    def run_text_encoder(self, input_info, keyframes=None, references=None):
        use_action_travel = self.action_prompt_travel_config.enabled and bool(input_info.action_prompts)
        encoder = self.text_encoders[0]
        if self.text_reference_context is None:
            self.text_reference_context = encoder.prepare_reference_context(references)
        if not use_action_travel:
            if (input_info.negative_prompt or "").strip():
                logger.warning("MiniMax-H3 is guidance-distilled; negative_prompt is ignored")
            text = encoder.infer(input_info.prompt, reference_context=self.text_reference_context)
            text["_h3_prompt"] = input_info.prompt
            return text
        reference_length = len(self.text_reference_context["token_ids"])
        self.action_travel = ActionPromptTravel(input_info.prompt, input_info.action_prompts, self.action_prompt_travel_config, encoder.tokenizer, reference_length)
        if isinstance(input_info.audio_path, dict) and self.action_prompt_travel_config.naive_travel:
            empty_length = self.action_travel.presentation_length(self.action_travel.assemble_prompt(""))
            self.action_travel.text_length = max(self.action_travel.text_length, empty_length + self.action_prompt_travel_config.action_slot_len)
        return self.encode_action_prompt()

    def encode_action_prompt(self, action_prompt=None):
        travel = self.action_travel
        prompt = travel.prompt if travel is not None else self.input_info.prompt
        if action_prompt is not None:
            body = str(action_prompt).strip()
            if body and travel is not None and travel.config.naive_travel and travel.config.prompt_form == "six_key" and not body.startswith("[Action"):
                body = f"[Action 1]: {body}"
            prompt = travel.assemble_prompt(body) if travel is not None else body
        return self._encode_live_prompt(prompt)

    def _encode_live_prompt(self, prompt):
        travel = self.action_travel
        text = self.text_encoders[0].infer(
            prompt,
            reference_context=self.text_reference_context,
            pad_to_len=travel.text_length if travel is not None and travel.config.naive_travel else None,
            pad_mode=travel.config.naive_pad if travel is not None else "token_id",
        )
        if travel is not None and text["text_token_tags"].numel() != travel.text_length:
            raise RuntimeError("H3 action prompt length changed while rolling")
        # Keep the exact presentation, including its action slot. A reference
        # change must re-encode it without assembling that slot a second time.
        text["_h3_prompt"] = prompt
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
        text = self._encode_reference_conditions(self._prepare_references())

        # stream audio input is dict
        if not isinstance(self.input_info.audio_path, str):
            return {"text_encoder_output": text}

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

    def _encode_reference_conditions(self, references):
        self.prepared_references = references
        self.text_reference_context = None
        self.action_travel = None
        cached = self.reference_cache.get(self.prepared_references) if self.reference_cache is not None else None
        if cached is not None:
            self.text_reference_context, latents = cached
        text = self.run_text_encoder(self.input_info, references=self.prepared_references)
        if cached is None:
            latents = self._encode_references(self.prepared_references)
            if self.reference_cache is not None:
                self.reference_cache.store(self.prepared_references, self.text_reference_context, latents)
        self.condition_video_latents, self.condition_audio_latents = latents
        return text

    def _prepare_references(self, image_path=None):
        paths = self._split_reference_paths(self.input_info.image_path if image_path is None else image_path)
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
        self._initialize_kv_cache()

    def _initialize_kv_cache(self):
        plan = self.scheduler.plan
        capacity = plan.cache_size
        if isinstance(self.input_info.audio_path, dict):
            # A short configured clip still needs a full rolling window when
            # used as an unbounded live stream. The first full chunk has the
            # largest (29-latent) audio block.
            capacity = max(capacity, plan.condition_layout.sequence_length + (plan.sink_chunks + plan.latest_chunks) * plan.chunks[0].positions.shape[0])
        self.model.initialize_kv_cache(capacity)

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
            self.stream_waveform = None
            self.last_rest_audio = None

    def _get_stream_chunk(self, chunk_index, start_chunk=0):
        chunk_index = int(chunk_index)
        tags = self.inputs["text_encoder_output"]["text_token_tags"]
        rotary_time = float(tags.shape[0])
        for reference in self.prepared_references:
            if reference.kind == "image":
                rotary_time += 1.0
            elif reference.kind == "audio":
                rotary_time += float(reference.num_audio_latents)
            elif reference.kind == "video":
                rotary_time += max(float(reference.num_audio_latents), temporal_position_span(reference.num_latent_frames))
            else:
                raise ValueError(f"Unknown MiniMax-H3 reference kind: {reference.kind!r}")
        chunk = build_causal_stream_chunk(
            chunk_index,
            self.request_height // 16,
            self.request_width // 16,
            rotary_time,
            tuple(self.config.get("patch_size", (1, 2, 2))),
        )
        if start_chunk:
            # Keep absolute audio slices (29/28/28 cadence) while starting a
            # fresh video/KV epoch near the new reference's rotary origin.
            positions = chunk.positions.clone()
            positions[:, 0] -= temporal_position_span(start_chunk * LATENTS_PER_CHUNK)
            start = (chunk_index - start_chunk) * LATENTS_PER_CHUNK
            rows = chunk.num_video_rows // LATENTS_PER_CHUNK
            chunk = replace(chunk, positions=positions, video_frames=slice(start, start + LATENTS_PER_CHUNK), video_rows=slice(start * rows, (start + LATENTS_PER_CHUNK) * rows))
        return chunk

    def _switch_image(self, image_path, origin_text_output):
        # Every rank must accept or reject the reference together, before any
        # text/VAE encoder can enter a collective operation.
        references, error = None, ""
        try:
            references = self._prepare_references(image_path)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        ready = torch.tensor([not error], dtype=torch.int32, device=AI_DEVICE)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        notify = getattr(self.va_controller.reader, "on_image_switched", None)
        if not ready.item():
            error = error or "Reference image is unavailable on another rank"
            logger.warning("H3 reference image switch rejected: {}", error)
            if notify is not None:
                notify(image_path, error=error)
            return None

        current_prompt = self.inputs["text_encoder_output"]["_h3_prompt"]
        original, original_label = origin_text_output
        pending = isinstance(original, str) and original == "pending"
        original_prompt = None if pending else original["_h3_prompt"]
        self.input_info.image_path = image_path
        self._encode_reference_conditions(references)
        current = self._encode_live_prompt(current_prompt)
        if not pending:
            original = current if original_prompt == current_prompt else self._encode_live_prompt(original_prompt)
        self.inputs["text_encoder_output"] = current
        self.model.reset_autoregressive_state()
        self.model.reset_cuda_graph()
        # Rebuild only model conditioning; keep the live reader, recorder,
        # decoder overlap, audio history and fractional audio tail intact.
        MiniMaxH3Runner.init_run(self)
        self._initialize_kv_cache()
        self.fill_condition_cache()
        if notify is not None:
            notify(image_path)
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("H3 reference image switched: {}", image_path)
        return original, original_label

    # origin_audio: (audio_end - audio_start) / AUDIO_LATENTS_PER_SECOND * sr, eg. (29 / 40 * 32000)
    def _encode_audio_for_each_chunk(self, chunk_idx: int, origin_audio: np.ndarray, chunk) -> torch.Tensor:
        hop = self.audio_vae.hop_length
        expect = chunk.num_audio_frames * hop
        block = torch.as_tensor(origin_audio, dtype=torch.float32).reshape(1, -1)
        if dist.get_rank() == 0:
            logger.debug(f"chunk[{chunk_idx}] origin_audio shape: {block.shape} {expect}")

        if not self.config.get("audio_mono", False):
            block = block.expand(2, -1).contiguous()

        if chunk_idx == 0 or self.stream_waveform is None:
            self.stream_waveform = block
        else:
            self.stream_waveform = torch.cat((self.stream_waveform, block), dim=-1)
        window_secs = float(self.config.get("audio_encode_window_secs", 15))
        if window_secs > 0:
            max_samples = max(hop, int(window_secs * self.audio_vae.sampling_rate) // hop * hop)
            if self.stream_waveform.shape[-1] > max_samples:
                if dist.get_rank() == 0:
                    logger.debug(f"stream_waveform truncate from {self.stream_waveform.shape} to {max_samples}")
                self.stream_waveform = self.stream_waveform[..., -max_samples:].contiguous()

        encode_waveform = self.stream_waveform[:1] if self.config.get("audio_mono", False) else self.stream_waveform
        latents = self.audio_vae.encode(encode_waveform, return_cpu=False)
        if encode_waveform.shape[0] == 1:
            latents = latents.expand(2, -1, -1)
        n_audio = chunk.num_audio_frames
        start = encode_waveform.shape[-1] // hop - n_audio
        clean = latents[..., start : start + n_audio].transpose(1, 2).reshape(-1, latents.shape[1]).contiguous()
        if dist.get_rank() == 0:
            logger.debug(f"chunk[{chunk_idx}] clean audio shape: {clean.shape}")
        return clean

    def _publish_live_frames(self, chunk_idx: int, frames, origin_audio, valid_duration):
        if self.va_controller.recorder is None or origin_audio is None:
            return

        # frames: (1, 3, 17, H, W) float [0, 1] -> [17, H, W, 3] for the recorder
        video_seg = frames[0].permute(1, 2, 3, 0).contiguous().cpu()
        n_frames = video_seg.shape[0]

        src_sr = self.va_controller.audio_sr
        dst_sr = self.va_controller.record_audio_sr
        audio = np.asarray(origin_audio, dtype=np.float32).reshape(-1)

        # 17/24 + 17/24 + 17/24 video seconds map to 29/40 + 28/40 + 28/40 audio seconds.
        # 32000 Hz gives 32000/24 samples per frame and cannot stay frame-aligned;
        # 24000 Hz gives 1000 samples per frame.
        if src_sr != dst_sr:
            audio = torchaudio.functional.resample(torch.from_numpy(audio).unsqueeze(0), src_sr, dst_sr).squeeze(0).contiguous().numpy()

        if self.last_rest_audio is not None:
            if dist.get_rank() == 0:
                logger.debug(f"concat last_rest_audio {self.last_rest_audio.shape} and origin_audio {audio.shape}")
            audio = np.concatenate((self.last_rest_audio, audio))
            self.last_rest_audio = None

        # 17 frames at 24000/24 need 17000 samples; chunk audio is 17400, 16800, 16800.
        # Carry the extra 400 samples from every third chunk into the next ones.
        sample_count = int(round(dst_sr / self.va_controller.record_fps * n_frames))
        if audio.shape[0] > sample_count:
            self.last_rest_audio = audio[sample_count:]
            audio = audio[:sample_count]
        elif audio.shape[0] < sample_count:
            logger.warning(f"origin_audio is not enough: {audio.shape[0]} < {sample_count}, pad {sample_count - audio.shape[0]} zero samples")
            audio = np.pad(audio, (0, sample_count - audio.shape[0]))

        audio_seg = torch.as_tensor(audio, dtype=torch.float32)
        gen_video = torch.full((1, 1, n_frames), int(chunk_idx), dtype=torch.int64, device=AI_DEVICE)
        self.va_controller.pub_livestream(video_seg, audio_seg, gen_video, valid_duration=valid_duration)

    def _commit_text_output(self, text_output, emit=False, prompt=None):
        self.inputs["text_encoder_output"] = text_output
        self.fill_condition_cache(overwrite=True)
        if not emit:
            return
        recorder = self.va_controller.recorder if self.va_controller is not None else None
        if recorder is not None and hasattr(recorder, "emit_prompt_update"):
            recorder.emit_prompt_update(prompt)

    def _update_origin_text_output(self, prompt, duration, superseded_prompt, origin_text_output):
        # 更新全局 prompt
        if duration == -1:
            origin_text_output = (self.inputs["text_encoder_output"], prompt)
            if dist.get_rank() == 0:
                logger.warning(f"reset global prompt to {prompt}")
        # 若前一个 -1 的 prompt 被后到的动作盖掉，还没编码，标记为 pending，待恢复时再算。
        # 若在恢复之前后续的 -1 的 prompt 又到了，origin_text_output 会被覆盖成最新的 prompt，恢复时也恢复的最新的
        elif duration > 0 and superseded_prompt:
            origin_text_output = ("pending", superseded_prompt)
            if dist.get_rank() == 0:
                logger.warning(f"global prompt {superseded_prompt} was overwritten before it applied")
        return origin_text_output

    def _update_restore_prompt_at(self, duration, chunk_idx):
        if duration > 0:
            return chunk_idx + max(1, math.ceil(duration * FPS / FRAMES_PER_CHUNK))
        return -1

    def run_main(self):
        try:
            self.init_run()
            self.va_controller = None
            self.va_controller = VAController(self)
            logger.info(f"init va_recorder: {self.va_controller.recorder} and va_reader: {self.va_controller.reader}")

            if self.va_controller.reader is None:
                return super().run_main()

            self.va_controller.start()
            logger.info("start ar audio generation")
            self.va_controller.before_control()
            self.fill_condition_cache()
            self.set_vae_decode_tile_shape()
            decoder = MiniMaxH3StreamingVideoDecoder(self.video_vae, self.request_num_frames, output_uint8=False)

            chunk_idx = 0
            stream_start_idx = 0
            fail_count, max_fail_count = 0, 10
            last_origin_audio = None
            last_chunk_idx = 0
            last_valid_duration = 0
            self.last_rest_audio = None
            origin_text_output = (self.inputs["text_encoder_output"], self.inputs["text_encoder_output"]["_h3_prompt"])
            restore_prompt_at = -1

            while True:
                control = self.va_controller.next_control()

                if control.action == "switch_image":
                    refreshed = self._switch_image(control.data, origin_text_output)
                    if refreshed is not None:
                        origin_text_output = refreshed
                        stream_start_idx = chunk_idx
                    continue

                elif control.action == "perform_action":
                    prompt, duration = control.data["prompt"], control.data["duration"]
                    text_output = self.encode_action_prompt(prompt)
                    self._commit_text_output(text_output, emit=True, prompt=prompt)
                    origin_text_output = self._update_origin_text_output(prompt, duration, control.data.get("superseded_prompt"), origin_text_output)
                    restore_prompt_at = self._update_restore_prompt_at(duration, chunk_idx)
                    if dist.get_rank() == 0:
                        logger.info(f"H3 prompt switched before chunk {chunk_idx}: {prompt} duration={duration} restore_prompt_at={restore_prompt_at}")
                    continue

                elif restore_prompt_at >= 0 and chunk_idx >= restore_prompt_at:
                    text_output, restored_prompt = origin_text_output
                    is_pending = isinstance(text_output, str) and text_output == "pending"
                    if is_pending:
                        text_output = self.encode_action_prompt(restored_prompt)
                        origin_text_output = (text_output, restored_prompt)
                    self._commit_text_output(text_output, emit=is_pending, prompt=restored_prompt)
                    restore_prompt_at = -1
                    if dist.get_rank() == 0:
                        logger.info(f"H3 prompt restored to origin before chunk {chunk_idx}: {restored_prompt}")
                    continue

                elif control.action == "wait":
                    time.sleep(0.01)
                    continue

                chunk = self._get_stream_chunk(chunk_idx, start_chunk=stream_start_idx)
                origin_audio, valid_duration = self.va_controller.reader.get_audio_segment(chunk.num_audio_frames / AUDIO_LATENTS_PER_SECOND)
                if origin_audio is None:
                    fail_count += 1
                    logger.warning(f"Failed to get audio chunk {fail_count} times")
                    if fail_count > max_fail_count:
                        raise Exception(f"Failed to get audio chunk {fail_count} times, stop reader")
                    continue

                with ProfilingContext4DebugL1(f"stream segment end2end {chunk_idx}"):
                    self.check_stop()
                    if dist.get_rank() == 0:
                        logger.debug(f"MiniMax-H3 causal live chunk {chunk_idx}")
                    clean = self._encode_audio_for_each_chunk(chunk_idx, origin_audio, chunk)
                    self.scheduler.prepare_chunk(chunk_idx - stream_start_idx, clean, chunk=chunk)
                    for step in range(self.scheduler.infer_steps):
                        self.check_stop()
                        self.scheduler.step_pre(step)
                        self.model.infer(self.inputs)
                        self.scheduler.step_post()

                    if dist.get_rank() == 0:
                        logger.debug(f"{chunk_idx} chunk video latents shape: {self.scheduler.video_latents.shape}")

                    video = unpatchify_video_tokens(
                        self.scheduler.video_latents,
                        chunk.video_frames.stop - chunk.video_frames.start,
                        self.scheduler.latent_height,
                        self.scheduler.latent_width,
                    )

                    for frames in decoder.push(video):
                        if dist.get_rank() == 0:
                            logger.debug(f"{chunk_idx} chunk video shape: {frames.shape}")
                        self._publish_live_frames(last_chunk_idx, frames, last_origin_audio, last_valid_duration)
                        break

                    last_origin_audio = origin_audio
                    last_chunk_idx = chunk_idx
                    last_valid_duration = valid_duration
                    chunk_idx += 1
                    fail_count = 0

        finally:
            if hasattr(self.model, "inputs"):
                self.end_run()
            if self.va_controller is not None:
                self.va_controller.clear()
                self.va_controller = None

    def close(self):
        try:
            if self.video_decode_graph is not None:
                self.video_decode_graph.clear()
        finally:
            super().close()
