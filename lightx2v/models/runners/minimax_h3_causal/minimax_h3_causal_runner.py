import io
import math
import time
import wave
from contextlib import suppress

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from PIL import Image
from loguru import logger

from lightx2v.common.kvcache.manager import KVCacheManager
from lightx2v.models.input_encoders.hf.minimax_h3_causal import MiniMaxH3CausalQwen3VLTextEncoder
from lightx2v.models.networks.minimax_h3.packing import (
    AUDIO_LATENTS_PER_SECOND,
    FPS,
    FRAMES_PER_CHUNK,
    align_num_frames,
    audio_latent_num_frames,
    temporal_position_span,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from lightx2v.models.networks.minimax_h3.packing_ref2av import MAX_REFERENCE_IMAGES, MiniMaxH3PreparedReference, resolve_reference_image_size
from lightx2v.models.networks.minimax_h3_causal.model import MiniMaxH3CausalModel
from lightx2v.models.networks.minimax_h3_causal.streaming import build_causal_stream_chunk
from lightx2v.models.runners.minimax_h3.minimax_h3_runner import MiniMaxH3Runner
from lightx2v.models.runners.minimax_h3_causal.action_prompt_travel import ActionPromptTravel, ActionPromptTravelConfig, coerce_action_prompts
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.schedulers.minimax_h3_causal.scheduler import MiniMaxH3CausalScheduler
from lightx2v.models.video_encoders.hf.minimax_h3_causal.streaming import MiniMaxH3StreamingVideoDecoder
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.input_info import INPUT_INFO_TYPES
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import seed_all
from lightx2v.utils.va_controller import VAController
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("minimax_h3_causal")
class MiniMaxH3CausalRunner(MiniMaxH3Runner):
    supported_request_fields_by_task = {
        "refa2v": COMMON_REQUEST_FIELDS | VIDEO_OUTPUT_FIELDS | {"image_path", "audio_path", "prompt", "action_prompts"},
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
        self.action_prompt_travel_config = ActionPromptTravelConfig.model_validate(config.get("action_prompt_travel") or {})
        super().__init__(config)
        self.kv_cache_manager = None
        self.stream_waveform = None
        self.last_rest_audio = None
        self.stream_contexts = {}
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
        self.stream_contexts = {}

    def run_text_encoder(self, input_info, keyframes=None, references=None):
        if not self.action_prompt_travel_config.enabled or not input_info.action_prompts:
            return super().run_text_encoder(input_info, keyframes=keyframes, references=references)
        encoder = self.text_encoders[0]
        self.text_reference_context = encoder.prepare_reference_context(references)
        reference_length = len(self.text_reference_context["token_ids"])
        self.action_travel = ActionPromptTravel(input_info.prompt, input_info.action_prompts, self.action_prompt_travel_config, encoder.tokenizer, reference_length)
        if isinstance(input_info.audio_path, dict) and self.action_prompt_travel_config.naive_travel:
            empty_length = self.action_travel.presentation_length(self.action_travel.assemble_prompt(""))
            self.action_travel.text_length = max(self.action_travel.text_length, empty_length + self.action_prompt_travel_config.action_slot_len)
        return self.encode_action_prompt()

    def encode_action_prompt(self, action_prompt=None):
        travel = self.action_travel
        prompt = travel.prompt
        if action_prompt is not None:
            body = str(action_prompt).strip()
            if body and travel.config.naive_travel and travel.config.prompt_form == "six_key" and not body.startswith("[Action"):
                body = f"[Action 1]: {body}"
            prompt = travel.assemble_prompt(body)
        text = self.text_encoders[0].infer(
            prompt,
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

    def fill_condition_cache(self, *, overwrite=False):
        self.scheduler.prepare_condition(overwrite=overwrite)
        for step in range(self.scheduler.infer_steps):
            self.check_stop()
            self.scheduler.step_pre(step)
            self.model.infer(self.inputs)

    def run_segment(self, segment_idx=0):
        self.fill_condition_cache()
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
        # Mono preprocessing creates identical channels; encode one and reuse
        # its latents while retaining the downstream left/right token layout.
        waveform = self.waveform[:1] if self.config.get("audio_mono", False) else self.waveform
        for index, chunk in enumerate(chunks):
            self.check_stop()
            logger.info(f"MiniMax-H3 causal chunk {index + 1}/{len(chunks)}")
            # Re-encode only the available waveform prefix, and commit only new
            # latents. Full-waveform encoding would leak future audio context.
            latents = self.audio_vae.encode(waveform[..., : chunk.audio_frames.stop * self.audio_vae.hop_length], return_cpu=False)
            if waveform.shape[0] == 1:
                latents = latents.expand(2, -1, -1)
            clean = latents[..., chunk.audio_frames].transpose(1, 2).reshape(-1, latents.shape[1]).contiguous()
            if dist.get_rank() == 0:
                logger.debug(f"{index} chunk clean audio shape: {latents.shape} {clean.shape}")
            self.scheduler.prepare_chunk(index, clean)
            for step in range(self.scheduler.infer_steps):
                self.check_stop()
                self.scheduler.step_pre(step)
                self.model.infer(self.inputs)
                self.scheduler.step_post()
                if self.progress_callback:
                    self.progress_callback(100 * (index * self.scheduler.infer_steps + step + 1) / (len(chunks) * self.scheduler.infer_steps), 100)
            if dist.get_rank() == 0:
                logger.debug(f"{index} chunk video latents shape: {self.scheduler.video_latents.shape} {chunk.video_frames.stop} {chunk.video_frames.start}")
            video = unpatchify_video_tokens(self.scheduler.video_latents, chunk.video_frames.stop - chunk.video_frames.start, self.scheduler.latent_height, self.scheduler.latent_width)
            for frames in decoder.push(video):
                if dist.get_rank() == 0:
                    logger.debug(f"{index} chunk video shape: {frames.shape}")
                videos.append(frames)
                if self.stream_callback is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                    self.stream_callback(frames, emitted, False)
                emitted += frames.shape[2]
            travel = self.action_travel
            if travel is not None and travel.config.min_slice_rolling and (index + 1) % travel.config.unit_transition == 0 and index + 1 < len(chunks):
                # Nonzero ranks do not receive decoded RGB in parallel VAE
                # mode, but the decoder advances this counter on every rank.
                seconds = (decoder.emitted_frames or (index + 1) * FRAMES_PER_CHUNK) / FPS
                if travel.update(seconds):
                    self.inputs["text_encoder_output"] = self.encode_action_prompt()
                    self.fill_condition_cache(overwrite=True)
                    logger.info(f"H3 action prompt updated after chunk {index + 1} at {seconds:.3f}s")
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
        self.stream_waveform = None
        self.last_rest_audio = None
        self.stream_contexts = {}

    def _get_stream_chunk(self, chunk_index):
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
        return build_causal_stream_chunk(
            chunk_index,
            self.request_height // 16,
            self.request_width // 16,
            rotary_time,
            tuple(self.config.get("patch_size", (1, 2, 2))),
        )

    # origin_audio: (audio_end - audio_start) / AUDIO_LATENTS_PER_SECOND * sr, eg. (29 / 40 * 32000)
    def _encode_audio_for_each_chunk(self, chunk_idx: int, origin_audio: np.ndarray, chunk) -> torch.Tensor:
        hop = self.audio_vae.hop_length
        expect = chunk.num_audio_frames * hop
        block = torch.as_tensor(origin_audio, dtype=torch.float32).reshape(1, -1)
        if dist.get_rank() == 0:
            logger.debug(f"chunk[{chunk_idx}] origin_audio shape: {block.shape} {expect}")

        if not self.config.get("audio_mono", False):
            block = block.expand(2, -1).contiguous()

        prefix_samples = chunk.audio_frames.start * hop
        if chunk_idx == 0 or self.stream_waveform is None:
            self.stream_waveform = block
        elif self.stream_waveform.shape[-1] != prefix_samples:
            raise ValueError(f"live audio prefix has {self.stream_waveform.shape[-1]} samples; chunk {chunk_idx} expects {prefix_samples}")
        else:
            self.stream_waveform = torch.cat((self.stream_waveform, block), dim=-1)

        encode_waveform = self.stream_waveform[:1] if self.config.get("audio_mono", False) else self.stream_waveform
        latents = self.audio_vae.encode(encode_waveform[..., : chunk.audio_frames.stop * hop], return_cpu=False)
        if encode_waveform.shape[0] == 1:
            latents = latents.expand(2, -1, -1)
        clean = latents[..., chunk.audio_frames].transpose(1, 2).reshape(-1, latents.shape[1]).contiguous()
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
            fail_count, max_fail_count = 0, 10
            last_origin_audio = None
            last_chunk_idx = 0
            last_valid_duration = 0
            next_t = None
            self.last_rest_audio = None

            while True:
                control = self.va_controller.next_control()
                if control.action == "perform_action":
                    self.inputs["text_encoder_output"] = self.encode_action_prompt(control.data)
                    self.fill_condition_cache(overwrite=True)
                    logger.info(f"H3 prompt switched before chunk {chunk_idx}: {control.data}")
                    continue

                chunk = self._get_stream_chunk(chunk_idx)
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
                        logger.info(f"MiniMax-H3 causal live chunk {chunk_idx}")
                    clean = self._encode_audio_for_each_chunk(chunk_idx, origin_audio, chunk)
                    self.scheduler.prepare_chunk(chunk_idx, clean, chunk=chunk)
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
                    if next_t is not None:
                        time.sleep(max(0.0, next_t - time.time()))
                    next_t = time.time() + 17 / 24.0

        finally:
            if hasattr(self.model, "inputs"):
                self.end_run()
            if self.va_controller is not None:
                self.va_controller.clear()
                self.va_controller = None
