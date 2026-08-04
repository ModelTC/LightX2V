import json
import os
from contextlib import suppress

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from loguru import logger

from lightx2v.models.audio_encoders.hf.minimax_h3 import MiniMaxH3AudioVAE
from lightx2v.models.input_encoders.hf.minimax_h3 import MiniMaxH3Qwen3VLTextEncoder
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3.packing import (
    TEXT_TAG,
    align_num_frames,
    prepare_keyframe_image,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    validate_t2av_geometry,
)
from lightx2v.models.networks.minimax_h3.packing_ref2av import (
    MAX_REFERENCES,
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    MiniMaxH3PreparedReference,
    decode_reference_audio,
    decode_reference_video,
    prepare_reference_frames,
    prepare_reference_image,
    prepare_reference_waveform,
    resample_reference_frames,
    resolve_reference_image_size,
    trim_reference_num_frames,
)
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.minimax_h3 import MiniMaxH3Scheduler
from lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops import Audio
from lightx2v.models.video_encoders.hf.minimax_h3 import MiniMaxH3VideoVAE
from lightx2v.utils.input_info import FL2AVInputInfo, I2AVInputInfo, L2AVInputInfo, Ref2AVInputInfo, T2AVInputInfo
from lightx2v.utils.ltx2_media_io import encode_video
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("minimax_h3")
class MiniMaxH3Runner(DefaultRunner):
    """Native MiniMax-H3 audio-video runner.

    The large components are executed sequentially on one accelerator:
    Qwen3-VL conditioner -> native packed DiT -> native video/audio VAEs.
    This mirrors the upstream modular pipeline while keeping Diffusers out of
    the runtime dependency graph.
    """

    def __init__(self, config):
        if config.get("task") not in {"t2av", "i2av", "l2av", "fl2av", "ref2av"}:
            raise ValueError("MiniMax-H3 supports t2av/i2av/l2av/fl2av/ref2av")
        self.loaded_transformer_partition = "transformer_ref" if config["task"] == "ref2av" else "transformer"
        if not config.get("cpu_offload", False):
            raise ValueError("MiniMax-H3 currently requires cpu_offload=true so Qwen, DiT, and both VAEs can run sequentially without residing on one GPU together")
        if not config.get("text_encoder_cpu_offload", True) or not config.get("vae_cpu_offload", True):
            raise ValueError("MiniMax-H3 requires text_encoder_cpu_offload=true and vae_cpu_offload=true; the conditioner, DiT, and VAEs are intentionally resident on the accelerator one at a time.")
        if config.get("lazy_load", False) or config.get("unload_modules", False):
            raise NotImplementedError("MiniMax-H3 does not support lazy_load or unload_modules yet; use the released sharded checkpoint with cpu_offload=true and offload_granularity='model'.")
        super().__init__(config)

    def init_modules(self):
        super().init_modules()
        self.run_input_encoder = self._run_input_encoder_local_h3

    def run_warmup(self):
        raise NotImplementedError("MiniMax-H3 warmup is not implemented")

    def init_scheduler(self):
        self.scheduler = MiniMaxH3Scheduler(self.config)

    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.video_vae, self.audio_vae = self.load_vae()

    def load_transformer(self):
        return MiniMaxH3Model(
            model_path=self.config["model_path"],
            config=self.config,
            device=self.init_device,
        )

    def load_text_encoder(self):
        return [MiniMaxH3Qwen3VLTextEncoder(self.config)]

    def load_vae(self):
        cpu_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        video_vae = MiniMaxH3VideoVAE.from_pretrained(self.config["model_path"], device=AI_DEVICE, cpu_offload=cpu_offload)
        audio_vae = MiniMaxH3AudioVAE.from_pretrained(self.config["model_path"], device=AI_DEVICE, cpu_offload=cpu_offload)
        configured_sample_rate = int(self.config.get("audio_sampling_rate", audio_vae.sampling_rate))
        if configured_sample_rate != audio_vae.sampling_rate:
            raise ValueError(f"MiniMax-H3 audio_sampling_rate must match the Audio VAE checkpoint: config={configured_sample_rate}, checkpoint={audio_vae.sampling_rate}")
        return video_vae, audio_vae

    def _resolve_request_geometry(self, geometry_image=None):
        if self.input_info.target_shape:
            if len(self.input_info.target_shape) != 2:
                raise ValueError(f"MiniMax-H3 target_shape must be [height, width], got {self.input_info.target_shape}")
            height, width = (int(value) for value in self.input_info.target_shape)
        elif geometry_image is not None:
            height, width = resolve_canvas_size(*geometry_image.size)
            self.input_info.target_shape = [height, width]
        else:
            height = int(self.config["target_height"])
            width = int(self.config["target_width"])
            self.input_info.target_shape = [height, width]

        requested_frames = int(self.input_info.target_video_length or self.config.get("target_video_length", 124))
        num_frames = align_num_frames(requested_frames)
        if num_frames != requested_frames:
            logger.warning(f"MiniMax-H3 frame count must be 17*n+5; aligning {requested_frames} upward to {num_frames}")
        validate_t2av_geometry(num_frames, height, width)
        self.input_info.target_video_length = num_frames
        self.request_height = height
        self.request_width = width
        self.request_num_frames = num_frames

    def run_text_encoder(self, input_info, keyframes=None, references=None):
        negative_prompt = (input_info.negative_prompt or "").strip()
        if negative_prompt:
            logger.warning("MiniMax-H3 is guidance-distilled; negative_prompt is ignored")
        return self.text_encoders[0].infer(input_info.prompt, image_list=keyframes, references=references)

    @staticmethod
    def _load_rgb_image(value):
        if isinstance(value, Image.Image):
            image = value
        else:
            image = Image.open(value)
        return ImageOps.exif_transpose(image).convert("RGB")

    def _prepare_keyframes(self):
        task = self.config["task"]
        if task == "t2av":
            if not isinstance(self.input_info, T2AVInputInfo):
                raise TypeError(f"MiniMax-H3 t2av expects T2AVInputInfo, got {type(self.input_info).__name__}")
            return [], ()
        if task == "i2av":
            if not isinstance(self.input_info, I2AVInputInfo) or not self.input_info.image_path:
                raise ValueError("MiniMax-H3 i2av requires exactly one --image_path")
            values, anchors = [self.input_info.image_path], ("first",)
        elif task == "l2av":
            if not isinstance(self.input_info, L2AVInputInfo) or not self.input_info.last_frame_path:
                raise ValueError("MiniMax-H3 l2av requires --last_frame_path")
            values, anchors = [self.input_info.last_frame_path], ("last",)
        elif task == "fl2av":
            if not isinstance(self.input_info, FL2AVInputInfo) or not self.input_info.image_path or not self.input_info.last_frame_path:
                raise ValueError("MiniMax-H3 fl2av requires --image_path and --last_frame_path")
            values, anchors = [self.input_info.image_path, self.input_info.last_frame_path], ("first", "last")
        else:
            return [], ()
        if any(isinstance(value, str) and "," in value for value in values):
            raise ValueError(f"MiniMax-H3 {task} accepts one file per frame argument, not comma-separated lists")
        images = [self._load_rgb_image(value) for value in values]
        self._resolve_request_geometry(images[0])
        images = [prepare_keyframe_image(image, self.request_height, self.request_width, stretch=index == 0) for index, image in enumerate(images)]
        return images, anchors

    @staticmethod
    def _parse_reference_entry(entry):
        if isinstance(entry, str):
            stripped = entry.strip()
            if stripped.startswith("{"):
                entry = json.loads(stripped)
            else:
                kind, separator, value = stripped.partition("=")
                if not separator or kind not in {"image", "video", "audio"}:
                    raise ValueError(f"Invalid ref2av reference {entry!r}; use image=/path, video=/path, or audio=/path")
                entry = {kind: value}
        if not isinstance(entry, dict):
            raise TypeError(f"A ref2av reference must be a string or mapping, got {type(entry).__name__}")
        media = [kind for kind in ("image", "video", "audio") if entry.get(kind) is not None]
        if media not in (["image"], ["video"], ["audio"], ["video", "audio"]):
            raise ValueError(f"A ref2av entry must contain image, video, audio, or video+audio; got {media}")
        return entry

    def _prepare_references(self):
        if not isinstance(self.input_info, Ref2AVInputInfo):
            raise TypeError(f"MiniMax-H3 ref2av expects Ref2AVInputInfo, got {type(self.input_info).__name__}")
        raw = self.input_info.references
        if isinstance(raw, (str, dict)):
            raw = [raw]
        if not raw:
            raise ValueError("MiniMax-H3 ref2av requires at least one --reference")
        if len(raw) > MAX_REFERENCES:
            raise ValueError(f"MiniMax-H3 ref2av accepts at most {MAX_REFERENCES} references")
        entries = [self._parse_reference_entry(entry) for entry in raw]
        kinds = ["image" if "image" in entry else "video" if "video" in entry else "audio" for entry in entries]
        if kinds.count("image") > MAX_REFERENCE_IMAGES or kinds.count("video") > MAX_REFERENCE_VIDEOS:
            raise ValueError("MiniMax-H3 ref2av reference image/video count exceeds 9/3")
        if all(kind == "audio" for kind in kinds):
            raise ValueError("MiniMax-H3 ref2av does not allow audio-only references")

        references = []
        audio_count = 0
        max_duration = self.request_num_frames / 24.0
        for entry, kind in zip(entries, kinds):
            if kind == "image":
                image = self._load_rgb_image(entry["image"])
                height, width = resolve_reference_image_size(*image.size)
                references.append(MiniMaxH3PreparedReference("image", image=prepare_reference_image(image, height, width)))
                continue
            if kind == "video":
                video = entry["video"]
                soundtrack = None
                if isinstance(video, (str, os.PathLike)):
                    frames, fps, decoded_soundtrack = decode_reference_video(video)
                    if decoded_soundtrack is not None:
                        waveform, sample_rate = decoded_soundtrack
                        soundtrack = Audio(
                            waveform=waveform,
                            sampling_rate=int(entry.get("sample_rate", sample_rate)),
                        )
                else:
                    frames = np.asarray(video)
                    fps = float(entry.get("fps", 24.0))
                frames = prepare_reference_frames(resample_reference_frames(frames, float(entry.get("fps", fps))), self.request_num_frames)
                explicit_audio = entry.get("audio")
                if explicit_audio is not None:
                    if isinstance(explicit_audio, (str, os.PathLike)):
                        waveform, sample_rate = decode_reference_audio(explicit_audio)
                        soundtrack = Audio(
                            waveform=waveform,
                            sampling_rate=int(entry.get("sample_rate", sample_rate)),
                        )
                    else:
                        soundtrack = Audio(
                            waveform=torch.as_tensor(explicit_audio),
                            sampling_rate=int(entry.get("sample_rate", self.audio_vae.sampling_rate)),
                        )
                reference = MiniMaxH3PreparedReference("video", has_audio=soundtrack is not None, frames=frames)
                if soundtrack is not None:
                    audio_count += 1
                    waveform = soundtrack.waveform.squeeze(0) if soundtrack.waveform.ndim == 3 else soundtrack.waveform
                    reference.waveform = prepare_reference_waveform(waveform, soundtrack.sampling_rate, self.audio_vae.sampling_rate, max_duration)
                references.append(reference)
                continue
            audio_count += 1
            value = entry["audio"]
            if isinstance(value, (str, os.PathLike)):
                waveform, sample_rate = decode_reference_audio(value)
                decoded = Audio(
                    waveform=waveform,
                    sampling_rate=int(entry.get("sample_rate", sample_rate)),
                )
            else:
                decoded = Audio(
                    waveform=torch.as_tensor(value),
                    sampling_rate=int(entry.get("sample_rate", self.audio_vae.sampling_rate)),
                )
            waveform = decoded.waveform.squeeze(0) if decoded.waveform.ndim == 3 else decoded.waveform
            references.append(
                MiniMaxH3PreparedReference(
                    "audio",
                    has_audio=True,
                    waveform=prepare_reference_waveform(waveform, decoded.sampling_rate, self.audio_vae.sampling_rate, max_duration),
                )
            )
        if audio_count > MAX_REFERENCE_AUDIOS:
            raise ValueError(f"MiniMax-H3 ref2av accepts at most {MAX_REFERENCE_AUDIOS} audio-bearing references")
        return references

    def _encode_keyframes(self, keyframes):
        latents = []
        for image in keyframes:
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None].float().div_(255.0)
            latents.append(self.video_vae.encode_condition(pixels, video=False))
        return latents

    def _encode_references(self, references):
        video_latents, audio_latents = [], []
        for reference in references:
            if reference.kind != "audio":
                if reference.kind == "image":
                    pixels = torch.from_numpy(np.asarray(reference.image).copy()).permute(2, 0, 1)[None, :, None].float().div_(255.0)
                    latent = self.video_vae.encode_condition(pixels, video=False)
                else:
                    frames = reference.frames[: trim_reference_num_frames(reference.frames.shape[0])]
                    pixels = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2)[None].float().div_(255.0)
                    latent = self.video_vae.encode_condition(pixels, video=True)
                reference.num_latent_frames = latent.shape[2]
                reference.latent_height, reference.latent_width = latent.shape[3:]
                video_latents.append(latent)
            if reference.has_audio:
                latent = self.audio_vae.encode(reference.waveform)
                reference.num_audio_latents = latent.shape[-1]
                audio_latents.append(latent)
        return video_latents, audio_latents

    def _run_input_encoder_local_h3(self):
        task = self.config["task"]
        requested_partition = "transformer_ref" if task == "ref2av" else "transformer"
        if requested_partition != self.loaded_transformer_partition:
            raise ValueError(
                "MiniMax-H3 cannot switch between the base and reference transformer partitions after initialization; "
                f"loaded {self.loaded_transformer_partition!r}, requested {requested_partition!r}. "
                "Create a separate LightX2VPipeline for ref2av."
            )
        self.condition_video_latents = []
        self.condition_audio_latents = []
        self.keyframe_anchors = ()
        self.prepared_references = None
        if task == "ref2av":
            self._resolve_request_geometry()
            self.prepared_references = self._prepare_references()
            text_encoder_output = self.run_text_encoder(self.input_info, references=self.prepared_references)
            self.condition_video_latents, self.condition_audio_latents = self._encode_references(self.prepared_references)
        else:
            keyframes, self.keyframe_anchors = self._prepare_keyframes()
            if task == "t2av":
                self._resolve_request_geometry()
            text_encoder_output = self.run_text_encoder(self.input_info, keyframes=keyframes)
            self.condition_video_latents = self._encode_keyframes(keyframes)
        tags = text_encoder_output["text_token_tags"]
        if tags.ndim != 1:
            raise ValueError("MiniMax-H3 conditioner token tags must be one-dimensional")
        if task == "t2av" and not bool((tags == TEXT_TAG).all()):
            raise ValueError("MiniMax-H3 t2av conditioner returned non-text modality rows")
        self.maybe_empty_cache(force=True, collect_garbage=True)
        return {"text_encoder_output": text_encoder_output}

    _run_input_encoder_local_t2av = _run_input_encoder_local_h3
    _run_input_encoder_local_i2av = _run_input_encoder_local_h3

    def init_run(self):
        prompt_embeds = self.inputs["text_encoder_output"]["prompt_embeds"]
        self.scheduler.prepare(
            seed=self.input_info.seed,
            num_frames=self.request_num_frames,
            height=self.request_height,
            width=self.request_width,
            text_token_tags=self.inputs["text_encoder_output"]["text_token_tags"],
            keyframe_anchors=self.keyframe_anchors,
            condition_video_latents=self.condition_video_latents,
            condition_audio_latents=self.condition_audio_latents,
            references=self.prepared_references,
        )
        logger.info(
            "MiniMax-H3 packed layout: "
            f"text={prompt_embeds.shape[0]}, audio={self.scheduler.audio_latents.shape[0]}, "
            f"video={self.scheduler.video_latents.shape[0]}, total={self.scheduler.layout.sequence_length}"
        )
        logger.info("Moving the native MiniMax-H3 transformer to the accelerator")
        self.model.to_cuda()
        torch_device_module.synchronize()

    def run_segment(self, segment_idx=0):
        infer_steps = self.scheduler.infer_steps
        for step_index in range(infer_steps):
            self.check_stop()
            logger.info(f"==> MiniMax-H3 step: {step_index + 1} / {infer_steps}")
            self.scheduler.step_pre(step_index)
            self.model.infer(self.inputs)
            self.scheduler.step_post()
            if self.progress_callback:
                self.progress_callback(((step_index + 1) / infer_steps) * 100, 100)
        return self.scheduler.video_latents, self.scheduler.audio_latents

    def _offload_transformer(self):
        logger.info("Offloading MiniMax-H3 transformer before VAE decode")
        self.model.to_cpu()
        torch_device_module.synchronize()
        self.maybe_empty_cache(force=True, collect_garbage=True)

    def run_vae_decoder(self, video_rows, audio_rows):
        video_rows = video_rows[self.scheduler.num_condition_video_rows :]
        audio_rows = audio_rows[self.scheduler.num_condition_audio_rows :]
        video_latents = unpatchify_video_tokens(
            video_rows,
            self.scheduler.num_latent_frames,
            self.scheduler.latent_height,
            self.scheduler.latent_width,
            channels=int(self.config.get("in_channels", 24)),
            patch_size=tuple(self.config.get("patch_size", (1, 2, 2))),
        )
        audio_latents = unpack_audio_tokens(audio_rows, self.scheduler.num_audio_latents)
        video = self.video_vae.decode(video_latents)
        audio = self.audio_vae.decode(audio_latents)
        return video, audio

    @staticmethod
    def _video_to_uint8_frames(video):
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
            raise ValueError(f"decoded H3 video must be [1,3,F,H,W], got {tuple(video.shape)}")
        return (video[0].permute(1, 2, 3, 0).float() * 255.0).round().to(torch.uint8).cpu()

    def process_images_after_vae_decoder(self):
        if self.input_info.return_result_tensor:
            return {
                # Match the public tensor layout of the reference pipeline:
                # [batch, frames, channels, height, width].
                "video": self.gen_video.permute(0, 2, 1, 3, 4).contiguous().cpu(),
                "audio": self.gen_audio.cpu(),
                "sampling_rate": self.audio_vae.sampling_rate,
            }

        output_path = self.input_info.save_result_path
        if output_path and (not dist.is_initialized() or dist.get_rank() == 0):
            if os.path.splitext(output_path)[1].lower() != ".mp4":
                raise ValueError(f"MiniMax-H3 AV output uses H.264/AAC; save_result_path must end in .mp4, got {output_path!r}")
            parent = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(parent, exist_ok=True)
            frames = self._video_to_uint8_frames(self.gen_video)
            waveform = self.gen_audio[0].float().cpu()
            audio = Audio(
                waveform=waveform,
                sampling_rate=self.audio_vae.sampling_rate,
            )
            logger.info(f"Saving MiniMax-H3 audio-video output to {output_path}")
            encode_video(
                video=frames,
                fps=int(self.config.get("fps", 24)),
                audio=audio,
                output_path=output_path,
                video_chunks_number=1,
            )
            logger.info(f"MiniMax-H3 output saved to {output_path}")
        return {"video": None, "audio": None}

    def run_main(self):
        transformer_offloaded = False
        try:
            self.init_run()
            try:
                video_rows, audio_rows = self.run_segment(0)
            finally:
                self._offload_transformer()
                transformer_offloaded = True

            self.gen_video, self.gen_audio = self.run_vae_decoder(video_rows, audio_rows)
            return self.process_images_after_vae_decoder()
        finally:
            # ``init_run`` can fail after a partial device transfer. Preserve
            # the original exception while still making a best-effort return
            # of the large transformer to host memory.
            if not transformer_offloaded:
                with suppress(Exception):
                    self._offload_transformer()
            try:
                self.end_run()
            finally:
                # Decoded FP32 video is large (roughly 1.5 GiB at the default
                # shape). Returned tensors keep their own references/copies;
                # the runner should not retain another request-sized result.
                self.gen_video = None
                self.gen_audio = None
