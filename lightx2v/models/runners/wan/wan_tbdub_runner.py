"""Audio-driven cropped-video editing on the native Wan2.2 dense lifecycle."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger

from lightx2v.models.input_encoders.hubert.model import HubertAudioEncoder, load_audio
from lightx2v.models.input_encoders.tbdub import continuation_frames, match_reference_colors, read_video, video_tensor
from lightx2v.models.networks.wan.tbdub_model import TBDubModel
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS, VIDEO_OUTPUT_FIELDS
from lightx2v.models.runners.wan.wan_runner import Wan22DenseRunner
from lightx2v.models.schedulers.wan.tbdub.scheduler import TBDubScheduler
from lightx2v.models.video_encoders.hf.wan.tbdub_vae import TBDubVAE
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v.utils.input_info import InputInfo
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import is_main_process, mux_audio_from_video, save_to_video
from lightx2v_platform.base.global_var import AI_DEVICE


@dataclass
class TBDubInputInfo(InputInfo):
    video_path: str = ""
    audio_path: str = ""
    num_frames: int | None = None
    size: list = field(default_factory=list)


@RUNNER_REGISTER("tbdub")
class TBDubRunner(Wan22DenseRunner):
    supported_lora_modes = ("merge",)

    input_info_cls_by_task = {"v2v": TBDubInputInfo}
    supported_request_fields_by_task = {"v2v": COMMON_REQUEST_FIELDS | VIDEO_OUTPUT_FIELDS | {"video_path", "audio_path"}}
    # These are checkpoint temporal conventions, not the requested output length.
    clip_frames = 77
    motion_frames = 5
    motion_latent_frames = 2
    stride_frames = 72
    fps = 25

    def __init__(self, config):
        if config.get("color_correction", True):
            try:
                import skimage.color  # noqa: F401
            except ImportError as error:
                raise ImportError("TBDub color correction requires scikit-image: pip install 'lightx2v[tbdub]' (or pip install scikit-image)") from error
        if config.get("model_variant", "student") != "student":
            raise ValueError("TBDub currently supports the standalone student checkpoint only")
        if config.get("fps", 25) != self.fps:
            raise ValueError("TBDub's audio/video conditioning requires fps=25")
        if GET_DTYPE() != torch.bfloat16 or GET_SENSITIVE_DTYPE() != torch.bfloat16:
            raise ValueError("TBDub student requires DTYPE=BF16 and SENSITIVE_LAYER_DTYPE=BF16 (or None)")
        for option in ("lazy_load", "unload_modules", "disagg_mode", "use_tae", "use_stream_vae", "use_tiling_vae", "warmup"):
            if config.get(option):
                raise NotImplementedError(f"TBDub does not support {option}; use native resident/model/block offload")
        if config.get("lora_configs"):
            raise NotImplementedError("TBDub is a standalone student; additional LoRA loading has not been validated")
        super().__init__(config)
        self.vae_cls = TBDubVAE

    def init_scheduler(self):
        self.scheduler = TBDubScheduler(self.config)

    def load_transformer(self):
        return TBDubModel(self.config["model_path"], self.config, self.init_device)

    def load_text_encoder(self):
        self.prompt_embedding = torch.load(self.config["prompt_embedding_path"], map_location="cpu", weights_only=True)
        if not isinstance(self.prompt_embedding, torch.Tensor) or self.prompt_embedding.shape not in ((512, 4096), (1, 512, 4096)):
            raise ValueError("TBDub null_prompt_emb.pt must contain a [512,4096] or [1,512,4096] tensor")
        return None

    def load_image_encoder(self):
        return None

    def load_vae_encoder(self):
        offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        return self.vae_cls(**self._build_wan22_vae_config(offload))

    def load_model(self):
        super().load_model()
        self.audio_encoder = HubertAudioEncoder(
            self.config["audio_encoder_path"],
            device=AI_DEVICE,
            cpu_offload=self.config.get("audio_encoder_cpu_offload", self.config.get("cpu_offload", False)),
            num_frames=self.clip_frames,
        )

    def prepare_request(self, request_data):
        # Output duration follows driving audio, not Wan's 4n+1 request rule.
        inputs = BaseRunner.prepare_request(self, request_data)
        for name in ("video_path", "audio_path"):
            if not getattr(inputs, name) or not Path(getattr(inputs, name)).is_file():
                raise ValueError(f"TBDub v2v requires an existing local {name}")
        if Path(inputs.audio_path).suffix.lower() == ".npy":
            raise ValueError("audio_path must be a media file (for example WAV), not an unframed NumPy waveform")
        if inputs.num_frames is not None and inputs.num_frames < 1:
            raise ValueError("num_frames must be positive; omit it to follow the driving audio")
        if list(inputs.size or self.config["size"]) != [512, 512]:
            raise ValueError("The released TBDub student edits 512x512 face crops")
        if not inputs.return_result_tensor and not inputs.save_result_path:
            raise ValueError("TBDub requires save_result_path or return_result_tensor=true")
        return inputs

    @torch.no_grad()
    def run_pipeline(self, input_info):
        self.input_info = input_info
        try:
            waveform = load_audio(input_info.audio_path)
            frame_count = len(waveform) * self.fps // 16000
            if input_info.num_frames is not None:
                frame_count = min(frame_count, input_info.num_frames)
            if frame_count < 1:
                raise ValueError("Driving audio must contain at least one video frame (40 ms)")
            self.video_segment_num = (frame_count + self.stride_frames - 1) // self.stride_frames
            required_frames = self.video_segment_num * self.stride_frames
            source = read_video(input_info.video_path, self.get_target_size(), self.fps, required_frames)
            reference, output_reference = continuation_frames(source, required_frames, self.motion_frames)
            features = self.audio_encoder.audio2feat(waveform, pad_end=True)
            context = self.prompt_embedding.to(device=AI_DEVICE, dtype=GET_DTYPE())
            motion = None
            segments = []
            for index in range(self.video_segment_num):
                self.check_stop()
                logger.info("TBDub window {}/{}", index + 1, self.video_segment_num)
                start = index * self.stride_frames
                pixels = video_tensor(reference[start : start + self.clip_frames], AI_DEVICE, GET_DTYPE())
                ref_latents = self.get_vae_encoder_output(pixels).to(GET_DTYPE())
                if motion is None:
                    motion = self.get_vae_encoder_output(pixels[:, : self.motion_frames]).to(GET_DTYPE())
                self.inputs = {
                    "ref_latents": ref_latents,
                    "motion_latents": motion,
                    "audio_feats": self.audio_encoder.crop_overlap_audio_window(features, start - self.motion_frames).to(device=AI_DEVICE, dtype=GET_DTYPE()),
                    "text_encoder_output": {"context": context},
                }
                self.scheduler.prepare_loop(ref_latents, input_info.seed + index, motion)
                latents = self.run_segment(index)
                motion = latents[:, -self.motion_latent_frames :].clone()
                segments.append(latents[:, self.motion_latent_frames if index else 0 :].cpu())
            # Decode the stitched latent stream once. Independent per-window
            # VAE calls discard temporal history and introduce boundary flicker.
            joined = torch.cat(segments, dim=1).to(device=AI_DEVICE, dtype=GET_DTYPE())
            decoded = self.vae_decoder.decode(joined)[0, :, self.motion_frames : self.motion_frames + frame_count]
            frames = ((decoded.to(GET_DTYPE()).permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
            if self.config.get("color_correction", True):
                frames = match_reference_colors(frames, output_reference[:frame_count])
            images = torch.from_numpy(frames).float().div_(255)
            if is_main_process() and input_info.save_result_path:
                save_to_video(images, input_info.save_result_path, fps=self.fps, method="ffmpeg", lossless=self.config.get("save_lossless", False))
                if mux_audio_from_video(input_info.audio_path, input_info.save_result_path, prefer_copy=False) is None:
                    raise RuntimeError("TBDub generated video, but muxing its driving audio failed")
            return {"video": images if input_info.return_result_tensor else None, "audio": None}
        except BaseException as error:
            self._cleanup_failed_request(error)
            raise
        finally:
            self.end_run()

    def _cleanup_failed_request(self, error):
        # A successful Wan block loop prefetches block 0 into the next slot.
        # A failed loop can leave either slot at an arbitrary block. Drain all
        # transfers before moving weights, and force initialization on reuse.
        actions = []
        infer = getattr(self.model, "transformer_infer", None)
        manager = getattr(infer, "offload_manager", None)
        if manager is not None:
            manager.need_init_first_buffer = True
            actions.extend((f"synchronize {name}", getattr(manager, name).synchronize) for name in ("init_stream", "cuda_load_stream", "compute_stream"))
        if self.config.get("cpu_offload"):
            if self.config.get("offload_granularity") == "model":
                actions.append(("offload DiT", self.model.to_cpu))
            else:
                actions.extend((("offload DiT pre-weights", self.model.pre_weight.to_cpu), ("offload DiT head", self.model.transformer_weights.non_block_weights_to_cpu)))
        vaes = [self.vae_encoder]
        if self.vae_decoder is not self.vae_encoder:
            vaes.append(self.vae_decoder)
        for vae in vaes:
            codec = getattr(vae, "model", None)
            if codec is not None:
                actions.append(("clear VAE temporal cache", codec.clear_cache))
            if vae.cpu_offload:
                actions.append(("offload VAE", vae.to_cpu))
        for description, cleanup in actions:
            try:
                cleanup()
            except BaseException as cleanup_error:
                # Preserve the generation error and still attempt the remaining
                # releases (a CUDA failure can also make synchronization fail).
                error.add_note(f"TBDub cleanup failed to {description}: {cleanup_error!r}")
