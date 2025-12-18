import gc
import io
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio as ta
import torchvision.transforms.functional as TF
from PIL import Image, ImageCms, ImageOps
from einops import rearrange
from loguru import logger

from lightx2v.deploy.common.va_controller import VAController
from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import AudioAdapter
from lightx2v.models.input_encoders.hf.seko_audio.audio_encoder import SekoAudioEncoderModel
from lightx2v.models.networks.wan.audio_model import WanAudioModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.audio.scheduler import EulerScheduler
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path, fixed_shape_resize, get_optimal_patched_size_with_sp, isotropic_crop_resize, load_weights, vae_to_comfyui_image_inplace
from lightx2v_platform.base.global_var import AI_DEVICE

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")


def resize_image(img, resize_mode="adaptive", bucket_shape=None, fixed_area=None, fixed_shape=None):
    assert resize_mode in ["adaptive", "keep_ratio_fixed_area", "fixed_min_area", "fixed_max_area", "fixed_shape", "fixed_min_side"]

    if resize_mode == "fixed_shape":
        assert fixed_shape is not None
        logger.info(f"[wan_audio] fixed_shape_resize fixed_height: {fixed_shape[0]}, fixed_width: {fixed_shape[1]}")
        return fixed_shape_resize(img, fixed_shape[0], fixed_shape[1])

    if bucket_shape is not None:
        """
        "adaptive_shape": {
            "0.667": [[480, 832], [544, 960], [720, 1280]],
            "1.500": [[832, 480], [960, 544], [1280, 720]],
            "1.000": [[480, 480], [576, 576], [704, 704], [960, 960]]
        }
        """
        bucket_config = {}
        for ratio, resolutions in bucket_shape.items():
            bucket_config[float(ratio)] = np.array(resolutions, dtype=np.int64)
        # logger.info(f"[wan_audio] use custom bucket_shape: {bucket_config}")
    else:
        bucket_config = {
            0.667: np.array([[480, 832], [544, 960], [720, 1280]], dtype=np.int64),
            1.500: np.array([[832, 480], [960, 544], [1280, 720]], dtype=np.int64),
            1.000: np.array([[480, 480], [576, 576], [704, 704], [960, 960]], dtype=np.int64),
        }
        # logger.info(f"[wan_audio] use default bucket_shape: {bucket_config}")

    ori_height = img.shape[-2]
    ori_weight = img.shape[-1]
    ori_ratio = ori_height / ori_weight

    if resize_mode == "adaptive":
        aspect_ratios = np.array(np.array(list(bucket_config.keys())))
        closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
        closet_ratio = aspect_ratios[closet_aspect_idx]
        if ori_ratio < 1.0:
            target_h, target_w = 480, 832
        elif ori_ratio == 1.0:
            target_h, target_w = 480, 480
        else:
            target_h, target_w = 832, 480
        for resolution in bucket_config[closet_ratio]:
            if ori_height * ori_weight >= resolution[0] * resolution[1]:
                target_h, target_w = resolution
    elif resize_mode == "keep_ratio_fixed_area":
        area_in_pixels = 480 * 832
        if fixed_area == "480p":
            area_in_pixels = 480 * 832
        elif fixed_area == "720p":
            area_in_pixels = 720 * 1280
        else:
            area_in_pixels = 480 * 832
        target_h = round(np.sqrt(area_in_pixels * ori_ratio))
        target_w = round(np.sqrt(area_in_pixels / ori_ratio))
    elif resize_mode == "fixed_min_area":
        aspect_ratios = np.array(np.array(list(bucket_config.keys())))
        closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
        closet_ratio = aspect_ratios[closet_aspect_idx]
        target_h, target_w = bucket_config[closet_ratio][0]
    elif resize_mode == "fixed_min_side":
        min_side = 720
        if fixed_area == "720p":
            min_side = 720
        elif fixed_area == "480p":
            min_side = 480
        else:
            logger.warning(f"[wan_audio] fixed_area is not '480p' or '720p', using default 480p: {fixed_area}")
            min_side = 480
        if ori_ratio < 1.0:
            target_h = min_side
            target_w = round(target_h / ori_ratio)
        else:
            target_w = min_side
            target_h = round(target_w * ori_ratio)
    elif resize_mode == "fixed_max_area":
        aspect_ratios = np.array(np.array(list(bucket_config.keys())))
        closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
        closet_ratio = aspect_ratios[closet_aspect_idx]
        target_h, target_w = bucket_config[closet_ratio][-1]

    cropped_img = isotropic_crop_resize(img, (target_h, target_w))
    logger.info(f"[wan_audio] resize_image: {img.shape} -> {cropped_img.shape}, resize_mode: {resize_mode}, target_h: {target_h}, target_w: {target_w}")
    return cropped_img, target_h, target_w


@dataclass
class AudioSegment:
    """Data class for audio segment information"""

    audio_array: torch.Tensor
    start_frame: int
    end_frame: int


class FramePreprocessorTorchVersion:
    """Handles frame preprocessing including noise and masking"""

    def __init__(self, noise_mean: float = -3.0, noise_std: float = 0.5, mask_rate: float = 0.1):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.mask_rate = mask_rate

    def add_noise(self, frames: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add noise to frames"""

        device = frames.device
        shape = frames.shape
        bs = 1 if len(shape) == 4 else shape[0]

        # Generate sigma values on the same device
        sigma = torch.normal(mean=self.noise_mean, std=self.noise_std, size=(bs,), device=device, generator=generator)
        sigma = torch.exp(sigma)

        for _ in range(1, len(shape)):
            sigma = sigma.unsqueeze(-1)

        # Generate noise on the same device
        noise = torch.randn(*shape, device=device, generator=generator) * sigma
        return frames + noise

    def add_mask(self, frames: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add mask to frames"""

        device = frames.device
        h, w = frames.shape[-2:]

        # Generate mask on the same device
        mask = torch.rand(h, w, device=device, generator=generator) > self.mask_rate
        return frames * mask

    def process_prev_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Process previous frames with noise and masking"""
        frames = self.add_noise(frames, torch.Generator(device=frames.device))
        frames = self.add_mask(frames, torch.Generator(device=frames.device))
        return frames


class AudioProcessor:
    """Handles audio loading and segmentation"""

    def __init__(self, audio_sr: int = 16000, target_fps: int = 16):
        self.audio_sr = audio_sr
        self.target_fps = target_fps
        self.audio_frame_rate = audio_sr // target_fps

    def load_audio(self, audio_path: str):
        audio_array, ori_sr = ta.load(audio_path)
        audio_array = ta.functional.resample(audio_array.mean(0), orig_freq=ori_sr, new_freq=self.audio_sr)
        return audio_array

    def load_multi_person_audio(self, audio_paths: List[str]):
        audio_arrays = []
        max_len = 0

        for audio_path in audio_paths:
            audio_array = self.load_audio(audio_path)
            audio_arrays.append(audio_array)
            max_len = max(max_len, audio_array.numel())

        num_files = len(audio_arrays)
        padded = torch.zeros(num_files, max_len, dtype=torch.float32)

        for i, arr in enumerate(audio_arrays):
            length = arr.numel()
            padded[i, :length] = arr

        return padded

    def get_audio_range(self, start_frame: int, end_frame: int) -> Tuple[int, int]:
        """Calculate audio range for given frame range"""
        return round(start_frame * self.audio_frame_rate), round(end_frame * self.audio_frame_rate)

    def segment_audio(self, audio_array: torch.Tensor, expected_frames: int, max_num_frames: int, prev_frame_length: int = 5) -> List[AudioSegment]:
        """
        Segment audio based on frame requirements
        audio_array is (N, T) tensor
        """
        segments = []
        segments_idx = self.init_segments_idx(expected_frames, max_num_frames, prev_frame_length)

        audio_start, audio_end = self.get_audio_range(0, expected_frames)
        audio_array_ori = audio_array[:, audio_start:audio_end]

        for idx, (start_idx, end_idx) in enumerate(segments_idx):
            audio_start, audio_end = self.get_audio_range(start_idx, end_idx)
            audio_array = audio_array_ori[:, audio_start:audio_end]

            if idx < len(segments_idx) - 1:
                end_idx = segments_idx[idx + 1][0]
            else:  # for last segments
                if audio_array.shape[1] < audio_end - audio_start:
                    padding_len = audio_end - audio_start - audio_array.shape[1]
                    audio_array = F.pad(audio_array, (0, padding_len))
                    # Adjust end_idx to account for the frames added by padding
                    end_idx = end_idx - padding_len // self.audio_frame_rate

            segments.append(AudioSegment(audio_array, start_idx, end_idx))
        del audio_array, audio_array_ori
        return segments

    def init_segments_idx(self, total_frame: int, clip_frame: int = 81, overlap_frame: int = 5) -> list[tuple[int, int, int]]:
        """Initialize segment indices with overlap"""
        start_end_list = []
        min_frame = clip_frame
        for start in range(0, total_frame, clip_frame - overlap_frame):
            is_last = start + clip_frame >= total_frame
            end = min(start + clip_frame, total_frame)
            if end - start < min_frame:
                end = start + min_frame
            if ((end - start) - 1) % 4 != 0:
                end = start + (((end - start) - 1) // 4) * 4 + 1
            start_end_list.append((start, end))
            if is_last:
                break
        return start_end_list


def load_image(image: Union[str, Image.Image], to_rgb: bool = True) -> Image.Image:
    _image = image
    if isinstance(image, str):
        if os.path.isfile(image):
            _image = Image.open(image)
        else:
            raise ValueError(f"Incorrect path. {image} is not a valid path.")
    # orientation transpose
    _image = ImageOps.exif_transpose(_image)
    # convert color space to sRGB
    icc_profile = _image.info.get("icc_profile")
    if icc_profile:
        srgb_profile = ImageCms.createProfile("sRGB")
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
        _image = ImageCms.profileToProfile(_image, input_profile, srgb_profile)
    # convert to "RGB"
    if to_rgb:
        _image = _image.convert("RGB")

    return _image


@RUNNER_REGISTER("seko_talk")
class WanAudioRunner(WanRunner):  # type:ignore
    def __init__(self, config):
        super().__init__(config)
        self.prev_frame_length = self.config.get("prev_frame_length", 5)
        self.frame_preprocessor = FramePreprocessorTorchVersion()

    def init_scheduler(self):
        """Initialize consistency model scheduler"""
        self.scheduler = EulerScheduler(self.config)

    def read_audio_input(self, audio_path):
        """Read audio input - handles both single and multi-person scenarios"""
        audio_sr = self.config.get("audio_sr", 16000)
        target_fps = self.config.get("target_fps", 16)
        self._audio_processor = AudioProcessor(audio_sr, target_fps)

        if not isinstance(audio_path, str):
            return [], 0, None, 0

        # Get audio files from person objects or legacy format
        audio_files, mask_files = self.get_audio_files_from_audio_path(audio_path)

        # Load audio based on single or multi-person mode
        if len(audio_files) == 1:
            audio_array = self._audio_processor.load_audio(audio_files[0])
            audio_array = audio_array.unsqueeze(0)  # Add batch dimension for consistency
        else:
            audio_array = self._audio_processor.load_multi_person_audio(audio_files)

        video_duration = self.config.get("video_duration", 5)
        audio_len = int(audio_array.shape[1] / audio_sr * target_fps)
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_audio_len.observe(audio_len)

        expected_frames = min(max(1, int(video_duration * target_fps)), audio_len)
        if expected_frames < int(video_duration * target_fps):
            logger.warning(f"Input video duration is greater than actual audio duration, using audio duration instead: audio_duration={audio_len / target_fps}, video_duration={video_duration}")

        # Segment audio
        audio_segments = self._audio_processor.segment_audio(audio_array, expected_frames, self.config.get("target_video_length", 81), self.prev_frame_length)

        # Mask latent for multi-person s2v
        if mask_files is not None:
            mask_latents = [self.process_single_mask(mask_file) for mask_file in mask_files]
            mask_latents = torch.cat(mask_latents, dim=0)
        else:
            mask_latents = None

        return audio_segments, expected_frames, mask_latents, len(audio_files)

    def get_audio_files_from_audio_path(self, audio_path):
        if os.path.isdir(audio_path):
            audio_files = []
            mask_files = []
            logger.info(f"audio_path is a directory, loading config.json from {audio_path}")
            audio_config_path = os.path.join(audio_path, "config.json")
            assert os.path.exists(audio_config_path), "config.json not found in audio_path"
            with open(audio_config_path, "r") as f:
                audio_config = json.load(f)
            for talk_object in audio_config["talk_objects"]:
                audio_files.append(os.path.join(audio_path, talk_object["audio"]))
                mask_files.append(os.path.join(audio_path, talk_object["mask"]))
        else:
            logger.info(f"audio_path is a file without mask: {audio_path}")
            audio_files = [audio_path]
            mask_files = None

        return audio_files, mask_files

    def process_single_mask(self, mask_file):
        mask_img = load_image(mask_file)
        mask_img = TF.to_tensor(mask_img).sub_(0.5).div_(0.5).unsqueeze(0).to(AI_DEVICE)

        if mask_img.shape[1] == 3:  # If it is an RGB three-channel image
            mask_img = mask_img[:, :1]  # Only take the first channel

        mask_img, h, w = resize_image(
            mask_img,
            resize_mode=self.config.get("resize_mode", "adaptive"),
            bucket_shape=self.config.get("bucket_shape", None),
            fixed_area=self.config.get("fixed_area", None),
            fixed_shape=self.config.get("fixed_shape", None),
        )

        mask_latent = torch.nn.functional.interpolate(
            mask_img,  # (1, 1, H, W)
            size=(h // 16, w // 16),
            mode="bicubic",
        )

        mask_latent = (mask_latent > 0).to(torch.int8)
        return mask_latent

    def read_image_input(self, img_path):
        if isinstance(img_path, Image.Image):
            ref_img = img_path
        else:
            ref_img = load_image(img_path)
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(0).to(AI_DEVICE)

        ref_img, h, w = resize_image(
            ref_img,
            resize_mode=self.config.get("resize_mode", "adaptive"),
            bucket_shape=self.config.get("bucket_shape", None),
            fixed_area=self.config.get("fixed_area", None),
            fixed_shape=self.config.get("fixed_shape", None),
        )
        logger.info(f"[wan_audio] resize_image target_h: {h}, target_w: {w}")
        patched_h = h // self.config["vae_stride"][1] // self.config["patch_size"][1]
        patched_w = w // self.config["vae_stride"][2] // self.config["patch_size"][2]

        patched_h, patched_w = get_optimal_patched_size_with_sp(patched_h, patched_w, 1)

        latent_h = patched_h * self.config["patch_size"][1]
        latent_w = patched_w * self.config["patch_size"][2]

        latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w)
        target_shape = [latent_h * self.config["vae_stride"][1], latent_w * self.config["vae_stride"][2]]

        logger.info(f"[wan_audio] target_h: {target_shape[0]}, target_w: {target_shape[1]}, latent_h: {latent_h}, latent_w: {latent_w}")

        ref_img = torch.nn.functional.interpolate(ref_img, size=(target_shape[0], target_shape[1]), mode="bicubic")
        return ref_img, latent_shape, target_shape

    @ProfilingContext4DebugL1(
        "Run Image Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_img_encode_duration,
        metrics_labels=["WanAudioRunner"],
    )
    def run_image_encoder(self, first_frame, last_frame=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.image_encoder = self.load_image_encoder()
        clip_encoder_out = self.image_encoder.visual([first_frame]).squeeze(0).to(GET_DTYPE()) if self.config.get("use_image_encoder", True) else None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.image_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return clip_encoder_out

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["WanAudioRunner"],
    )
    def run_vae_encoder(self, img):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()

        img = rearrange(img, "1 C H W -> 1 C 1 H W")
        vae_encoder_out = self.vae_encoder.encode(img.to(GET_DTYPE()))

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return vae_encoder_out

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_s2v(self):
        img, latent_shape, target_shape = self.read_image_input(self.input_info.image_path)
        if self.config.get("f2v_process", False):
            self.ref_img = img
        self.input_info.latent_shape = latent_shape  # Important: set latent_shape in input_info
        self.input_info.target_shape = target_shape  # Important: set target_shape in input_info
        clip_encoder_out = self.run_image_encoder(img) if self.config.get("use_image_encoder", True) else None
        vae_encode_out = self.run_vae_encoder(img)

        audio_segments, expected_frames, person_mask_latens, audio_num = self.read_audio_input(self.input_info.audio_path)
        self.input_info.audio_num = audio_num
        self.input_info.with_mask = person_mask_latens is not None
        text_encoder_output = self.run_text_encoder(self.input_info)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encode_out,
            },
            "audio_segments": audio_segments,
            "expected_frames": expected_frames,
            "person_mask_latens": person_mask_latens,
        }

    def prepare_prev_latents(self, prev_video: Optional[torch.Tensor], prev_frame_length: int) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare previous latents for conditioning"""
        dtype = GET_DTYPE()

        tgt_h, tgt_w = self.input_info.target_shape[0], self.input_info.target_shape[1]
        prev_frames = torch.zeros((1, 3, self.config["target_video_length"], tgt_h, tgt_w), device=AI_DEVICE)

        if prev_video is not None:
            # Extract and process last frames
            last_frames = prev_video[:, :, -prev_frame_length:].clone().to(AI_DEVICE)
            if self.config["model_cls"] != "wan2.2_audio" and not self.config.get("f2v_process", False):
                last_frames = self.frame_preprocessor.process_prev_frames(last_frames)
            prev_frames[:, :, :prev_frame_length] = last_frames
            prev_len = (prev_frame_length - 1) // 4 + 1
        else:
            prev_len = 0

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()

        _, nframe, height, width = self.model.scheduler.latents.shape
        with ProfilingContext4DebugL1(
            "vae_encoder in init run segment",
            recorder_mode=GET_RECORDER_MODE(),
            metrics_func=monitor_cli.lightx2v_run_vae_encoder_pre_latent_duration,
            metrics_labels=["WanAudioRunner"],
        ):
            if self.config["model_cls"] == "wan2.2_audio":
                if prev_video is not None:
                    prev_latents = self.vae_encoder.encode(prev_frames.to(dtype))
                else:
                    prev_latents = None
                prev_mask = self.model.scheduler.mask
            else:
                prev_latents = self.vae_encoder.encode(prev_frames.to(dtype))

            frames_n = (nframe - 1) * 4 + 1
            prev_mask = torch.ones((1, frames_n, height, width), device=AI_DEVICE, dtype=dtype)
            prev_frame_len = max((prev_len - 1) * 4 + 1, 0)
            prev_mask[:, prev_frame_len:] = 0
            prev_mask = self._wan_mask_rearrange(prev_mask)

        if prev_latents is not None:
            if prev_latents.shape[-2:] != (height, width):
                logger.warning(f"Size mismatch: prev_latents {prev_latents.shape} vs scheduler latents (H={height}, W={width}). Config tgt_h={tgt_h}, tgt_w={tgt_w}")
                prev_latents = torch.nn.functional.interpolate(prev_latents, size=(height, width), mode="bilinear", align_corners=False)

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()

        return {"prev_latents": prev_latents, "prev_mask": prev_mask, "prev_len": prev_len}

    def _wan_mask_rearrange(self, mask: torch.Tensor) -> torch.Tensor:
        """Rearrange mask for WAN model"""
        if mask.ndim == 3:
            mask = mask[None]
        assert mask.ndim == 4
        _, t, h, w = mask.shape
        assert t == ((t - 1) // 4 * 4 + 1)
        mask_first_frame = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask = torch.concat([mask_first_frame, mask[:, 1:]], dim=1)
        mask = mask.view(mask.shape[1] // 4, 4, h, w)
        return mask.transpose(0, 1).contiguous()

    def get_video_segment_num(self):
        self.video_segment_num = len(self.inputs["audio_segments"])

    def init_run(self):
        super().init_run()
        self.scheduler.set_audio_adapter(self.audio_adapter)
        if self.config.get("f2v_process", False):
            self.prev_video = self.ref_img.unsqueeze(2)
        else:
            self.prev_video = None
        if self.input_info.return_result_tensor:
            self.gen_video_final = torch.zeros((self.inputs["expected_frames"], self.input_info.target_shape[0], self.input_info.target_shape[1], 3), dtype=torch.float32, device="cpu")
            self.cut_audio_final = torch.zeros((self.inputs["expected_frames"] * self._audio_processor.audio_frame_rate), dtype=torch.float32, device="cpu")
        else:
            self.gen_video_final = None
            self.cut_audio_final = None

    @ProfilingContext4DebugL1(
        "Init run segment",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_init_run_segment_duration,
        metrics_labels=["WanAudioRunner"],
    )
    def init_run_segment(self, segment_idx, audio_array=None):
        self.segment_idx = segment_idx
        if audio_array is not None:
            end_idx = audio_array.shape[0] // self._audio_processor.audio_frame_rate - self.prev_frame_length
            audio_tensor = torch.Tensor(audio_array).float().unsqueeze(0)
            self.segment = AudioSegment(audio_tensor, 0, end_idx)
        else:
            self.segment = self.inputs["audio_segments"][segment_idx]

        self.input_info.seed = self.input_info.seed + segment_idx
        torch.manual_seed(self.input_info.seed)
        # logger.info(f"Processing segment {segment_idx + 1}/{self.video_segment_num}, seed: {self.config.seed}")

        if (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)) and not hasattr(self, "audio_encoder"):
            self.audio_encoder = self.load_audio_encoder()

        features_list = []
        for i in range(self.segment.audio_array.shape[0]):
            feat = self.audio_encoder.infer(self.segment.audio_array[i])
            feat = self.audio_adapter.forward_audio_proj(feat, self.model.scheduler.latents.shape[1])
            features_list.append(feat.squeeze(0))
        audio_features = torch.stack(features_list, dim=0)

        self.inputs["audio_encoder_output"] = audio_features
        self.inputs["previmg_encoder_output"] = self.prepare_prev_latents(self.prev_video, prev_frame_length=self.prev_frame_length)

        # Reset scheduler for non-first segments
        if segment_idx > 0:
            self.model.scheduler.reset(self.input_info.seed, self.input_info.latent_shape, self.inputs["previmg_encoder_output"])

    @ProfilingContext4DebugL1(
        "End run segment",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_end_run_segment_duration,
        metrics_labels=["WanAudioRunner"],
    )
    def end_run_segment(self, segment_idx):
        self.gen_video = torch.clamp(self.gen_video, -1, 1).to(torch.float)
        useful_length = self.segment.end_frame - self.segment.start_frame
        video_seg = self.gen_video[:, :, :useful_length].cpu()
        audio_seg = self.segment.audio_array[:, : useful_length * self._audio_processor.audio_frame_rate]
        audio_seg = audio_seg.sum(dim=0)  # Multiple audio tracks, mixed into one track
        video_seg = vae_to_comfyui_image_inplace(video_seg)

        # [Warning] Need check whether video segment interpolation works...
        if "video_frame_interpolation" in self.config and self.vfi_model is not None:
            target_fps = self.config["video_frame_interpolation"]["target_fps"]
            logger.info(f"Interpolating frames from {self.config.get('fps', 16)} to {target_fps}")
            video_seg = self.vfi_model.interpolate_frames(
                video_seg,
                source_fps=self.config.get("fps", 16),
                target_fps=target_fps,
            )

        if "video_super_resolution" in self.config and self.vsr_model is not None:
            # logger.info(f"Applying video super resolution with scale {self.config['video_super_resolution']['scale']}")
            video_seg = self.vsr_model.super_resolve_frames(
                video_seg,
                seed=self.config["video_super_resolution"]["seed"],
                scale=self.config["video_super_resolution"]["scale"],
            )

        if self.va_controller.recorder is not None:
            self.va_controller.pub_livestream(video_seg, audio_seg, self.gen_video[:, :, :useful_length])
        elif self.input_info.return_result_tensor:
            self.gen_video_final[self.segment.start_frame : self.segment.end_frame].copy_(video_seg)
            self.cut_audio_final[self.segment.start_frame * self._audio_processor.audio_frame_rate : self.segment.end_frame * self._audio_processor.audio_frame_rate].copy_(audio_seg)

        # Update prev_video for next iteration
        self.prev_video = self.gen_video

        del video_seg, audio_seg
        torch.cuda.empty_cache()

    @ProfilingContext4DebugL1(
        "End run segment stream",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_end_run_segment_duration,
        metrics_labels=["WanAudioRunner"],
    )
    def end_run_segment_stream(self, latents):
        valid_length = self.segment.end_frame - self.segment.start_frame
        frame_segments = []
        frame_idx = 0

        # frame_segment: 1*C*1*H*W, 1*C*4*H*W, 1*C*4*H*W, ...
        for origin_seg in self.run_vae_decoder_stream(latents):
            origin_seg = torch.clamp(origin_seg, -1, 1).to(torch.float)
            valid_T = min(valid_length - frame_idx, origin_seg.shape[2])

            video_seg = vae_to_comfyui_image_inplace(origin_seg[:, :, :valid_T].cpu())
            audio_start = frame_idx * self._audio_processor.audio_frame_rate
            audio_end = (frame_idx + valid_T) * self._audio_processor.audio_frame_rate
            audio_seg = self.segment.audio_array[:, audio_start:audio_end].sum(dim=0)

            if self.va_controller.recorder is not None:
                self.va_controller.pub_livestream(video_seg, audio_seg, origin_seg[:, :, :valid_T])

            frame_segments.append(origin_seg)
            frame_idx += valid_T
            del video_seg, audio_seg

        # Update prev_video for next iteration
        self.prev_video = torch.cat(frame_segments, dim=2)
        torch.cuda.empty_cache()

    def run_main(self):
        try:
            self.va_controller = None
            self.va_controller = VAController(self)
            logger.info(f"init va_recorder: {self.va_controller.recorder} and va_reader: {self.va_controller.reader}")

            # fixed audio segments inputs
            if self.va_controller.reader is None:
                return super().run_main()

            self.va_controller.start()
            self.init_run()
            if self.config.get("compile", False) and hasattr(self.model, "comple"):
                self.model.select_graph_for_compile(self.input_info)
            # steam audio input, video segment num is unlimited
            self.video_segment_num = 1000000
            segment_idx = 0
            fail_count, max_fail_count = 0, 10
            self.va_controller.before_control()

            while True:
                with ProfilingContext4DebugL1(f"stream segment get audio segment {segment_idx}"):
                    control = self.va_controller.next_control()
                    if control.action == "immediate":
                        self.prev_video = control.data
                    elif control.action == "wait":
                        time.sleep(0.01)
                        continue

                    audio_array = self.va_controller.reader.get_audio_segment()
                    if audio_array is None:
                        fail_count += 1
                        logger.warning(f"Failed to get audio chunk {fail_count} times")
                        if fail_count > max_fail_count:
                            raise Exception(f"Failed to get audio chunk {fail_count} times, stop reader")
                        continue

                with ProfilingContext4DebugL1(f"stream segment end2end {segment_idx}"):
                    try:
                        # reset pause signal
                        self.pause_signal = False
                        self.init_run_segment(segment_idx, audio_array)
                        self.check_stop()
                        latents = self.run_segment(segment_idx)
                        self.check_stop()
                        if self.config.get("use_stream_vae", False):
                            self.end_run_segment_stream(latents)
                        else:
                            self.gen_video = self.run_vae_decoder(latents)
                            self.check_stop()
                            self.end_run_segment(segment_idx)
                        segment_idx += 1
                        fail_count = 0
                    except Exception as e:
                        if "pause_signal, pause running" in str(e):
                            logger.warning(f"model infer audio pause: {e}, should continue")
                        else:
                            raise
        finally:
            if hasattr(self.model, "inputs"):
                self.end_run()
            if self.va_controller is not None:
                self.va_controller.clear()
                self.va_controller = None

    @ProfilingContext4DebugL1("Process after vae decoder")
    def process_images_after_vae_decoder(self):
        if self.input_info.return_result_tensor:
            audio_waveform = self.cut_audio_final.unsqueeze(0).unsqueeze(0)
            comfyui_audio = {"waveform": audio_waveform, "sample_rate": self._audio_processor.audio_sr}
            return {"video": self.gen_video_final, "audio": comfyui_audio}
        return {"video": None, "audio": None}

    def load_transformer(self):
        """Load transformer with LoRA support"""
        base_model = WanAudioModel(self.config["model_path"], self.config, self.init_device)
        if self.config.get("lora_configs") and self.config["lora_configs"]:
            assert not self.config.get("dit_quantized", False)
            lora_wrapper = WanLoraWrapper(base_model)
            for lora_config in self.config["lora_configs"]:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")

        return base_model

    def load_audio_encoder(self):
        audio_encoder_path = self.config.get("audio_encoder_path", os.path.join(self.config["model_path"], "TencentGameMate-chinese-hubert-large"))
        audio_encoder_offload = self.config.get("audio_encoder_cpu_offload", self.config.get("cpu_offload", False))
        model = SekoAudioEncoderModel(audio_encoder_path, self.config["audio_sr"], audio_encoder_offload)
        return model

    def load_audio_adapter(self):
        audio_adapter_offload = self.config.get("audio_adapter_cpu_offload", self.config.get("cpu_offload", False))
        if audio_adapter_offload:
            device = torch.device("cpu")
        else:
            device = torch.device(AI_DEVICE)
        audio_adapter = AudioAdapter(
            attention_head_dim=self.config["dim"] // self.config["num_heads"],
            num_attention_heads=self.config["num_heads"],
            base_num_layers=self.config["num_layers"],
            interval=1,
            audio_feature_dim=1024,
            time_freq_dim=256,
            projection_transformer_layers=4,
            mlp_dims=(1024, 1024, 32 * 1024),
            quantized=self.config.get("adapter_quantized", False),
            quant_scheme=self.config.get("adapter_quant_scheme", None),
            cpu_offload=audio_adapter_offload,
        )

        audio_adapter.to(device)
        load_from_rank0 = self.config.get("load_from_rank0", False)
        weights_dict = load_weights(self.config["adapter_model_path"], cpu_offload=audio_adapter_offload, remove_key="ca", load_from_rank0=load_from_rank0)
        audio_adapter.load_state_dict(weights_dict, strict=False)
        return audio_adapter.to(dtype=GET_DTYPE())

    def load_model(self):
        super().load_model()
        with ProfilingContext4DebugL2("Load audio encoder and adapter"):
            self.audio_encoder = self.load_audio_encoder()
            self.audio_adapter = self.load_audio_adapter()

    def get_latent_shape_with_lat_hw(self, latent_h, latent_w):
        latent_shape = [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            latent_h,
            latent_w,
        ]
        return latent_shape


@RUNNER_REGISTER("wan2.2_audio")
class Wan22AudioRunner(WanAudioRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_vae_decoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device(AI_DEVICE)
        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", "Wan2.2_VAE.pth"),
            "device": vae_device,
            "cpu_offload": vae_offload,
            "offload_cache": self.config.get("vae_offload_cache", False),
        }
        vae_decoder = Wan2_2_VAE(**vae_config)
        return vae_decoder

    def load_vae_encoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device(AI_DEVICE)
        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", "Wan2.2_VAE.pth"),
            "device": vae_device,
            "cpu_offload": vae_offload,
            "offload_cache": self.config.get("vae_offload_cache", False),
        }
        if self.config.task not in ["i2v", "s2v"]:
            return None
        else:
            return Wan2_2_VAE(**vae_config)

    def load_vae(self):
        vae_encoder = self.load_vae_encoder()
        vae_decoder = self.load_vae_decoder()
        return vae_encoder, vae_decoder
