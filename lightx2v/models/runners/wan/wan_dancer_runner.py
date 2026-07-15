import math

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from loguru import logger
from torchvision.transforms import InterpolationMode

from lightx2v.models.input_encoders.hf.wan.wan_dancer.wan_dancer import extract_music_features, split_music_features
from lightx2v.models.networks.wan.dancer_model import WanDancerModel
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.dancer import WanDancerScheduler, WanDancerStepDistillScheduler
from lightx2v.models.video_encoders.hf.wan.dancer_vae import WanDancerVAE
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path, mux_audio_from_video, save_to_video
from lightx2v_platform.base.global_var import AI_DEVICE


def _is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


@RUNNER_REGISTER("wan_dancer")
class WanDancerRunner(WanRunner):
    def __init__(self, config):
        super().__init__(config)
        self.vae_cls = WanDancerVAE
        self.vae_name = "Wan2.1_VAE.pth"

    def init_scheduler(self):
        scheduler_cls = WanDancerStepDistillScheduler if self.config.get("denoising_step_list") else WanDancerScheduler
        self.scheduler = scheduler_cls(self.config)

    def _run_input_encoder_local_s2v(self):
        # DefaultRunner binds this symbol during initialization; Dancer owns its
        # two-stage orchestration in run_pipeline and never calls the binding.
        raise RuntimeError("Wan-Dancer uses its stage-specific run_pipeline.")

    def load_transformer(self):
        model = WanDancerModel(self.config["model_path"], self.config, self.init_device)
        lora_configs = self.config.get("lora_configs")
        if lora_configs:
            if self.config.get("lora_dynamic_apply", False):
                raise NotImplementedError("Wan-Dancer 4-step LoRA uses merged mode.")
            model.apply_merged_lora(lora_configs)
        return model

    def load_image_encoder(self):
        # Reuse Wan's exact CLIP loader; Dancer's public task is s2v but its DiT is I2V.
        with self.config.temporarily_unlocked():
            task = self.config["task"]
            self.config["task"] = "i2v"
            try:
                return super().load_image_encoder()
            finally:
                self.config["task"] = task

    def _vae_config(self):
        offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        return {
            "vae_path": find_torch_model_path(self.config, "vae_path", self.vae_name),
            "device": torch.device("cpu") if offload else torch.device(AI_DEVICE),
            "parallel": False,
            "use_tiling": False,
            "cpu_offload": offload,
            "load_from_rank0": self.config.get("load_from_rank0", False),
            "dummy_model": self.config.get("dummy_model", False),
            "dtype": GET_DTYPE(),
        }

    def load_vae_encoder(self):
        return self.vae_cls(**self._vae_config())

    def load_vae_decoder(self):
        return self.vae_cls(**self._vae_config())

    @staticmethod
    def _crop_and_resize(image, width, height):
        image = image.convert("RGB")
        scale = min(width / image.width, height / image.height)
        resized_width, resized_height = round(image.width * scale), round(image.height * scale)
        resized = TF.resize(image, (resized_height, resized_width), interpolation=InterpolationMode.BILINEAR)
        canvas = np.full((height, width, 3), 127, dtype=np.uint8)
        left, top = (width - resized_width) // 2, (height - resized_height) // 2
        canvas[top : top + resized_height, left : left + resized_width] = np.asarray(resized, dtype=np.uint8)
        return Image.fromarray(canvas), (left, top, left + resized_width, top + resized_height)

    @staticmethod
    def _image_tensor(image):
        return TF.to_tensor(image).sub_(0.5).div_(0.5).unsqueeze(0).to(AI_DEVICE)

    @staticmethod
    def _conditioning_mask(mask, latent_height, latent_width):
        mask = torch.as_tensor(mask, dtype=GET_DTYPE()).view(1, -1, 1, 1).expand(1, -1, latent_height, latent_width)
        mask = torch.cat([mask[:, :1].repeat_interleave(4, dim=1), mask[:, 1:]], dim=1)
        return mask.view(1, mask.shape[1] // 4, 4, latent_height, latent_width).transpose(1, 2)[0]

    def _encode_keyframes(self, keyframes, mask):
        height, width = self.config["target_height"], self.config["target_width"]
        video = torch.full((1, 3, 149, height, width), -1, dtype=GET_DTYPE())
        for frame_index, image in keyframes.items():
            video[0, :, frame_index].copy_(TF.to_tensor(image).mul_(2).sub_(1).to(GET_DTYPE()))
        latent = self.vae_encoder.encode(video)
        condition_mask = self._conditioning_mask(mask, height // 8, width // 8)
        return torch.cat([condition_mask, latent], dim=0).to(AI_DEVICE)

    def _make_inputs(self, prompt_output, first_frame, reference, vae_output, music_feature, input_fps):
        clip = self.run_image_encoder(self._image_tensor(first_frame)).to(AI_DEVICE)
        ref_clip = self.run_image_encoder(self._image_tensor(reference)).to(AI_DEVICE)
        return {
            "text_encoder_output": prompt_output,
            "image_encoder_output": {
                "clip_encoder_out": clip,
                "ref_clip_encoder_out": ref_clip,
                "vae_encoder_out": vae_output,
                "music_feature": music_feature,
                "input_fps": float(input_fps),
                "enable_skip_layer": True,
            },
        }

    def _denoise(self, inputs, seed, segment_index=0, segment_count=1):
        latent_shape = [16, 38, self.config["target_height"] // 8, self.config["target_width"] // 8]
        self.inputs = inputs
        self.video_segment_num = segment_count
        self.scheduler.reset(seed, latent_shape)
        return self.run_segment(segment_index)

    @staticmethod
    def _to_comfy(video, crop_box):
        left, top, right, bottom = crop_box
        video = video[0, :, :, top:bottom, left:right]
        return video.permute(1, 2, 3, 0).float().add_(1).mul_(0.5).clamp_(0, 1).cpu()

    def _global_pipeline(self):
        width, height = self.config["target_width"], self.config["target_height"]
        reference, crop_box = self._crop_and_resize(Image.open(self.input_info.image_path), width, height)
        music_feature = extract_music_features(self.input_info.audio_path)
        divisor = max(1, int(music_feature.shape[0] / 149.0 + 0.5))
        input_fps = float(f"{30.0 / divisor:.4f}")
        original_prompt = self.input_info.prompt
        self.input_info.prompt = f"{original_prompt}帧率是{input_fps:.4f}"
        text_output = self.run_text_encoder(self.input_info)
        self.input_info.prompt = original_prompt

        mask = np.zeros(149, dtype=np.int32)
        mask[0] = 1
        vae_output = self._encode_keyframes({0: reference}, mask)
        inputs = self._make_inputs(text_output, reference, reference, vae_output, music_feature, input_fps)
        latents = self._denoise(inputs, self.input_info.seed)

        result = None
        if _is_main_process():
            decoded = self.vae_decoder.decode_framewise(latents.to(GET_DTYPE()))
            result = self._to_comfy(decoded, crop_box)
            if self.input_info.save_result_path:
                save_to_video(result, self.input_info.save_result_path, fps=self.config["fps"], method="ffmpeg")
                logger.info(f"Wan-Dancer global video saved to {self.input_info.save_result_path}")
        _barrier()
        self.scheduler.clear()
        return {"video": result if self.input_info.return_result_tensor and _is_main_process() else None}

    def _read_global_plan(self, total_frames):
        capture = cv2.VideoCapture(self.input_info.video_path)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if not frames:
            raise ValueError(f"No frames found in global plan: {self.input_info.video_path}")

        segment_count = math.ceil(total_frames / 149)
        interval = total_frames / len(frames)
        masks = []
        for segment_index in range(segment_count):
            mask = np.zeros(149, dtype=np.int32)
            if segment_index != segment_count - 1:
                count = 0
                while count * interval < 149 - interval:
                    mask[int(math.ceil(interval * count))] = 1
                    count += 1
            else:
                end_index = total_frames - 149 * segment_index - 1
                mask[end_index] = 1
                count = 0
                while count * interval < end_index - interval:
                    mask[int(math.ceil(interval * count))] = 1
                    count += 1
            masks.append(mask)

        width, height = self.config["target_width"], self.config["target_height"]
        keyframes, source_index = [], 0
        for mask in masks:
            mapping = {}
            for frame_index in np.flatnonzero(mask):
                image = Image.fromarray(frames[source_index])
                mapping[int(frame_index)] = self._crop_and_resize(image, width, height)[0]
                source_index += 1
            keyframes.append(mapping)
        for index in range(len(keyframes) - 1):
            keyframes[index][148] = keyframes[index + 1][0]
            masks[index][148] = 1
        return keyframes, masks

    def _local_pipeline(self):
        music_features, duration = split_music_features(self.input_info.audio_path)
        fps = float(self.config["fps"])
        total_frames = int(duration * fps)
        keyframes, masks = self._read_global_plan(total_frames)
        if len(music_features) != len(keyframes):
            raise ValueError(f"Audio/global-plan segment mismatch: {len(music_features)} vs {len(keyframes)}")

        width, height = self.config["target_width"], self.config["target_height"]
        reference, crop_box = self._crop_and_resize(Image.open(self.input_info.image_path), width, height)
        first = keyframes[0][0]
        if min(first.size) < 512:
            keyframes[0][0] = reference

        original_prompt = self.input_info.prompt
        self.input_info.prompt = f"{original_prompt}, 帧率是{fps:g}fps。"
        text_output = self.run_text_encoder(self.input_info)
        self.input_info.prompt = original_prompt

        decoded_segments = []
        for index, (mapping, mask, music_feature) in enumerate(zip(keyframes, masks, music_features)):
            logger.info(f"Wan-Dancer local segment {index + 1}/{len(keyframes)}")
            vae_output = self._encode_keyframes(mapping, mask)
            inputs = self._make_inputs(text_output, mapping[0], reference, vae_output, music_feature, fps)
            latents = self._denoise(inputs, self.input_info.seed + index * 10, index, len(keyframes))
            if _is_main_process():
                decoded_segments.append(self.vae_decoder.decode(latents.to(GET_DTYPE())))
            _barrier()

        result = None
        if _is_main_process():
            decoded = torch.cat(decoded_segments, dim=2)
            keep_frames = max(1, int((duration - 0.2) * fps))
            decoded = decoded[:, :, :keep_frames]
            result = self._to_comfy(decoded, crop_box)
            if self.input_info.save_result_path:
                save_to_video(result, self.input_info.save_result_path, fps=fps, method="ffmpeg")
                mux_audio_from_video(self.input_info.audio_path, self.input_info.save_result_path)
                logger.info(f"Wan-Dancer final video saved to {self.input_info.save_result_path}")
        _barrier()
        self.scheduler.clear()
        return {"video": result if self.input_info.return_result_tensor and _is_main_process() else None}

    def run_pipeline(self, input_info):
        self.input_info = input_info
        if not input_info.image_path or not input_info.audio_path:
            raise ValueError("Wan-Dancer requires image_path and audio_path.")
        stage = self.config["dancer_stage"]
        if stage == "global":
            return self._global_pipeline()
        if stage == "local":
            if not input_info.video_path:
                raise ValueError("Wan-Dancer local stage requires video_path from the global stage.")
            return self._local_pipeline()
        raise ValueError(f"Unknown Wan-Dancer stage: {stage}")
