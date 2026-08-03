# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from loguru import logger

from lightx2v.models.networks.bagel.sensenova_tasks import (
    TEXT_OUTPUT_MODES,
    clean_text_output,
    ensure_image_placeholders,
    get_mode_profile,
    get_omni_vision_task_spec,
    resolve_prompt,
)
from lightx2v.models.networks.bagel.sensenova_transforms import build_sensenova_transforms
from lightx2v.models.networks.bagel.sensenova_vision_model import SenseNovaVisionModel
from lightx2v.models.runners.bagel.bagel_runner import BagelRunner
from lightx2v.models.runners.bagel.sensenova_postprocess import load_official_postprocess, resolve_pose_string
from lightx2v.models.video_encoders.hf.bagel.sensenova_vae import SenseNovaVisionVae
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def _is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _set_request_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@RUNNER_REGISTER("sensenova_vision")
class SenseNovaVisionRunner(BagelRunner):
    def load_bagel_model(self):
        return SenseNovaVisionModel(self.config)

    def load_vae_decoder(self):
        return SenseNovaVisionVae(self.config)

    def init_modules(self):
        super().init_modules()
        self.sensenova_transforms = build_sensenova_transforms()

    def _configure_mode(self, mode):
        profile = get_mode_profile(mode)
        if "num_timesteps" not in profile:
            return profile

        inference_hyper = {
            key: profile[key]
            for key in (
                "cfg_text_scale",
                "cfg_img_scale",
                "cfg_interval",
                "timestep_shift",
                "cfg_renorm_min",
                "cfg_renorm_type",
            )
        }
        self.model.inference_hyper = inference_hyper
        self.scheduler.infer_steps = int(profile["num_timesteps"])
        self.scheduler.timestep_shift = float(profile["timestep_shift"])
        self.scheduler.set_timesteps()
        return profile

    @staticmethod
    def _parse_image_paths(image_path, allow_empty=False):
        paths = [item.strip() for item in str(image_path or "").split(",") if item.strip()]
        if not paths:
            if allow_empty:
                return []
            raise ValueError("SenseNova-Vision requires --image_path with one or more comma-separated images.")
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"SenseNova-Vision input image(s) not found: {missing}")
        return paths

    @staticmethod
    def _load_images(paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return images

    @staticmethod
    def _build_interleaved_inputs(prompt, images):
        prompt = ensure_image_placeholders(prompt, len(images))
        parts = prompt.split("<image>")
        inputs = []
        for index, part in enumerate(parts):
            text = part.strip()
            if text:
                inputs.append(text)
            if index < len(images):
                inputs.append(images[index])
        return inputs

    def _select_transforms(self, task):
        if task == "recon3d":
            return self.sensenova_transforms["recon3d_vae"], self.sensenova_transforms["recon3d_vit"]
        if task == "camera_pose":
            return self.sensenova_transforms["vae"], self.sensenova_transforms["camera_vit"]
        return self.sensenova_transforms["vae"], self.sensenova_transforms["vit"]

    def _build_decode_info(self, prepared):
        image_shapes = [prepared.image_shape] * len(prepared.output_packed_seqlens)
        return {
            "packed_seqlens": prepared.output_packed_seqlens,
            "image_shapes": image_shapes,
            "latent_downsample": self.model.latent_downsample,
            "latent_channel": self.model.latent_channel,
            "latent_patch_size": self.model.latent_patch_size,
            "return_result_tensor": bool(getattr(self.input_info, "return_result_tensor", False)),
            "output_raw_tensor": prepared.output_raw_tensor,
        }

    @staticmethod
    def _derive_path(save_path, suffix, default_extension):
        path = Path(save_path)
        if path.suffix.lower() == default_extension:
            return path
        if path.suffix:
            return path.with_name(f"{path.stem}{suffix}{default_extension}")
        return path.with_suffix(default_extension)

    def _save_text(self, text, input_info, has_image_output, task):
        if not _is_main_process() or not text:
            return None
        print(f"SenseNova-Vision text output:\n{text}")
        save_path = getattr(input_info, "save_result_path", None)
        if not save_path:
            return None
        path = Path(save_path)
        if has_image_output:
            path = path.with_name(f"{path.stem}_text.txt")
        elif path.suffix.lower() != ".txt":
            path = path.with_suffix(".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.info(f"SenseNova-Vision text saved: {path}")

        if task == "camera_pose":
            pose = resolve_pose_string(text)
            pose_path = path.with_name(f"{path.stem}_pose.json")
            pose_path.write_text(json.dumps(pose, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"SenseNova-Vision parsed camera pose saved: {pose_path}")
        return str(path)

    def _save_images(self, images, input_info, log_prefix="SenseNova-Vision image saved"):
        if not _is_main_process() or getattr(input_info, "return_result_tensor", False):
            return []
        save_path = getattr(input_info, "save_result_path", None)
        if not save_path:
            return []
        path = Path(save_path)
        extension = path.suffix or ".png"
        stem = path.stem if path.suffix else path.name
        output_paths = []
        for index, image in enumerate(images):
            output_path = path.with_name(f"{stem}_{index:05d}{extension}") if len(images) > 1 else path.with_suffix(extension)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            logger.info(f"{log_prefix}: {output_path}")
            output_paths.append(str(output_path))
        return output_paths

    def _postprocess_recon3d(self, pointmaps, prepared, input_info, actual_image_count):
        pointmaps = np.asarray(pointmaps[:actual_image_count], dtype=np.float32)
        raw_path = getattr(input_info, "raw_output_path", "")
        if not raw_path:
            save_path = getattr(input_info, "save_result_path", "") or "sensenova_recon3d.npy"
            raw_path = str(self._derive_path(save_path, "_raw", ".npy"))
        raw_path = Path(raw_path)
        if _is_main_process():
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(raw_path, pointmaps)
            logger.info(f"SenseNova-Vision raw point maps saved: {raw_path}")

        scene = None
        glb_path = getattr(input_info, "glb_output_path", "")
        requested_postprocess = getattr(input_info, "postprocess_predictions", None)
        if requested_postprocess is None:
            requested_postprocess = self.config.get("postprocess_predictions", False)
        postprocess = bool(requested_postprocess or glb_path)
        if postprocess:
            source_path = self.config.get(
                "sensenova_source_path",
                "/data/nvme0/lhd_codes/sensenova-vision-v2",
            )
            postprocess_reconstruction = load_official_postprocess(source_path)
            scene = postprocess_reconstruction(
                list(pointmaps),
                prepared.preprocessed_images[:actual_image_count],
                mask_edge=True,
                mask_sky=False,
                mask_black_bg=False,
                mask_white_bg=False,
            )
            if not glb_path:
                glb_path = str(raw_path.with_name(f"{raw_path.stem}_scene.glb"))
            if _is_main_process():
                Path(glb_path).parent.mkdir(parents=True, exist_ok=True)
                scene.export(file_obj=glb_path)
                logger.info(f"SenseNova-Vision reconstructed scene saved: {glb_path}")
        return pointmaps, scene, str(raw_path), glb_path or None

    def run_pipeline(self, input_info):
        self.input_info = input_info
        _set_request_seed(getattr(input_info, "seed", 42))
        subtask, task_spec = get_omni_vision_task_spec(getattr(input_info, "omni_vision_subtask", ""))
        input_info.omni_vision_subtask = subtask
        task = task_spec.runner_task
        mode = task_spec.mode
        self._configure_mode(mode)

        image_paths = self._parse_image_paths(
            input_info.image_path,
            allow_empty=mode in {"generate", "think_generate"},
        )
        images = self._load_images(image_paths)
        actual_image_count = len(images)
        prompt = resolve_prompt(task, input_info.prompt)
        if actual_image_count < task_spec.min_images or actual_image_count > task_spec.max_images:
            expected = str(task_spec.min_images) if task_spec.min_images == task_spec.max_images else f"{task_spec.min_images}-{task_spec.max_images}"
            raise ValueError(f"SenseNova-Vision subtask={subtask!r} requires {expected} input image(s), got {actual_image_count}.")
        if task == "recon3d":
            if actual_image_count > 10:
                logger.warning("SenseNova-Vision recon3d accepts at most 10 images; truncating input.")
                images = images[:10]
                actual_image_count = 10
            if actual_image_count == 1:
                logger.info("SenseNova-Vision recon3d duplicates a single input view, matching official inference.")
                images = images * 2
            input_lists = [*images, prompt]
        else:
            input_lists = self._build_interleaved_inputs(prompt, images)

        vae_transform, vit_transform = self._select_transforms(task)
        prepared = self.model.prepare_sensenova_inputs(
            input_info=input_info,
            scheduler=self.scheduler,
            vae_model=self.vae_decoder,
            input_lists=input_lists,
            mode=mode,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            output_multiple_vae=task == "recon3d",
            output_raw_tensor=task == "recon3d",
        )
        text = clean_text_output(prepared.text_outputs[0]) if prepared.text_outputs else ""

        if mode in TEXT_OUTPUT_MODES:
            self._save_text(text, input_info, has_image_output=False, task=task)
            self.end_run()
            return {"images": [], "text": text}

        self.inputs = prepared.inputs
        latents, generator = self.run_dit()
        outputs = self.run_vae_decoder(latents, self._build_decode_info(prepared))

        result = {"images": outputs, "text": text}
        if task == "recon3d":
            pointmaps, scene, raw_path, glb_path = self._postprocess_recon3d(
                outputs,
                prepared,
                input_info,
                actual_image_count,
            )
            result = {
                "pts3d": pointmaps,
                "scene": scene,
                "text": text,
                "raw_output_path": raw_path,
                "glb_output_path": glb_path,
            }
        else:
            self._save_images(outputs, input_info)
            self._save_text(text, input_info, has_image_output=True, task=task)

        del latents, generator
        torch_device_module.empty_cache()
        gc.collect()
        self.end_run()
        return result
