"""Runner for Hunyuan3D-2.1 shape generation (image -> mesh)."""

from __future__ import annotations

import gc
import os

import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.hunyuan3d.encoder import Hunyuan3DConditionEncoder, Hunyuan3DImagePreprocessor
from lightx2v.models.networks.hunyuan3d.model import Hunyuan3DDiTModel
from lightx2v.models.networks.hunyuan3d.utils import torchvision_fix
from lightx2v.models.networks.hunyuan3d.utils.checkpoint import load_checkpoint_dict, resolve_ckpt_paths, resolve_model_dir
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.hunyuan3d.scheduler import Hunyuan3DShapeScheduler
from lightx2v.models.video_encoders.hf.hunyuan3d.decoder import Hunyuan3DShapeVAEDecoder
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


@RUNNER_REGISTER("hunyuan3d")
class Hunyuan3DShapeRunner(DefaultRunner):
    """Image-to-3D-mesh runner for Hunyuan3D-2.1 shape pipeline."""

    def __init__(self, config):
        super().__init__(config)
        self._ckpt = None
        self.image_preprocessor = Hunyuan3DImagePreprocessor(enable_rembg=config.get("enable_rembg", True))

    def init_scheduler(self):
        self.scheduler = Hunyuan3DShapeScheduler(self.config)

    def init_modules(self):
        logger.info("Initializing runner modules...")
        if not self.config.get("lazy_load", False) and not self.config.get("unload_modules", False):
            self.load_model()
        if hasattr(self, "model") and self.model is not None:
            self.model.set_scheduler(self.scheduler)
        self.run_input_encoder = self._run_input_encoder_local_i23d
        self.config.lock()

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self._apply_torchvision_fix()
        self._ckpt = self._load_checkpoint()
        self.model = self.load_transformer()
        self.image_encoder = self.load_image_encoder()
        _, self.vae_decoder = self.load_vae()

    def _apply_torchvision_fix(self) -> None:
        try:
            torchvision_fix.apply_fix()
        except Exception as exc:
            logger.warning(f"Failed to apply torchvision fix: {exc}")

    def _load_checkpoint(self):
        model_path = self.config["model_path"]
        subfolder = self.config.get("subfolder", "hunyuan3d-dit-v2-1")
        use_safetensors = bool(self.config.get("use_safetensors", False))
        variant = self.config.get("variant", "fp16")
        model_dir = resolve_model_dir(model_path, subfolder)
        _, ckpt_path = resolve_ckpt_paths(model_dir, use_safetensors=use_safetensors, variant=variant)
        return load_checkpoint_dict(ckpt_path, use_safetensors=use_safetensors)

    def load_transformer(self):
        with ProfilingContext4DebugL1("Load Hunyuan3D DiT"):
            model_dir = resolve_model_dir(self.config["model_path"], self.config.get("subfolder", "hunyuan3d-dit-v2-1"))
            model = Hunyuan3DDiTModel(
                model_path=model_dir,
                config=self.config,
                device=self.init_device,
                weight_dict=self._build_dit_weight_dict(),
            )
            model.set_scheduler(self.scheduler)
            return model

    def _build_dit_weight_dict(self):
        state_dict = self._ckpt["model"]
        dtype = Hunyuan3DDiTModel._resolve_dit_dtype_from_config(self.config)
        weight_dict = {}
        for key, tensor in state_dict.items():
            weight_dict[key] = tensor.to(dtype=dtype)
        return weight_dict

    def load_text_encoder(self):
        return []

    def load_image_encoder(self):
        with ProfilingContext4DebugL1("Load Hunyuan3D condition encoder"):
            return Hunyuan3DConditionEncoder.from_pretrained(self.config, self._ckpt)

    def load_vae(self):
        with ProfilingContext4DebugL1("Load Hunyuan3D shape VAE"):
            decoder = Hunyuan3DShapeVAEDecoder.from_pretrained(self.config, self._ckpt)
            return None, decoder

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_i23d(self):
        image_path = self.input_info.image_path
        if not image_path:
            raise ValueError("input_info.image_path must be set")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path does not exist: {image_path}")

        with ProfilingContext4DebugL1("Prepare input image"):
            image = self.image_preprocessor(image_path)

        guidance_scale = float(self.config.get("guidance_scale", 5.0))
        do_classifier_free_guidance = guidance_scale >= 0 and not (hasattr(self.model, "guidance_embed") and self.model.guidance_embed is True)

        with ProfilingContext4DebugL1("Encode condition"):
            cond_inputs = self.image_encoder.prepare_image(image, mask=None)
            image_tensor = cond_inputs.pop("image")
            encoded = self.image_encoder.encode_cond(
                image_tensor=image_tensor,
                additional_cond_inputs=cond_inputs,
                do_classifier_free_guidance=do_classifier_free_guidance,
                dual_guidance=False,
            )
            cond = self._cast_cond_dtype(encoded["cond"], self.image_encoder.dtype)
            uncond = encoded.get("uncond")
            if uncond is not None:
                uncond = self._cast_cond_dtype(uncond, self.image_encoder.dtype)

        guidance = None
        if hasattr(self.model, "guidance_embed") and self.model.guidance_embed is True:
            batch_size = image_tensor.shape[0]
            guidance = torch.tensor(
                [guidance_scale] * batch_size,
                device=self.image_encoder.device,
                dtype=self.image_encoder.dtype,
            )

        torch_device_module.empty_cache()
        gc.collect()
        return {
            "cond": cond,
            "uncond": uncond,
            "image_tensor": image_tensor,
            "do_classifier_free_guidance": do_classifier_free_guidance,
            "guidance_scale": guidance_scale,
            "guidance": guidance,
        }

    @staticmethod
    def _cast_cond_dtype(cond, dtype):
        if isinstance(cond, torch.Tensor):
            return cond.to(dtype=dtype)
        return {k: Hunyuan3DShapeRunner._cast_cond_dtype(v, dtype) for k, v in cond.items()}

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        latent_shape = (self.inputs["image_tensor"].shape[0], *self.vae_decoder.vae.latent_shape)
        self.scheduler.prepare(
            seed=getattr(self.input_info, "seed", None),
            batch_size=latent_shape[0],
            latent_shape=latent_shape,
        )

        dit_inputs = {
            "cond": self.inputs["cond"],
            "uncond": self.inputs.get("uncond"),
            "guidance": self.inputs.get("guidance"),
            "guidance_scale": self.inputs["guidance_scale"],
            "do_classifier_free_guidance": self.inputs["do_classifier_free_guidance"],
        }

        enable_pbar = bool(self.config.get("enable_pbar", True))
        iterator = range(self.scheduler.infer_steps)
        if enable_pbar:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Diffusion Sampling:")

        for step_index in iterator:
            with ProfilingContext4DebugL1("step_pre"):
                self.scheduler.step_pre(step_index)
            with ProfilingContext4DebugL1("infer_main"):
                self.model.infer(dit_inputs)
            with ProfilingContext4DebugL1("step_post"):
                self.scheduler.step_post()

        mesh = self._export_mesh(self.scheduler.latents)
        save_result_path = self.input_info.save_result_path
        if not save_result_path:
            raise ValueError("input_info.save_result_path must be set")
        return self._save_mesh(mesh, save_result_path)

    @ProfilingContext4DebugL1("Export mesh")
    def _export_mesh(self, latents):
        return self.vae_decoder.decode_mesh(
            latents,
            box_v=float(self.config.get("box_v", 1.01)),
            mc_level=float(self.config.get("mc_level", 0.0)),
            num_chunks=int(self.config.get("num_chunks", 8000)),
            octree_resolution=int(self.config.get("octree_resolution", 384)),
            mc_algo=self.config.get("mc_algo"),
            enable_pbar=bool(self.config.get("enable_pbar", True)),
        )

    @ProfilingContext4DebugL1("Save mesh")
    def _save_mesh(self, mesh, save_path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        mesh.export(save_path)
        logger.info(f"[Hunyuan3D] Saved mesh to {save_path}")
        return save_path

    def end_run(self):
        if hasattr(self, "inputs"):
            del self.inputs
        self.input_info = None
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.clear()
        torch_device_module.empty_cache()
        gc.collect()

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["Hunyuan3DShapeRunner"],
    )
    @torch.inference_mode()
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info
        self.inputs = self.run_input_encoder()
        result = self.run_main()
        self.end_run()
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        return result
