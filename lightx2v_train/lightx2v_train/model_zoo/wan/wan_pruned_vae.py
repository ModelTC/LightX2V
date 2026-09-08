"""Wan2.1 residual-branch search and recovery through frozen VAE components."""

import json
from contextlib import nullcontext
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file

from lightx2v_train.model_capabilities import VAEDistillationCapability
from lightx2v_train.model_zoo.native.wan.pruned_vae import (
    WAN_VAE_CONFIG,
    WanPrunedDecoder,
    WanPrunedEncoder,
    normalized_posterior_stats,
    teacher_suffix,
)
from lightx2v_train.runtime.ddp import unwrap_ddp_module
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from ..base import BaseModel
from .capability_adapters.wan_vae_distillation_capability import WanVAEDistillationCapability


class WanPrunedVAEModel(BaseModel):
    pipeline_cls = None
    component = None

    def register_capabilities(self):
        super().register_capabilities()
        self.capabilities.register(
            VAEDistillationCapability,
            WanVAEDistillationCapability(self, self.config["training"]["vae_distillation"]),
        )

    def load_components(self, *, load_transformer, load_vae, load_condition_encoder):
        del load_vae, load_condition_encoder
        config = self.config["model"]
        path = Path(config["pretrained_model_name_or_path"])
        self.pretrained_model_path = path / "Wan2.1_VAE.pth" if path.is_dir() else path
        self.teacher_config = {**WAN_VAE_CONFIG, **config.get("vae_architecture", {})}
        self.teacher_autocast_dtype = get_running_dtype(config.get("teacher_autocast_dtype", "fp32"))
        self.student_autocast = bool(config.get("student_autocast", True))
        student_dtype = get_running_dtype(config.get("student_param_dtype", "fp32"))
        pruning = dict(config[f"pruned_{self.component}"])
        selection_path = pruning.pop("selection_path", None)
        self.search_config = pruning.get("search")
        if selection_path:
            if self.search_config is not None or "kept_residual_indices" in pruning:
                raise ValueError("Use selection_path alone to initialize a fixed Wan VAE component.")
            selection = json.loads(Path(selection_path).read_text())
            if selection["component"] != self.component or selection["teacher_architecture"] != self.teacher_config:
                raise ValueError("The Wan pruning selection belongs to a different component or architecture.")
            pruning["kept_residual_indices"] = selection["kept_residual_indices"]
        if self.search_config is None and "kept_residual_indices" not in pruning:
            raise ValueError("Wan recovery requires selection_path or kept_residual_indices.")

        state = torch.load(self.pretrained_model_path, map_location="cpu", weights_only=True)
        component_cls = WanPrunedEncoder if self.component == "encoder" else WanPrunedDecoder
        self.transformer = None
        if load_transformer:
            self.transformer = component_cls(teacher_config=self.teacher_config, **pruning)
            self.transformer.initialize_from_teacher_state(state)
            self.transformer.to(device=self.device, dtype=student_dtype)
            if selection_path:
                weights = Path(selection_path).parent / self.transformer.weights_name
                self.transformer.load_state_dict(load_file(str(weights)), strict=True)
                logger.info("[wan-vae] loaded {} search export: {}", self.component, weights)
            if config.get("initial_weights_path"):
                self._load_initial_weights(config["initial_weights_path"])

        self.teacher_encoder = WanPrunedEncoder(teacher_config=self.teacher_config)
        self.teacher_decoder = WanPrunedDecoder(teacher_config=self.teacher_config)
        for teacher in (self.teacher_encoder, self.teacher_decoder):
            teacher.initialize_from_teacher_state(state)
            teacher.requires_grad_(False).eval().to(device=self.device, dtype=torch.float32)
            teacher.enable_gradient_checkpointing()
        logger.info(
            "[wan-vae] component={} stage={} teacher={} kept={} grouping={} groups={}",
            self.component,
            "search" if self.search_config is not None else "recovery",
            self.pretrained_model_path,
            self.transformer.selected_layers() if self.transformer is not None else None,
            self.transformer.search_grouping if self.transformer is not None and self.search_config is not None else None,
            self.transformer.residual_groups if self.transformer is not None else None,
        )

    def denoiser_module(self):
        return self.transformer

    @property
    def latent_channels(self):
        return int(self.teacher_config["z_dim"])

    def _autocast(self, dtype):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)
        return nullcontext()

    def set_full_trainable(self):
        student = unwrap_ddp_module(self.transformer)
        if self.search_config is not None:
            student.configure_search_trainable()
        else:
            student.requires_grad_(True)
        student.train()

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

    def fsdp2_shard_plan(self, fsdp_config):
        # Stateless functional traversal lives inside the component's root forward.
        # Shard the root, not child modules whose forward may be bypassed by traversal.
        return [{
            "module": self.transformer,
            "reshard_after_forward": fsdp_config.get("reshard_after_forward", {}).get("root_reshard", False),
        }]

    @staticmethod
    def _unpack(output, return_features, auxiliary_feature_index):
        if auxiliary_feature_index is not None:
            prediction, features, auxiliary = output
            return prediction, features, auxiliary
        if return_features:
            prediction, features = output
            return prediction, features, None
        return output, (), None

    def distillation_forward(self, video, *, running_dtype, return_features=False, auxiliary_feature_index=None):
        pixels = video.to(device=self.device, dtype=torch.float32) * 2.0 - 1.0
        need_auxiliary = auxiliary_feature_index is not None
        result = {}
        with torch.no_grad(), self._autocast(self.teacher_autocast_dtype):
            teacher_output = self.teacher_encoder(pixels, return_features=return_features and self.component == "encoder")
            moments, encoder_features, _ = self._unpack(
                teacher_output, return_features and self.component == "encoder", None,
            )
            teacher_mu, teacher_std = normalized_posterior_stats(moments)
            result.update(teacher_mu=teacher_mu, teacher_std=teacher_std, teacher_latents=teacher_mu)
            if self.component == "encoder":
                result["teacher_features"] = encoder_features
            if need_auxiliary or (return_features and self.component == "decoder"):
                teacher_output = self.teacher_decoder(teacher_mu, return_features=return_features and self.component == "decoder")
                prediction, features, _ = self._unpack(
                    teacher_output, return_features and self.component == "decoder", None,
                )
                result["teacher_prediction"] = prediction.float()
                if self.component == "decoder":
                    result["teacher_features"] = features

        inputs = pixels if self.component == "encoder" else teacher_mu
        compute_dtype = running_dtype if self.student_autocast else torch.float32
        with self._autocast(compute_dtype):
            output = self.transformer(
                inputs,
                return_features=return_features,
                auxiliary_feature_index=auxiliary_feature_index,
            )
        prediction, features, auxiliary = self._unpack(output, return_features, auxiliary_feature_index)
        result["student_features"] = features
        if self.component == "encoder":
            mu, std = normalized_posterior_stats(prediction)
            result.update(student_mu=mu, student_std=std)
            # Frozen parameters still transmit pixel-loss gradients into the student encoder.
            with self._autocast(self.teacher_autocast_dtype):
                prediction = self.teacher_decoder(mu)
        result["prediction"] = prediction.float()

        if need_auxiliary:
            teacher = self.teacher_encoder if self.component == "encoder" else self.teacher_decoder
            with self._autocast(self.teacher_autocast_dtype):
                auxiliary_prediction = teacher_suffix(teacher, auxiliary.float(), auxiliary_feature_index)
                if self.component == "encoder":
                    mu, _ = normalized_posterior_stats(auxiliary_prediction)
                    auxiliary_prediction = self.teacher_decoder(mu)
            result["auxiliary_prediction"] = auxiliary_prediction.float()
        return result

    @torch.no_grad()
    def reconstruct(self, video):
        pixels = video.to(device=self.device, dtype=torch.float32) * 2.0 - 1.0
        self.transformer.eval()
        with self._autocast(self.teacher_autocast_dtype):
            if self.component == "decoder":
                moments = self.teacher_encoder(pixels)
            else:
                with self._autocast(self.running_dtype if self.student_autocast else torch.float32):
                    moments = self.transformer(pixels)
            latents, _ = normalized_posterior_stats(moments)
            if self.component == "encoder":
                prediction = self.teacher_decoder(latents)
            else:
                with self._autocast(self.running_dtype if self.student_autocast else torch.float32):
                    prediction = self.transformer(latents)
        return ((prediction.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    def consolidated_safetensors_metadata(self):
        return {
            "format": "pt",
            "model_type": f"wan21_pruned_{self.component}",
            "architecture": json.dumps(self.transformer.architecture_config, sort_keys=True),
        }

    def _load_initial_weights(self, checkpoint_path):
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        expected = self.consolidated_safetensors_metadata()
        if any(metadata.get(key) != expected[key] for key in ("model_type", "architecture")):
            raise ValueError(f"Wan checkpoint architecture does not match the selected student: {checkpoint_path}")
        self.transformer.load_state_dict(load_file(str(checkpoint_path)), strict=True)
        logger.info("[wan-vae] loaded initial weights: {}", checkpoint_path)

    @torch.no_grad()
    def export_pruned_component(self, output_dir, gate_logits):
        student = unwrap_ddp_module(self.transformer)
        kept = student.selected_layers(logits=gate_logits)
        component_cls = WanPrunedEncoder if self.component == "encoder" else WanPrunedDecoder
        exported = component_cls(teacher_config=self.teacher_config, kept_residual_indices=kept)
        original_state = torch.load(self.pretrained_model_path, map_location="cpu", weights_only=True)
        exported.initialize_from_teacher_state(original_state)
        output_dir = Path(output_dir)
        exported.save_pretrained(output_dir)
        selection = {
            "schema_version": 1,
            "component": self.component,
            "teacher_architecture": self.teacher_config,
            "original_residual_count": student.depth,
            "kept_residual_indices": kept,
            "search_grouping": student.search_grouping,
            "residual_groups": student.residual_groups,
            "kept_per_group": [len(set(group).intersection(kept)) for group in student.residual_groups],
            "initialization": "original_teacher_weights_without_search_lora",
        }
        # Publish last so recovery cannot observe a selection before weights exist.
        temporary = output_dir / "kept_layers.json.tmp"
        temporary.write_text(json.dumps(selection, indent=2) + "\n")
        temporary.replace(output_dir / "kept_layers.json")
        logger.info("[wan-vae] exported {} retained residuals={} to {}", self.component, kept, output_dir)


@MODEL_REGISTER("wan21_pruned_encoder")
class WanPrunedEncoderModel(WanPrunedVAEModel):
    component = "encoder"


@MODEL_REGISTER("wan21_pruned_decoder")
class WanPrunedDecoderModel(WanPrunedVAEModel):
    component = "decoder"
