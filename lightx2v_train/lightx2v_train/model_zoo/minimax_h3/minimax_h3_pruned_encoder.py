"""Distill an H3 encoder through the frozen, original H3 decoder."""

import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from lightx2v_train.model_capabilities import VAEDistillationCapability
from lightx2v_train.model_zoo.native.minimax_h3.pruned_encoder import (
    MiniMaxH3PrunedVideoEncoder,
    teacher_encoder_forward,
    teacher_encoder_suffix,
)
from lightx2v_train.model_zoo.native.minimax_h3.video_vae import (
    imagenet_preprocess,
    load_minimax_h3_video_vae,
    normalize_video_latents,
    resolve_video_vae_dir,
)
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from ..base import BaseModel
from .capability_adapters.minimax_h3_encoder_distillation_capability import MiniMaxH3EncoderDistillationCapability
from .minimax_h3_pruned_vae import teacher_config_fingerprint
from .minimax_h3_vae import MiniMaxH3VAEModel


@MODEL_REGISTER("minimax_h3_pruned_encoder")
class MiniMaxH3PrunedEncoderModel(MiniMaxH3VAEModel):
    def register_capabilities(self):
        BaseModel.register_capabilities(self)
        self.capabilities.register(
            VAEDistillationCapability,
            MiniMaxH3EncoderDistillationCapability(self, self.config["training"]["vae_distillation"]),
        )

    def load_components(self, *, load_transformer, load_vae, load_condition_encoder):
        del load_vae, load_condition_encoder
        config = self.config["model"]
        self.pretrained_model_path = config["pretrained_model_name_or_path"]
        self.student_param_dtype = get_running_dtype(config.get("student_param_dtype", "fp32"))
        self.student_autocast = config.get("student_autocast", True)
        self.teacher_encoder_dtype = get_running_dtype(config.get("teacher_encoder_dtype", "fp32"))
        self.teacher_autocast_dtype = get_running_dtype(config.get("teacher_autocast_dtype", "fp32"))
        distillation = self.config["training"].get("vae_distillation", {})
        self.feature_distillation_weight = self._maximum_distillation_weight(distillation, "feature")
        with (resolve_video_vae_dir(self.pretrained_model_path) / "config.json").open() as handle:
            self.teacher_config = json.load(handle)

        encoder_config = dict(config["pruned_encoder"])
        selection_path = encoder_config.pop("selection_path", None)
        self.search_config = encoder_config.get("search")
        if selection_path:
            if self.search_config is not None or "kept_residual_indices" in encoder_config:
                raise ValueError("selection_path defines the fixed encoder; do not also specify search or kept_residual_indices.")
            with Path(selection_path).open() as handle:
                selection = json.load(handle)
            if selection["teacher_config_sha256"] != teacher_config_fingerprint(self.teacher_config):
                raise ValueError("The encoder selection was exported for a different H3 VAE config.")
            encoder_config["kept_residual_indices"] = selection["kept_residual_indices"]
        if self.search_config is None and "kept_residual_indices" not in encoder_config:
            raise ValueError("Encoder recovery requires selection_path or kept_residual_indices.")
        if self.search_config is not None and self.config["training"]["method"] != "vae_encoder_pruning":
            raise ValueError("A search encoder requires training.method=vae_encoder_pruning.")

        teacher = load_minimax_h3_video_vae(
            self.pretrained_model_path,
            torch_dtype=torch.float32,
            local_files_only=config.get("local_files_only", True),
        )
        self.transformer = None
        if load_transformer:
            self.transformer = MiniMaxH3PrunedVideoEncoder(teacher_config=self.teacher_config, **encoder_config)
            self.transformer.initialize_from_teacher_state(teacher.state_dict())
            if selection_path:
                weights_path = Path(selection_path).parent / self.transformer.weights_name
                self.transformer.load_state_dict(load_file(str(weights_path)), strict=True)
                logger.info("[encoder] loaded stage-one selected initialization from {}", weights_path)
            self.transformer.to(device=self.device, dtype=self.student_param_dtype)
            if config.get("initial_weights_path"):
                self._load_initial_weights(config["initial_weights_path"])
            if self.search_config is not None:
                pruning = self.transformer.encoder
                logger.info(
                    "[encoder-prune] grouping={} keep={} groups={} candidates={}",
                    pruning.search_grouping, pruning.keep_residuals,
                    pruning.residual_groups if pruning.search_grouping == "stage" else None,
                    len(pruning.candidate_masks),
                )

        # Frozen parameters still provide the Jacobian needed to train the encoder.
        self.teacher_vae = teacher.requires_grad_(False).eval().to(self.device)
        self.teacher_vae.decoder.gradient_checkpointing = True
        self.teacher_vae.decoder._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        logger.info(
            "[encoder] original teacher encoder and decoder frozen; decoder input gradients enabled; retained residuals={}",
            self.transformer.selected_layers() if self.transformer is not None else None,
        )

    @property
    def latent_channels(self):
        return int(self.teacher_config["latent_channels"])

    def set_full_trainable(self):
        student = self.denoiser_module()
        if self.search_config is not None:
            student.configure_search_trainable()
        else:
            student.requires_grad_(True)
        student.train()
        self.teacher_vae.requires_grad_(False).eval()

    def student_encode_clip(self, pixel_clip, running_dtype, *, return_features=False, auxiliary_index=None):
        context = self._autocast(running_dtype) if self.student_autocast else nullcontext()
        with context:
            return self.transformer(pixel_clip, return_features=return_features, auxiliary_index=auxiliary_index)

    @torch.no_grad()
    def teacher_encode_clip(self, pixel_clip, *, return_features=False):
        with self._autocast(self.teacher_encoder_dtype):
            return teacher_encoder_forward(self.teacher_vae, pixel_clip.float(), return_features=return_features)

    def teacher_encoder_suffix(self, stage_feature, stage_index, *, gradient_checkpointing=True):
        with self._autocast(self.teacher_encoder_dtype):
            return teacher_encoder_suffix(
                self.teacher_vae,
                stage_feature.float(),
                stage_index,
                gradient_checkpointing=gradient_checkpointing,
            )

    def _denormalize_latents(self, normalized_latents):
        latents = normalized_latents.float()
        mean = latents.new_tensor(self.teacher_config["latents_mean"]).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(self.teacher_config["latents_std"]).view(1, -1, 1, 1, 1)
        return latents * std + mean

    def decode_student_latent_window(self, normalized_latents, running_dtype=None):
        del running_dtype
        with self._autocast(self.teacher_autocast_dtype):
            return self.teacher_vae.decoder(self.teacher_vae.post_quant_conv(self._denormalize_latents(normalized_latents))).float()

    @torch.no_grad()
    def encode_video(self, video):
        """Use the student with the original 17-frame/tiled posterior protocol."""
        student = self.denoiser_module()
        video = imagenet_preprocess(video.to(device=self.device, dtype=torch.float32))
        clip_length = self.teacher_config["clip_length"]
        padding = (-video.shape[2]) % clip_length
        if padding:
            video = torch.cat((video, video[:, :, -1:].expand(-1, -1, padding, -1, -1)), dim=2)
        height, width = video.shape[-2:]
        if student.use_tiling:
            ys, hs, y_overlaps = self.teacher_vae._split_tiles(height, student.tile_sample_min_height, student.tile_sample_min_overlap_height)
            xs, ws, x_overlaps = self.teacher_vae._split_tiles(width, student.tile_sample_min_width, student.tile_sample_min_overlap_width)
        else:
            ys, hs, y_overlaps = [0], [height], []
            xs, ws, x_overlaps = [0], [width], []
        ratio = self.teacher_vae.spatial_compression_ratio
        moments = []
        for start in range(0, video.shape[2], clip_length):
            rows = []
            for y, h in zip(ys, hs, strict=True):
                row = []
                for x, w in zip(xs, ws, strict=True):
                    pixel_clip = video[:, :, start : start + clip_length, y : y + h, x : x + w]
                    row.append(self.student_encode_clip(pixel_clip, self.running_dtype).float())
                rows.append(row)
            moments.append(self.teacher_vae._stitch_tiles(rows, [v // ratio for v in y_overlaps], [v // ratio for v in x_overlaps]))
        moments = torch.cat(moments, dim=2)
        token_drop = self.teacher_config["token_drop"]
        if token_drop:
            moments = moments[:, :, :-token_drop]
        return normalize_video_latents(self.teacher_vae, moments[:, : self.latent_channels])

    @torch.no_grad()
    def decode_latents(self, normalized_latents):
        normalized_latents = normalized_latents.to(device=self.device, dtype=torch.float32)
        with self._autocast(self.teacher_autocast_dtype):
            raw_video = self.teacher_vae._decode(self._denormalize_latents(normalized_latents)).float()
        return self.postprocess_raw_video(raw_video)

    def consolidated_safetensors_metadata(self):
        return {
            "format": "pt",
            "model_type": "minimax_h3_pruned_encoder",
            "architecture": json.dumps(self.denoiser_module().architecture_config, sort_keys=True, separators=(",", ":")),
        }

    def validate_consolidated_metadata(self, metadata, checkpoint_path):
        expected = self.consolidated_safetensors_metadata()
        if any(metadata.get(key) != expected[key] for key in ("model_type", "architecture")):
            raise ValueError(f"Checkpoint does not match the configured pruned H3 encoder: {checkpoint_path}")

    @torch.no_grad()
    def export_pruned_encoder(self, output_dir, gate_logits):
        student = self.denoiser_module()
        kept = student.selected_layers(logits=gate_logits)
        export_config = {
            key: getattr(student, key)
            for key in (
                "use_tiling", "tile_sample_min_height", "tile_sample_min_width",
                "tile_sample_min_overlap_height", "tile_sample_min_overlap_width",
            )
        }
        exported = MiniMaxH3PrunedVideoEncoder(self.teacher_config, kept_residual_indices=kept, **export_config)
        exported.initialize_from_teacher_state(self.teacher_vae.state_dict())
        output_dir = Path(output_dir)
        exported.save_pretrained(output_dir)
        selection = {
            "schema_version": 1,
            "teacher_config_sha256": teacher_config_fingerprint(self.teacher_config),
            "teacher_model_path": str(self.pretrained_model_path),
            "original_num_residuals": len(self.teacher_config["block_out_channels"]) * self.teacher_config["layers_per_block"],
            "kept_residual_indices": kept,
            "teacher_feature_indices": list(range(len(self.teacher_config["block_out_channels"]))),
            "initialization": "original_teacher_weights_without_search_lora",
        }
        if student.search_config is not None:
            pruning = student.encoder
            selection["grouping"] = pruning.search_grouping
            if pruning.search_grouping == "stage":
                selection["residual_groups"] = pruning.residual_groups
                selection["group_budgets"] = [len(set(group).intersection(kept)) for group in pruning.residual_groups]
        temporary = output_dir / "kept_layers.json.tmp"
        temporary.write_text(json.dumps(selection, indent=2) + "\n")
        temporary.replace(output_dir / "kept_layers.json")
        logger.info("[encoder-prune] exported retained residuals={} to {}", kept, output_dir)
