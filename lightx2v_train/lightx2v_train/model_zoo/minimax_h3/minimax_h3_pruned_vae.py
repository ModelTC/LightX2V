"""Search and recover a shallow decoder without changing H3's latent interface."""

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

from lightx2v_train.model_zoo.native.minimax_h3.pruned_vae import MiniMaxH3PrunedVideoVAE, decode_teacher_suffix
from lightx2v_train.model_zoo.native.minimax_h3.video_vae import load_minimax_h3_video_vae, resolve_video_vae_dir
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .minimax_h3_vae import MiniMaxH3VAEModel


def teacher_config_fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@MODEL_REGISTER("minimax_h3_pruned_vae")
class MiniMaxH3PrunedVAEModel(MiniMaxH3VAEModel):
    def load_components(self, *, load_transformer, load_vae, load_condition_encoder):
        del load_condition_encoder
        config = self.config["model"]
        self.pretrained_model_path = config["pretrained_model_name_or_path"]
        self.student_param_dtype = get_running_dtype(config.get("student_param_dtype", "fp32"))
        self.teacher_encoder_dtype = get_running_dtype(config.get("teacher_encoder_dtype", "fp32"))
        self.teacher_autocast_dtype = get_running_dtype(config.get("teacher_autocast_dtype", "fp16"))
        self.student_autocast = config.get("student_autocast", True)
        distillation = self.config["training"].get("vae_distillation", {})
        self.feature_distillation_weight = self._maximum_distillation_weight(distillation, "feature")
        self.auxiliary_distillation_weight = self._maximum_distillation_weight(distillation, "auxiliary")
        intermediate_supervision = self.feature_distillation_weight or self.auxiliary_distillation_weight
        self.teacher_vae = None
        self.transformer = None
        with (resolve_video_vae_dir(self.pretrained_model_path) / "config.json").open() as handle:
            self.teacher_config = json.load(handle)

        decoder_config = dict(config["pruned_decoder"])
        selection_path = decoder_config.pop("selection_path", None)
        self.search_config = decoder_config.get("search")
        if selection_path:
            if self.search_config is not None or "kept_layers" in decoder_config:
                raise ValueError("selection_path specifies the fixed decoder; do not also set search or kept_layers.")
            with Path(selection_path).open() as handle:
                selection = json.load(handle)
            if selection["teacher_config_sha256"] != teacher_config_fingerprint(self.teacher_config):
                raise ValueError("The pruning selection was exported for a different H3 VAE config.")
            decoder_config["kept_layers"] = selection["kept_layers"]
            if intermediate_supervision:
                anchors = selection["teacher_feature_indices"]
                configured = distillation.setdefault("teacher_feature_indices", anchors)
                if list(configured) != anchors:
                    raise ValueError("Teacher feature indices must match the exported pruning group endpoints.")
        if self.search_config is None and "kept_layers" not in decoder_config:
            raise ValueError("Recovery requires a pruning selection_path or explicit kept_layers.")
        if self.search_config is not None:
            if self.config["training"]["method"] != "vae_pruning":
                raise ValueError("A search decoder requires training.method=vae_pruning.")
            if self.feature_distillation_weight:
                raise ValueError("Layer search uses the reconstruction task, not intermediate feature KD.")
            if self.auxiliary_distillation_weight:
                raise ValueError("Auxiliary teacher-suffix decoding is a recovery loss, not a layer-search loss.")

        if load_transformer:
            self.transformer = MiniMaxH3PrunedVideoVAE(teacher_config=self.teacher_config, **decoder_config)

        # Load once on CPU, initialize the student, then discard unused frozen components.
        teacher = load_minimax_h3_video_vae(
            self.pretrained_model_path,
            torch_dtype=torch.float32,
            local_files_only=config.get("local_files_only", True),
        )
        if self.transformer is not None:
            self.transformer.initialize_from_teacher_state(teacher.state_dict())
            self.transformer.to(device=self.device, dtype=self.student_param_dtype)
        teacher_supervision = intermediate_supervision or self._maximum_distillation_weight(distillation, "teacher_output")
        keep_decoder = load_vae and config.get("load_teacher_decoder", bool(teacher_supervision))
        keep_encoder = load_vae and config.get("load_teacher_encoder", False)
        if teacher_supervision and not keep_decoder:
            raise ValueError("Teacher-supervised losses require load_teacher_decoder=true.")
        if not keep_decoder:
            teacher.decoder = None
        if not keep_encoder:
            teacher.encoder = None
            teacher.quant_conv = None
        if keep_decoder or keep_encoder:
            self.teacher_vae = teacher.requires_grad_(False).eval().to(self.device)

        if load_transformer and selection_path:
            exported_weights = Path(selection_path).parent / self.transformer.weights_name
            self.transformer.load_state_dict(load_file(str(exported_weights)), strict=True)
            logger.info("[model] loaded pruned initialization from {}", exported_weights)
        if load_transformer and config.get("initial_weights_path"):
            self._load_initial_weights(config["initial_weights_path"])
        if intermediate_supervision:
            anchors = distillation.get("teacher_feature_indices", (17, 35))
            if len(anchors) != len(self.transformer.decoder.transformer_blocks):
                raise ValueError("Recovery requires one teacher feature anchor per student block.")
            if not anchors or list(anchors) != sorted(set(anchors)) or anchors[0] < 0 or anchors[-1] >= self.teacher_config["decoder_num_layers"]:
                raise ValueError("Teacher feature anchor is outside the original decoder.")

    def student_decode_window_with_aux(self, normalized_latent_window, running_dtype, *, auxiliary_feature_index, return_features=False):
        context = self._autocast(running_dtype) if self.student_autocast else nullcontext()
        with context:
            return self.denoiser_module()(
                normalized_latent_window,
                return_features=return_features,
                auxiliary_feature_index=auxiliary_feature_index,
            )

    def teacher_decode_suffix(self, full_tokens, latent_shape, teacher_feature_index, *, gradient_checkpointing=True):
        with self._autocast(self.teacher_autocast_dtype):
            return decode_teacher_suffix(
                self.teacher_vae.decoder,
                full_tokens,
                latent_shape,
                teacher_feature_index,
                gradient_checkpointing=gradient_checkpointing,
            ).float()

    def set_full_trainable(self):
        student = self.denoiser_module()
        if self.search_config is not None:
            student.configure_search_trainable()
        else:
            student.requires_grad_(True)
        student.train()

    def consolidated_safetensors_metadata(self):
        return {
            "format": "pt",
            "model_type": "minimax_h3_pruned_vae",
            "architecture": json.dumps(self.denoiser_module().architecture_config, sort_keys=True, separators=(",", ":")),
        }

    def validate_consolidated_metadata(self, metadata, checkpoint_path):
        expected = self.consolidated_safetensors_metadata()
        if any(metadata.get(key) != expected[key] for key in ("model_type", "architecture")):
            raise ValueError(f"Checkpoint does not match the configured pruned H3 decoder: {checkpoint_path}")

    def fsdp2_shard_plan(self, fsdp_config):
        # H3 normalizes in FP32. Keep original parameter dtype; autocast handles the matmuls.
        param_dtype = fsdp_config.get("mixed_precision", {}).get("param_dtype")
        if param_dtype not in {None, "fp32", "float32"}:
            raise ValueError("H3 ViT VAE FSDP param_dtype must be null/fp32; use model.running_dtype for autocast.")
        return super().fsdp2_shard_plan(fsdp_config)

    @torch.no_grad()
    def export_pruned_decoder(self, output_dir, gate_logits):
        student = self.denoiser_module()
        kept = student.selected_layers(logits=gate_logits)
        group_size = self.search_config.get("group_size", 3)
        total_layers = self.teacher_config["decoder_num_layers"]
        export_config = {
            "kept_layers": kept,
            "use_tiling": True,
            "tile_sample_min_height": student.tile_sample_min_height,
            "tile_sample_min_width": student.tile_sample_min_width,
            "tile_sample_min_overlap_height": student.tile_sample_min_overlap_height,
            "tile_sample_min_overlap_width": student.tile_sample_min_overlap_width,
        }
        exported = MiniMaxH3PrunedVideoVAE(teacher_config=self.teacher_config, **export_config)
        teacher = load_minimax_h3_video_vae(self.pretrained_model_path, torch_dtype=torch.float32)
        exported.initialize_from_teacher_state(teacher.state_dict())
        del teacher
        output_dir = Path(output_dir)
        exported.save_pretrained(output_dir)
        selection = {
            "schema_version": 1,
            "teacher_config_sha256": teacher_config_fingerprint(self.teacher_config),
            "original_num_layers": total_layers,
            "group_size": group_size,
            "kept_layers": kept,
            "teacher_feature_indices": list(range(group_size - 1, total_layers, group_size)),
            "initialization": "original_teacher_weights_without_search_lora",
        }
        # Publish selection last: recovery must not see it before the weights are complete.
        temporary = output_dir / "kept_layers.json.tmp"
        temporary.write_text(json.dumps(selection, indent=2) + "\n")
        temporary.replace(output_dir / "kept_layers.json")
        logger.info("[prune] exported retained layers={} to {}", kept, output_dir)
