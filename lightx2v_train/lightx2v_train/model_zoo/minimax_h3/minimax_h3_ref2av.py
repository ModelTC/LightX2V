from math import prod

import torch

from lightx2v_train.utils.registry import MODEL_REGISTER

from .minimax_h3_fl2av import MiniMaxH3FL2AVModel


@MODEL_REGISTER("minimax_h3_ref2av")
class MiniMaxH3Ref2AVModel(MiniMaxH3FL2AVModel):
    transformer_component = "transformer_ref"

    def prepare_text_condition(self, condition):
        prepared = self._prepare_text_rows(condition)
        if not bool((prepared["text_token_tags"] == 0).any()):
            raise ValueError("Ref2AV requires image tokens in the multimodal prompt cache.")
        references = condition["references"]
        if not references:
            raise ValueError("Ref2AV requires at least one reference image.")
        _, patch_h, patch_w = self.patch_size
        video_dim = self.video_latent_channels * prod(self.patch_size)
        sizes, rows = [], []
        for reference in references:
            if reference["kind"] != "image" or reference["num_latent_frames"] != 1 or not reference["normalized"]:
                raise ValueError("Ref2AV training requires normalized single-frame image references.")
            height, width = reference["latent_height"], reference["latent_width"]
            if height <= 0 or width <= 0 or height % patch_h or width % patch_w:
                raise ValueError("Ref2AV image geometry must be divisible by the latent patch size.")
            value = reference["video_latents"]
            expected = (height // patch_h * (width // patch_w), video_dim)
            if value.shape != expected:
                raise ValueError(f"Ref2AV expected image rows {expected}, got {tuple(value.shape)}.")
            sizes.append((height, width))
            rows.append(value)

        return {
            **prepared,
            "reference_sizes": tuple(sizes),
            "condition_video_latents": torch.cat(rows).to(self.device, dtype=torch.float32),
            "target_height": condition["target_height"],
            "target_width": condition["target_width"],
            "target_num_frames": condition["target_num_frames"],
        }
