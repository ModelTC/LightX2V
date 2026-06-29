import torch


class Cosmos3PostInfer:
    def __init__(self, config):
        self.config = config
        self.latent_patch_size = config["latent_patch_size"]
        self.latent_channel = config["latent_channel"]

    def _unpatchify_and_unpack_latents(
        self,
        packed_mse_preds,
        token_shapes_vision,
        noisy_frame_indexes_vision,
        original_latent_shapes,
    ):
        p = self.latent_patch_size
        unpatchified_latents = []
        start_idx = 0
        for token_shape, noisy_frame_indexes, original_shape in zip(
            token_shapes_vision,
            noisy_frame_indexes_vision,
            original_latent_shapes,
        ):
            t_c = token_shape[0]
            _, h_orig, w_orig = original_shape
            h_padded = ((h_orig + p - 1) // p) * p
            w_padded = ((w_orig + p - 1) // p) * p
            h_patches = h_padded // p
            w_patches = w_padded // p
            t_n = len(noisy_frame_indexes)
            output_tensor = torch.zeros(
                (self.latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent_patches = latent_patches.reshape(t_n, h_patches, w_patches, p, p, self.latent_channel)
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)
                latent = latent.reshape(self.latent_channel, t_n, h_patches * p, w_patches * p)
                output_tensor[:, noisy_frame_indexes] = latent[:, :, :h_orig, :w_orig]
                start_idx = end_idx
            unpatchified_latents.append(output_tensor.unsqueeze(0))
        return unpatchified_latents

    def infer(self, weights, transformer_out, pre_infer_out):
        last_hidden_state = torch.cat(
            [
                weights.norm.apply(transformer_out.und_seq),
                weights.norm_moe_gen.apply(transformer_out.gen_seq),
            ],
            dim=0,
        )
        preds_vision_packed = weights.proj_out.apply(last_hidden_state[pre_infer_out.vision_mse_loss_indexes])
        preds_vision = self._unpatchify_and_unpack_latents(
            preds_vision_packed,
            token_shapes_vision=pre_infer_out.vision_token_shapes,
            noisy_frame_indexes_vision=pre_infer_out.vision_noisy_frame_indexes,
            original_latent_shapes=pre_infer_out.original_latent_shapes,
        )
        return preds_vision[0]
