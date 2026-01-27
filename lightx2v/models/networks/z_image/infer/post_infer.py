import torch.nn.functional as F


class ZImagePostInfer:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("cpu_offload", False)
        self.zero_cond_t = config.get("zero_cond_t", False)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, temb_img_silu, image_tokens_len=None):
        """
        Post-inference processing: apply norm_out, proj_out, and unpatchify.
        All processing is done without batch dimension: [T, D] instead of [B, T, D].

        Args:
            weights: PostInfer weights
            hidden_states: Hidden states [T, D] (no batch dimension)
            temb_img_silu: Time embedding [1, D]
            image_tokens_len: Image tokens length (optional)

        Returns:
            output_4d: Output tensor [C, H, W] (no batch dimension)
        """
        # hidden_states is already [T, D] from pre_infer (no batch dimension)
        # temb_img_silu is [1, D] from pre_infer

        # Apply norm_out_linear: [1, D] -> [1, D]
        temb_silu = F.silu(temb_img_silu)  # [1, D]
        temb1 = weights.norm_out_linear.apply(temb_silu)  # [1, D]

        # Apply modulation: scale = 1.0 + temb1
        scale = 1.0 + temb1  # [1, D]
        normed = weights.norm_out.apply(hidden_states)  # [T, D]
        scaled_norm = normed * scale  # [T, D] * [1, D] -> [T, D]

        # Apply proj_out_linear: [T, D] -> [T, out_dim]
        output = weights.proj_out_linear.apply(scaled_norm)  # [T, out_dim]

        # Trim to image_tokens_len if specified
        if image_tokens_len is not None:
            output = output[:image_tokens_len, :]  # [image_tokens_len, out_dim]

        # Get output dimension
        T, out_dim = output.shape

        # Validate output dimension
        patch_size = self.config.get("patch_size", 2)
        f_patch_size = self.config.get("f_patch_size", 1)
        transformer_out_channels = out_dim // (patch_size * patch_size * f_patch_size)
        expected_out_dim = patch_size * patch_size * f_patch_size * transformer_out_channels

        if out_dim != expected_out_dim:
            raise ValueError(f"out_dim mismatch: {out_dim} != {expected_out_dim} (transformer_out_channels={transformer_out_channels})")

        out_channels = transformer_out_channels
        target_shape = self.scheduler.input_info.target_shape

        _, _, height, width = target_shape
        num_frames = 1
        pH = pW = patch_size
        pF = f_patch_size
        F_tokens = num_frames // pF
        H_tokens = height // pH
        W_tokens = width // pW

        expected_T = F_tokens * H_tokens * W_tokens
        if T != expected_T:
            raise ValueError(f"Token count mismatch: T={T} != expected_T={expected_T} (from target_shape={target_shape})")

        # Unpatchify: [T, out_dim] -> [C, H, W]
        # Reshape: [T, out_dim] -> [F_tokens, H_tokens, W_tokens, pF, pH, pW, out_channels]
        output_reshaped = output.view(F_tokens, H_tokens, W_tokens, pF, pH, pW, out_channels)
        # Permute: [F_tokens, H_tokens, W_tokens, pF, pH, pW, out_channels]
        #       -> [out_channels, F_tokens, pF, H_tokens, pH, W_tokens, pW]
        output_permuted = output_reshaped.permute(6, 0, 3, 1, 4, 2, 5)
        # Reshape: [out_channels, F_tokens, pF, H_tokens, pH, W_tokens, pW]
        #       -> [out_channels, num_frames, height, width]
        output_4d = output_permuted.reshape(out_channels, num_frames, height, width)
        # Remove frame dimension: [out_channels, 1, height, width] -> [out_channels, height, width]
        output_4d = output_4d.squeeze(1)

        return output_4d
