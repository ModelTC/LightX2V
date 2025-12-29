import torch.nn.functional as F


class ZImagePostInfer:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("cpu_offload", False)
        self.zero_cond_t = config.get("zero_cond_t", False)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, temb_img_silu, image_tokens_len=None):
        temb_silu = F.silu(temb_img_silu)
        temb1 = weights.norm_out_linear.apply(temb_silu)

        scale = 1.0 + temb1
        normed = weights.norm_out.apply(hidden_states)
        scaled_norm = normed * scale.unsqueeze(1)
        B, T, D = scaled_norm.shape
        hidden_states_2d = scaled_norm.reshape(B * T, D)

        output_2d = weights.proj_out_linear.apply(hidden_states_2d)
        out_dim = output_2d.shape[-1]
        output = output_2d.reshape(B, T, out_dim)

        if image_tokens_len is not None:
            output = output[:, :image_tokens_len, :]

        patch_size = self.config.get("patch_size", 2)
        f_patch_size = 1
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
        if output.shape[1] != expected_T:
            raise ValueError(f"Token count mismatch: output.shape[1]={output.shape[1]} != expected_T={expected_T} (from target_shape={target_shape})")

        output_reshaped = output.view(B, F_tokens, H_tokens, W_tokens, pF, pH, pW, out_channels)
        output_permuted = output_reshaped.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output_4d = output_permuted.reshape(B, out_channels, num_frames, height, width)
        output_4d = output_4d.squeeze(2)

        return output_4d
