import torch
from einops import rearrange, repeat
from lightx2v.models.schedulers.mgcdr.scheduler import MagicDriverScheduler
from lightx2v.models.networks.mgcdr.weights.post_weights import MagicDrivePostWeights


class MagicDrivePostInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = self.config.get('patch_size', (1,2,2))
        self.out_channels = self.config.get('in_channels', 16)
        
    def set_scheduler(self, scheduler: MagicDriverScheduler):
        self.scheduler = scheduler
        
    def t2i_modulate(self, x, shift, scale):
        return x * (1 + scale) + shift
    
    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x
        
    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x
        
    def infer(self, weights: MagicDrivePostWeights, x, t, x_mask, t0, S, NC, T, H, W, Tx, Hx, Wx):
        # import pdb; pdb.set_trace()
        t = repeat(t, "b d -> (b NC) d", NC=NC)
        t0 = repeat(t0, "b d -> (b NC) d", NC=NC)
        shift, scale = (weights.final_layer_scale_shift_table.tensor[None] + t[:, None]).chunk(2, dim=1)  
        x = self.t2i_modulate(weights.final_layer_norm.apply(x), shift, scale)
        shift_zero, scale_zero = (weights.final_layer_scale_shift_table.tensor[None] + t0[:, None]).chunk(2, dim=1)
        x_zero = self.t2i_modulate(weights.final_layer_norm.apply(x), shift_zero, scale_zero)
        x = self.t_mask_select(x_mask, x, x_zero, T, S)
        
        # x_shape = x.shape
        # x = x.reshape(-1, x_shape[-1])
        x = weights.final_layer_linear.apply(x)
        # x = x.reshape(x_shape[0], x_shape[1], -1)
        # import pdb; pdb.set_trace()
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)
        x = rearrange(x, "(B NC) C T ... -> B (C NC) T ...", NC=NC)
        
        return x
        