import torch
import torch.nn as nn
from einops import rearrange
from lightx2v.models.networks.wan.model import WanModel as LightX2VWanModel
from lightx2v.models.networks.mova.transformer_infer import MOVATransformerInfer

def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    pos = torch.arange(end, device=freqs.device)
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


class MOVACompatVideoDiT(LightX2VWanModel):
    @property
    def blocks(self):
        if not hasattr(self, 'transformer_weights'):
            return []
        return self.transformer_weights.blocks
    @property
    def head(self):
        if not hasattr(self, 'transformer_weights'):
            return None
        return self.transformer_weights.head
    @property
    def infer_engine(self):
        if not hasattr(self, 'transformer_infer'):
            return None
        return self.transformer_infer
    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = MOVATransformerInfer(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()
    transformer_infer_class = MOVATransformerInfer    
    def __init__(self, model_path, config, device, mova_state_dict, **kwargs):
        self.mova_state_dict = mova_state_dict
        config['has_image_input'] = False
        config['task'] = 't2v'
        config['lazy_load'] = False

        # 先手动创建权重容器
        self.pre_weight = self.pre_weight_class(config)
        self.transformer_weights = self.transformer_weight_class(config)
        if hasattr(self, "post_weight_class") and self.post_weight_class is not None:
            self.post_weight = self.post_weight_class(config)

        # 再调用父类初始化
        super().__init__(model_path, config, device, **kwargs)

        # 初始化 freqs
        head_dim = config['dim'] // config['num_heads']
        self.freqs = precompute_freqs_cis_3d(head_dim)
        self.freqs = tuple(f.to(device) for f in self.freqs)

    def _init_weights(self, weight_dict=None):
        # 完全跳过自动加载，等待手动调用 load_mova_weights
        pass

    def load_mova_weights(self, state_dict):
        remapped = {}
        for key, tensor in state_dict.items():
            if key == "img_emb.weight":
                remapped["img_emb.proj.0.weight"] = tensor
            elif key == "img_emb.bias":
                remapped["img_emb.proj.0.bias"] = tensor
            elif key.startswith("img_emb.proj."):
                remapped[key] = tensor
            else:
                remapped[key] = tensor

        self.pre_weight.load(remapped)
        self.transformer_weights.load(remapped)
        if hasattr(self, "post_weight"):
            self.post_weight.load(remapped)
        print("[MOVA] Video DiT weights manually loaded.")

    def patchify(self, x):
        # x: [B, C, T, H, W] 视频 latent
        x = x.to(self.pre_weight.patch_embedding.weight.dtype)
        x = self.pre_weight.patch_embedding.apply(x)  # [B, dim, t, h, w]
        grid_size = x.shape[2:]  # (t, h, w)
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        return x, grid_size

    def unpatchify(self, x, grid_size):
        t, h, w = grid_size
        out_dim = self.config['out_dim']  # 视频 latent 通道数，通常为16
        p_t, p_h, p_w = self.config['patch_size']  # 例如 (1,2,2)
        # x shape: [B, L, D], D = out_dim * p_t * p_h * p_w
        x = x.view(1, t, h, w, p_t, p_h, p_w, out_dim)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(1, out_dim, t * p_t, h * p_h, w * p_w)
        return x