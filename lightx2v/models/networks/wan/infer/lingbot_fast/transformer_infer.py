import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.autoregressive.causal_transformer_infer import WanCausalTransformerInfer


class WanLingbotFastTransformerInfer(WanCausalTransformerInfer):
    def prepare_cross_attn_input(self, block, x, conditional_dict):
        if conditional_dict and "c2ws_plucker_emb" in conditional_dict and block is not None:
            cam = conditional_dict["c2ws_plucker_emb"]
            if cam.dim() == 3:
                cam = cam.squeeze(0)
            if cam.shape[0] < x.shape[0]:
                cam = F.pad(cam, (0, 0, 0, x.shape[0] - cam.shape[0]))
            elif cam.shape[0] > x.shape[0]:
                cam = cam[: x.shape[0]]
            cam = cam.to(dtype=x.dtype, device=x.device)
            cam_hidden = block.cam_injector_layer2.apply(F.silu(block.cam_injector_layer1.apply(cam))) + cam
            x = (1.0 + block.cam_scale_layer.apply(cam_hidden)) * x + block.cam_shift_layer.apply(cam_hidden)

        return x
