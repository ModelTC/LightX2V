import torch

from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer


class WanLingbotPreInfer(WanPreInfer):
    def _build_lingbot_conditional_dict(self, weights, inputs, x_tokens: torch.Tensor) -> dict:
        image_encoder_output = inputs.get("image_encoder_output") or {}
        dit_cond_dict = image_encoder_output.get("dit_cond_dict") or {}
        c2ws_plucker_emb = dit_cond_dict.get("c2ws_plucker_emb", None)
        if c2ws_plucker_emb is None:
            return {}
        if isinstance(c2ws_plucker_emb, (list, tuple)):
            if len(c2ws_plucker_emb) == 0:
                return {}
            c2ws_plucker_emb = c2ws_plucker_emb[0]
        if c2ws_plucker_emb.dim() == 4:
            c2ws_plucker_emb = c2ws_plucker_emb.unsqueeze(0)
        if c2ws_plucker_emb.dim() != 5:
            return {}
        if not hasattr(weights, "patch_embedding_wancamctrl"):
            return {}

        b, c, f, h, w = c2ws_plucker_emb.shape
        p0, p1, p2 = self.config["patch_size"]
        if f % p0 != 0 or h % p1 != 0 or w % p2 != 0:
            return {}

        cam = c2ws_plucker_emb.reshape(b, c, f // p0, p0, h // p1, p1, w // p2, p2)
        cam = cam.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().reshape(b, -1, c * p0 * p1 * p2)
        cam = cam.to(dtype=x_tokens.dtype, device=x_tokens.device)
        cam = weights.patch_embedding_wancamctrl.apply(cam.squeeze(0)).unsqueeze(0)
        cam_hidden = weights.c2ws_hidden_states_layer2.apply(torch.nn.functional.silu(weights.c2ws_hidden_states_layer1.apply(cam.squeeze(0))))
        cam = cam.squeeze(0) + cam_hidden

        seq_len = x_tokens.shape[0]
        cam_len = cam.shape[0]
        if cam_len < seq_len:
            cam = torch.nn.functional.pad(cam, (0, 0, 0, seq_len - cam_len))
        elif cam_len > seq_len:
            cam = cam[:seq_len]

        return {"c2ws_plucker_emb": cam}

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        out = super().infer(weights, inputs, kv_start=kv_start, kv_end=kv_end)
        out.conditional_dict = self._build_lingbot_conditional_dict(weights, inputs, out.x)
        return out
