import torch
import torch.distributed as dist

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.neopp.infer.post_infer import NeoppPostInfer
from lightx2v.models.networks.neopp.infer.pre_infer import NeoppPreInfer
from lightx2v.models.networks.neopp.infer.transformer_infer import NeoppTransformerInfer
from lightx2v.models.networks.neopp.weights.post_weights import NeoppPostWeights
from lightx2v.models.networks.neopp.weights.pre_weights import NeoppPreWeights
from lightx2v.models.networks.neopp.weights.transformer_weights import NeoppTransformerWeights
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class NeoppModel(BaseTransformerModel):
    pre_weight_class = NeoppPreWeights
    transformer_weight_class = NeoppTransformerWeights
    post_weight_class = NeoppPostWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)
        self.preserved_keys = ["fm_modules", "mot_gen"]
        self._init_infer_class()
        self._init_infer()
        self._init_weights()
        self.cfg_interval = self.config.get("cfg_interval", (-1, 2))
        self.cfg_scale = self.config.get("cfg_scale", 7.0)
        self.img_cfg_scale = self.config.get("img_cfg_scale", 1.5)
        self.patch_size = self.config.get("patch_size", 16)
        self.merge_size = 2

    def _init_infer_class(self):
        self.pre_infer_class = NeoppPreInfer
        self.transformer_infer_class = NeoppTransformerInfer
        self.post_infer_class = NeoppPostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)

    @torch.no_grad()
    def infer(self, inputs):
        pre_infer_out = self.pre_infer.infer(self.pre_weight)

        if self.config["task"] == "i2i":
            v_pred = self._infer_i2i(inputs, pre_infer_out)
        else:
            v_pred = self._infer_t2i(inputs, pre_infer_out)

        t = self.scheduler.timesteps[self.scheduler.step_index]
        t_next = self.scheduler.timesteps[self.scheduler.step_index + 1]
        z = pre_infer_out.z + (t_next - t) * v_pred
        self.scheduler.image_prediction = self.unpatchify(
            z,
            self.patch_size * self.merge_size,
            self.scheduler.image_prediction.shape[-2],
            self.scheduler.image_prediction.shape[-1],
        )
        return z

    def _infer_t2i(self, inputs, pre_infer_out):
        t = self.scheduler.timesteps[self.scheduler.step_index]
        use_cfg = t > self.cfg_interval[0] and t < self.cfg_interval[1] and self.cfg_scale > 1

        if self.config.get("cfg_parallel", False):
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
            cfg_p_rank = dist.get_rank(cfg_p_group)

            if use_cfg:
                v_pred = self._infer_pass(inputs, pre_infer_out, "cond" if cfg_p_rank == 0 else "uncond")
                v_pred_list = [torch.zeros_like(v_pred) for _ in range(2)]
                dist.all_gather(v_pred_list, v_pred, group=cfg_p_group)
                v_pred_cond, v_pred_uncond = v_pred_list[0], v_pred_list[1]
                return v_pred_uncond + self.cfg_scale * (v_pred_cond - v_pred_uncond)
            else:
                return self._infer_pass(inputs, pre_infer_out, "cond")
        else:
            v_pred_condition = self._infer_pass(inputs, pre_infer_out, "cond")
            if use_cfg:
                v_pred_uncond = self._infer_pass(inputs, pre_infer_out, "uncond")
                return v_pred_uncond + self.cfg_scale * (v_pred_condition - v_pred_uncond)
            return v_pred_condition

    def _infer_i2i(self, inputs, pre_infer_out):
        t = self.scheduler.timesteps[self.scheduler.step_index]
        use_cfg = t > self.cfg_interval[0] and t < self.cfg_interval[1]

        if self.config.get("cfg_parallel", False):
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            assert dist.get_world_size(cfg_p_group) == 3, "cfg_p_world_size must be equal to 3 for i2i"
            cfg_p_rank = dist.get_rank(cfg_p_group)

            if use_cfg:
                if cfg_p_rank == 0:
                    v_pred = self._infer_pass(inputs, pre_infer_out, "cond")
                elif cfg_p_rank == 1:
                    v_pred = self._infer_pass(inputs, pre_infer_out, "text_uncond") if self.cfg_scale > 1 else torch.zeros_like(pre_infer_out.z)
                else:  # cfg_p_rank == 2
                    v_pred = self._infer_pass(inputs, pre_infer_out, "img_uncond") if self.img_cfg_scale > 1 else torch.zeros_like(pre_infer_out.z)
                v_pred_list = [torch.zeros_like(v_pred) for _ in range(3)]
                dist.all_gather(v_pred_list, v_pred, group=cfg_p_group)
                v_pred_condition = v_pred_list[0]
                v_pred_text_uncond = v_pred_list[1] if self.cfg_scale > 1 else 0
                v_pred_img_uncond = v_pred_list[2] if self.img_cfg_scale > 1 else 0
                v_pred_text = v_pred_text_uncond + self.cfg_scale * (v_pred_condition - v_pred_text_uncond)
                return v_pred_img_uncond + self.img_cfg_scale * (v_pred_text - v_pred_img_uncond)
            else:
                return self._infer_pass(inputs, pre_infer_out, "cond")
        else:
            v_pred_condition = self._infer_pass(inputs, pre_infer_out, "cond")
            if use_cfg:
                v_pred_text_uncond = self._infer_pass(inputs, pre_infer_out, "text_uncond") if self.cfg_scale > 1 else 0
                v_pred_img_uncond = self._infer_pass(inputs, pre_infer_out, "img_uncond") if self.img_cfg_scale > 1 else 0
                v_pred_text = v_pred_text_uncond + self.cfg_scale * (v_pred_condition - v_pred_text_uncond)
                return v_pred_img_uncond + self.img_cfg_scale * (v_pred_text - v_pred_img_uncond)
            return v_pred_condition

    def _infer_pass(self, inputs, pre_infer_out, pass_name):
        """Run one forward pass. pass_name: 'cond' | 'uncond' | 'text_uncond' | 'img_uncond'"""
        self.scheduler.infer_pass = pass_name
        hidden_states = self.transformer_infer.infer(self.transformer_weights, pre_infer_out, inputs)
        v_pred = self.post_infer.infer(self.post_weight, pre_infer_out, hidden_states)
        return v_pred

    def _infer_cond_uncond(self, inputs, pre_infer_out, infer_condition=True):
        pass

    def _seq_parallel_post_process(self, pre_infer_out):
        pass

    def _seq_parallel_pre_process(self, pre_infer_out):
        pass

    def unpatchify(sle, x, patch_size, h=None, w=None):
        """
        x: (N, L, patch_size**2 *3)
        images: (N, 3, H, W)
        """
        if h is None or w is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h = h // patch_size
            w = w // patch_size
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        images = x.reshape(shape=(x.shape[0], 3, h * patch_size, w * patch_size))
        return images
