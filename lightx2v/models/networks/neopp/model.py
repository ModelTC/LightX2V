import torch
import torch.distributed as dist
import torch.nn.functional as F

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
        self.cfg_scale = self.config.get("cfg_scale", 4.0)
        self.cfg_norm = self.config.get("cfg_norm", "global")
        self.patch_size = self.config.get("patch_size", 16)
        self.merge_size = 2
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

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
        # logger.info(f"infer: cfg_scale={self.cfg_scale}")
        # logger.info(f"infer: cfg_interval={self.cfg_interval}")
        # logger.info(f"infer: cfg_norm={self.cfg_norm}")
        pre_infer_out = self.pre_infer.infer(self.pre_weight)

        # if self.config["task"] == "i2i":
        #     v_pred = self._infer_i2i(inputs, pre_infer_out)
        # else:
        #     v_pred = self._infer_t2i(inputs, pre_infer_out)

        v_pred = self._infer_t2i_i2i(inputs, pre_infer_out)

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

    def cfg_norm_func(self, v_pred, v_pred_condition):
        if self.cfg_norm == "global":
            logger.info(f"cfg_norm is global, applying global normalization")
            norm_v_condition = torch.norm(v_pred_condition, dim=(1, 2), keepdim=True)
            norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
            v_pred = v_pred * scale
        elif self.cfg_norm == "channel":
            logger.info(f"cfg_norm is channel, applying channel normalization")
            norm_v_condition = torch.norm(v_pred_condition, dim=-1, keepdim=True)
            norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
            scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
            v_pred = v_pred * scale
        elif self.cfg_norm == "none":
            logger.info(f"cfg_norm is none, no normalization will be applied")
        else:
            raise ValueError(f"Invalid cfg_norm: {self.cfg_norm}")
        return v_pred

    def _infer_t2i_i2i(self, inputs, pre_infer_out):
        t = self.scheduler.timesteps[self.scheduler.step_index]
        use_cfg = t >= self.cfg_interval[0] and t <= self.cfg_interval[1] and self.cfg_scale > 1

        # 预计算各 pass 的 image_embeds：seq_parallel 时切分为本 rank 的 shard，否则直接引用原张量
        # 这样 _infer_cond_uncond 无需在每次调用时反复 chunk/restore，避免多次 pass 间互相污染
        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            cur_rank = dist.get_rank(self.seq_p_group)
            image_embeds = pre_infer_out.image_embeds
            seq_len = image_embeds.shape[1]
            padding_size = (world_size - (seq_len % world_size)) % world_size
            if padding_size > 0:
                image_embeds = F.pad(image_embeds, (0, 0, 0, padding_size))
            shard = torch.chunk(image_embeds, world_size, dim=1)[cur_rank]
            pre_infer_out.image_embeds_cond = shard
            pre_infer_out.image_embeds_uncond = shard
        else:
            pre_infer_out.image_embeds_cond = pre_infer_out.image_embeds
            pre_infer_out.image_embeds_uncond = pre_infer_out.image_embeds

        if self.config.get("cfg_parallel", False):
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            # assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
            cfg_p_rank = dist.get_rank(cfg_p_group)

            cfg_p_world_size = dist.get_world_size(cfg_p_group)
            if use_cfg:
                if cfg_p_rank == 0:
                    v_pred = self._infer_cond_uncond(inputs, pre_infer_out, True)
                else:
                    v_pred = self._infer_cond_uncond(inputs, pre_infer_out, False)
                v_pred_list = [torch.zeros_like(v_pred) for _ in range(cfg_p_world_size)]
                dist.all_gather(v_pred_list, v_pred, group=cfg_p_group)
                v_pred_cond, v_pred_uncond = v_pred_list[0], v_pred_list[1]
                v_pred = v_pred_uncond + self.cfg_scale * (v_pred_cond - v_pred_uncond)
                v_pred = self.cfg_norm_func(v_pred, v_pred_cond)
                return v_pred
            else:
                # cfg 区间外只有 rank 0 做 cond 推理，其余 rank 用 all_gather 接收结果，无需持有 cond kvcache
                if cfg_p_rank == 0:
                    v_pred = self._infer_cond_uncond(inputs, pre_infer_out, True)
                else:
                    v_pred = torch.zeros_like(pre_infer_out.z)
                v_pred_list = [torch.zeros_like(v_pred) for _ in range(cfg_p_world_size)]
                dist.all_gather(v_pred_list, v_pred, group=cfg_p_group)
                return v_pred_list[0]
        else:
            v_pred_condition = self._infer_cond_uncond(inputs, pre_infer_out, True)
            if use_cfg:
                v_pred_uncond = self._infer_cond_uncond(inputs, pre_infer_out, False)
                v_pred = v_pred_uncond + self.cfg_scale * (v_pred_condition - v_pred_uncond)
                v_pred = self.cfg_norm_func(v_pred, v_pred_condition)
                return v_pred
            return v_pred_condition

    def _infer_cond_uncond(self, inputs, pre_infer_out, infer_condition: bool):
        self.scheduler.infer_condition = infer_condition
        pre_infer_out.image_embeds = pre_infer_out.image_embeds_cond if infer_condition else pre_infer_out.image_embeds_uncond

        hidden_states = self.transformer_infer.infer(self.transformer_weights, pre_infer_out, inputs)

        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            gathered_hidden_states = [torch.empty_like(hidden_states) for _ in range(world_size)]
            dist.all_gather(gathered_hidden_states, hidden_states, group=self.seq_p_group)
            hidden_states = torch.cat(gathered_hidden_states, dim=1)
            hidden_states = hidden_states[:, : pre_infer_out.image_token_num, :]

        v_pred = self.post_infer.infer(self.post_weight, pre_infer_out, hidden_states)
        return v_pred

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
