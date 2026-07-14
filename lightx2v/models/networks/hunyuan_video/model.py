import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.hunyuan_video.infer.feature_caching.transformer_infer import HunyuanTransformerInferTeaCaching, HunyuanVideo15TransformerInferMagCaching
from lightx2v.models.networks.hunyuan_video.infer.offload.transformer_infer import HunyuanVideo15OffloadTransformerInfer
from lightx2v.models.networks.hunyuan_video.infer.post_infer import HunyuanVideo15PostInfer
from lightx2v.models.networks.hunyuan_video.infer.pre_infer import HunyuanVideo15PreInfer
from lightx2v.models.networks.hunyuan_video.infer.transformer_infer import HunyuanVideo15TransformerInfer
from lightx2v.models.networks.hunyuan_video.weights.post_weights import HunyuanVideo15PostWeights
from lightx2v.models.networks.hunyuan_video.weights.pre_weights import HunyuanVideo15PreWeights
from lightx2v.models.networks.hunyuan_video.weights.transformer_weights import HunyuanVideo15TransformerWeights
from lightx2v.utils.custom_compiler import compiled_method


class HunyuanVideo15Model(BaseTransformerModel):
    pre_weight_class = HunyuanVideo15PreWeights
    transformer_weight_class = HunyuanVideo15TransformerWeights
    post_weight_class = HunyuanVideo15PostWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)
        self.remove_keys.extend(["byt5_in", "vision_in"])
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        self.pre_infer_class = HunyuanVideo15PreInfer
        self.post_infer_class = HunyuanVideo15PostInfer
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = HunyuanVideo15TransformerInfer if not self.cpu_offload else HunyuanVideo15OffloadTransformerInfer
        elif self.config["feature_caching"] == "Mag":
            self.transformer_infer_class = HunyuanVideo15TransformerInferMagCaching
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = HunyuanTransformerInferTeaCaching
        else:
            raise NotImplementedError

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)

        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        x = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)

        if self.config["seq_parallel"]:
            x = self._seq_parallel_post_process(x)

        noise_pred = self.post_infer.infer(x, pre_infer_out)[0]

        return noise_pred

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        seqlen = pre_infer_out.img.shape[1]
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)

        padding_size = (world_size - (seqlen % world_size)) % world_size
        if padding_size > 0:
            pre_infer_out.img = F.pad(pre_infer_out.img, (0, 0, 0, padding_size))

        pre_infer_out.img = torch.chunk(pre_infer_out.img, world_size, dim=1)[cur_rank]
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        world_size = dist.get_world_size(self.seq_p_group)
        gathered_x = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_x, x, group=self.seq_p_group)
        combined_output = torch.cat(gathered_x, dim=1)
        return combined_output

    @compiled_method()
    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        if self.config["enable_cfg"]:
            if self.config["cfg_parallel"]:
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=True).contiguous()
                else:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=False).contiguous()

                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
                noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)

            self.scheduler.noise_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            # ==================== No CFG ====================
            self.scheduler.noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
