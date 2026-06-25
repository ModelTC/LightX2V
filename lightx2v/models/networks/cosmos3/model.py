import torch
import torch.distributed as dist
from torch.nn import functional as F

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.cosmos3.infer.module_io import Cosmos3TransformerInferModuleOutput
from lightx2v.models.networks.cosmos3.infer.offload.transformer_infer import Cosmos3OffloadTransformerInfer
from lightx2v.models.networks.cosmos3.infer.post_infer import Cosmos3PostInfer
from lightx2v.models.networks.cosmos3.infer.pre_infer import Cosmos3PreInfer
from lightx2v.models.networks.cosmos3.infer.transformer_infer import Cosmos3TransformerInfer
from lightx2v.models.networks.cosmos3.weights.post_weights import Cosmos3PostWeights
from lightx2v.models.networks.cosmos3.weights.pre_weights import Cosmos3PreWeights
from lightx2v.models.networks.cosmos3.weights.transformer_weights import Cosmos3TransformerWeights
from lightx2v.utils.custom_compiler import compiled_method


class Cosmos3TransformerModel(BaseTransformerModel):
    pre_weight_class = Cosmos3PreWeights
    transformer_weight_class = Cosmos3TransformerWeights
    post_weight_class = Cosmos3PostWeights

    def __init__(self, model_path, config, device):
        if config.get("lazy_load", False):
            raise NotImplementedError("Cosmos3 LightX2V native transformer does not support lazy_load yet.")
        if config.get("cpu_offload", False) and config.get("offload_granularity", "block") != "block":
            raise NotImplementedError("Cosmos3 LightX2V native transformer supports only block-level cpu_offload.")
        super().__init__(model_path, config, device)
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        self.pre_infer_class = Cosmos3PreInfer
        self.transformer_infer_class = Cosmos3OffloadTransformerInfer if self.cpu_offload else Cosmos3TransformerInfer
        self.post_infer_class = Cosmos3PostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)

        und_len = pre_infer_out.und_len
        und_states = pre_infer_out.hidden_states[:und_len]
        gen_states = pre_infer_out.hidden_states[und_len:]
        und_position_ids = pre_infer_out.position_ids[:, :und_len]
        gen_position_ids = pre_infer_out.position_ids[:, und_len:]

        gen_len = gen_states.shape[0]
        padding_size = (world_size - (gen_len % world_size)) % world_size
        if padding_size > 0:
            gen_states = F.pad(gen_states, (0, 0, 0, padding_size))
            pad_position_ids = gen_position_ids[:, -1:].expand(-1, padding_size)
            gen_position_ids = torch.cat([gen_position_ids, pad_position_ids], dim=1)

        local_gen_states = torch.chunk(gen_states, world_size, dim=0)[cur_rank]
        local_gen_position_ids = torch.chunk(gen_position_ids, world_size, dim=1)[cur_rank]

        pre_infer_out.hidden_states = torch.cat([und_states, local_gen_states], dim=0)
        pre_infer_out.position_ids = torch.cat([und_position_ids, local_gen_position_ids], dim=1)
        pre_infer_out.seq_p_gen_len = gen_len
        pre_infer_out.seq_p_gen_padding_size = padding_size
        pre_infer_out.seq_p_local_gen_len = local_gen_states.shape[0]
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, transformer_out, pre_infer_out):
        world_size = dist.get_world_size(self.seq_p_group)
        local_gen_seq = transformer_out.gen_seq.contiguous()
        gathered_gen_seq = [torch.empty_like(local_gen_seq) for _ in range(world_size)]
        dist.all_gather(gathered_gen_seq, local_gen_seq, group=self.seq_p_group)
        gen_seq = torch.cat(gathered_gen_seq, dim=0)[: pre_infer_out.seq_p_gen_len]
        return Cosmos3TransformerInferModuleOutput(und_seq=transformer_out.und_seq, gen_seq=gen_seq)

    @torch.no_grad()
    def _infer_cond_uncond(self, input_ids):
        latents = self.scheduler.latents
        timestep = self.scheduler.timesteps[self.scheduler.step_index]
        pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            input_ids=input_ids,
            latents=latents,
            timestep=timestep,
        )
        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        transformer_out = self.transformer_infer.infer(
            block_weights=self.transformer_weights,
            pre_infer_out=pre_infer_out,
        )
        if self.config["seq_parallel"]:
            transformer_out = self._seq_parallel_post_process(transformer_out, pre_infer_out)

        return self.post_infer.infer(self.post_weight, transformer_out, pre_infer_out)

    @compiled_method()
    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()

        text_encoder_output = inputs["text_encoder_output"]
        do_cfg = self.config.get("enable_cfg", True) and self.scheduler.sample_guide_scale != 1.0
        if do_cfg and self.config.get("cfg_parallel", False):
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
            cfg_p_rank = dist.get_rank(cfg_p_group)
            input_ids = text_encoder_output["cond_input_ids"] if cfg_p_rank == 0 else text_encoder_output["uncond_input_ids"]
            noise_pred = self._infer_cond_uncond(input_ids)
            noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
            dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
            cond, uncond = noise_pred_list[0], noise_pred_list[1]
            self.scheduler.noise_pred = uncond + self.scheduler.sample_guide_scale * (cond - uncond)
        elif do_cfg:
            cond = self._infer_cond_uncond(text_encoder_output["cond_input_ids"])
            uncond = self._infer_cond_uncond(text_encoder_output["uncond_input_ids"])
            self.scheduler.noise_pred = uncond + self.scheduler.sample_guide_scale * (cond - uncond)
        else:
            cond = self._infer_cond_uncond(text_encoder_output["cond_input_ids"])
            self.scheduler.noise_pred = cond

        if self.cpu_offload:
            self.pre_weight.to_cpu()
            self.post_weight.to_cpu()
