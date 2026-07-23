import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.wan.infer.feature_caching.transformer_infer import (
    WanTransformerInferAdaCaching,
    WanTransformerInferCustomCaching,
    WanTransformerInferDualBlock,
    WanTransformerInferDynamicBlock,
    WanTransformerInferFirstBlock,
    WanTransformerInferMagCaching,
    WanTransformerInferTaylorCaching,
    WanTransformerInferTeaCaching,
)
from lightx2v.models.networks.wan.infer.offload.transformer_infer import (
    WanOffloadTransformerInfer,
)
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *
from lightx2v_platform.base.global_var import AI_DEVICE


class WanModel(BaseTransformerModel):
    pre_weight_class = WanPreWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device, model_type="wan2.1", lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)
        if self.lazy_load:
            self.remove_keys.extend(["blocks."])
        self.sensitive_layer = {
            "norm",
            "embedding",
            "modulation",
            "time",
            "img_emb.proj.0",
            "img_emb.proj.4",
            "before_proj",  # vace
            "after_proj",  # vace
        }
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    # ------------------------------------------------------------------ TP --
    def _rank_device(self):
        if dist.is_initialized():
            return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
        return torch.device(AI_DEVICE)

    def _get_split_type(self, key):
        if not key.endswith(".weight"):
            return None
        col_infixes = (
            ".self_attn.q.",
            ".self_attn.k.",
            ".self_attn.v.",
            ".self_attn.norm_q.",
            ".self_attn.norm_k.",
            ".cross_attn.q.",
            ".cross_attn.k.",
            ".cross_attn.v.",
            ".cross_attn.norm_q.",
            ".cross_attn.norm_k.",
            ".cross_attn.k_img.",
            ".cross_attn.v_img.",
        )
        row_infixes = (".self_attn.o.", ".cross_attn.o.", ".ffn.2.")
        if any(s in key for s in col_infixes):
            return "col"
        if any(s in key for s in row_infixes):
            return "row"
        if ".ffn.0." in key:
            return "col"
        return None

    def _split_bias_for_tp(self, bias, split_type, tp_size):
        if split_type == "col":
            return list(torch.chunk(bias, tp_size, dim=0))
        raise ValueError(f"Unsupported bias split_type: {split_type}")

    def _split_weight_for_tp(self, key, weight, tp_size):
        split_type = self._get_split_type(key)
        if split_type == "col":
            return list(torch.chunk(weight, tp_size, dim=0))
        if split_type == "row":
            return list(torch.chunk(weight, tp_size, dim=1))
        raise ValueError(f"Unknown split_type for {key}")

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        if not self.use_tp:
            return super()._load_weights_from_rank0(weight_dict, is_weight_loader)

        src_rank = 0
        target_device = self._rank_device()

        if is_weight_loader:
            processed, meta, processed_bias = {}, {}, set()
            for key, tensor in weight_dict.items():
                split_type = self._get_split_type(key)
                if key.endswith(".weight") and split_type is not None:
                    shards = self._split_weight_for_tp(key, tensor, self.tp_size)
                    for r, shard in enumerate(shards):
                        processed[f"{key}__tp_{r}"] = shard.contiguous()
                    meta[key] = {"shape": shards[0].shape, "dtype": shards[0].dtype, "is_tp": True}
                    bias_key = key.replace(".weight", ".bias")
                    if bias_key in weight_dict and split_type == "col":
                        bias_shards = self._split_bias_for_tp(weight_dict[bias_key], split_type, self.tp_size)
                        for r, shard in enumerate(bias_shards):
                            processed[f"{bias_key}__tp_{r}"] = shard.contiguous()
                        meta[bias_key] = {"shape": bias_shards[0].shape, "dtype": bias_shards[0].dtype, "is_tp": True}
                        processed_bias.add(bias_key)
                elif key not in processed_bias:
                    processed[key] = tensor
                    meta[key] = {"shape": tensor.shape, "dtype": tensor.dtype, "is_tp": False}
            obj_list = [meta]
        else:
            obj_list = [None]

        dist.broadcast_object_list(obj_list, src=src_rank)
        synced_meta = obj_list[0]

        distributed = {k: torch.empty(m["shape"], dtype=m["dtype"], device=target_device) for k, m in synced_meta.items()}

        for key in sorted(synced_meta.keys()):
            m = synced_meta[key]
            if m["is_tp"]:
                for r in range(self.tp_size):
                    buf = processed[f"{key}__tp_{r}"].to(target_device) if is_weight_loader else torch.empty(m["shape"], dtype=m["dtype"], device=target_device)
                    dist.broadcast(buf, src=src_rank, group=self.tp_group)
                    if r == self.tp_rank:
                        distributed[key].copy_(buf)
                    del buf
            else:
                if is_weight_loader:
                    distributed[key].copy_(processed[key].to(target_device))
                dist.broadcast(distributed[key], src=src_rank, group=self.tp_group)

        return distributed

    # ------------------------------------------------------------------ TP --

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer

        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WanTransformerInfer if not self.cpu_offload else WanOffloadTransformerInfer
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = WanTransformerInferTeaCaching
        elif self.config["feature_caching"] == "TaylorSeer":
            self.transformer_infer_class = WanTransformerInferTaylorCaching
        elif self.config["feature_caching"] == "Ada":
            self.transformer_infer_class = WanTransformerInferAdaCaching
        elif self.config["feature_caching"] == "Custom":
            self.transformer_infer_class = WanTransformerInferCustomCaching
        elif self.config["feature_caching"] == "FirstBlock":
            self.transformer_infer_class = WanTransformerInferFirstBlock
        elif self.config["feature_caching"] == "DualBlock":
            self.transformer_infer_class = WanTransformerInferDualBlock
        elif self.config["feature_caching"] == "DynamicBlock":
            self.transformer_infer_class = WanTransformerInferDynamicBlock
        elif self.config["feature_caching"] == "Mag":
            self.transformer_infer_class = WanTransformerInferMagCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        if hasattr(self.pre_infer, "set_rope"):
            first_attn = self.transformer_weights.blocks[0].compute_phases[0]
            rope = getattr(first_attn, "dreamzero_rope", None)
            self.pre_infer.set_rope(rope if rope is not None else first_attn.rope)
        if hasattr(self.pre_infer, "set_audio_rope"):
            self.pre_infer.set_audio_rope(self.transformer_weights.blocks[0].compute_phases[2].rope_1d)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    def _should_init_empty_model(self):
        if self.config.get("lora_configs") and self.config["lora_configs"] and not self.config.get("lora_dynamic_apply", False):
            if self.model_type in ["wan2.1"]:
                return True
            if self.model_type in ["wan2.2_moe_high_noise"]:
                for lora_config in self.config["lora_configs"]:
                    if lora_config["name"] == "high_noise_model":
                        return True
            if self.model_type in ["wan2.2_moe_low_noise"]:
                for lora_config in self.config["lora_configs"]:
                    if lora_config["name"] == "low_noise_model":
                        return True
        return False

    @compiled_method()
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

        if self.clean_cuda_cache:
            del x, pre_infer_out
            torch.cuda.empty_cache()

        return noise_pred

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        x = pre_infer_out.x
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        padding_size = (world_size - (x.shape[0] % world_size)) % world_size
        if padding_size > 0:
            x = F.pad(x, (0, 0, 0, padding_size))

        pre_infer_out.x = torch.chunk(x, world_size, dim=0)[cur_rank]

        if self.config["model_cls"] in ["wan2.2", "wan2.2_audio"] and self.config["task"] in ["i2v", "s2v", "rs2v"]:
            embed, embed0 = pre_infer_out.embed, pre_infer_out.embed0

            padding_size = (world_size - (embed.shape[0] % world_size)) % world_size
            if padding_size > 0:
                embed = F.pad(embed, (0, 0, 0, padding_size))
                embed0 = F.pad(embed0, (0, 0, 0, 0, 0, padding_size))

            pre_infer_out.embed = torch.chunk(embed, world_size, dim=0)[cur_rank]
            pre_infer_out.embed0 = torch.chunk(embed0, world_size, dim=0)[cur_rank]

        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        world_size = dist.get_world_size(self.seq_p_group)
        gathered_x = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_x, x, group=self.seq_p_group)
        combined_output = torch.cat(gathered_x, dim=0)
        return combined_output

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
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
                else:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=False)

                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
                noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)

            noise_pred_guided = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
            self.scheduler.noise_pred_cond = noise_pred_cond
            self.scheduler.noise_pred_uncond = noise_pred_uncond
            self.scheduler.noise_pred_guided = noise_pred_guided
            self.scheduler.noise_pred = noise_pred_guided
        else:
            # ==================== No CFG ====================
            noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
            self.scheduler.noise_pred_cond = noise_pred
            self.scheduler.noise_pred_uncond = None
            self.scheduler.noise_pred_guided = noise_pred
            self.scheduler.noise_pred = noise_pred

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()
