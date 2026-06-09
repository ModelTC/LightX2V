import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.networks.wan.infer.infinitetalk.pre_infer import WanInfiniteTalkPreInfer
from lightx2v.models.networks.wan.infer.infinitetalk.transformer_infer import WanInfiniteTalkTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.infinitetalk.pre_weights import WanInfiniteTalkPreWeights
from lightx2v.models.networks.wan.weights.infinitetalk.transformer_weights import WanInfiniteTalkTransformerWeights
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


class WanInfiniteTalkModel(WanModel):
    pre_weight_class = WanInfiniteTalkPreWeights
    transformer_weight_class = WanInfiniteTalkTransformerWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type="infinitetalk", lora_path=lora_path, lora_strength=lora_strength)

    def _init_infer_class(self):
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("InfiniteTalk parity path requires feature_caching=NoCaching.")
        offload_granularity = self.config.get("offload_granularity", "block")
        if self.config.get("cpu_offload", False) and offload_granularity not in {"block", "model"}:
            raise NotImplementedError(f"InfiniteTalk currently supports block/model offload, not {offload_granularity} offload.")
        self.pre_infer_class = WanInfiniteTalkPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = WanInfiniteTalkTransformerInfer

    def _load_adapter_ckpt(self):
        adapter_model_path = self.config.get("adapter_model_path", None)
        if adapter_model_path is None:
            raise ValueError("InfiniteTalk requires adapter_model_path to point to the single/multi adapter checkpoint.")
        logger.info(f"Loading InfiniteTalk adapter weights from {adapter_model_path}")
        sensitive_layer = set(self.sensitive_layer)
        sensitive_layer.update({"audio_proj.norm", "norm_x"})
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        return self._load_safetensor_to_dict(adapter_model_path, unified_dtype, sensitive_layer)

    @torch.no_grad()
    def _infer_infinitetalk_branch(self, inputs, infer_condition=True, use_audio=True):
        branch_inputs = dict(inputs)
        self.scheduler.infer_condition = infer_condition
        if not use_audio:
            branch_inputs["audio_encoder_output"] = torch.zeros_like(inputs["audio_encoder_output"])[-1:]
        return self._infer_cond_uncond(branch_inputs, infer_condition=infer_condition)

    def _infinitetalk_cfg_branches(self):
        if self.config.get("enable_text_cfg", True):
            return [
                ("cond", True, True),
                ("drop_text", False, True),
                ("uncond", False, False),
            ]
        return [
            ("cond", True, True),
            ("drop_audio", True, False),
        ]

    def _run_infinitetalk_cfg_serial(self, inputs, branches):
        return {name: self._infer_infinitetalk_branch(inputs, infer_condition=infer_condition, use_audio=use_audio) for name, infer_condition, use_audio in branches}

    def _run_infinitetalk_cfg_parallel(self, inputs, branches):
        cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
        cfg_p_world_size = dist.get_world_size(cfg_p_group)
        cfg_p_rank = dist.get_rank(cfg_p_group)
        max_local_branches = (len(branches) + cfg_p_world_size - 1) // cfg_p_world_size

        local_outputs = []
        template = None
        for slot_idx in range(max_local_branches):
            branch_idx = slot_idx * cfg_p_world_size + cfg_p_rank
            if branch_idx < len(branches):
                _, infer_condition, use_audio = branches[branch_idx]
                noise_pred = self._infer_infinitetalk_branch(inputs, infer_condition=infer_condition, use_audio=use_audio)
                template = noise_pred
            else:
                noise_pred = None
            local_outputs.append(noise_pred)

        if template is None:
            template = torch.zeros_like(self.scheduler.latents, dtype=torch.float32)

        local_stack = torch.stack(
            [noise_pred if noise_pred is not None else torch.zeros_like(template) for noise_pred in local_outputs],
            dim=0,
        )
        gathered_stacks = [torch.zeros_like(local_stack) for _ in range(cfg_p_world_size)]
        dist.all_gather(gathered_stacks, local_stack, group=cfg_p_group)

        outputs = {}
        for branch_idx, (name, _, _) in enumerate(branches):
            branch_rank = branch_idx % cfg_p_world_size
            branch_slot = branch_idx // cfg_p_world_size
            outputs[name] = gathered_stacks[branch_rank][branch_slot]
        return outputs

    @torch.no_grad()
    def infer(self, inputs):
        if self.config.get("use_apg", False):
            raise NotImplementedError("InfiniteTalk APG is not implemented in the LightX2V parity path yet.")

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        if not self.config["enable_cfg"]:
            noise_pred_cond = self._infer_infinitetalk_branch(inputs, infer_condition=True, use_audio=True)
            noise_pred_guided = noise_pred_cond
            self.scheduler.noise_pred_uncond = None
            self.scheduler.noise_pred_drop_text = None
        elif self.config.get("cfg_parallel", False):
            branches = self._infinitetalk_cfg_branches()
            branch_outputs = self._run_infinitetalk_cfg_parallel(inputs, branches)
            noise_pred_cond = branch_outputs["cond"]
            if self.config.get("enable_text_cfg", True):
                noise_pred_drop_text = branch_outputs["drop_text"]
                noise_pred_uncond = branch_outputs["uncond"]
                noise_pred_guided = (
                    noise_pred_uncond
                    + self.scheduler.sample_text_guide_scale * (noise_pred_cond - noise_pred_drop_text)
                    + self.scheduler.sample_audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
                )
                self.scheduler.noise_pred_uncond = noise_pred_uncond
                self.scheduler.noise_pred_drop_text = noise_pred_drop_text
            else:
                noise_pred_drop_audio = branch_outputs["drop_audio"]
                noise_pred_guided = noise_pred_drop_audio + self.scheduler.sample_audio_guide_scale * (noise_pred_cond - noise_pred_drop_audio)
                self.scheduler.noise_pred_uncond = noise_pred_drop_audio
                self.scheduler.noise_pred_drop_text = None
        else:
            branches = self._infinitetalk_cfg_branches()
            branch_outputs = self._run_infinitetalk_cfg_serial(inputs, branches)
            noise_pred_cond = branch_outputs["cond"]
            if self.config.get("enable_text_cfg", True):
                noise_pred_drop_text = branch_outputs["drop_text"]
                noise_pred_uncond = branch_outputs["uncond"]
                noise_pred_guided = (
                    noise_pred_uncond
                    + self.scheduler.sample_text_guide_scale * (noise_pred_cond - noise_pred_drop_text)
                    + self.scheduler.sample_audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
                )
                self.scheduler.noise_pred_uncond = noise_pred_uncond
                self.scheduler.noise_pred_drop_text = noise_pred_drop_text
            else:
                noise_pred_drop_audio = branch_outputs["drop_audio"]
                noise_pred_guided = noise_pred_drop_audio + self.scheduler.sample_audio_guide_scale * (noise_pred_cond - noise_pred_drop_audio)
                self.scheduler.noise_pred_uncond = noise_pred_drop_audio
                self.scheduler.noise_pred_drop_text = None

        self.scheduler.noise_pred_cond = noise_pred_cond
        self.scheduler.noise_pred_guided = noise_pred_guided
        self.scheduler.noise_pred = -noise_pred_guided

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        x = pre_infer_out.x
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        padding_size = (world_size - (x.shape[0] % world_size)) % world_size
        if padding_size > 0:
            x = F.pad(x, (0, 0, 0, padding_size))
        pre_infer_out.x = torch.chunk(x, world_size, dim=0)[cur_rank]
        return pre_infer_out
