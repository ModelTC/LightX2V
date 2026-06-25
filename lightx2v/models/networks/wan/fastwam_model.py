from pathlib import Path

import torch
from loguru import logger

from lightx2v.common.ops import *  # noqa: F403,F401
from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.wan.infer.fastwam import FastWAMPreInfer, FastWAMTransformerInfer
from lightx2v.models.networks.wan.weights.fastwam import FastWAMPreWeights, FastWAMTransformerWeights
from lightx2v.models.schedulers.wan.fastwam import FastWAMActionScheduler
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


class FastWAMNativeModel(BaseTransformerModel):
    pre_weight_class = FastWAMPreWeights
    transformer_weight_class = FastWAMTransformerWeights

    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device, model_type="fastwam")
        self.sensitive_layer = {
            "norm",
            "modulation",
            "time",
            "embedding",
        }
        self._init_infer_class()
        self._init_weights()
        self._init_infer()
        self.action_scheduler = FastWAMActionScheduler(self.config)
        self.set_scheduler(self.action_scheduler)

    def _init_infer_class(self):
        self.pre_infer_class = FastWAMPreInfer
        self.transformer_infer_class = FastWAMTransformerInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        adapter_path = self.config.get("adapter_model_path")
        if not adapter_path:
            raise ValueError("FastWAM requires `adapter_model_path`.")
        adapter_path = Path(str(adapter_path)).expanduser()
        if not adapter_path.is_absolute():
            raise ValueError(f"FastWAM requires an absolute adapter model path, got: {adapter_path}")
        adapter_path = adapter_path.resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(str(adapter_path))
        logger.info(f"Loading FastWAM native weights from {adapter_path}")
        payload = torch.load(str(adapter_path), map_location="cpu", weights_only=True)
        if "mot" not in payload:
            raise ValueError(f"FastWAM checkpoint must contain `mot`, got keys={list(payload.keys())}")
        state = dict(payload["mot"])
        if "proprio_encoder" in payload:
            state["proprio_encoder.weight"] = payload["proprio_encoder"]["weight"]
            state["proprio_encoder.bias"] = payload["proprio_encoder"]["bias"]

        weight_dict = {}
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.is_floating_point():
                dtype = GET_DTYPE() if unified_dtype or all(s not in key for s in sensitive_layer) else GET_SENSITIVE_DTYPE()
                weight_dict[key] = tensor.to(device=self.device, dtype=dtype)
            else:
                weight_dict[key] = tensor.to(device=self.device)
        return weight_dict

    def _infer_cond_uncond(self, inputs):
        action_pre = self.pre_infer.infer_action(
            self.pre_weight,
            self.scheduler.latents,
            self.scheduler.current_timestep,
            inputs["context"],
            inputs["context_mask"],
        )
        noise_pred = self.transformer_infer.action_with_video_cache(
            self.transformer_weights,
            action_pre,
            inputs["video_kv_cache"],
            inputs["video_seq_len"],
            inputs["attention_mask"],
        )
        return noise_pred.unsqueeze(0)

    def _seq_parallel_pre_process(self, pre_infer_out):
        raise NotImplementedError("FastWAM native port currently supports single-device serial inference only.")

    def _seq_parallel_post_process(self, x):
        raise NotImplementedError("FastWAM native port currently supports single-device serial inference only.")

    @torch.no_grad()
    def infer(self, inputs):
        self.scheduler.noise_pred = self._infer_cond_uncond(inputs)

    def _append_robot_state_to_context(self, context, context_mask, robot_state):
        if robot_state is None:
            return context, context_mask
        robot_state = torch.as_tensor(robot_state, device=context.device, dtype=context.dtype).reshape(1, -1)
        if robot_state.shape[1] != int(self.config["robot_state_dim"]):
            raise ValueError(f"FastWAM robot_state must have {self.config['robot_state_dim']} dims, got {robot_state.shape[1]}")
        robot_state_token = self.pre_weight.proprio_encoder.apply(robot_state)
        robot_state_mask = torch.ones((1,), dtype=torch.bool, device=context.device)
        return torch.cat([context, robot_state_token], dim=0), torch.cat([context_mask, robot_state_mask], dim=0)

    def _prepare_video_cache(self, first_frame_latents, context, context_mask):
        video_pre = self.pre_infer.infer_video(
            self.pre_weight,
            first_frame_latents,
            context,
            context_mask,
        )
        video_kv_cache = self.transformer_infer.prefill_video_cache(self.transformer_weights, video_pre)
        return video_pre, video_kv_cache

    @torch.no_grad()
    def prepare_action_inputs(
        self,
        first_frame_latents,
        context,
        context_mask,
        action_chunk_size,
        robot_state=None,
    ):
        if first_frame_latents.ndim == 4:
            first_frame_latents = first_frame_latents.unsqueeze(0)
        first_frame_latents = first_frame_latents.to(device=self.device, dtype=GET_DTYPE())
        context = context.to(device=self.device, dtype=GET_DTYPE())
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        context, context_mask = self._append_robot_state_to_context(context, context_mask, robot_state)

        video_pre, video_kv_cache = self._prepare_video_cache(first_frame_latents, context, context_mask)
        action_chunk_size = int(action_chunk_size)
        attention_mask = self.transformer_infer.build_mot_attention_mask(
            video_seq_len=video_pre.tokens.shape[0],
            action_seq_len=action_chunk_size,
            video_tokens_per_frame=video_pre.tokens_per_frame,
            device=first_frame_latents.device,
        )

        return {
            "context": context,
            "context_mask": context_mask,
            "video_kv_cache": video_kv_cache,
            "video_seq_len": video_pre.tokens.shape[0],
            "attention_mask": attention_mask,
        }, (1, action_chunk_size, int(self.config["action_dim"]))
