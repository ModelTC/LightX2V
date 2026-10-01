import torch

from lightx2v.models.networks.wan.fastwam_model import FastWAMNativeModel
from lightx2v.models.networks.wan.infer.realtimewam import RealtimeWAMPreInfer, RealtimeWAMTransformerInfer
from lightx2v.models.networks.wan.weights.realtimewam import RealtimeWAMTransformerWeights
from lightx2v.models.networks.wan.weights.realtimewam.transformer_weights import condition_layers
from lightx2v.utils.envs import GET_DTYPE


class RealtimeWAM(FastWAMNativeModel):
    model_type = "realtimewam"
    transformer_weight_class = RealtimeWAMTransformerWeights

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = RealtimeWAMPreInfer
        self.transformer_infer_class = RealtimeWAMTransformerInfer

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        weights = super()._load_ckpt(unified_dtype, sensitive_layer)
        conditions = condition_layers(self.config)
        fusion = self.config.get("kv_fusion", False)
        keys = {key for key in weights if key.startswith("video_kv_fusion_logits.")}
        expected = {f"video_kv_fusion_logits.{i}" for i in range(len(conditions))} if fusion else set()
        if keys != expected:
            raise ValueError("kv_fusion/checkpoint mismatch: unexpected video KV fusion weights")
        for index in range(int(self.config["num_layers"])):
            prefix = f"mixtures.action.blocks.{index}"
            conditioned = index in conditions
            width = int(self.config["dim"]) if conditioned else 8 * (int(self.config["dim"]) // int(self.config["num_heads"]))
            if (f"{prefix}.cross_attn.q.weight" in weights) != conditioned or weights[f"{prefix}.self_attn.q.weight"].shape[0] != width:
                raise ValueError(f"kv_fusion/checkpoint mismatch at action layer {index}")
        for i, index in enumerate(conditions if fusion else ()):
            key = f"video_kv_fusion_logits.{i}"
            previous = conditions[i - 1] if i else -1
            if weights[key].shape != (index - previous,):
                raise ValueError(f"KV fusion logits do not match interval ending at layer {index}")
        return weights

    def _init_weights(self, weight_dict=None):
        super()._init_weights(weight_dict)
        self.transformer_weights.pack_projections()

    def prepare_video_noise(self, latent_shape, seed):
        frames = self.config.get("num_video_frames", 9)
        if type(frames) is not int or frames < 5 or frames % 4 != 1:
            raise ValueError("FasterWAM num_video_frames must be >= 5 and satisfy T % 4 == 1")
        shape = list(latent_shape)
        shape[2] = (frames - 1) // 4 + 1
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu").to(device=self.device, dtype=GET_DTYPE())

    @staticmethod
    def merge_video_latents(first_frame_latents, noise):
        video = noise.clone()
        video[:, :, :1] = first_frame_latents
        return video

    @torch.no_grad()
    def prepare_action_inputs(self, first_frame_latents, context, context_mask, action_chunk_size, robot_state=None, seed=None):
        if self.config.get("kv_fusion", False):
            if first_frame_latents.ndim == 4:
                first_frame_latents = first_frame_latents.unsqueeze(0)
            noise = self.prepare_video_noise(first_frame_latents.shape, seed)
            full_frame_latents = self.merge_video_latents(first_frame_latents, noise)
        else:
            full_frame_latents = first_frame_latents
        return super().prepare_action_inputs(full_frame_latents, context, context_mask, action_chunk_size, robot_state)
