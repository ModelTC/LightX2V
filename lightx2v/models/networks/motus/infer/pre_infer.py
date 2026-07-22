import torch

from lightx2v.models.networks.wan.infer.module_io import GridOutput
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer

from .module_io import MotusPreInferModuleOutput


class MotusPreInfer(WanPreInfer):
    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        self.scheduler = None
        self._grid_cache = {}

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        del weights, kv_start, kv_end
        if self.scheduler is None:
            raise RuntimeError("MotusPreInfer requires a scheduler before infer().")

        first_frame = inputs["motus_first_frame"]
        state = inputs["motus_state"]
        instruction = inputs["motus_instruction"]
        t5_context = inputs["motus_t5_embeddings"]
        processed_t5_context = inputs["motus_processed_t5_context"]
        vlm_inputs = inputs["motus_vlm_inputs"]
        image_context = inputs["motus_image_context"]
        und_tokens = inputs["motus_und_tokens"]

        video_latents = self.scheduler.video_latents
        if video_latents.dim() != 5:
            raise RuntimeError(f"Expected video latents with shape [B, C, T, H, W], got {tuple(video_latents.shape)}")
        batch_size = state.shape[0]
        _, _, latent_t, latent_h, latent_w = video_latents.shape
        grid_tuple = (
            latent_t,
            latent_h // self.model.video_backbone.patch_size[1],
            latent_w // self.model.video_backbone.patch_size[2],
        )
        grid_key = (batch_size, grid_tuple, state.device.type, state.device.index)
        grid_output = self._grid_cache.get(grid_key)
        if grid_output is None:
            grid_sizes = torch.tensor([grid_tuple], dtype=torch.long, device=state.device).expand(batch_size, -1)
            grid_output = GridOutput(tensor=grid_sizes, tuple=grid_tuple)
            self._grid_cache[grid_key] = grid_output

        if self.cos_sin is None or self.grid_sizes != grid_output.tuple:
            self.grid_sizes = grid_output.tuple
            self.cos_sin = self.prepare_rope_cache(self.prepare_cos_sin(grid_output.tuple, self.freqs.clone()))

        dummy_embed = torch.empty(0, device=state.device, dtype=processed_t5_context.dtype)

        return MotusPreInferModuleOutput(
            embed=dummy_embed,
            grid_sizes=grid_output,
            x=self.scheduler.video_latents,
            embed0=dummy_embed,
            context=processed_t5_context,
            cos_sin=self.cos_sin,
            rope_positions=self.rope_positions,
            first_frame=first_frame,
            state=state,
            instruction=instruction,
            t5_embeddings=t5_context,
            vlm_inputs=vlm_inputs,
            image_context=image_context,
            und_tokens=und_tokens,
            condition_frame_latent=self.scheduler.condition_frame_latent,
            adapter_args={"instruction": instruction},
        )
