import torch

from .module_io import MotusPreInferModuleOutput


class MotusPreInfer:
    def __init__(self, adapter, config):
        self.adapter = adapter
        self.config = config
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, image_path: str, prompt: str, state_value, seed: int | None = None):
        if self.scheduler is None:
            raise RuntimeError("MotusPreInfer requires a scheduler before infer().")

        first_frame = self.adapter.prepare_frame(image_path)
        state = self.adapter.prepare_state(state_value)
        instruction = self.adapter.build_instruction(prompt)
        t5_embeddings = self.adapter.build_t5_embeddings(instruction)
        vlm_inputs = [self.adapter.build_vlm_inputs(instruction, first_frame)]
        condition_frame_latent = self.adapter.encode_condition_frame(first_frame)
        processed_t5_context = self.adapter.model.video_module.preprocess_t5_embeddings(t5_embeddings)
        und_tokens = self.adapter.model.und_module.extract_und_features(vlm_inputs)
        image_context = self.adapter.model.und_module.extract_image_context(vlm_inputs)

        self.scheduler.prepare(
            seed=seed,
            condition_frame_latent=condition_frame_latent,
            action_shape=(state.shape[0], self.adapter.model.config.action_chunk_size, self.adapter.model.config.action_dim),
            dtype=self.adapter.model.dtype,
            device=self.adapter.device,
        )

        return MotusPreInferModuleOutput(
            first_frame=first_frame,
            state=state,
            instruction=instruction,
            t5_embeddings=t5_embeddings,
            vlm_inputs=vlm_inputs,
            processed_t5_context=processed_t5_context,
            image_context=image_context,
            und_tokens=und_tokens,
            condition_frame_latent=condition_frame_latent,
            grid_sizes=self.adapter.model.grid_sizes[: state.shape[0]],
        )
