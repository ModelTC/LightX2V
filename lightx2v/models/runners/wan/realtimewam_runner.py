"""Teacher-distilled RealtimeWAM policy. No ROS or external WAM imports."""

import numpy as np
import torch

from lightx2v.models.networks.wan.realtimewam_model import RealtimeWAMNativeModel
from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy


class RealtimeWAMPolicy(FastWAMPolicy):
    model_class = RealtimeWAMNativeModel

    def encode_prompt(self, prompt):
        if not hasattr(self, "_prompt_cache"):
            self._prompt_cache = {}
        if prompt not in self._prompt_cache:
            self._prompt_cache[prompt] = super().encode_prompt(prompt)
        return self._prompt_cache[prompt]

    @torch.inference_mode()
    def predict_action_chunk(self, images, state, task_description, seed=None):
        seed = self.seed if seed is None else seed
        image = self.build_image_tensor(images)
        latents = self.encode_image_latents(image)
        context, mask = self.encode_prompt(self.default_prompt.format(task_prompt=task_description))
        inputs, shape = self.model.prepare_action_inputs(latents, context, mask, self.action_chunk_size, self.state_normalizer.forward(state), seed=seed)
        # Teacher-flow distillation uses the same Euler velocity update as the
        # teacher; one NFE at sigma=1 is x0 = noise - predicted_velocity.
        action = self._run_action_denoising(inputs, shape, self.action_infer_steps, seed)
        action = self.action_normalizer.backward(action).numpy()
        if self.gripper_postprocess:
            action[..., -1] = -(2 * action[..., -1] - 1)
            if self.binarize_gripper:
                action[..., -1] = np.sign(action[..., -1])
        if not np.isfinite(action).all():
            raise ValueError("RealtimeWAM produced non-finite actions")
        return action.astype(np.float32)
