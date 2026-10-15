"""Transport raw prompts and original references to the official Qwen pipeline."""

from __future__ import annotations

from uuid import uuid4

from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from verl.utils.profiler import simple_timer
from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput


class QwenImage21AgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        extra = kwargs.get("extra_info") or {}
        prompt = kwargs.get("prompt", extra.get("prompt"))
        task_type = kwargs.get("task_type", extra.get("task_type"))
        references = kwargs.get("source_images", extra.get("source_images", []))
        if not isinstance(prompt, str) or task_type not in {"t2i", "i2i"}:
            raise ValueError("Qwen rollout requires the dataset's raw prompt and task_type.")
        if (task_type == "t2i" and len(references)) or (task_type == "i2i" and not len(references)):
            raise ValueError("Reference images do not match the sample's task_type.")

        def load_images():
            images = []
            for path in references:
                with Image.open(path) as source:
                    images.append(source.convert("RGBA"))
            return images

        images = await self.loop.run_in_executor(None, load_images)
        # These IDs serve transport/logging only. The original text travels
        # separately, so no tokenize/decode round-trip changes official prompts.
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError("Prompt exceeds rollout.prompt_length; increase it rather than truncate silently.")
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images or None,
                mm_processor_kwargs={"qwen_image21_prompt": prompt, "task_type": task_type},
            )
        metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        return DiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=output.diffusion_output,
            response_logprobs=output.log_probs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
