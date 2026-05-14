from pathlib import Path

import torch

from lightx2v_train.utils.registry import INFERENCER_REGISTER

from .base import BaseInferencer


@INFERENCER_REGISTER("native_pipeline")
class NativePipelineInferencer(BaseInferencer):
    @torch.no_grad()
    def infer(self):
        prompts = [sample["prompt"] for sample in self.dataloader_eval.dataset.samples]
        enable_cfg = self.infer_config.get("enable_cfg", False)
        negative_prompt = self.infer_config.get("negative_prompt", " ") if enable_cfg else None
        base_seed = self.infer_config.get("seed", 42)

        # Model-specific kwargs (e.g. QwenImage uses `true_cfg_scale` instead of `guidance_scale`)
        pipeline_kwargs = self.model.get_pipeline_infer_kwargs(self.infer_config)

        # Use the pipeline's original pretrained scheduler for bit-exact alignment with diffusers
        pipe = self.model.assemble_pipeline()

        # self.lora_path = self.infer_config.get("lora_path", None)
        # if self.lora_path:
        #     pipe.load_lora_weights(self.lora_path)

        saved_paths = []
        self.model.transformer.eval()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                generator = torch.Generator(device=self.model.device).manual_seed(base_seed + i)
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    generator=generator,
                    **pipeline_kwargs,
                )

                if self.output_infer_dir is not None:
                    save_path = Path(self.output_infer_dir) / f"{i:05d}.png"
                    result.images[0].save(save_path)
                    print(f"Saved to {save_path}")
                    saved_paths.append(str(save_path))

        # if self.lora_path:
        #     self.model.unload_lora_for_infer()

        return saved_paths
