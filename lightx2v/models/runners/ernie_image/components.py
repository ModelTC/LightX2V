class ErnieImagePipelineComponents:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(cls, model_path, pretrained_kwargs, target_device, cpu_offload=False):
        try:
            from diffusers import ErnieImagePipeline
        except ImportError as exc:
            raise ImportError("ERNIE-Image requires a Diffusers release with ErnieImagePipeline support.") from exc

        pipeline = ErnieImagePipeline.from_pretrained(model_path, **pretrained_kwargs)
        if cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(target_device)
        return cls(pipeline)

    @property
    def transformer(self):
        return self.pipeline.transformer

    @property
    def scheduler(self):
        return self.pipeline.scheduler

    @property
    def vae(self):
        return self.pipeline.vae

    @property
    def vae_scale_factor(self):
        return self.pipeline.vae_scale_factor

    @property
    def execution_device(self):
        return getattr(self.pipeline, "_execution_device", None)

    @property
    def text_in_dim(self):
        return self.transformer.config.text_in_dim

    @property
    def has_prompt_enhancer(self):
        return getattr(self.pipeline, "pe", None) is not None and getattr(self.pipeline, "pe_tokenizer", None) is not None

    def enhance_prompt_with_pe(self, prompt, device, width, height):
        return self.pipeline._enhance_prompt_with_pe(prompt, device, width=width, height=height)

    def encode_prompt(self, prompt, device, num_images_per_prompt):
        return self.pipeline.encode_prompt(prompt, device, num_images_per_prompt)

    def pad_text(self, text_hiddens, device, dtype):
        return self.pipeline._pad_text(
            text_hiddens=text_hiddens,
            device=device,
            dtype=dtype,
            text_in_dim=self.text_in_dim,
        )

    @property
    def unpatchify_latents(self):
        return getattr(self.pipeline, "_unpatchify_latents", None)

    def maybe_free_model_hooks(self):
        if hasattr(self.pipeline, "maybe_free_model_hooks"):
            self.pipeline.maybe_free_model_hooks()

    def run_pipeline_call(self, kwargs):
        return self.pipeline(**kwargs)
