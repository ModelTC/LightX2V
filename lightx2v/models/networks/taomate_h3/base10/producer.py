"""Native Base10 audio milestone producer for TaoMate Stage3 requests."""

from pathlib import Path

import torch

from lightx2v_platform.base.global_var import AI_DEVICE

from .model import TaoMateBase10Model
from .scheduler import TaoMateBase10Scheduler


class TaoMateBase10Producer:
    """Run the released base DiT's nine audio-only denoising forwards."""

    def __init__(self, model_root, config, *, model=None):
        self.config = dict(config)
        self.model_root = Path(model_root)
        self.scheduler = TaoMateBase10Scheduler(self.config)
        self.model = model
        if self.model is not None:
            self.model.set_scheduler(self.scheduler)

    def load(self):
        if self.model is not None:
            return self.model
        config = dict(self.config)
        config.setdefault("hidden_size", 5376)
        config.setdefault("ffn_dim", 14336)
        config.setdefault("num_attention_heads", 56)
        # Base10 is an unadapted BF16 teacher with its own operator profile.
        # Stage3 attention, norm, RoPE, and adapter choices change its audio
        # milestones, so none of those choices may leak into this model.
        config.update(
            dit_original_ckpt=str(self.model_root / "transformer"),
            seq_parallel=False,
            feature_caching="NoCaching",
            text_refiner_tensor_parallel=False,
            use_fused_qkv=True,
            use_fused_qkv_norm_rope=False,
            attn_type="flash_attn3",
            refiner_attn_type="flash_attn3",
            rms_type="torch_native",
            rope_type="torch_real_rope",
            dit_quantized=False,
            dit_quantized_ckpt=None,
            dit_quant_scheme="Default",
            weight_auto_quant=False,
            adapter_model_path=None,
            lora_configs=None,
            lora_dynamic_apply=False,
        )
        device = "cpu" if config.get("cpu_offload", False) else AI_DEVICE
        self.model = TaoMateBase10Model(str(self.model_root), config, device)
        self.model.set_scheduler(self.scheduler)
        if self.model.prepost_resident:
            self.model.pre_weight.to_cuda()
            self.model.post_weight.to_cuda()
        return self.model

    @torch.no_grad()
    def produce(self, prompt_embeds, text_tags, seed, request_index, previous_clean=None):
        """Return Base10 states 3, 6, and 9 as CPU float32 audio rows.

        ``seed`` is the first request's audio seed; the official producer uses
        consecutive seeds for subsequent requests. ``previous_clean`` is the
        previous request's last milestone, including both stereo channels.
        """
        model = self.load()
        height, width = self.config["size"]
        scheduler = self.scheduler
        scheduler.prepare_request(
            text_token_tags=text_tags,
            audio_seed=int(seed) + int(request_index),
            request_index=int(request_index),
            previous_clean=previous_clean,
            height=int(height),
            width=int(width),
        )
        if self.config.get("cpu_offload", False) and not model.block_offload:
            model.to_cuda()
        inputs = {"text_encoder_output": {"prompt_embeds": prompt_embeds.to(AI_DEVICE)}}
        try:
            for step_index in range(scheduler.infer_steps):
                scheduler.step_pre(step_index)
                model.infer(inputs)
                scheduler.step_post()
            return scheduler.finish_request()
        finally:
            if self.config.get("cpu_offload", False) and not model.block_offload:
                model.to_cpu()
            scheduler.clear()

    def close(self):
        if self.model is None:
            return
        self.model.transformer_infer.reset_cuda_graph()
        self.model.to_cpu()
        self.model.release_block_offload_buffers()
        self.scheduler.clear()
        self.scheduler.transformer_infer = None
        self.model = None
