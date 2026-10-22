"""Native TaoMate-H3 DiT with the released Stage3 adapter.

This class uses LightX2V's MiniMax-H3 weight containers and inference ops.
Its separate phase scheduler and clean-history attention are attached by the
TaoMate runner; no upstream model modules are imported.
"""

import torch

from lightx2v.models.networks.autoregressive_h3 import H3AutoregressiveModel
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.taomate_h3.infer.clean_kv import TaoMateCleanKVCache
from lightx2v.models.networks.taomate_h3.infer.post_infer import TaoMateH3PostInfer
from lightx2v.models.networks.taomate_h3.infer.pre_infer import TaoMateH3PreInfer
from lightx2v.models.networks.taomate_h3.infer.transformer_infer import TaoMateH3OffloadTransformerInfer, TaoMateH3TransformerInfer
from lightx2v.models.networks.taomate_h3.weights import inspect_taomate_adapter, load_taomate_lora


class TaoMateH3NativeModel(H3AutoregressiveModel, MiniMaxH3Model):
    """Load FL2VA-equivalent base weights and apply the TaoMate LoRA bank."""

    def __init__(self, model_path, config, device):
        self.adapter_info = inspect_taomate_adapter(config["adapter_path"])
        native_config = dict(config)
        # The refiner sees only a handful of text rows. Replicating it avoids
        # TP reduction rounding before the long autoregressive denoise loop.
        native_config.setdefault("text_refiner_tensor_parallel", False)
        native_config["lora_dynamic_apply"] = True
        native_config["lora_configs"] = [
            {
                "path": str(self.adapter_info.path / "adapter_model.safetensors"),
                "strength": 1.0,
                "alpha": self.adapter_info.alpha,
            }
        ]
        super().__init__(
            model_path,
            native_config,
            device,
            lora_path=str(self.adapter_info.path / "adapter_model.safetensors"),
            lora_strength=1.0,
            lora_alpha=self.adapter_info.alpha,
        )

    def _init_infer_class(self):
        super()._init_infer_class()
        self.pre_infer_class = TaoMateH3PreInfer
        self.transformer_infer_class = TaoMateH3OffloadTransformerInfer if self.cpu_offload else TaoMateH3TransformerInfer
        self.post_infer_class = TaoMateH3PostInfer

    def _seq_parallel_pre_process(self, pre):
        # Stage3 keeps the conditioner prefix replicated and shards only media.
        return MiniMaxH3Model._seq_parallel_pre_process(self, pre)

    def _seq_parallel_post_process(self, output, pre):
        return MiniMaxH3Model._seq_parallel_post_process(self, output, pre)

    def initialize_kv_cache(self, capacity):
        pool = self._bind_rolling_kv_cache(capacity, step_cache=False)
        cache = getattr(self, "clean_kv_cache", None)
        if cache is None or cache.pool is not pool:
            cache = TaoMateCleanKVCache(pool, len(self.transformer_weights.blocks))
            self.clean_kv_cache = cache
            self.transformer_infer.set_clean_kv_cache(cache)
        self.scheduler.cache = self.clean_kv_cache
        return self.clean_kv_cache

    @torch.no_grad()
    def commit_clean_chunk(self, inputs, chunk_index):
        """Run the clean transformer once to store final KV without output heads."""

        cache = self.clean_kv_cache
        cache.begin_clean_commit(chunk_index)
        move_pre = self.block_offload and not self.prepost_resident
        if move_pre:
            self.pre_weight.to_cuda()
        try:
            prompt = inputs["text_encoder_output"]["prompt_embeds"]
            pre = self.pre_infer.infer(self.pre_weight, prompt)
            if self.config.get("seq_parallel", False):
                pre = self._seq_parallel_pre_process(pre)
            self.transformer_infer.infer(self.transformer_weights, pre)
            cache.commit()
        except Exception:
            cache.rollback()
            raise
        finally:
            if move_pre:
                self.pre_weight.to_cpu()

    def reset_autoregressive_state(self):
        cache = getattr(self, "clean_kv_cache", None)
        if cache is not None:
            cache.clear()
        super().reset_autoregressive_state()

    def _load_lora_file(self, file_path, alpha=None):
        if alpha is not None and float(alpha) != self.adapter_info.alpha:
            raise ValueError("TaoMate-H3 Stage3 alpha differs from the released adapter")
        weights = load_taomate_lora(self.adapter_info.path)
        self._pending_dynamic_lora_model_keys = {f"{name[: -len('.lora_down.weight')]}.weight" for name in weights if name.endswith(".lora_down.weight")}
        return weights


__all__ = ["TaoMateH3NativeModel"]
