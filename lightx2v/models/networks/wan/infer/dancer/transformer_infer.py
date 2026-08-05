import torch

from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


def apply_dancer_rope(xq, xk, freqs):
    """The upstream implementation deliberately computes RoPE in complex128."""
    length, heads = freqs.shape[0], xq.shape[1]

    def apply(x):
        dtype = x.dtype
        value = torch.view_as_complex(x[:length].to(torch.float64).reshape(length, heads, -1, 2))
        value = torch.view_as_real(value * freqs).flatten(2).to(dtype)
        return torch.cat([value, x[length:]], dim=0) if length < x.shape[0] else value

    return apply(xq), apply(xk)


class WanDancerTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.task = "i2v"
        self.apply_rope_func = apply_dancer_rope
        self.inject_layer_to_id = {layer: index for index, layer in enumerate(config["music_inject_layers"])}

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        self._dancer_weights = weights
        self._dancer_use_global = pre_infer_out.adapter_args["use_global"]
        return super().infer(weights, pre_infer_out)

    def infer_block(self, block, x, pre_infer_out):
        if pre_infer_out.adapter_args["enable_skip_layer"] and not self.scheduler.infer_condition and self.block_idx == 9:
            return x
        x = super().infer_block(block, x, pre_infer_out)
        injector_id = self.inject_layer_to_id.get(self.block_idx)
        if injector_id is not None:
            audio = pre_infer_out.adapter_args["music_context"]
            residual = self._dancer_weights.music_injectors[injector_id].v.apply(audio)
            residual = self._dancer_weights.music_injectors[injector_id].o.apply(residual)
            if x.shape[0] % residual.shape[0]:
                raise ValueError(f"Visual/audio token mismatch: {x.shape[0]} vs {residual.shape[0]}")
            x.add_(torch.repeat_interleave(residual, x.shape[0] // residual.shape[0], dim=0))
        return x

    def infer_non_blocks(self, weights, x, e):
        modulation = weights.head_global_modulation.tensor if self._dancer_use_global else weights.head_modulation.tensor
        head = weights.head_global if self._dancer_use_global else weights.head
        e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = weights.norm.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype)
        x.mul_(1 + e[1].squeeze()).add_(e[0].squeeze())
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.infer_dtype)
        return head.apply(x)
