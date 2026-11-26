"""Image LayerNorm and time-step modulation for the BF16 Qwen Image block."""

import torch
import triton
import triton.language as tl
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from triton.language.extra.cuda import libdevice


@triton.jit
def _modulate(Normalized, Scale, Shift, Out, n, WIDTH: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = offsets < n
    normalized = tl.load(Normalized + offsets, mask=valid, other=0)
    scale = tl.load(Scale + offsets % WIDTH).to(tl.float32)
    shift = tl.load(Shift + offsets % WIDTH).to(tl.float32)
    factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    # The eager multiply rounds to FP32 before adding the shift.
    value = libdevice.add_rn(libdevice.mul_rn(normalized, factor), shift)
    tl.store(Out + offsets, value, mask=valid)


@triton.jit
def _prepare_backward(X, Grad, Scale, InputFP32, GradFP32, n, WIDTH: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = offsets < n
    x = tl.load(X + offsets, mask=valid, other=0).to(tl.float32)
    grad = tl.load(Grad + offsets, mask=valid, other=0).to(tl.float32)
    scale = tl.load(Scale + offsets % WIDTH).to(tl.float32)
    factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    tl.store(InputFP32 + offsets, x, mask=valid)
    tl.store(GradFP32 + offsets, grad * factor, mask=valid)


class _ModulatedLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, shift, eps, output_dtype):
        # Keep native FP32 reductions and save the original BF16 input for backward.
        width = x.size(-1)
        normalized, mean, rstd = torch.ops.aten.native_layer_norm.default(x.float(), [width], None, None, eps)
        output = torch.empty_like(x, dtype=output_dtype)
        grid = (triton.cdiv(x.numel(), 512),)
        _modulate[grid](normalized, scale, shift, output, x.numel(), width, 512, enable_fp_fusion=False)
        ctx.save_for_backward(x, scale, mean, rstd)
        return output

    @staticmethod
    def backward(ctx, grad):
        x, scale, mean, rstd = ctx.saved_tensors
        width = x.size(-1)
        x_fp32, grad_fp32 = torch.empty((2, *x.shape), device=x.device, dtype=torch.float32).unbind()
        grid = (triton.cdiv(x.numel(), 512),)
        _prepare_backward[grid](x, grad.contiguous(), scale, x_fp32, grad_fp32, x.numel(), width, 512, enable_fp_fusion=False)
        dx, _, _ = torch.ops.aten.native_layer_norm_backward.default(grad_fp32, x_fp32, [width], mean, rstd, None, None, [True, False, False])
        return dx.to(x.dtype), None, None, None, None


def _image_modulation(block, norm, states, params, output_dtype):
    shift, scale, gate = params.chunk(3, dim=-1)
    if (
        type(norm) is torch.nn.LayerNorm
        and norm.normalized_shape == (states.size(-1),)
        and norm.weight is None
        and norm.bias is None
        and not (norm._forward_hooks or norm._forward_pre_hooks or norm._backward_hooks or norm._backward_pre_hooks)
        and not params.requires_grad
        and states.is_contiguous()
        and shift.is_contiguous()
        and scale.is_contiguous()
        and states.dtype == params.dtype == torch.bfloat16
        and params.shape == (1, 3 * states.size(-1))
    ):
        return _ModulatedLayerNorm.apply(states, scale, shift, norm.eps, output_dtype), gate.unsqueeze(1)
    return block._modulate(norm(states), params)


def forward_with_modulated_norm(
    self,
    hidden_states,
    encoder_hidden_states,
    encoder_hidden_states_mask,
    temb,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    modulate_index=None,
):
    if (
        not self.training
        or self.zero_cond_t
        or modulate_index is not None
        or not hidden_states.is_cuda
        or hidden_states.ndim != 3
        or hidden_states.size(0) != 1
        or hidden_states.size(-1) != 3072
        or hidden_states.dtype != torch.bfloat16
        or encoder_hidden_states.dtype != torch.bfloat16
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        return QwenImageTransformerBlock.forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb, joint_attention_kwargs, modulate_index)

    img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
    txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)
    image, img_gate = _image_modulation(self, self.img_norm1, hidden_states, img_mod1, torch.float32)
    text, txt_gate = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
    image, text = self.attn(
        hidden_states=image,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        image_rotary_emb=image_rotary_emb,
        **(joint_attention_kwargs or {}),
    )
    hidden_states = hidden_states + img_gate * image
    encoder_hidden_states = encoder_hidden_states + txt_gate * text

    # The standard GELU MLP consumes BF16 at its first linear projection.
    activation = self.img_mlp.net[0] if type(self.img_mlp) is FeedForward else None
    bf16_linear = (
        type(activation) is GELU
        and type(activation.proj) is torch.nn.Linear
        and activation.proj.weight.dtype == torch.bfloat16
        and not any(layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks or layer._backward_pre_hooks for layer in (self.img_mlp, activation, activation.proj))
    )
    image, img_gate = _image_modulation(self, self.img_norm2, hidden_states, img_mod2, torch.bfloat16 if bf16_linear else torch.float32)
    hidden_states = hidden_states + img_gate * self.img_mlp(image)
    text, txt_gate = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
    return encoder_hidden_states + txt_gate * self.txt_mlp(text), hidden_states
