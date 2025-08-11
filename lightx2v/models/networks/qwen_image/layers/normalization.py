import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Type


__all__ = ["LayerNorm", "RMSNorm"]


class DefaultLayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class DefaultRMSNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


def replace_layernorm_with_custom(model: nn.Module, CustomLayerNorm: Type[nn.Module]) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm): 
            normalized_shape = module.normalized_shape
            eps = module.eps
            elementwise_affine = module.elementwise_affine
            
            custom_layernorm = CustomLayerNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine
            )
            
            if elementwise_affine:
                with torch.no_grad():
                    custom_layernorm.weight.copy_(module.weight)
                    custom_layernorm.bias.copy_(module.bias)          
            setattr(model, name, custom_layernorm)
        else:
            replace_layernorm_with_custom(module, CustomLayerNorm)
    
    return model


def replace_rmsnorm_with_custom(model: nn.Module, CustomRMSNorm: Type[nn.Module]) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.RMSNorm):
            normalized_shape = module.normalized_shape
            eps = getattr(module, 'eps', 1e-6) 
            elementwise_affine = getattr(module, 'elementwise_affine', True)
            
            custom_rmsnorm = CustomRMSNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine
            )
            
            if elementwise_affine:
                with torch.no_grad():
                    custom_rmsnorm.weight.copy_(module.weight)
                    if hasattr(module, 'bias') and hasattr(custom_rmsnorm, 'bias'):
                        custom_rmsnorm.bias.copy_(module.bias)
            
            setattr(model, name, custom_rmsnorm)
        else:
            replace_rmsnorm_with_custom(module, CustomRMSNorm)
    
    return model