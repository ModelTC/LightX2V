import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type


class DefaultLinear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


def replace_linear_with_custom(model: nn.Module, CustomLinear: Type[nn.Module]) -> nn.Module:
    """
    递归地将模型中的所有nn.Linear层替换为CustomLinear层
    
    参数:
        model: 要处理的PyTorch模型
        CustomLinear: 自定义的Linear类，参数接口应与nn.Linear一致
        
    返回:
        替换后的模型
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 获取原始Linear层的参数
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            
            # 创建CustomLinear层
            custom_linear = CustomLinear(
                in_features=in_features,
                out_features=out_features,
                bias=bias
            )
            
            # 复制权重和偏置
            with torch.no_grad():
                custom_linear.weight.copy_(module.weight)
                if bias:
                    custom_linear.bias.copy_(module.bias)
            
            # 替换模块
            setattr(model, name, custom_linear)
        else:
            # 递归处理子模块
            replace_linear_with_custom(module, CustomLinear)
    
    return model