from lightx2v_platform.base.global_var import AI_DEVICE
class WeightModule:
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def is_empty(self):
        return len(self._modules) == 0 and len(self._parameters) == 0

    def add_module(self, name, module):
        self._modules[name] = module
        setattr(self, name, module)

    def register_parameter(self, name, param):
        self._parameters[name] = param
        setattr(self, name, param)

    def load(self, weight_dict):
        for _, module in self._modules.items():
            if hasattr(module, "load"):
                module.load(weight_dict)

        for _, parameter in self._parameters.items():
            if hasattr(parameter, "load"):
                parameter.load(weight_dict)

    def register_diff(self, weight_dict):
        for _, module in self._modules.items():
            if hasattr(module, "register_diff"):
                module.register_diff(weight_dict)

        for _, parameter in self._parameters.items():
            if hasattr(parameter, "register_diff"):
                parameter.register_diff(weight_dict)

    def register_lora(self, weight_dict, strength):
        for _, module in self._modules.items():
            if hasattr(module, "register_lora"):
                module.register_lora(weight_dict, strength)

        for _, parameter in self._parameters.items():
            if hasattr(parameter, "register_lora"):
                parameter.register_lora(weight_dict, strength)

    def update_lora(self, weight_dict, strength):
        for _, module in self._modules.items():
            if hasattr(module, "update_lora"):
                module.update_lora(weight_dict, strength)

        for _, parameter in self._parameters.items():
            if hasattr(parameter, "update_lora"):
                parameter.update_lora(weight_dict, strength)

    def remove_lora(self):
        for _, module in self._modules.items():
            if hasattr(module, "remove_lora"):
                module.remove_lora()

        for _, parameter in self._parameters.items():
            if hasattr(parameter, "remove_lora"):
                parameter.remove_lora()

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        for _, param in self._parameters.items():
            if param is not None:
                param.state_dict(destination)
        for _, module in self._modules.items():
            if module is not None:
                module.state_dict(destination)
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if destination is None:
            destination = {}
        for _, param in self._parameters.items():
            if param is not None:
                param.load_state_dict(destination, block_index, adapter_block_index)
        for _, module in self._modules.items():
            if module is not None:
                module.load_state_dict(destination, block_index, adapter_block_index)
        return destination

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        for _, param in self._parameters.items():
            if param is not None:
                param.load_state_dict_from_disk(block_index, adapter_block_index)
        for _, module in self._modules.items():
            if module is not None:
                module.load_state_dict_from_disk(block_index, adapter_block_index)

    def named_parameters(self, prefix=""):
        for name, param in self._parameters.items():
            if param is not None:
                yield prefix + name, param
        for name, module in self._modules.items():
            if module is not None:
                yield from module.named_parameters(prefix + name + ".")

    def to_cpu(self, non_blocking=False):
        for name, param in self._parameters.items():
            if param is not None:
                if hasattr(param, "cpu"):
                    self._parameters[name] = param.cpu()
                    setattr(self, name, self._parameters[name])
                elif hasattr(param, "to_cpu"):
                    self._parameters[name].to_cpu()
                    setattr(self, name, self._parameters[name])
        for module in self._modules.values():
            if isinstance(module, WeightModuleList):
                for i in range(len(module)):
                    for m in module[i]._modules.values():
                        if m is not None and hasattr(m, "to_cpu"):
                            m.to_cpu()
                    for m in module[i]._parameters.values():
                        if m is not None and hasattr(m, "to_cpu"):
                            m.to_cpu()
            else:
                if module is not None and hasattr(module, "to_cpu"):
                    module.to_cpu()

    def to_cuda(self, non_blocking=False):
        """Move parameters to GPU device (supports cuda/intel xpu)"""
        target_device = AI_DEVICE
        for name, param in self._parameters.items():
            if param is not None:
                if hasattr(param, "cuda"):
                    self._parameters[name] = param.to(target_device, non_blocking=non_blocking)
                elif hasattr(param, "to_cuda"):
                    self._parameters[name].to_cuda()
                setattr(self, name, self._parameters[name])
        for module in self._modules.values():
            if isinstance(module, WeightModuleList):
                for i in range(len(module)):
                    for m in module[i]._modules.values():
                        if m is not None and hasattr(m, "to_cuda"):
                            m.to_cuda()
                    for m in module[i]._parameters.values():
                        if m is not None and hasattr(m, "to_cuda"):
                            m.to_cuda()
            else:
                if module is not None and hasattr(module, "to_cuda"):
                    module.to_cuda()

    def to_cpu_async(self, non_blocking=True):
        for name, param in self._parameters.items():
            if param is not None:
                if hasattr(param, "cpu"):
                    self._parameters[name] = param.cpu(non_blocking=True)
                    setattr(self, name, self._parameters[name])
                elif hasattr(param, "to_cpu"):
                    self._parameters[name].to_cpu(non_blocking=True)
                    setattr(self, name, self._parameters[name])
        for module in self._modules.values():
            if isinstance(module, WeightModuleList):
                for i in range(len(module)):
                    for m in module[i]._modules.values():
                        if m is not None and hasattr(m, "to_cpu"):
                            m.to_cpu(non_blocking=True)
                    for m in module[i]._parameters.values():
                        if m is not None and hasattr(m, "to_cpu"):
                            m.to_cpu(non_blocking=True)
            else:
                if module is not None and hasattr(module, "to_cpu"):
                    module.to_cpu(non_blocking=True)

    def to_cuda_async(self, non_blocking=True):
        target_device = AI_DEVICE
        for name, param in self._parameters.items():
            if param is not None:
                if hasattr(param, "cuda"):
                    self._parameters[name] = param.to(target_device, non_blocking=non_blocking)
                elif hasattr(param, "to_cuda"):
                    self._parameters[name].to_cuda(non_blocking=True)
                setattr(self, name, self._parameters[name])
        for module in self._modules.values():
            if isinstance(module, WeightModuleList):
                for i in range(len(module)):
                    for m in module[i]._modules.values():
                        if m is not None and hasattr(m, "to_cuda"):
                            m.to_cuda(non_blocking=True)
                    for m in module[i]._parameters.values():
                        if m is not None and hasattr(m, "to_cuda"):
                            m.to_cuda(non_blocking=True)
            else:
                if module is not None and hasattr(module, "to_cuda"):
                    module.to_cuda(non_blocking=True)


class WeightModuleList(WeightModule):
    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        if modules is not None:
            for idx, module in enumerate(modules):
                self.append(module)

    def append(self, module):
        idx = len(self._list)
        self._list.append(module)
        self.add_module(str(idx), module)

    def __getitem__(self, idx):
        return self._list[idx]

    def __setitem__(self, idx, module):
        self._list[idx] = module
        self.add_module(str(idx), module)

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)
