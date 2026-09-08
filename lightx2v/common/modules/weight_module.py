from loguru import logger

from lightx2v.common.offload.config import get_offload_granularity, get_offload_plan
from lightx2v_platform.base.global_var import AI_DEVICE


def resolve_resident_block_indices(resident_count, blocks_num):
    count = blocks_num if resident_count == "all" else resident_count
    if not 0 <= count <= blocks_num:
        raise ValueError(f"resident block count must be between 0 and {blocks_num}, got {count}")
    if count == 0:
        return frozenset()
    return frozenset((index * blocks_num) // count for index in range(count))


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
            if param is not None and hasattr(param, "state_dict"):
                param.state_dict(destination)
        for _, module in self._modules.items():
            if module is not None:
                module.state_dict(destination)
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if destination is None:
            destination = {}
        for _, param in self._parameters.items():
            if param is not None and hasattr(param, "load_state_dict"):
                param.load_state_dict(destination, block_index, adapter_block_index)
        for _, module in self._modules.items():
            if module is not None:
                module.load_state_dict(destination, block_index, adapter_block_index)
        return destination

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        for _, param in self._parameters.items():
            if param is not None and hasattr(param, "load_state_dict_from_disk"):
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
                    self._parameters[name] = param.to("cpu", non_blocking=non_blocking)
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
        for name, param in self._parameters.items():
            if param is not None:
                if hasattr(param, "cuda"):
                    self._parameters[name] = param.to(AI_DEVICE, non_blocking=non_blocking)
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
                    self._parameters[name] = param.to("cpu", non_blocking=non_blocking)
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
        for name, param in self._parameters.items():
            if param is not None:
                if hasattr(param, "cuda"):
                    self._parameters[name] = param.to(AI_DEVICE, non_blocking=non_blocking)
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

    def release_device_weights(self):
        self.to_cpu()

    def release_non_block_weights(self):
        block_groups = tuple(self.get_offload_block_groups().values())
        excluded_modules = {id(blocks) for blocks in block_groups}
        for blocks in block_groups:
            for name in ("offload_cuda_buffers", "offload_cpu_buffers"):
                buffers = getattr(blocks, name, None)
                if buffers is not None:
                    excluded_modules.add(id(buffers))

        for module in self._modules.values():
            if module is not None and id(module) not in excluded_modules and hasattr(module, "to_cpu"):
                module.to_cpu()
        for name, parameter in self._parameters.items():
            if parameter is None or id(parameter) in excluded_modules:
                continue
            if hasattr(parameter, "cpu"):
                self._parameters[name] = parameter.to("cpu")
                setattr(self, name, self._parameters[name])
            elif hasattr(parameter, "to_cpu"):
                parameter.to_cpu()

    def register_offload_block_group(self, config, name, blocks):
        granularity = get_offload_granularity(config)
        if not config.get("cpu_offload", False):
            resident_count = "all"
        elif granularity == "block":
            resident_count = get_offload_plan(config).get("resident_blocks", {}).get(name, 0)
        else:
            resident_count = 0

        resident_indices = resolve_resident_block_indices(resident_count, len(blocks))
        blocks.offload_group_name = name
        blocks.resident_block_indices = resident_indices
        blocks.offload_block_indices = tuple(index for index in range(len(blocks)) if index not in resident_indices)

        if not hasattr(self, "_offload_block_groups"):
            self._offload_block_groups = {}
        self._offload_block_groups[name] = blocks
        slot_count = min(2, len(blocks.offload_block_indices)) if granularity == "block" and config.get("cpu_offload", False) else 0
        if config.get("cpu_offload", False) and granularity == "block":
            logger.info(
                "Block offload group '{}': resident={}/{}, indices={}, staging_slots={}",
                name,
                len(resident_indices),
                len(blocks),
                tuple(sorted(resident_indices)),
                slot_count,
            )
        return slot_count

    def register_offload_block_buffers(self, name, cuda_buffers, cpu_buffers=None):
        blocks = self._offload_block_groups[name]
        blocks.offload_cuda_buffers = cuda_buffers
        blocks.offload_cpu_buffers = cpu_buffers

    def validate_offload_block_groups(self, config):
        if not config.get("cpu_offload", False) or get_offload_granularity(config) != "block":
            return

        plan = get_offload_plan(config)
        configured_groups = set(plan.get("resident_blocks", {}))
        registered_groups = set(self.get_offload_block_groups())
        unknown_groups = configured_groups - registered_groups
        if unknown_groups:
            raise ValueError(f"Unknown resident block groups {sorted(unknown_groups)}; available groups are {sorted(registered_groups)}")

    def get_offload_block_groups(self):
        return getattr(self, "_offload_block_groups", {})

    def resident_blocks_to_cuda(self, non_blocking=True):
        for blocks in self.get_offload_block_groups().values():
            for block_index in blocks.resident_block_indices:
                blocks[block_index].to_cuda(non_blocking=non_blocking)

    def release_resident_blocks(self):
        for blocks in self.get_offload_block_groups().values():
            for block_index in blocks.resident_block_indices:
                blocks[block_index].release_device_weights()


class WeightModuleList(WeightModule):
    def __init__(self, modules=None):
        super().__init__()
        # Transformer block lists receive these values when registered for offload.
        self.offload_group_name = None
        self.resident_block_indices = frozenset()
        self.offload_block_indices = ()
        self.offload_cuda_buffers = None
        self.offload_cpu_buffers = None
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
