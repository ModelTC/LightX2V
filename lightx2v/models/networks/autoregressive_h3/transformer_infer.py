"""Shared compilation and CUDA Graph lifecycle for autoregressive H3 blocks."""

import torch

from lightx2v.common.transformer_infer.autoregressive_cuda_graph import AutoregressiveCUDAGraphStack, CUDAGraphPool


class H3AutoregressiveTransformerInferMixin:
    @staticmethod
    def graph_weight_bindings(block, device):
        """Keep the live device weights bound to a captured block graph.

        WeightModule.state_dict() prefers pinned CPU copies when a model was
        loaded on CPU. Graph signatures must follow the GPU tensors used by
        the kernels, including unregistered fused QKV storage.
        """
        bindings = []
        seen_modules = set()
        seen_tensors = set()
        pending = [block]
        while pending:
            module = pending.pop()
            if module is None or id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            values = list(getattr(module, "__dict__", {}).values())
            values.extend(getattr(module, "_parameters", {}).values())
            for value in values:
                if isinstance(value, torch.Tensor) and value.device == device and id(value) not in seen_tensors:
                    bindings.append(value)
                    seen_tensors.add(id(value))
            pending.extend(getattr(module, "_modules", {}).values())
            # TP linears and fused QKV storage are not registered children.
            pending.append(getattr(module, "_mm", None))
            pending.append(getattr(module, "to_qkv", None))
            pending.append(getattr(module, "_qkv_storage", None))
            pending.append(getattr(module, "target", None))
            pending.extend(getattr(module, "sources", ()))
        return tuple(bindings)

    def _create_compile_backend(self):
        if self.compile_backend == "aot_eager":
            return "aot_eager"
        return super()._create_compile_backend()

    def __init__(self, config):
        super().__init__(config)
        self.cuda_graph_stack = AutoregressiveCUDAGraphStack() if config.get("use_cuda_graph", False) else None
        self.block_graph_pool = None
        if self.use_compile and self.compile_backend == "default":
            self.compile_options.setdefault("emulate_precision_casts", True)
        if self.cuda_graph_stack is not None:
            self.compile_options["triton.cudagraphs"] = False

    def reset_cuda_graph(self):
        if self.cuda_graph_stack is not None:
            self.cuda_graph_stack.clear()
        self.block_graph_pool = None

    def run_captured_block(self, key, runner, inputs, *, bindings=(), mutated_buffers=(), variant=None, on_capture=None):
        if self.block_graph_pool is None:
            self.block_graph_pool = CUDAGraphPool(inputs[0].device)
        return self.cuda_graph_stack.run(
            key,
            runner,
            inputs,
            bindings=bindings,
            mutated_buffers=mutated_buffers,
            variant=key if variant is None else variant,
            memory_pool=self.block_graph_pool,
            on_capture=on_capture or getattr(self, "before_graph_capture", None),
        )
