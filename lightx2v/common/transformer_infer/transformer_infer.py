import math
from abc import ABC, abstractmethod

import torch
from loguru import logger

from lightx2v.common.offload.config import get_offload_granularity, use_event_offload
from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class BaseTransformerInfer(ABC):
    @staticmethod
    def create_block_offload_manager(config, **kwargs):
        manager_class = EventSlotWeightAsyncStreamManager if use_event_offload(config) else WeightAsyncStreamManager
        return manager_class(offload_granularity="block", **kwargs)

    def init_block_offload(self, config, transformer_weights):
        self._block_offload_managers = {}
        shared_streams = {}
        block_offload = config.get("cpu_offload", False) and get_offload_granularity(config) == "block"

        for name, blocks in transformer_weights.get_offload_block_groups().items():
            if not block_offload or not blocks.offload_block_indices:
                self._block_offload_managers[name] = None
                continue

            cuda_buffers = getattr(blocks, "offload_cuda_buffers", None)
            if cuda_buffers is None:
                raise RuntimeError(f"Block offload group {name!r} has no staging buffers")
            expected_buffers = min(2, len(blocks.offload_block_indices))
            if len(cuda_buffers) != expected_buffers:
                raise RuntimeError(f"Block offload group {name!r} requires {expected_buffers} staging buffers, got {len(cuda_buffers)}")

            manager = self.create_block_offload_manager(config, **shared_streams)
            manager.init_cuda_buffer(cuda_buffers)
            if config.get("lazy_load", False):
                manager.init_cpu_buffer(blocks.offload_cpu_buffers)
                manager.init_lazy_load(config.get("num_disk_workers", 4))
            self._block_offload_managers[name] = manager

            if manager.uses_events and not shared_streams:
                shared_streams = {
                    "load_stream": manager.cuda_load_stream,
                    "compute_stream": manager.compute_stream,
                }

    def get_block_offload_manager(self, blocks):
        return self._block_offload_managers[blocks.offload_group_name]

    def has_block_offload_manager(self):
        return any(manager is not None for manager in getattr(self, "_block_offload_managers", {}).values())

    def init_compile(self, config):
        self.use_compile = config.get("use_compile", False)
        self.compiled_blocks = {}
        if self.use_compile:
            logger.info(f"[Compile] Using torch.compile for {type(self).__name__}")

    def get_compiled_block(self, block_idx, block):
        key = self.get_compile_block_key(block_idx, block)
        cached = self.compiled_blocks.get(key)
        if cached is not None and cached[0] is block:
            return cached[1]

        def block_runner(*args):
            return self.infer_block(block, *args)

        compiled = torch.compile(block_runner, dynamic=None)
        self.compiled_blocks[key] = (block, compiled)
        return compiled

    def get_compile_block_key(self, block_idx, block):
        return block_idx

    def run_block(self, block_idx, block, *args):
        if self.use_compile:
            return self.get_compiled_block(block_idx, block)(*args)
        return self.infer_block(block, *args)

    @staticmethod
    def _adapter_block_index(adapter_block_index, block_index):
        return adapter_block_index(block_index) if adapter_block_index is not None else None

    @staticmethod
    def _run_offload_block(manager, run_block, block_index, block):
        if AI_DEVICE == "xpu":
            return run_block(block_index, block)
        with torch_device_module.stream(manager.compute_stream):
            return run_block(block_index, block)

    @staticmethod
    def _record_block_output_stream(block_output, stream):
        if isinstance(block_output, torch.Tensor):
            block_output.record_stream(stream)
        elif isinstance(block_output, (tuple, list)):
            for value in block_output:
                BaseTransformerInfer._record_block_output_stream(value, stream)

    def _finish_offload_group(self, manager, caller_stream, block_output):
        if AI_DEVICE == "xpu":
            return
        with torch_device_module.stream(manager.compute_stream):
            done = manager.compute_stream.record_event()
        caller_stream.wait_event(done)
        self._record_block_output_stream(block_output, caller_stream)

    def _run_blocks_with_stream_offload(
        self,
        manager,
        blocks,
        run_block,
        adapter_block_index=None,
        state_dict_transform=None,
    ):
        offloaded_indices = blocks.offload_block_indices
        if not offloaded_indices:
            block_output = None
            for block_index, block in enumerate(blocks):
                block_output = run_block(block_index, block)
            return block_output

        caller_stream = torch_device_module.current_stream()
        if AI_DEVICE != "xpu":
            manager.compute_stream.wait_stream(caller_stream)

        original_buffers = tuple(manager.cuda_buffers)
        first_index = offloaded_indices[0]
        first_adapter_index = self._adapter_block_index(adapter_block_index, first_index)
        block_output = None
        try:
            if manager.loaded_state_dict_transform is not state_dict_transform or manager.loaded_first_adapter_block_index != first_adapter_index:
                manager.need_init_first_buffer = True
            if manager.need_init_first_buffer:
                manager.init_first_buffer(
                    blocks,
                    first_adapter_index,
                    block_idx=first_index,
                    state_dict_transform=state_dict_transform,
                )

            offloaded_position = 0
            for block_index, block in enumerate(blocks):
                if block_index in blocks.resident_block_indices:
                    block_output = self._run_offload_block(manager, run_block, block_index, block)
                    continue

                if len(offloaded_indices) > 1:
                    next_position = (offloaded_position + 1) % len(offloaded_indices)
                    next_index = offloaded_indices[next_position]
                    manager.prefetch_weights_to_buffer(
                        1,
                        next_index,
                        blocks,
                        self._adapter_block_index(adapter_block_index, next_index),
                        state_dict_transform,
                    )

                block_output = self._run_offload_block(manager, run_block, block_index, manager.cuda_buffers[0])
                if len(offloaded_indices) > 1:
                    manager.swap_blocks()
                else:
                    manager.wait_for_block_compute()
                offloaded_position += 1
            manager.loaded_state_dict_transform = state_dict_transform
            manager.loaded_first_adapter_block_index = first_adapter_index
        except BaseException:
            torch_device_module.synchronize()
            manager.need_init_first_buffer = True
            manager.loaded_state_dict_transform = None
            manager.loaded_first_adapter_block_index = None
            manager.cuda_buffers[:] = original_buffers
            raise

        self._finish_offload_group(manager, caller_stream, block_output)
        return block_output

    def _run_blocks_with_event_offload(
        self,
        manager,
        blocks,
        run_block,
        adapter_block_index=None,
        state_dict_transform=None,
    ):
        offloaded_indices = blocks.offload_block_indices
        if not offloaded_indices:
            block_output = None
            for block_index, block in enumerate(blocks):
                block_output = run_block(block_index, block)
            return block_output

        caller_stream = torch_device_module.current_stream()
        compute_stream = caller_stream if AI_DEVICE == "xpu" else manager.compute_stream
        if AI_DEVICE != "xpu":
            compute_stream.wait_stream(caller_stream)
        scheduled_slots = {}
        next_position = 0
        block_output = None

        def prefetch_next(slot_index):
            nonlocal next_position
            if next_position == len(offloaded_indices):
                return
            block_index = offloaded_indices[next_position]
            manager.prefetch_to_slot(
                slot_index,
                block_index,
                blocks,
                self._adapter_block_index(adapter_block_index, block_index),
                state_dict_transform,
            )
            scheduled_slots[block_index] = slot_index
            next_position += 1

        try:
            for slot_index in range(manager.slot_count):
                prefetch_next(slot_index)

            for block_index, block in enumerate(blocks):
                if block_index in blocks.resident_block_indices:
                    block_output = self._run_offload_block(manager, run_block, block_index, block)
                    continue

                slot_index = scheduled_slots.pop(block_index)
                staged_block = manager.wait_ready(slot_index, compute_stream)
                block_output = self._run_offload_block(manager, run_block, block_index, staged_block)
                manager.record_free(slot_index, compute_stream)
                prefetch_next(slot_index)
        except BaseException:
            torch_device_module.synchronize()
            manager.reset_slots()
            raise

        self._finish_offload_group(manager, caller_stream, block_output)
        return block_output

    def run_blocks_with_offload(
        self,
        blocks,
        run_block,
        adapter_block_index=None,
        state_dict_transform=None,
    ):
        """Run blocks in model order, staging only the non-resident weights.

        ``run_block`` returns the tensor state that leaves the block group so
        its lifetime can be transferred back to the caller stream.
        """
        manager = self.get_block_offload_manager(blocks)
        if manager is None:
            block_output = None
            for block_index, block in enumerate(blocks):
                block_output = run_block(block_index, block)
            return block_output
        if manager.uses_events:
            return self._run_blocks_with_event_offload(
                manager,
                blocks,
                run_block,
                adapter_block_index,
                state_dict_transform,
            )
        return self._run_blocks_with_stream_offload(
            manager,
            blocks,
            run_block,
            adapter_block_index,
            state_dict_transform,
        )

    def get_offload_managers(self):
        managers = [manager for manager in getattr(self, "_block_offload_managers", {}).values() if manager is not None]
        manager = getattr(self, "offload_manager", None)
        if manager is not None and manager not in managers:
            managers.append(manager)
        return managers

    def clear_offload_managers(self):
        self._block_offload_managers = {}
        if hasattr(self, "offload_manager"):
            del self.offload_manager

    @abstractmethod
    def infer(self):
        pass

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.scheduler.transformer_infer = self


class BaseTaylorCachingTransformerInfer(BaseTransformerInfer):
    @abstractmethod
    def infer_calculating(self):
        pass

    @abstractmethod
    def infer_using_cache(self):
        pass

    @abstractmethod
    def get_taylor_step_diff(self):
        pass

    # 1. when fully calcualted, stored in cache
    def derivative_approximation(self, block_cache, module_name, out):
        if module_name not in block_cache:
            block_cache[module_name] = {0: out}
        else:
            step_diff = self.get_taylor_step_diff()

            previous_out = block_cache[module_name][0]
            block_cache[module_name][0] = out
            block_cache[module_name][1] = (out - previous_out) / step_diff

    def taylor_formula(self, tensor_dict):
        x = self.get_taylor_step_diff()

        output = 0
        for i in range(len(tensor_dict)):
            output += (1 / math.factorial(i)) * tensor_dict[i] * (x**i)

        return output
