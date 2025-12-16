import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

from loguru import logger
class WanOffloadTransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                self.infer_func = self.infer_with_blocks_offload
            elif offload_granularity == "phase":
                self.infer_func = self.infer_with_phases_offload
                self.phase_params = {
                    "shift_msa": None,
                    "scale_msa": None,
                    "gate_msa": None,
                    "c_shift_msa": None,
                    "c_scale_msa": None,
                    "c_gate_msa": None,
                    "y_out": None,
                    "attn_out": None,
                    "y": None,
                }
            elif offload_granularity == "model":
                self.infer_func = self.infer_without_offload

            if offload_granularity != "model":
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)
            self.lazy_load = self.config.get("lazy_load", False)
            if self.lazy_load and offload_granularity == "phase":
                self.offload_manager.init_lazy_load(num_workers=self.config.get("num_disk_workers", 4))

    def infer_with_blocks_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)

            self.offload_manager.prefetch_weights((block_idx + 1) % len(blocks), blocks)
            with torch_device_module.stream(self.offload_manager.compute_stream):
                x = self.infer_block(self.offload_manager.cuda_buffers[0], x, pre_infer_out)

            self.offload_manager.swap_blocks()

        if self.clean_cuda_cache:
            del (
                pre_infer_out.embed0,
                pre_infer_out.context,
            )
            torch_device_module.empty_cache()

        return x

    def infer_with_phases_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            logger.info(block_idx)
            self.block_idx = block_idx
            if self.lazy_load:
                next_prefetch = (block_idx + 1) % len(blocks)
                self.offload_manager.start_prefetch_block(next_prefetch)

            x = self.infer_phases(block_idx, blocks, x, pre_infer_out)
            if self.clean_cuda_cache:
                del (
                    self.phase_params["attn_out"],
                    self.phase_params["y_out"],
                    self.phase_params["y"],
                )
                torch_device_module.empty_cache()

        if self.clean_cuda_cache:
            self.clear_offload_params(pre_infer_out)

        return x

    def infer_phases(self, block_idx, blocks, x, pre_infer_out):
        for phase_idx in range(self.phases_num):
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)
            next_block_idx = (block_idx + 1) % len(blocks) if phase_idx == self.phases_num - 1 else block_idx
            next_phase_idx = (phase_idx + 1) % self.phases_num
            if self.lazy_load:
                if phase_idx == self.phases_num - 1:
                    self.offload_manager.swap_cpu_buffers()
            self.offload_manager.prefetch_phase(next_block_idx, next_phase_idx, blocks)
            with torch_device_module.stream(self.offload_manager.compute_stream):
                x = self.infer_phase(phase_idx, self.offload_manager.cuda_buffers[phase_idx], x, pre_infer_out)

            self.offload_manager.swap_phases()

        return x

    def infer_phase(self, cur_phase_idx, cur_phase, x, pre_infer_out):
        if cur_phase_idx == 0:
            if hasattr(cur_phase, "before_proj") and cur_phase.before_proj.weight is not None:
                x = cur_phase.before_proj.apply(x) + pre_infer_out.x
            (
                self.phase_params["shift_msa"],
                self.phase_params["scale_msa"],
                self.phase_params["gate_msa"],
                self.phase_params["c_shift_msa"],
                self.phase_params["c_scale_msa"],
                self.phase_params["c_gate_msa"],
            ) = self.pre_process(cur_phase.modulation, pre_infer_out.embed0)
            self.phase_params["y_out"] = self.infer_self_attn(
                cur_phase,
                x,
                self.phase_params["shift_msa"],
                self.phase_params["scale_msa"],
            )
        elif cur_phase_idx == 1:
            x, self.phase_params["attn_out"] = self.infer_cross_attn(
                cur_phase,
                x,
                pre_infer_out.context,
                self.phase_params["y_out"],
                self.phase_params["gate_msa"],
            )
        elif cur_phase_idx == 2:
            self.phase_params["y"] = self.infer_ffn(
                cur_phase,
                x,
                self.phase_params["attn_out"],
                self.phase_params["c_shift_msa"],
                self.phase_params["c_scale_msa"],
            )
            x = self.post_process(x, self.phase_params["y"], self.phase_params["c_gate_msa"], pre_infer_out)
            if hasattr(cur_phase, "after_proj"):
                pre_infer_out.adapter_args["hints"].append(cur_phase.after_proj.apply(x))
        elif cur_phase_idx == 3:
            x = self.infer_post_adapter(cur_phase, x, pre_infer_out)
        return x

    def clear_offload_params(self, pre_infer_out):
        del (
            self.phase_params["shift_msa"],
            self.phase_params["scale_msa"],
            self.phase_params["gate_msa"],
        )
        del (
            self.phase_params["c_shift_msa"],
            self.phase_params["c_scale_msa"],
            self.phase_params["c_gate_msa"],
        )
        del (
            pre_infer_out.embed0,
            pre_infer_out.context,
        )
        torch_device_module.empty_cache()
