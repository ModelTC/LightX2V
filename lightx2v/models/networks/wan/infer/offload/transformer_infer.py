import torch

from lightx2v.common.offload.manager import (
    LazyWeightAsyncStreamManager,
    WeightAsyncStreamManager,
)

from ..transformer_infer import WanTransformerInfer


class WanOffloadTransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            if "offload_ratio" in self.config:
                offload_ratio = self.config["offload_ratio"]
            else:
                offload_ratio = 1
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                if not self.config.get("lazy_load", False):
                    self.infer_func = self.infer_with_offload
                else:
                    self.infer_func = self.infer_with_lazy_offload
            elif offload_granularity == "phase":
                if not self.config.get("lazy_load", False):
                    self.infer_func = self.infer_with_phases_offload
                else:
                    self.infer_func = self.infer_with_phases_lazy_offload
            elif offload_granularity == "model":
                self.infer_func = self.infer_without_offload

            if offload_granularity != "model":
                if not self.config.get("lazy_load", False):
                    self.weights_stream_mgr = WeightAsyncStreamManager(
                        blocks_num=self.blocks_num,
                        offload_ratio=offload_ratio,
                        phases_num=self.phases_num,
                    )
                else:
                    self.weights_stream_mgr = LazyWeightAsyncStreamManager(
                        blocks_num=self.blocks_num,
                        offload_ratio=offload_ratio,
                        phases_num=self.phases_num,
                        num_disk_workers=self.config.get("num_disk_workers", 2),
                        max_memory=self.config.get("max_memory", 2),
                        offload_gra=offload_granularity,
                    )

    def infer_with_offload(self, weights, x, pre_infer_out):
        for block_idx in range(self.blocks_num):
            self.block_idx = block_idx
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = weights.blocks[0]
                self.weights_stream_mgr.active_weights[0].to_cuda()

            if block_idx < self.blocks_num - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, weights.blocks)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                x = self.infer_block(weights.blocks[block_idx], x, pre_infer_out)
            self.weights_stream_mgr.swap_weights()

        return x

    def infer_with_lazy_offload(self, weights, x, pre_infer_out):
        self.weights_stream_mgr.prefetch_weights_from_disk(weights.blocks)

        for block_idx in range(self.blocks_num):
            if block_idx == 0:
                block = self.weights_stream_mgr.pin_memory_buffer.get(block_idx)
                block.to_cuda()
                self.weights_stream_mgr.active_weights[0] = (block_idx, block)

            if block_idx < self.blocks_num - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, weights.blocks)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                x = self.infer_block(weights.blocks[block_idx], x, pre_infer_out)

            self.weights_stream_mgr.swap_weights()

            if block_idx == self.blocks_num - 1:
                self.weights_stream_mgr.pin_memory_buffer.pop_front()

            self.weights_stream_mgr._async_prefetch_block(weights.blocks)

        if self.clean_cuda_cache:
            del pre_infer_out.grid_sizes, pre_infer_out.embed0, pre_infer_out.seq_lens, pre_infer_out.freqs, pre_infer_out.context
            torch.cuda.empty_cache()

        return x

    def infer_with_phases_offload(self, weights, x, pre_infer_out):
        for block_idx in range(weights.blocks_num):
            self.block_idx = block_idx
            for phase_idx in range(self.phases_num):
                if block_idx == 0 and phase_idx == 0:
                    phase = weights.blocks[block_idx].compute_phases[phase_idx]
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (phase_idx, phase)

                with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                    cur_phase_idx, cur_phase = self.weights_stream_mgr.active_weights[0]
                    if cur_phase_idx == 0:
                        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_modulation(cur_phase, pre_infer_out.embed0)

                    elif cur_phase_idx == 1:
                        y_out = self.infer_self_attn(
                            cur_phase,
                            pre_infer_out.grid_sizes,
                            x,
                            pre_infer_out.seq_lens,
                            pre_infer_out.freqs,
                            shift_msa,
                            scale_msa,
                        )
                    elif cur_phase_idx == 2:
                        x, attn_out = self.infer_cross_attn(cur_phase, x, pre_infer_out.context, y_out, gate_msa)
                    elif cur_phase_idx == 3:
                        y = self.infer_ffn(cur_phase, x, attn_out, c_shift_msa, c_scale_msa)
                        x = self.post_process(x, y, c_gate_msa, pre_infer_out)

                is_last_phase = block_idx == weights.blocks_num - 1 and phase_idx == self.phases_num - 1
                if not is_last_phase:
                    next_block_idx = block_idx + 1 if phase_idx == self.phases_num - 1 else block_idx
                    next_phase_idx = (phase_idx + 1) % self.phases_num
                    self.weights_stream_mgr.prefetch_phase(next_block_idx, next_phase_idx, weights.blocks)

                self.weights_stream_mgr.swap_phases()

            if self.clean_cuda_cache:
                del attn_out, y_out, y
                torch.cuda.empty_cache()

        if self.clean_cuda_cache:
            del shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa
            del pre_infer_out.grid_sizes, pre_infer_out.embed0, pre_infer_out.seq_lens, pre_infer_out.freqs, pre_infer_out.context
            torch.cuda.empty_cache()

        return x

    def infer_with_phases_lazy_offload(self, weights, x, pre_infer_out):
        self.weights_stream_mgr.prefetch_weights_from_disk(weights.blocks)

        for block_idx in range(weights.blocks_num):
            self.block_idx = block_idx
            for phase_idx in range(self.weights_stream_mgr.phases_num):
                if block_idx == 0 and phase_idx == 0:
                    obj_key = (block_idx, phase_idx)
                    phase = self.weights_stream_mgr.pin_memory_buffer.get(obj_key)
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (obj_key, phase)

                with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                    (
                        (
                            _,
                            cur_phase_idx,
                        ),
                        cur_phase,
                    ) = self.weights_stream_mgr.active_weights[0]

                    if cur_phase_idx == 0:
                        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_modulation(cur_phase, pre_infer_out.embed0)

                    elif cur_phase_idx == 1:
                        y_out = self.infer_self_attn(
                            cur_phase,
                            pre_infer_out.grid_sizes,
                            x,
                            pre_infer_out.seq_lens,
                            pre_infer_out.freqs,
                            shift_msa,
                            scale_msa,
                        )
                    elif cur_phase_idx == 2:
                        x, attn_out = self.infer_cross_attn(cur_phase, x, pre_infer_out.context, y_out, gate_msa)
                    elif cur_phase_idx == 3:
                        y = self.infer_ffn(cur_phase, x, attn_out, c_shift_msa, c_scale_msa)
                        x = self.post_process(x, y, c_gate_msa, pre_infer_out)

                if not (block_idx == weights.blocks_num - 1 and phase_idx == self.phases_num - 1):
                    next_block_idx = block_idx + 1 if phase_idx == self.phases_num - 1 else block_idx
                    next_phase_idx = (phase_idx + 1) % self.weights_stream_mgr.phases_num
                    self.weights_stream_mgr.prefetch_phase(next_block_idx, next_phase_idx, weights.blocks)

                self.weights_stream_mgr.swap_phases()

            self.weights_stream_mgr._async_prefetch_block(weights.blocks)

            if self.clean_cuda_cache:
                del attn_out, y_out, y
                torch.cuda.empty_cache()

        if self.clean_cuda_cache:
            del shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa
            del pre_infer_out.grid_sizes, pre_infer_out.embed0, pre_infer_out.seq_lens, pre_infer_out.freqs, pre_infer_out.context
            torch.cuda.empty_cache()

        return x
