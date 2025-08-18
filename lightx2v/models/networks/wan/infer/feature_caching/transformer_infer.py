import gc

import numpy as np
import torch

from lightx2v.common.transformer_infer.transformer_infer import BaseTaylorCachingTransformerInfer

from ..transformer_infer import WanTransformerInfer


class WanTransformerInferCaching(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.must_calc_steps = []
        if self.config.get("changing_resolution", False):
            self.must_calc_steps = self.config["changing_resolution_steps"]

    def must_calc(self, step_index):
        if step_index in self.must_calc_steps:
            return True
        return False


class WanTransformerInferTeaCaching(WanTransformerInferCaching):
    def __init__(self, config):
        super().__init__(config)
        self.teacache_thresh = config.teacache_thresh
        self.accumulated_rel_l1_distance_even = 0
        self.previous_e0_even = None
        self.previous_residual_even = None
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_odd = None
        self.previous_residual_odd = None
        self.use_ret_steps = config.use_ret_steps
        if self.use_ret_steps:
            self.coefficients = self.config.coefficients[0]
            self.ret_steps = 5
            self.cutoff_steps = self.config.infer_steps
        else:
            self.coefficients = self.config.coefficients[1]
            self.ret_steps = 1
            self.cutoff_steps = self.config.infer_steps - 1

    # calculate should_calc
    @torch.no_grad()
    def calculate_should_calc(self, embed, embed0):
        # 1. timestep embedding
        modulated_inp = embed0 if self.use_ret_steps else embed

        # 2. L1 calculate
        should_calc = False
        if self.scheduler.infer_condition:
            if self.scheduler.step_index < self.ret_steps or self.scheduler.step_index >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp - self.previous_e0_even.cuda()).abs().mean() / self.previous_e0_even.cuda().abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()
            if self.config["cpu_offload"]:
                self.previous_e0_even = self.previous_e0_even.cpu()

        else:
            if self.scheduler.step_index < self.ret_steps or self.scheduler.step_index >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp - self.previous_e0_odd.cuda()).abs().mean() / self.previous_e0_odd.cuda().abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

            if self.config["cpu_offload"]:
                self.previous_e0_odd = self.previous_e0_odd.cpu()

        if self.config["cpu_offload"]:
            modulated_inp = modulated_inp.cpu()
            del modulated_inp
            torch.cuda.empty_cache()
            gc.collect()

        if self.clean_cuda_cache:
            del embed, embed0
            torch.cuda.empty_cache()

        # 3. return the judgement
        return should_calc

    def infer_main_blocks(self, weights, pre_infer_out):
        if self.scheduler.infer_condition:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(pre_infer_out.embed, pre_infer_out.embed0)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index] or self.must_calc(index):
                x = self.infer_calculating(weights, pre_infer_out)
            else:
                x = self.infer_using_cache(pre_infer_out.x)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(pre_infer_out.embed, pre_infer_out.embed0)
                self.scheduler.caching_records_2[index] = should_calc

            if caching_records_2[index] or self.must_calc(index):
                x = self.infer_calculating(weights, pre_infer_out)
            else:
                x = self.infer_using_cache(pre_infer_out.x)

        if self.clean_cuda_cache:
            del grid_sizes, embed, embed0, seq_lens, freqs, context
            torch.cuda.empty_cache()

        return x

    def infer_calculating(self, weights, pre_infer_out):
        ori_x = pre_infer_out.x.clone()

        x = super().infer_main_blocks(weights, pre_infer_out)
        if self.scheduler.infer_condition:
            self.previous_residual_even = x - ori_x
            if self.config["cpu_offload"]:
                self.previous_residual_even = self.previous_residual_even.cpu()
        else:
            self.previous_residual_odd = x - ori_x
            if self.config["cpu_offload"]:
                self.previous_residual_odd = self.previous_residual_odd.cpu()

        if self.config["cpu_offload"]:
            ori_x = ori_x.to("cpu")
            del ori_x
            torch.cuda.empty_cache()
            gc.collect()
        return x

    def infer_using_cache(self, x):
        if self.scheduler.infer_condition:
            x.add_(self.previous_residual_even.cuda())
        else:
            x.add_(self.previous_residual_odd.cuda())
        return x

    def clear(self):
        if self.previous_residual_even is not None:
            self.previous_residual_even = self.previous_residual_even.cpu()
        if self.previous_residual_odd is not None:
            self.previous_residual_odd = self.previous_residual_odd.cpu()
        if self.previous_e0_even is not None:
            self.previous_e0_even = self.previous_e0_even.cpu()
        if self.previous_e0_odd is not None:
            self.previous_e0_odd = self.previous_e0_odd.cpu()

        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.previous_e0_even = None
        self.previous_e0_odd = None

        torch.cuda.empty_cache()


class WanTransformerInferTaylorCaching(WanTransformerInferCaching, BaseTaylorCachingTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

        self.blocks_cache_even = [{} for _ in range(self.blocks_num)]
        self.blocks_cache_odd = [{} for _ in range(self.blocks_num)]

    # 1. get taylor step_diff when there is two caching_records in scheduler
    def get_taylor_step_diff(self):
        step_diff = 0
        if self.infer_conditional:
            current_step = self.scheduler.step_index
            last_calc_step = current_step - 1
            while last_calc_step >= 0 and not self.scheduler.caching_records[last_calc_step]:
                last_calc_step -= 1
            step_diff = current_step - last_calc_step
        else:
            current_step = self.scheduler.step_index
            last_calc_step = current_step - 1
            while last_calc_step >= 0 and not self.scheduler.caching_records_2[last_calc_step]:
                last_calc_step -= 1
            step_diff = current_step - last_calc_step

        return step_diff

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records

            if caching_records[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2

            if caching_records_2[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        if self.config.enable_cfg:
            self.switch_status()

        return x

    def infer_calculating(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_modulation(weights.blocks[block_idx].compute_phases[0], embed0)

            y_out = self.infer_self_attn(weights.blocks[block_idx].compute_phases[1], grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa)
            if self.infer_conditional:
                self.derivative_approximation(self.blocks_cache_even[block_idx], "self_attn_out", y_out)
            else:
                self.derivative_approximation(self.blocks_cache_odd[block_idx], "self_attn_out", y_out)

            x, attn_out = self.infer_cross_attn(weights.blocks[block_idx].compute_phases[2], x, context, y_out, gate_msa)
            if self.infer_conditional:
                self.derivative_approximation(self.blocks_cache_even[block_idx], "cross_attn_out", attn_out)
            else:
                self.derivative_approximation(self.blocks_cache_odd[block_idx], "cross_attn_out", attn_out)

            y_out = self.infer_ffn(weights.blocks[block_idx].compute_phases[3], x, attn_out, c_shift_msa, c_scale_msa)
            if self.infer_conditional:
                self.derivative_approximation(self.blocks_cache_even[block_idx], "ffn_out", y_out)
            else:
                self.derivative_approximation(self.blocks_cache_odd[block_idx], "ffn_out", y_out)

            x = self.post_process(x, y_out, c_gate_msa)
        return x

    def infer_using_cache(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            x = self.infer_block(weights.blocks[block_idx], grid_sizes, embed, x, embed0, seq_lens, freqs, context, block_idx)
        return x

    # 1. taylor using caching
    def infer_block(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, i):
        # 1. shift, scale, gate
        _, _, gate_msa, _, _, c_gate_msa = self.infer_modulation(weights.compute_phases[0], embed0)

        # 2. residual and taylor
        if self.infer_conditional:
            out = self.taylor_formula(self.blocks_cache_even[i]["self_attn_out"])
            out = out * gate_msa.squeeze(0)
            x = x + out

            out = self.taylor_formula(self.blocks_cache_even[i]["cross_attn_out"])
            x = x + out

            out = self.taylor_formula(self.blocks_cache_even[i]["ffn_out"])
            out = out * c_gate_msa.squeeze(0)
            x = x + out

        else:
            out = self.taylor_formula(self.blocks_cache_odd[i]["self_attn_out"])
            out = out * gate_msa.squeeze(0)
            x = x + out

            out = self.taylor_formula(self.blocks_cache_odd[i]["cross_attn_out"])
            x = x + out

            out = self.taylor_formula(self.blocks_cache_odd[i]["ffn_out"])
            out = out * c_gate_msa.squeeze(0)
            x = x + out

        return x

    def clear(self):
        for cache in self.blocks_cache_even:
            for key in cache:
                if cache[key] is not None:
                    if isinstance(cache[key], torch.Tensor):
                        cache[key] = cache[key].cpu()
                    elif isinstance(cache[key], dict):
                        for k, v in cache[key].items():
                            if isinstance(v, torch.Tensor):
                                cache[key][k] = v.cpu()
            cache.clear()

        for cache in self.blocks_cache_odd:
            for key in cache:
                if cache[key] is not None:
                    if isinstance(cache[key], torch.Tensor):
                        cache[key] = cache[key].cpu()
                    elif isinstance(cache[key], dict):
                        for k, v in cache[key].items():
                            if isinstance(v, torch.Tensor):
                                cache[key][k] = v.cpu()
            cache.clear()
        torch.cuda.empty_cache()


class WanTransformerInferAdaCaching(WanTransformerInferCaching):
    def __init__(self, config):
        super().__init__(config)

        # 1. fixed args
        self.decisive_double_block_id = self.blocks_num // 2
        self.codebook = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}

        # 2. Create two instances of AdaArgs
        self.args_even = AdaArgs(config)
        self.args_odd = AdaArgs(config)

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records

            if caching_records[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

                # 1. calculate the skipped step length
                if index <= self.scheduler.infer_steps - 2:
                    self.args_even.skipped_step_length = self.calculate_skip_step_length()
                    for i in range(1, self.args_even.skipped_step_length):
                        if (index + i) <= self.scheduler.infer_steps - 1:
                            self.scheduler.caching_records[index + i] = False
            else:
                x = self.infer_using_cache(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

        else:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records_2

            if caching_records[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

                # 1. calculate the skipped step length
                if index <= self.scheduler.infer_steps - 2:
                    self.args_odd.skipped_step_length = self.calculate_skip_step_length()
                    for i in range(1, self.args_odd.skipped_step_length):
                        if (index + i) <= self.scheduler.infer_steps - 1:
                            self.scheduler.caching_records_2[index + i] = False
            else:
                x = self.infer_using_cache(xt)

        if self.config.enable_cfg:
            self.switch_status()

        return x

    def infer_calculating(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(self.blocks_num):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_modulation(weights.blocks[block_idx].compute_phases[0], embed0)

            y_out = self.infer_self_attn(weights.blocks[block_idx].compute_phases[1], grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa)
            if block_idx == self.decisive_double_block_id:
                if self.infer_conditional:
                    self.args_even.now_residual_tiny = y_out * gate_msa.squeeze(0)
                else:
                    self.args_odd.now_residual_tiny = y_out * gate_msa.squeeze(0)

            x, attn_out = self.infer_cross_attn(weights.blocks[block_idx].compute_phases[2], x, context, y_out, gate_msa)
            y_out = self.infer_ffn(weights.blocks[block_idx].compute_phases[3], x, attn_out, c_shift_msa, c_scale_msa)
            x = self.post_process(x, y_out, c_gate_msa)

        if self.infer_conditional:
            self.args_even.previous_residual = x - ori_x
        else:
            self.args_odd.previous_residual = x - ori_x
        return x

    def infer_using_cache(self, x):
        if self.infer_conditional:
            x += self.args_even.previous_residual
        else:
            x += self.args_odd.previous_residual
        return x

    def calculate_skip_step_length(self):
        if self.infer_conditional:
            if self.args_even.previous_residual_tiny is None:
                self.args_even.previous_residual_tiny = self.args_even.now_residual_tiny
                return 1
            else:
                cache = self.args_even.previous_residual_tiny
                res = self.args_even.now_residual_tiny
                norm_ord = self.args_even.norm_ord
                cache_diff = (cache - res).norm(dim=(0, 1), p=norm_ord) / cache.norm(dim=(0, 1), p=norm_ord)
                cache_diff = cache_diff / self.args_even.skipped_step_length

                if self.args_even.moreg_steps[0] <= self.scheduler.step_index <= self.args_even.moreg_steps[1]:
                    moreg = 0
                    for i in self.args_even.moreg_strides:
                        moreg_i = (res[i * self.args_even.spatial_dim :, :] - res[: -i * self.args_even.spatial_dim, :]).norm(p=norm_ord)
                        moreg_i /= res[i * self.args_even.spatial_dim :, :].norm(p=norm_ord) + res[: -i * self.args_even.spatial_dim, :].norm(p=norm_ord)
                        moreg += moreg_i
                    moreg = moreg / len(self.args_even.moreg_strides)
                    moreg = ((1 / self.args_even.moreg_hyp[0] * moreg) ** self.args_even.moreg_hyp[1]) / self.args_even.moreg_hyp[2]
                else:
                    moreg = 1.0

                mograd = self.args_even.mograd_mul * (moreg - self.args_even.previous_moreg) / self.args_even.skipped_step_length
                self.args_even.previous_moreg = moreg
                moreg = moreg + abs(mograd)
                cache_diff = cache_diff * moreg

                metric_thres, cache_rates = list(self.codebook.keys()), list(self.codebook.values())
                if cache_diff < metric_thres[0]:
                    new_rate = cache_rates[0]
                elif cache_diff < metric_thres[1]:
                    new_rate = cache_rates[1]
                elif cache_diff < metric_thres[2]:
                    new_rate = cache_rates[2]
                elif cache_diff < metric_thres[3]:
                    new_rate = cache_rates[3]
                elif cache_diff < metric_thres[4]:
                    new_rate = cache_rates[4]
                else:
                    new_rate = cache_rates[-1]

                self.args_even.previous_residual_tiny = self.args_even.now_residual_tiny
                return new_rate

        else:
            if self.args_odd.previous_residual_tiny is None:
                self.args_odd.previous_residual_tiny = self.args_odd.now_residual_tiny
                return 1
            else:
                cache = self.args_odd.previous_residual_tiny
                res = self.args_odd.now_residual_tiny
                norm_ord = self.args_odd.norm_ord
                cache_diff = (cache - res).norm(dim=(0, 1), p=norm_ord) / cache.norm(dim=(0, 1), p=norm_ord)
                cache_diff = cache_diff / self.args_odd.skipped_step_length

                if self.args_odd.moreg_steps[0] <= self.scheduler.step_index <= self.args_odd.moreg_steps[1]:
                    moreg = 0
                    for i in self.args_odd.moreg_strides:
                        moreg_i = (res[i * self.args_odd.spatial_dim :, :] - res[: -i * self.args_odd.spatial_dim, :]).norm(p=norm_ord)
                        moreg_i /= res[i * self.args_odd.spatial_dim :, :].norm(p=norm_ord) + res[: -i * self.args_odd.spatial_dim, :].norm(p=norm_ord)
                        moreg += moreg_i
                    moreg = moreg / len(self.args_odd.moreg_strides)
                    moreg = ((1 / self.args_odd.moreg_hyp[0] * moreg) ** self.args_odd.moreg_hyp[1]) / self.args_odd.moreg_hyp[2]
                else:
                    moreg = 1.0

                mograd = self.args_odd.mograd_mul * (moreg - self.args_odd.previous_moreg) / self.args_odd.skipped_step_length
                self.args_odd.previous_moreg = moreg
                moreg = moreg + abs(mograd)
                cache_diff = cache_diff * moreg

                metric_thres, cache_rates = list(self.codebook.keys()), list(self.codebook.values())
                if cache_diff < metric_thres[0]:
                    new_rate = cache_rates[0]
                elif cache_diff < metric_thres[1]:
                    new_rate = cache_rates[1]
                elif cache_diff < metric_thres[2]:
                    new_rate = cache_rates[2]
                elif cache_diff < metric_thres[3]:
                    new_rate = cache_rates[3]
                elif cache_diff < metric_thres[4]:
                    new_rate = cache_rates[4]
                else:
                    new_rate = cache_rates[-1]

                self.args_odd.previous_residual_tiny = self.args_odd.now_residual_tiny
                return new_rate

    def clear(self):
        if self.args_even.previous_residual is not None:
            self.args_even.previous_residual = self.args_even.previous_residual.cpu()
        if self.args_even.previous_residual_tiny is not None:
            self.args_even.previous_residual_tiny = self.args_even.previous_residual_tiny.cpu()
        if self.args_even.now_residual_tiny is not None:
            self.args_even.now_residual_tiny = self.args_even.now_residual_tiny.cpu()

        if self.args_odd.previous_residual is not None:
            self.args_odd.previous_residual = self.args_odd.previous_residual.cpu()
        if self.args_odd.previous_residual_tiny is not None:
            self.args_odd.previous_residual_tiny = self.args_odd.previous_residual_tiny.cpu()
        if self.args_odd.now_residual_tiny is not None:
            self.args_odd.now_residual_tiny = self.args_odd.now_residual_tiny.cpu()

        self.args_even.previous_residual = None
        self.args_even.previous_residual_tiny = None
        self.args_even.now_residual_tiny = None

        self.args_odd.previous_residual = None
        self.args_odd.previous_residual_tiny = None
        self.args_odd.now_residual_tiny = None

        torch.cuda.empty_cache()


class AdaArgs:
    def __init__(self, config):
        # Cache related attributes
        self.previous_residual_tiny = None
        self.now_residual_tiny = None
        self.norm_ord = 1
        self.skipped_step_length = 1
        self.previous_residual = None

        # Moreg related attributes
        self.previous_moreg = 1.0
        self.moreg_strides = [1]
        self.moreg_steps = [int(0.1 * config.infer_steps), int(0.9 * config.infer_steps)]
        self.moreg_hyp = [0.385, 8, 1, 2]
        self.mograd_mul = 10
        self.spatial_dim = 1536


class WanTransformerInferCustomCaching(WanTransformerInferCaching, BaseTaylorCachingTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.cnt = 0
        self.teacache_thresh = config.teacache_thresh
        self.accumulated_rel_l1_distance_even = 0
        self.previous_e0_even = None
        self.previous_residual_even = None
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_odd = None
        self.previous_residual_odd = None
        self.cache_even = {}
        self.cache_odd = {}
        self.use_ret_steps = config.use_ret_steps
        if self.use_ret_steps:
            self.coefficients = self.config.coefficients[0]
            self.ret_steps = 5 * 2
            self.cutoff_steps = self.config.infer_steps * 2
        else:
            self.coefficients = self.config.coefficients[1]
            self.ret_steps = 1 * 2
            self.cutoff_steps = self.config.infer_steps * 2 - 2

    # 1. get taylor step_diff when there is two caching_records in scheduler
    def get_taylor_step_diff(self):
        step_diff = 0
        if self.infer_conditional:
            current_step = self.scheduler.step_index
            last_calc_step = current_step - 1
            while last_calc_step >= 0 and not self.scheduler.caching_records[last_calc_step]:
                last_calc_step -= 1
            step_diff = current_step - last_calc_step
        else:
            current_step = self.scheduler.step_index
            last_calc_step = current_step - 1
            while last_calc_step >= 0 and not self.scheduler.caching_records_2[last_calc_step]:
                last_calc_step -= 1
            step_diff = current_step - last_calc_step

        return step_diff

    # calculate should_calc
    def calculate_should_calc(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        # 1. timestep embedding
        modulated_inp = embed0 if self.use_ret_steps else embed

        # 2. L1 calculate
        should_calc = False
        if self.infer_conditional:
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()

        else:
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp - self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

        # 3. return the judgement
        return should_calc

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(x)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                self.scheduler.caching_records_2[index] = should_calc

            if caching_records_2[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(x)

        if self.config.enable_cfg:
            self.switch_status()

        self.cnt += 1

        return x

    def infer_calculating(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(self.blocks_num):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_modulation(weights.blocks[block_idx].compute_phases[0], embed0)

            y_out = self.infer_self_attn(weights.blocks[block_idx].compute_phases[1], grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa)
            x, attn_out = self.infer_cross_attn(weights.blocks[block_idx].compute_phases[2], x, context, y_out, gate_msa)
            y_out = self.infer_ffn(weights.blocks[block_idx].compute_phases[3], x, attn_out, c_shift_msa, c_scale_msa)
            x = self.post_process(x, y_out, c_gate_msa)

        if self.infer_conditional:
            self.previous_residual_even = x - ori_x
            self.derivative_approximation(self.cache_even, "previous_residual", self.previous_residual_even)
        else:
            self.previous_residual_odd = x - ori_x
            self.derivative_approximation(self.cache_odd, "previous_residual", self.previous_residual_odd)
        return x

    def infer_using_cache(self, x):
        if self.infer_conditional:
            x += self.taylor_formula(self.cache_even["previous_residual"])
        else:
            x += self.taylor_formula(self.cache_odd["previous_residual"])
        return x

    def clear(self):
        if self.previous_residual_even is not None:
            self.previous_residual_even = self.previous_residual_even.cpu()
        if self.previous_residual_odd is not None:
            self.previous_residual_odd = self.previous_residual_odd.cpu()
        if self.previous_e0_even is not None:
            self.previous_e0_even = self.previous_e0_even.cpu()
        if self.previous_e0_odd is not None:
            self.previous_e0_odd = self.previous_e0_odd.cpu()

        for key in self.cache_even:
            if self.cache_even[key] is not None and hasattr(self.cache_even[key], "cpu"):
                self.cache_even[key] = self.cache_even[key].cpu()
        self.cache_even.clear()

        for key in self.cache_odd:
            if self.cache_odd[key] is not None and hasattr(self.cache_odd[key], "cpu"):
                self.cache_odd[key] = self.cache_odd[key].cpu()
        self.cache_odd.clear()

        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.previous_e0_even = None
        self.previous_e0_odd = None

        torch.cuda.empty_cache()


class WanTransformerInferFirstBlock(WanTransformerInferCaching):
    def __init__(self, config):
        super().__init__(config)

        self.residual_diff_threshold = config.residual_diff_threshold
        self.prev_first_block_residual_even = None
        self.prev_remaining_blocks_residual_even = None
        self.prev_first_block_residual_odd = None
        self.prev_remaining_blocks_residual_odd = None
        self.downsample_factor = self.config.downsample_factor

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()
        x = super().infer_block(weights.blocks[0], grid_sizes, embed, x, embed0, seq_lens, freqs, context)
        x_residual = x - ori_x
        del ori_x

        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(x_residual)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(x)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(x_residual)
                self.scheduler.caching_records_2[index] = should_calc

            if caching_records_2[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(x)

        if self.config.enable_cfg:
            self.switch_status()

        return x

    def calculate_should_calc(self, x_residual):
        diff = 1.0
        x_residual_downsampled = x_residual[..., :: self.downsample_factor]
        if self.infer_conditional:
            if self.prev_first_block_residual_even is not None:
                t1 = self.prev_first_block_residual_even
                t2 = x_residual_downsampled
                mean_diff = (t1 - t2).abs().mean()
                mean_t1 = t1.abs().mean()
                diff = (mean_diff / mean_t1).item()
            self.prev_first_block_residual_even = x_residual_downsampled
        else:
            if self.prev_first_block_residual_odd is not None:
                t1 = self.prev_first_block_residual_odd
                t2 = x_residual_downsampled
                mean_diff = (t1 - t2).abs().mean()
                mean_t1 = t1.abs().mean()
                diff = (mean_diff / mean_t1).item()
            self.prev_first_block_residual_odd = x_residual_downsampled

        return diff >= self.residual_diff_threshold

    def infer_calculating(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(1, self.blocks_num):
            x = super().infer_block(
                weights.blocks[block_idx],
                grid_sizes,
                embed,
                x,
                embed0,
                seq_lens,
                freqs,
                context,
            )

        if self.infer_conditional:
            self.prev_remaining_blocks_residual_even = x - ori_x
        else:
            self.prev_remaining_blocks_residual_odd = x - ori_x
        del ori_x

        return x

    def infer_using_cache(self, x):
        if self.infer_conditional:
            return x.add_(self.prev_remaining_blocks_residual_even)
        else:
            return x.add_(self.prev_remaining_blocks_residual_odd)

    def clear(self):
        self.prev_first_block_residual_even = None
        self.prev_remaining_blocks_residual_even = None
        self.prev_first_block_residual_odd = None
        self.prev_remaining_blocks_residual_odd = None
        torch.cuda.empty_cache()


class WanTransformerInferDualBlock(WanTransformerInferCaching):
    def __init__(self, config):
        super().__init__(config)

        self.residual_diff_threshold = config.residual_diff_threshold
        self.prev_front_blocks_residual_even = None
        self.prev_middle_blocks_residual_even = None
        self.prev_front_blocks_residual_odd = None
        self.prev_middle_blocks_residual_odd = None
        self.downsample_factor = self.config.downsample_factor

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()
        for block_idx in range(0, 5):
            x = super().infer_block(
                weights.blocks[block_idx],
                grid_sizes,
                embed,
                x,
                embed0,
                seq_lens,
                freqs,
                context,
            )
        x_residual = x - ori_x
        del ori_x

        if self.infer_conditional:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(x_residual)
                self.scheduler.caching_records[index] = should_calc

            if caching_records[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(x)

        else:
            index = self.scheduler.step_index
            caching_records_2 = self.scheduler.caching_records_2
            if index <= self.scheduler.infer_steps - 1:
                should_calc = self.calculate_should_calc(x_residual)
                self.scheduler.caching_records_2[index] = should_calc

            if caching_records_2[index] or self.must_calc(index):
                x = self.infer_calculating(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
            else:
                x = self.infer_using_cache(x)

        for block_idx in range(self.blocks_num - 5, self.blocks_num):
            x = super().infer_block(
                weights.blocks[block_idx],
                grid_sizes,
                embed,
                x,
                embed0,
                seq_lens,
                freqs,
                context,
            )

        if self.config.enable_cfg:
            self.switch_status()

        return x

    def calculate_should_calc(self, x_residual):
        diff = 1.0
        x_residual_downsampled = x_residual[..., :: self.downsample_factor]
        if self.infer_conditional:
            if self.prev_front_blocks_residual_even is not None:
                t1 = self.prev_front_blocks_residual_even
                t2 = x_residual_downsampled
                mean_diff = (t1 - t2).abs().mean()
                mean_t1 = t1.abs().mean()
                diff = (mean_diff / mean_t1).item()
            self.prev_front_blocks_residual_even = x_residual_downsampled
        else:
            if self.prev_front_blocks_residual_odd is not None:
                t1 = self.prev_front_blocks_residual_odd
                t2 = x_residual_downsampled
                mean_diff = (t1 - t2).abs().mean()
                mean_t1 = t1.abs().mean()
                diff = (mean_diff / mean_t1).item()
            self.prev_front_blocks_residual_odd = x_residual_downsampled

        return diff >= self.residual_diff_threshold

    def infer_calculating(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        ori_x = x.clone()

        for block_idx in range(5, self.blocks_num - 5):
            x = super().infer_block(
                weights.blocks[block_idx],
                grid_sizes,
                embed,
                x,
                embed0,
                seq_lens,
                freqs,
                context,
            )

        if self.infer_conditional:
            self.prev_middle_blocks_residual_even = x - ori_x
        else:
            self.prev_middle_blocks_residual_odd = x - ori_x
        del ori_x

        return x

    def infer_using_cache(self, x):
        if self.infer_conditional:
            return x.add_(self.prev_middle_blocks_residual_even)
        else:
            return x.add_(self.prev_middle_blocks_residual_odd)

    def clear(self):
        self.prev_front_blocks_residual_even = None
        self.prev_middle_blocks_residual_even = None
        self.prev_front_blocks_residual_odd = None
        self.prev_middle_blocks_residual_odd = None
        torch.cuda.empty_cache()


class WanTransformerInferDynamicBlock(WanTransformerInferCaching):
    def __init__(self, config):
        super().__init__(config)
        self.residual_diff_threshold = config.residual_diff_threshold
        self.downsample_factor = self.config.downsample_factor

        self.block_in_cache_even = {i: None for i in range(self.blocks_num)}
        self.block_residual_cache_even = {i: None for i in range(self.blocks_num)}
        self.block_in_cache_odd = {i: None for i in range(self.blocks_num)}
        self.block_residual_cache_odd = {i: None for i in range(self.blocks_num)}

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            x = self.infer_block(weights.blocks[block_idx], grid_sizes, embed, x, embed0, seq_lens, freqs, context, block_idx)

        return x

    def infer_block(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, block_idx):
        ori_x = x.clone()

        if self.infer_conditional:
            if self.block_in_cache_even[block_idx] is not None:
                should_calc = self.are_two_tensor_similar(self.block_in_cache_even[block_idx], x)
                if should_calc or self.must_calc(block_idx):
                    x = super().infer_block(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                else:
                    x += self.block_residual_cache_even[block_idx]

            else:
                x = super().infer_block(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

            self.block_in_cache_even[block_idx] = ori_x
            self.block_residual_cache_even[block_idx] = x - ori_x
            del ori_x

        else:
            if self.block_in_cache_odd[block_idx] is not None:
                should_calc = self.are_two_tensor_similar(self.block_in_cache_odd[block_idx], x)
                if should_calc or self.must_calc(block_idx):
                    x = super().infer_block(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)
                else:
                    x += self.block_residual_cache_odd[block_idx]

            else:
                x = super().infer_block(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

            self.block_in_cache_odd[block_idx] = ori_x
            self.block_residual_cache_odd[block_idx] = x - ori_x
            del ori_x

        return x

    def are_two_tensor_similar(self, t1, t2):
        diff = 1.0
        t1_downsampled = t1[..., :: self.downsample_factor]
        t2_downsampled = t2[..., :: self.downsample_factor]
        mean_diff = (t1_downsampled - t2_downsampled).abs().mean()
        mean_t1 = t1_downsampled.abs().mean()
        diff = (mean_diff / mean_t1).item()

        return diff >= self.residual_diff_threshold

    def clear(self):
        for i in range(self.blocks_num):
            self.block_in_cache_even[i] = None
            self.block_residual_cache_even[i] = None
            self.block_in_cache_odd[i] = None
            self.block_residual_cache_odd[i] = None
        torch.cuda.empty_cache()


class WanTransformerInferMagCaching(WanTransformerInferCaching):
    def __init__(self, config):
        super().__init__(config)
        self.cnt = 0
        self.num_steps = config.infer_steps * 2
        self.magcache_thresh = config.magcache_thresh
        self.K = config.magcache_K
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
        self.accumulated_ratio = [1.0, 1.0]
        self.retention_ratio = config.magcache_retention_ratio
        self.residual_cache = [None, None]
        self.mag_ratios = np.array(config.magcache_ratios)

    def infer_main_blocks(self, weights, pre_infer_out):
        skip_forward = False
        if self.cnt >= int(self.num_steps * self.retention_ratio):
            # conditional and unconditional in one list
            cur_mag_ratio = self.mag_ratios[self.cnt]
            # magnitude ratio between current step and the cached step
            self.accumulated_ratio[self.cnt % 2] = self.accumulated_ratio[self.cnt % 2] * cur_mag_ratio
            self.accumulated_steps[self.cnt % 2] += 1  # skip steps plus 1
            # skip error of current steps
            cur_skip_err = np.abs(1 - self.accumulated_ratio[self.cnt % 2])
            # accumulated error of multiple steps
            self.accumulated_err[self.cnt % 2] += cur_skip_err

            if self.accumulated_err[self.cnt % 2] < self.magcache_thresh and self.accumulated_steps[self.cnt % 2] <= self.K:
                skip_forward = True
            else:
                self.accumulated_err[self.cnt % 2] = 0
                self.accumulated_steps[self.cnt % 2] = 0
                self.accumulated_ratio[self.cnt % 2] = 1.0

        if not skip_forward:
            x = self.infer_calculating(weights, pre_infer_out)
        else:
            x = self.infer_using_cache(pre_infer_out.x)

        if self.config.enable_cfg:
            self.switch_status()

        self.cnt += 1

        if self.cnt >= self.num_steps:  # clear the history of current video and prepare for generating the next video.
            self.cnt = 0
            self.accumulated_ratio = [1.0, 1.0]
            self.accumulated_err = [0.0, 0.0]
            self.accumulated_steps = [0, 0]

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return x

    def infer_calculating(self, weights, pre_infer_out):
        ori_x = pre_infer_out.x.clone()

        x = super().infer_main_blocks(weights, pre_infer_out)

        previous_residual = x - ori_x
        if self.config["cpu_offload"]:
            previous_residual = previous_residual.cpu()

        self.residual_cache[self.cnt % 2] = previous_residual

        if self.config["cpu_offload"]:
            ori_x = ori_x.to("cpu")
            del ori_x
            torch.cuda.empty_cache()
            gc.collect()
        return x

    def infer_using_cache(self, x):
        residual_x = self.residual_cache[self.cnt % 2]
        x.add_(residual_x.cuda())
        return x

    def clear(self):
        self.residual_cache = [None, None]
        torch.cuda.empty_cache()
