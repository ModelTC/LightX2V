import torch

from lightx2v.models.networks.flux2.infer.transformer_infer import Flux2TransformerInfer


class Flux2TransformerInferCaching(Flux2TransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.must_calc_steps = []
        if self.config.get("changing_resolution", False):
            self.must_calc_steps = self.config["changing_resolution_steps"]

    def must_calc(self, step_index):
        return step_index in self.must_calc_steps


class Flux2AdaArgs:
    def __init__(self, config):
        self.previous_residual_tiny = None
        self.now_residual_tiny = None
        self.norm_ord = 1
        self.skipped_step_length = 1
        self.previous_residual = None

        self.previous_moreg = 1.0
        self.moreg_strides = [1]
        self.moreg_steps = [int(0.1 * config["infer_steps"]), int(0.9 * config["infer_steps"])]
        self.moreg_hyp = [0.385, 8, 1, 2]
        self.mograd_mul = 10
        self.spatial_dim = config.get("adacache_spatial_dim", 0)


class Flux2TransformerInferAdaCaching(Flux2TransformerInferCaching):
    def __init__(self, config):
        super().__init__(config)
        self.decisive_double_block_id = config.get("num_layers", 10) // 2
        self.codebook = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}
        self.args_even = Flux2AdaArgs(config)
        self.args_odd = Flux2AdaArgs(config)

    def infer(self, block_weights, pre_infer_out):
        if self.scheduler.infer_condition:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records

            if caching_records[index] or self.must_calc(index):
                hidden_states = self.infer_calculating(block_weights, pre_infer_out)

                if index <= self.scheduler.infer_steps - 2:
                    self.args_even.skipped_step_length = self.calculate_skip_step_length()
                    for i in range(1, self.args_even.skipped_step_length):
                        if (index + i) <= self.scheduler.infer_steps - 1:
                            self.scheduler.caching_records[index + i] = False
            else:
                hidden_states = self.infer_using_cache(pre_infer_out)
        else:
            index = self.scheduler.step_index
            caching_records = self.scheduler.caching_records_2

            if caching_records[index] or self.must_calc(index):
                hidden_states = self.infer_calculating(block_weights, pre_infer_out)

                if index <= self.scheduler.infer_steps - 2:
                    self.args_odd.skipped_step_length = self.calculate_skip_step_length()
                    for i in range(1, self.args_odd.skipped_step_length):
                        if (index + i) <= self.scheduler.infer_steps - 1:
                            self.scheduler.caching_records_2[index + i] = False
            else:
                hidden_states = self.infer_using_cache(pre_infer_out)

        return hidden_states

    def infer_calculating(self, block_weights, pre_infer_out):
        ori_hidden_states = pre_infer_out.hidden_states.clone()
        ada_args = self.args_even if self.scheduler.infer_condition else self.args_odd

        def on_decisive_block(gated_img_attn):
            ada_args.now_residual_tiny = gated_img_attn.squeeze(0)

        hidden_states = self._infer_forward(
            block_weights,
            pre_infer_out,
            decisive_block_id=self.decisive_double_block_id,
            on_decisive_block=on_decisive_block,
        )

        ada_args.previous_residual = hidden_states - ori_hidden_states
        return hidden_states

    def infer_using_cache(self, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        if self.scheduler.infer_condition:
            hidden_states = hidden_states + self.args_even.previous_residual
        else:
            hidden_states = hidden_states + self.args_odd.previous_residual
        return hidden_states

    def _update_spatial_dim(self, ada_args, residual):
        if ada_args.spatial_dim <= 0:
            ada_args.spatial_dim = residual.shape[0]

    def _calculate_skip_step_length_for_args(self, ada_args):
        if ada_args.previous_residual_tiny is None:
            ada_args.previous_residual_tiny = ada_args.now_residual_tiny
            return 1

        cache = ada_args.previous_residual_tiny
        res = ada_args.now_residual_tiny
        self._update_spatial_dim(ada_args, res)
        norm_ord = ada_args.norm_ord
        cache_diff = (cache - res).norm(dim=(0, 1), p=norm_ord) / cache.norm(dim=(0, 1), p=norm_ord)
        cache_diff = cache_diff / ada_args.skipped_step_length

        if ada_args.moreg_steps[0] <= self.scheduler.step_index <= ada_args.moreg_steps[1]:
            moreg = 0
            for i in ada_args.moreg_strides:
                moreg_i = (res[i * ada_args.spatial_dim :, :] - res[: -i * ada_args.spatial_dim, :]).norm(p=norm_ord)
                moreg_i /= res[i * ada_args.spatial_dim :, :].norm(p=norm_ord) + res[: -i * ada_args.spatial_dim, :].norm(p=norm_ord)
                moreg += moreg_i
            moreg = moreg / len(ada_args.moreg_strides)
            moreg = ((1 / ada_args.moreg_hyp[0] * moreg) ** ada_args.moreg_hyp[1]) / ada_args.moreg_hyp[2]
        else:
            moreg = 1.0

        mograd = ada_args.mograd_mul * (moreg - ada_args.previous_moreg) / ada_args.skipped_step_length
        ada_args.previous_moreg = moreg
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

        ada_args.previous_residual_tiny = ada_args.now_residual_tiny
        return new_rate

    def calculate_skip_step_length(self):
        if self.scheduler.infer_condition:
            return self._calculate_skip_step_length_for_args(self.args_even)
        return self._calculate_skip_step_length_for_args(self.args_odd)

    def clear(self):
        for ada_args in (self.args_even, self.args_odd):
            if ada_args.previous_residual is not None:
                ada_args.previous_residual = ada_args.previous_residual.cpu()
            if ada_args.previous_residual_tiny is not None:
                ada_args.previous_residual_tiny = ada_args.previous_residual_tiny.cpu()
            if ada_args.now_residual_tiny is not None:
                ada_args.now_residual_tiny = ada_args.now_residual_tiny.cpu()

            ada_args.previous_residual = None
            ada_args.previous_residual_tiny = None
            ada_args.now_residual_tiny = None

        torch.cuda.empty_cache()
