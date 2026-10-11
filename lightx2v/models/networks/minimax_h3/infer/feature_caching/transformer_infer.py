import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.models.networks.minimax_h3.infer.offload import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class MiniMaxH3TransformerInferMagCaching(MiniMaxH3TransformerInfer):
    """Wan-style MagCache over H3's packed text/audio/video block residual.

    H3 has one guidance-distilled branch. Ratios are a flat list (or a single
    nested list), with a leading 1.0 for the first, always-computed step.
    """

    def __init__(self, config):
        self.enable_magcache_calibration = bool(config.get("magcache_calibration", False))
        self.magcache_thresh = float(config.get("magcache_thresh", 0.24))
        self.K = int(config.get("magcache_K", 6))
        self.retention_ratio = float(config.get("magcache_retention_ratio", 0.2))
        if not math.isfinite(self.magcache_thresh) or self.magcache_thresh < 0 or self.K < 0 or not 0 <= self.retention_ratio <= 1:
            raise ValueError("MiniMax-H3 MagCache requires a finite nonnegative threshold, K >= 0, and retention_ratio in [0, 1]")
        self.mag_ratios = []
        if not self.enable_magcache_calibration:
            ratios = config.get("magcache_ratios", [])
            if not ratios and config.get("magcache_ratios_path"):
                with Path(config["magcache_ratios_path"]).expanduser().open() as source:
                    ratios = json.load(source)
            if len(ratios) == 1 and isinstance(ratios[0], list):
                ratios = ratios[0]
            if not ratios or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in ratios):
                raise ValueError("MiniMax-H3 MagCache requires finite nonnegative magcache_ratios; run tools/cache_minimax_h3_magcache/run_cache_minimax_h3_magcache.sh first")
            self.mag_ratios = list(ratios)
        super().__init__(config)
        self._reset()

    def _reset(self):
        self.accumulated_err = 0.0
        self.accumulated_steps = 0
        self.accumulated_ratio = 1.0
        self.residual_cache = None
        self.norm_ratio = []
        self.norm_std = []
        self.cos_dis = []
        self.last_step = None

    def infer(self, block_weights, pre_infer_out):
        step = self.scheduler.step_index
        steps = self.scheduler.infer_steps
        if step == 0:
            self._reset()
            if not self.enable_magcache_calibration and len(self.mag_ratios) != steps:
                raise ValueError(f"MiniMax-H3 MagCache needs {steps} ratios for this schedule, got {len(self.mag_ratios)}; recalibrate with the inference config")

        # The output heads run at every step and require the CURRENT norm-out
        # modulation even when all transformer blocks are skipped.
        if self.use_adaln_cache:
            self._prepare_adaln_cache(pre_infer_out)
        hidden = pre_infer_out.hidden_states
        compatible = self.residual_cache is not None and self.residual_cache.shape == hidden.shape and self.last_step == step - 1
        skip = False
        if not self.enable_magcache_calibration and compatible and step >= int(steps * self.retention_ratio):
            self.accumulated_ratio *= self.mag_ratios[step]
            self.accumulated_steps += 1
            self.accumulated_err += abs(1 - self.accumulated_ratio)
            skip = self.accumulated_err < self.magcache_thresh and self.accumulated_steps <= self.K

        if skip:
            output = hidden + self.residual_cache.to(device=hidden.device, dtype=hidden.dtype)
            logger.debug("MiniMax-H3 MagCache reusing residual at step {}", step)
        else:
            self.accumulated_err = 0.0
            self.accumulated_steps = 0
            self.accumulated_ratio = 1.0
            original = hidden.clone()
            output = self.infer_func(block_weights.blocks, hidden, pre_infer_out)
            residual = output - original
            if self.enable_magcache_calibration:
                if step != len(self.norm_ratio):
                    raise ValueError("MiniMax-H3 MagCache calibration must visit every step in order")
                stats = self._calibration_stats(residual, pre_infer_out) if compatible else (1.0, 0.0, 0.0)
                self.norm_ratio.append(stats[0])
                self.norm_std.append(stats[1])
                self.cos_dis.append(stats[2])
                logger.info("MiniMax-H3 MagCache step {}: ratio={:.5f}, std={:.5f}, cosine_distance={:.5f}", step, *stats)
            self.residual_cache = residual.cpu() if self.config.get("cpu_offload", False) else residual
        self.last_step = step
        return output

    def _calibration_stats(self, residual, pre_infer_out):
        # Accumulate in chunks to avoid materializing two full FP32 copies of
        # the long joint sequence. Count replicated SP prefix rows only once.
        state = pre_infer_out.sequence_parallel_state
        start = state.aux_length if state is not None and dist.get_rank(self.seq_p_group) != 0 else 0
        totals = torch.zeros(4, dtype=torch.float64, device=residual.device)
        for offset in range(start, residual.shape[0], 1024):
            current = residual[offset : offset + 1024].float()
            previous = self.residual_cache[offset : offset + 1024].to(device=residual.device, dtype=torch.float32)
            current_norm = current.norm(dim=-1)
            previous_norm = previous.norm(dim=-1)
            ratios = current_norm / previous_norm.clamp_min(1e-8)
            ratios = torch.where((current_norm < 1e-8) & (previous_norm < 1e-8), 1.0, ratios).double()
            cosine = (1 - F.cosine_similarity(current, previous, dim=-1, eps=1e-8)).double()
            totals += torch.stack((ratios.sum(), ratios.square().sum(), cosine.sum(), ratios.new_tensor(ratios.numel())))
        if state is not None:
            dist.all_reduce(totals, group=self.seq_p_group)
        ratio_sum, square_sum, cosine_sum, count = totals.tolist()
        mean = ratio_sum / count
        std = math.sqrt(max(0.0, (square_sum - ratio_sum * mean) / max(count - 1, 1)))
        return mean, std, cosine_sum / count

    def clear(self):
        try:
            if self.enable_magcache_calibration and len(self.norm_ratio) == self.scheduler.infer_steps:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    output_dir = Path(self.config.get("magcache_calibration_dir", ".")).expanduser()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for name, values in (("mag_ratio", self.norm_ratio), ("mag_std", self.norm_std), ("cos_dis", self.cos_dis)):
                        path = output_dir / f"minimax_h3_{name}.json"
                        path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
                    logger.info("MiniMax-H3 MagCache calibration saved to {}", output_dir)
        finally:
            self._reset()
            self._clear_adaln_cache()


class MiniMaxH3OffloadTransformerInferMagCaching(MiniMaxH3TransformerInferMagCaching, MiniMaxH3OffloadTransformerInfer):
    """Use the existing model/block offload path for every computed step."""
