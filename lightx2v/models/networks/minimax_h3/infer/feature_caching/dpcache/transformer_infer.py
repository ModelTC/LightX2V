"""DPCache for H3's packed text/audio/video transformer output."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.models.networks.minimax_h3.infer.offload import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer

from .core import calc_cost_matrix_v2, taylor_formula
from .schedule import select_steps


def sample_indices(numel, samples, device):
    # Float32 linspace can round numel-1 UP to numel beyond 2**24.
    count = min(samples, numel)
    return torch.arange(count, device=device, dtype=torch.int64) * (numel - 1) // max(count - 1, 1)


def calibration_signature(config):
    keys = (
        "infer_steps",
        "num_frames",
        "size",
        "fps",
        "video_flow_shift",
        "audio_flow_shift",
        "model_path",
        "model_variant",
        "model-variant",
        "task",
        "dit_quantized_ckpt",
        "attn_type",
        "parallel",
        "dpcache_order",
        "dpcache_samples",
    )
    signature = {key: config.get(key) for key in keys}
    # Sparse attention parameters change the calibrated feature trajectory.
    # Keep existing dense-attention signatures compatible.
    if config.get("attn_type") == "sol_attn":
        signature["sol_attn_setting"] = config.get("sol_attn_setting", {})
        signature["refiner_attn_type"] = config.get("refiner_attn_type")
    return signature


class MiniMaxH3TransformerInferDPCaching(MiniMaxH3TransformerInfer):
    def __init__(self, config):
        self.calibrating = bool(config.get("dpcache_calibration", False))
        self.order = int(config.get("dpcache_order", 2))
        self.samples = int(config.get("dpcache_samples", 32768))
        self.first = int(config.get("dpcache_first_steps", 3))
        self.last = int(config.get("dpcache_last_steps", 1))
        self.steps = int(config["infer_steps"])
        if self.order not in (1, 2) or self.samples < 1 or self.first < self.order + 1 or self.steps < self.first:
            raise ValueError("H3 DPCache needs order 1/2, positive samples, and steps >= first_steps >= order + 1")
        self.cost_path = Path(config["dpcache_cost_path"]).expanduser()
        self.signature = calibration_signature(config)
        if self.calibrating:
            self.selected = list(range(self.steps))
        else:
            with np.load(self.cost_path, allow_pickle=False) as data:
                calibrated = json.loads(str(data["signature"]))
                if calibrated != self.signature:
                    differences = "; ".join(
                        f"{key}: calibrated={calibrated.get(key)!r}, requested={self.signature.get(key)!r}"
                        for key in sorted(calibrated.keys() | self.signature.keys())
                        if key not in calibrated or key not in self.signature or calibrated[key] != self.signature[key]
                    )
                    raise ValueError(
                        f"DPCache calibration config differs from inference ({self.cost_path}): {differences}. "
                        "Run with dpcache_calibration=true using the inference attention settings and a separate dpcache_cost_path."
                    )
                self.selected = select_steps(data["cost"], self.steps, int(config["dpcache_budget"]), self.first, self.last)
        super().__init__(config)
        self._reset()

    def _reset(self):
        self.factors = {}
        self.history = []
        self.previous = None
        self.last_visited = -1
        self.shape = None
        self.sample_indices = None

    def infer(self, block_weights, pre_infer_out):
        step = self.scheduler.step_index
        if step == 0:
            self._reset()
            logger.info("H3 DPCache selected steps ({} / {}): {}", len(self.selected), self.steps, self.selected)
        if self.scheduler.infer_steps != self.steps or step != self.last_visited + 1:
            raise ValueError("DPCache requires the calibrated schedule, visited in order from step zero")
        hidden = pre_infer_out.hidden_states
        if self.shape is not None and self.shape != tuple(hidden.shape):
            raise ValueError("DPCache packed sequence shape changed during denoising")
        self.shape = tuple(hidden.shape)
        # Norm-out modulation must correspond to the current step, even when
        # no transformer blocks execute. Preserve the existing AdaLN contract.
        if self.use_adaln_cache:
            self._prepare_adaln_cache(pre_infer_out)
        if step in self.selected:
            output = self.infer_func(block_weights.blocks, hidden, pre_infer_out)
            if self.calibrating:
                self._record(output, pre_infer_out)
            else:
                updated = {0: output.float().clone()}
                for degree in range(self.order):
                    if degree not in self.factors:
                        break
                    previous = self.factors[degree].to(output.device)
                    updated[degree + 1] = (updated[degree] - previous) / (step - self.previous)
                self.factors = {degree: value.cpu() if self.config.get("cpu_offload", False) else value for degree, value in updated.items()}
                self.previous = step
            action = "compute"
        else:
            factors = {degree: value.to(hidden.device) for degree, value in self.factors.items()}
            output = taylor_formula(factors, step - self.previous).to(hidden.dtype)
            action = "predict"
        self.last_visited = step
        logger.info("H3 DPCache step {}: {}", step, action)
        if step == self.steps - 1:
            if self.calibrating:
                self._save_cost()
            self._reset()
        return output

    def _record(self, output, pre):
        # Use identical coordinates at every step. Exclude replicated SP prefix
        # on nonzero ranks; reduce sampled L1 costs, never gather full features.
        state = pre.sequence_parallel_state
        start = state.aux_length if state is not None and dist.get_rank(self.seq_p_group) != 0 else 0
        flat = output[start:].reshape(-1)
        if self.sample_indices is None:
            self.sample_indices = sample_indices(flat.numel(), self.samples, flat.device)
        self.history.append([flat[self.sample_indices].float().clone(), self.scheduler.step_index])

    def _save_cost(self):
        # Upstream smoothed sentinel, with deterministic mean rather than its
        # negligible 1e-8 random perturbation (does not consume the user's RNG).
        sentinel = torch.stack([entry[0] for entry in self.history[-(self.order + 1) :]]).mean(0)
        self.history.append([sentinel, self.steps])
        cache = {
            "current": {"stream": "joint"},
            "cache_module_list": ["output"],
            "mode": "Taylor-DP",
            "num_steps": self.steps,
            "order": self.order,
            "history": {"joint": {"output": self.history}},
            "model_config": {"clip_fp16": False},
        }
        cost = calc_cost_matrix_v2(cache)[0]
        if self.config.get("seq_parallel", False):
            tensor = torch.from_numpy(cost).to(sentinel.device)
            dist.all_reduce(tensor, group=self.seq_p_group)
            cost = (tensor / dist.get_world_size(self.seq_p_group)).cpu().numpy()
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.cost_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cost_path.open("wb") as target:
                np.savez(target, cost=cost, signature=json.dumps(self.signature, sort_keys=True))
            logger.info("H3 DPCache calibrated cost saved to {}", self.cost_path)

    def clear(self):
        self._reset()
        self._clear_adaln_cache()


class MiniMaxH3OffloadTransformerInferDPCaching(MiniMaxH3TransformerInferDPCaching, MiniMaxH3OffloadTransformerInfer):
    """Keep H3's block/model offload implementation for computed steps."""
