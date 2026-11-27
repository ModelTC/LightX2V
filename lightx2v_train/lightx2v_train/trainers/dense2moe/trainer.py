"""H3 T2AV Dense2MoE on cached native video, audio and prompt tensors."""

import gc
import json
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from .checkpoint import latest_checkpoint, load_model_checkpoint, publish_checkpoint
from .data import CachedAVDataset, ManifestAVDataset
from .encoding import StageTimer
from .loss import BlockFeatures, feature_distillation_loss, flow_matching_losses, modality_losses
from .moe import SparseSwiGLU, taylor_channels
from .optimization import learning_rate_factor, optimizer_groups, parameter_samples, sample_base_sigma
from .runtime import AVSample, load_model, save_ffns, shift_sigma


class Dense2MoETrainer:
    def __init__(self, config, stage):
        self.config, self.stage = config, stage
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.output = Path(config["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.layers = config["moe"]["layers"]
        self.stage_seed = {"calibrate": 0, "shared": 100000, "moe": 200000}[stage]
        seed = config["seed"] + self.stage_seed
        if config.get("negative_dir"):
            self.dataset = ManifestAVDataset(config["prompts"], seed, config["negative_dir"])
        else:
            self.dataset = CachedAVDataset(config["prompts"], config["cache_dir"], seed)
        self.flow_weight = config.get("flow_weight", 0.0) if stage in config.get("flow_stages", ("shared", "moe")) else 0.0
        self.cfg_scale = config.get("training_cfg_scale", 1.0) if self.flow_weight else 1.0

    @torch.no_grad()
    def unconditional_velocity(self, student, sample, index, noisy, sigmas):
        if self.cfg_scale <= 1:
            return None
        negative = AVSample(self.dataset.negative(index), self.dataset.records[index], self.device, keyframe=sample.keyframe)
        return negative.forward(student, noisy, sigmas)

    def flow_prediction(self, prediction, uncond):
        if uncond is None:
            return prediction
        return {name: (value.float() + (self.cfg_scale - 1) * uncond[name].float()) / self.cfg_scale
                for name, value in prediction.items()}

    def log(self, values):
        if self.rank == 0:
            print(json.dumps(values), flush=True)
            with (self.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(values) + "\n")

    def checkpoint(self, student, optimizer, step, scores=None):
        self.log({"stage": self.stage, "event": "save_checkpoint", "step": step})
        root = self.output / "checkpoints"
        name = f"{self.stage}-{step:08d}"
        temporary = root / ("." + name + ".partial")
        if self.rank == 0:
            root.mkdir(exist_ok=True)
            if temporary.exists():
                shutil.rmtree(temporary)
        dist.barrier()
        if scores is None:
            options = StateDictOptions(ignore_frozen_params=True)
            model, optim = get_state_dict(student, optimizer, options=options)
            dcp.save({"model": model, "optimizer": optim}, checkpoint_id=temporary)
        elif self.rank == 0:
            temporary.mkdir()
            torch.save(scores, temporary / "scores.pt")
        dist.barrier()
        if self.rank == 0:
            publish_checkpoint(root, temporary, name, {"stage": self.stage, "step": step, "config": self.config}, self.config["save_limit"])
        dist.barrier()

    def resume(self, student, optimizer, scores=None):
        directory = latest_checkpoint(self.output / "checkpoints", self.stage)
        if directory is None:
            if scores is None and self.config.get("initialize_checkpoint"):
                count = load_model_checkpoint(student, self.config["initialize_checkpoint"])
                self.log({"event": "initialize_checkpoint", "parameters": count, "source": self.config["initialize_checkpoint"], "optimizer": "fresh"})
            return 0
        metadata = json.loads((directory / "progress.json").read_text())
        if metadata["config"] != self.config:
            raise ValueError("Resume requires the same Dense2MoE recipe and output directory")
        if scores is not None:
            scores.update(torch.load(directory / "scores.pt", map_location="cpu", weights_only=True))
            return metadata["step"]
        options = StateDictOptions(ignore_frozen_params=True)
        model, optim = get_state_dict(student, optimizer, options=options)
        dcp.load({"model": model, "optimizer": optim}, checkpoint_id=directory)
        set_state_dict(student, optimizer, model_state_dict=model, optim_state_dict=optim, options=options)
        return metadata["step"]

    @torch.no_grad()
    def validate(self, teacher, student, step):
        """Fixed training examples/noise: a comparable diagnostic, not a held-out score."""
        settings = self.config["validation"]
        index = settings["indices"][self.rank % len(settings["indices"])]
        condition, clean = self.dataset.load(index, self.device)
        clean = {name: value.float() for name, value in clean.items()}
        sample = AVSample(condition, self.dataset.records[index], self.device)
        generator = torch.Generator(device=self.device).manual_seed(settings["seed"] + self.rank)
        noise = {name: torch.randn(value.shape, device=self.device, generator=generator) for name, value in clean.items()}
        was_training = student.training
        student.eval()
        for base in settings["base_sigmas"]:
            sigmas = {name: shift_sigma(torch.tensor(base), self.config["flow_shift"][name]) for name in clean}
            noisy = {name: (1 - sigmas[name].to(self.device)) * value + sigmas[name].to(self.device) * noise[name] for name, value in clean.items()}
            target = sample.forward(teacher, noisy, sigmas)
            prediction = sample.forward(student, noisy, sigmas)
            errors = {"kd_" + name: loss for name, loss in modality_losses(prediction, target).items()}
            uncond = self.unconditional_velocity(student, sample, index, noisy, sigmas)
            errors.update({"flow_" + name: loss for name, loss in flow_matching_losses(self.flow_prediction(prediction, uncond), clean, noise).items()})
            errors.update({"teacher_flow_" + name: loss for name, loss in flow_matching_losses(target, clean, noise).items()})
            values = torch.stack(list(errors.values()))
            dist.all_reduce(values)
            self.log({"event": "validation", "stage": self.stage, "step": step, "base_sigma": base, **dict(zip(errors, (values / self.world).tolist()))})
        student.train(was_training)

    def train(self):
        config, stage = self.config, self.stage
        torch.manual_seed(config["seed"])
        permutations = torch.load(self.output / "partition.pt", weights_only=True) if stage != "calibrate" else None
        shared = torch.load(self.output / "shared.pt", weights_only=True) if stage == "moe" else None
        self.log({"stage": stage, "event": "load_teacher"})
        teacher = load_model(config)
        self.log({"stage": stage, "event": "load_student"})
        student = load_model(config, stage, permutations, shared)
        self.log(
            {
                "stage": stage,
                "event": "precision",
                "student_master_dtypes": sorted({str(p.dtype) for p in student.parameters()}),
                "fsdp_compute_dtype": "torch.bfloat16",
                "gradient_reduce_dtype": "torch.float32",
            }
        )
        del shared
        gc.collect()
        teacher_features = BlockFeatures(teacher, self.layers, offload=True)
        student_features = BlockFeatures(student, self.layers)
        params = [p for p in student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(optimizer_groups(student, config), foreach=False)
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        scores = {} if stage == "calibrate" else None
        start = self.resume(student, optimizer, scores)
        iterations = config["iterations"][stage]
        self.log({"event": "optimizer", "groups": [{"name": group["name"], "lr": group["lr"], "tensors": len(group["params"]), "parameters": sum(p.numel() for p in group["params"])} for group in optimizer.param_groups]})
        # Keep uneven model-loading time out of the first training-step timings.
        dist.barrier()
        if stage != "calibrate" and config.get("validation"):
            self.validate(teacher, student, start)
        for step in range(start, iterations):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            timer = StageTimer(self.device)
            torch.manual_seed(config["seed"] + self.stage_seed + step * self.world + self.rank)
            record_index, record = self.dataset.sample(step, self.rank, self.world)
            if self.rank == 0:
                batch = []
                for rank in range(self.world):
                    index, item = self.dataset.sample(step, rank, self.world)
                    batch.append({"rank": rank, "record_index": index, **{key: item[key] for key in ("width", "height", "num_frames")}})
                self.log({"stage": stage, "step": step + 1, "event": "batch", "samples": batch})
            with timer.measure("cache_load"):
                condition, clean = self.dataset.load(record_index, self.device)
                clean = {name: value.float() for name, value in clean.items()}
            sample = AVSample(condition, record, self.device)
            noise = {name: torch.randn_like(value) for name, value in clean.items()}
            base_sigma = sample_base_sigma(config)
            sigmas = {name: shift_sigma(base_sigma, config["flow_shift"][name]) for name in clean}
            noisy = {name: (1 - sigmas[name].to(self.device)) * value + sigmas[name].to(self.device) * noise[name] for name, value in clean.items()}
            if stage == "calibrate":
                with timer.measure("student_forward_loss"):
                    prediction = sample.forward(student, noisy, sigmas)
                    target = {name: value - noise[name] for name, value in clean.items()}
                    errors = modality_losses(prediction, target)
                    loss = sum(config["loss_weights"][name] * value for name, value in errors.items())
                    metrics = {name + "_loss": float(value.detach()) for name, value in errors.items()}
                with timer.measure("backward"):
                    loss.backward()
                for index in self.layers:
                    value = taylor_channels(student.transformer_blocks[index].ff)
                    scores[str(index)] = scores.get(str(index), torch.zeros_like(value)) + value
                del target, errors
            else:
                with timer.measure("teacher_forward"), torch.no_grad():
                    teacher_features.start()
                    target = sample.forward(teacher, noisy, sigmas)
                    reference = teacher_features.stop()
                with timer.measure("student_forward_loss"):
                    uncond = self.unconditional_velocity(student, sample, record_index, noisy, sigmas)
                    student_features.start()
                    prediction = sample.forward(student, noisy, sigmas)
                    features = student_features.stop()
                    errors = modality_losses(prediction, target)
                    output_loss = sum(config["loss_weights"][name] * value for name, value in errors.items())
                    feature_loss = feature_distillation_loss(output_loss, features, reference, config.get("feature_loss", "output_balanced"))
                    flow_errors = flow_matching_losses(self.flow_prediction(prediction, uncond), clean, noise)
                    flow_loss = sum(config["loss_weights"][name] * value for name, value in flow_errors.items())
                    auxiliary = [b.ff.balance_loss for b in student.transformer_blocks if isinstance(b.ff, SparseSwiGLU)]
                    balance = torch.stack(auxiliary).mean() if auxiliary else output_loss.new_zeros(())
                    # Router gradients enter locally inside each FFN.
                    loss = output_loss + config["feature_weight"] * feature_loss + self.flow_weight * flow_loss + config["balance_weight"] * balance
                    metrics = {name + "_loss": float(value.detach()) for name, value in errors.items()}
                    metrics.update(output_loss=float(output_loss.detach()), feature_loss=float(feature_loss.detach()), balance_loss=float(balance.detach()))
                    metrics.update(flow_loss=float(flow_loss.detach()), **{"flow_" + name: float(value.detach()) for name, value in flow_errors.items()})
                    if not torch.isfinite(loss.detach()):
                        raise FloatingPointError(f"Nonfinite {stage} loss at step {step + 1}")
                with timer.measure("backward"):
                    loss.backward()
                teacher_features.values.clear()
                student_features.values.clear()
                del target, reference, features, output_loss, feature_loss, flow_loss, flow_errors, auxiliary, balance, errors
                with timer.measure("optimizer"):
                    norm = torch.nn.utils.clip_grad_norm_(params, config.get("max_grad_norm", 1.0), error_if_nonfinite=True)
                    metrics["grad_norm"] = float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)
                    factor = learning_rate_factor(step, iterations, config.get("warmup_steps", 0), config.get("min_lr_ratio", 1.0))
                    for group, lr in zip(optimizer.param_groups, base_lrs):
                        group["lr"] = lr * factor
                    before = parameter_samples(optimizer)
                    optimizer.step()
                    after = parameter_samples(optimizer)
                    for name in before:
                        metrics[name + "_update_rms"] = float((after[name] - before[name]).square().mean().sqrt())
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Nonfinite {stage} loss at step {step + 1}")
            metrics_tensor = torch.tensor([float(loss.detach()), *metrics.values()], device=self.device)
            dist.all_reduce(metrics_tensor)
            metrics_tensor /= self.world
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            timings = [None] * self.world
            dist.all_gather_object(timings, {"rank": self.rank, **timer.seconds, "total": elapsed, "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
            self.log(
                {
                    "stage": stage,
                    "step": step + 1,
                    "base_sigma_rank0": float(base_sigma),
                    "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                    "loss": metrics_tensor[0].item(),
                    **dict(zip(metrics, metrics_tensor[1:].tolist())),
                    "seconds": max(item["total"] for item in timings),
                    "peak_gib": max(item["peak_gib"] for item in timings),
                    "timings_by_rank": timings,
                }
            )
            del loss, prediction, sample, clean, noisy, noise, condition
            if stage != "calibrate" and config.get("validation") and (step + 1) % config["validation"]["every"] == 0:
                self.validate(teacher, student, step + 1)
            if (step + 1) % config["save_every"] == 0 or step + 1 in config.get("save_steps", []) or step + 1 == iterations:
                self.checkpoint(student, optimizer, step + 1, scores)
        if stage == "calibrate":
            if self.rank == 0:
                torch.save({k: v.argsort(descending=True) for k, v in scores.items()}, self.output / "partition.pt")
                torch.save(scores, self.output / "taylor_scores.pt")
        else:
            self.log({"stage": stage, "event": "export_ffns"})
            save_ffns(student, self.layers, self.output / f"{stage}.pt")
        self.log({"stage": stage, "event": "complete", "steps": iterations})
        dist.barrier()
