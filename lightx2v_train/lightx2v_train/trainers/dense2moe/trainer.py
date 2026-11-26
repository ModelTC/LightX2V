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

from .checkpoint import latest_checkpoint, publish_checkpoint
from .data import CachedAVDataset
from .encoding import StageTimer
from .loss import BlockFeatures, feature_distillation_loss, modality_losses
from .moe import SparseSwiGLU, taylor_channels
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
        self.dataset = CachedAVDataset(config["prompts"], config["cache_dir"], config["seed"] + self.stage_seed)

    def log(self, values):
        if self.rank == 0:
            print(json.dumps(values), flush=True)
            with (self.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(values) + "\n")

    def checkpoint(self, student, optimizer, step):
        self.log({"stage": self.stage, "event": "save_checkpoint", "step": step})
        options = StateDictOptions(ignore_frozen_params=True)
        model, optim = get_state_dict(student, optimizer, options=options)
        root = self.output / "checkpoints"
        name = f"{self.stage}-{step:08d}"
        temporary = root / ("." + name + ".partial")
        if self.rank == 0:
            root.mkdir(exist_ok=True)
            if temporary.exists():
                shutil.rmtree(temporary)
        dist.barrier()
        dcp.save({"model": model, "optimizer": optim}, checkpoint_id=temporary)
        dist.barrier()
        if self.rank == 0:
            publish_checkpoint(root, temporary, name, {"stage": self.stage, "step": step, "config": self.config}, self.config["save_limit"])
        dist.barrier()

    def resume(self, student, optimizer):
        directory = latest_checkpoint(self.output / "checkpoints", self.stage)
        if directory is None:
            return 0
        metadata = json.loads((directory / "progress.json").read_text())
        if metadata["config"] != self.config:
            raise ValueError("Resume requires the same Dense2MoE recipe and output directory")
        options = StateDictOptions(ignore_frozen_params=True)
        model, optim = get_state_dict(student, optimizer, options=options)
        dcp.load({"model": model, "optimizer": optim}, checkpoint_id=directory)
        set_state_dict(student, optimizer, model_state_dict=model, optim_state_dict=optim, options=options)
        return metadata["step"]

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
        optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=0.01, foreach=False)
        start = self.resume(student, optimizer) if stage != "calibrate" else 0
        scores = {}
        iterations = config["iterations"][stage]
        # Keep uneven model-loading time out of the first training-step timings.
        dist.barrier()
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
            sample = AVSample(condition, record, self.device)
            for name, value in clean.items():
                if value.shape != sample.shapes[name] or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid encoded {name} latent for {record['video_path']}: {tuple(value.shape)}")
            noise = {name: torch.randn_like(value) for name, value in clean.items()}
            base_sigma = torch.rand((), device="cpu")
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
                    student_features.start()
                    prediction = sample.forward(student, noisy, sigmas)
                    features = student_features.stop()
                    errors = modality_losses(prediction, target)
                    output_loss = sum(config["loss_weights"][name] * value for name, value in errors.items())
                    feature_loss = feature_distillation_loss(output_loss, features, reference)
                    auxiliary = [b.ff.balance_loss for b in student.transformer_blocks if isinstance(b.ff, SparseSwiGLU)]
                    balance = torch.stack(auxiliary).mean() if auxiliary else output_loss.new_zeros(())
                    # Router gradients enter locally inside each FFN.
                    loss = output_loss + config["feature_weight"] * feature_loss + config["balance_weight"] * balance
                    metrics = {name + "_loss": float(value.detach()) for name, value in errors.items()}
                    metrics.update(output_loss=float(output_loss.detach()), feature_loss=float(feature_loss.detach()), balance_loss=float(balance.detach()))
                with timer.measure("backward"):
                    loss.backward()
                teacher_features.values.clear()
                student_features.values.clear()
                del target, reference, features, output_loss, feature_loss, auxiliary, balance, errors
                with timer.measure("optimizer"):
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
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
                    "loss": metrics_tensor[0].item(),
                    **dict(zip(metrics, metrics_tensor[1:].tolist())),
                    "seconds": max(item["total"] for item in timings),
                    "peak_gib": max(item["peak_gib"] for item in timings),
                    "timings_by_rank": timings,
                }
            )
            del loss, prediction, sample, clean, noisy, noise, condition
            if stage != "calibrate" and ((step + 1) % config["save_every"] == 0 or step + 1 == iterations):
                self.checkpoint(student, optimizer, step + 1)
        if stage == "calibrate":
            if self.rank == 0:
                torch.save({k: v.argsort(descending=True) for k, v in scores.items()}, self.output / "partition.pt")
                torch.save(scores, self.output / "taylor_scores.pt")
        else:
            self.log({"stage": stage, "event": "export_ffns"})
            save_ffns(student, self.layers, self.output / f"{stage}.pt")
        self.log({"stage": stage, "event": "complete", "steps": iterations})
        dist.barrier()
