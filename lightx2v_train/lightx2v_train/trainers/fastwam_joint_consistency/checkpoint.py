import os

import torch
import yaml

from lightx2v_train.runtime.checkpoint import find_latest_checkpoint, prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size, is_main_process

from .roles import load_role_state_dict, role_state_dict


def _atomic_save(payload, path, *, yaml_output=False):
    temporary = f"{path}.tmp-rank-{get_rank():05d}-pid-{os.getpid()}"
    try:
        if yaml_output:
            with open(temporary, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        else:
            torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class JointConsistencyCheckpointManager:
    def __init__(self, trainer):
        self.trainer = trainer

    def resolve_resume(self):
        resume = self.trainer.config.get("resume", {})
        if resume.get("resume_ckpt_path"):
            path = str(resume["resume_ckpt_path"])
            return path, int(os.path.basename(path).split("-")[-1])
        if not resume.get("auto_resume", False):
            return None, 0
        return find_latest_checkpoint(self.trainer.output_dir)

    def save(self, iteration):
        trainer = self.trainer
        if is_main_process():
            prune_checkpoints(trainer.output_dir, trainer.save_total_limit)
        save_dir = os.path.join(trainer.output_dir, f"checkpoint-{iteration:09d}")
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()

        rng = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state()
        _atomic_save(rng, os.path.join(save_dir, f"rng-rank-{get_rank():05d}.pt"))
        barrier()
        if is_main_process():
            train_type = trainer.parsed.student.train_type
            _atomic_save(role_state_dict(trainer.roles.student, train_type), os.path.join(save_dir, "student_action.pt"))
            _atomic_save(role_state_dict(trainer.roles.target, train_type), os.path.join(save_dir, "ema_action.pt"))
            if trainer.parsed.train_video:
                _atomic_save(role_state_dict(trainer.video_roles.student, train_type), os.path.join(save_dir, "student_video.pt"))
                _atomic_save(role_state_dict(trainer.video_roles.target, train_type), os.path.join(save_dir, "ema_video.pt"))
            _atomic_save(trainer.runtime_config, os.path.join(save_dir, "config.yaml"), yaml_output=True)
            _atomic_save(
                {
                    "iteration": iteration,
                    "world_size": get_world_size(),
                    "student_train_type": train_type,
                    "train_video": trainer.parsed.train_video,
                    "optimizer": trainer.optimizer.state_dict(),
                    "scheduler": trainer.scheduler.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )
        barrier()

    def load(self, checkpoint_dir):
        trainer = self.trainer
        state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location="cpu", weights_only=False)
        if int(state["world_size"]) != get_world_size():
            raise RuntimeError(f"Checkpoint world_size={state['world_size']} does not match current world_size={get_world_size()}.")
        train_type = trainer.parsed.student.train_type
        if state["student_train_type"] != train_type:
            raise RuntimeError("Checkpoint student train type does not match the current configuration.")
        if state.get("train_video", False) != trainer.parsed.train_video:
            raise RuntimeError("Checkpoint train_video does not match the current configuration.")
        roles = [(trainer.roles.student, "student_action.pt"), (trainer.roles.target, "ema_action.pt")]
        if trainer.parsed.train_video:
            roles += [(trainer.video_roles.student, "student_video.pt"), (trainer.video_roles.target, "ema_video.pt")]
        for _, filename in roles:
            if not os.path.isfile(os.path.join(checkpoint_dir, filename)):
                raise RuntimeError(f"Checkpoint is missing role weights: {filename}")
        for role, filename in roles:
            load_role_state_dict(
                role,
                train_type,
                torch.load(os.path.join(checkpoint_dir, filename), map_location="cpu", weights_only=True),
            )
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.scheduler.load_state_dict(state["scheduler"])

        rng_path = os.path.join(checkpoint_dir, f"rng-rank-{get_rank():05d}.pt")
        if not os.path.isfile(rng_path):
            raise RuntimeError(f"Checkpoint RNG state is missing for rank {get_rank()}: {rng_path}")
        rng = torch.load(rng_path, map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        if torch.cuda.is_available() and "cuda" in rng:
            torch.cuda.set_rng_state(rng["cuda"])
        return int(state["iteration"])
