import torch
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.model_zoo.native.wan.fastwam.action_distill import sample_action_one_step, sample_action_teacher
from lightx2v_train.trainers.fastwam_action_consistency.trainer import FastWAMActionConsistencyTrainer
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .config import FastWAMActionTBSMConfig
from .loss import action_scattering_loss


@TRAINER_REGISTER("fastwam_action_tbsm")
class FastWAMActionTBSMTrainer(FastWAMActionConsistencyTrainer):
    """Pure instant TBSM; inherit only action-training lifecycle and evaluation."""

    config_class = FastWAMActionTBSMConfig

    def _training_details(self):
        return f"positive_source={self.parsed.positive_source} teacher_steps={self.parsed.teacher_steps} ema_decay={self.parsed.ema_decay}"

    def _loss(self, inputs, condition, valid_mask):
        action = inputs["action"]
        module = self.model.unwrap_module()
        num_timesteps = module.train_action_scheduler.num_train_timesteps
        online = self.student_denoiser
        if isinstance(online, DistributedDataParallel):
            online = online.module

        # Reference sampling bypasses DDP's reducer and always uses the current
        # student, not EMA. All samples share one frozen observation cache.
        # Do not cache no-grad casts of FP32 student/LoRA weights: the main
        # sample runs with gradients in the same outer autocast scope.
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=action.device.type,
                dtype=torch.get_autocast_dtype(action.device.type),
                enabled=torch.is_autocast_enabled(action.device.type),
                cache_enabled=False,
            ),
        ):
            negative = sample_action_one_step(online, torch.randn_like(action), condition, num_timesteps)
            positive = action
            if self.parsed.positive_source == "teacher":
                positive = sample_action_teacher(
                    self.teacher_denoiser,
                    torch.randn_like(action),
                    condition,
                    module.infer_action_scheduler,
                    self.parsed.teacher_steps,
                )

        projectile = sample_action_one_step(self.student_denoiser, torch.randn_like(action), condition, num_timesteps)
        loss, raw_loss = action_scattering_loss(projectile, positive, negative, valid_mask)
        return loss, {"scattering": raw_loss}
