"""VAE component distillation with an optional adversarial objective."""

import hashlib
import json

from loguru import logger

from lightx2v_train.model_capabilities import (
    VAEDistillationCapability,
    VAEDistillationStepContext,
)
from lightx2v_train.runtime.distributed import is_sequence_parallel_enabled
from lightx2v_train.utils.registry import TRAINER_REGISTER

from ..base import BaseTrainer
from ..optimizer import OptimizerTrainer
from .adversarial import VAEAdversarialObjective


@TRAINER_REGISTER("vae_distillation")
class VAEDistillationTrainer(OptimizerTrainer):
    trainer_name = "vae_distillation"
    required_capabilities = (
        *BaseTrainer.required_capabilities,
        VAEDistillationCapability,
    )

    def __init__(self, config):
        super().__init__(config)
        if self.train_type != "full":
            raise ValueError("VAE distillation requires training.train_type='full'.")
        distillation_config = self.training_config.get("vae_distillation", {})
        serialized_config = json.dumps(distillation_config, sort_keys=True, separators=(",", ":"))
        self.schedule_fingerprint = hashlib.sha256(serialized_config.encode("utf-8")).hexdigest()
        self.stage_iteration_offset = int(self.training_config.get("vae_distillation_stage_iteration_offset", 0))
        self.adversarial = None
        self._micro_step = 0

    def set_model(self, model):
        super().set_model(model)
        self.vae_distillation = model.capabilities.require(VAEDistillationCapability)

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path=None)
        adversarial_config = self.training_config.get("vae_distillation", {}).get("gan", {})
        if adversarial_config.get("enabled", False):
            if is_sequence_parallel_enabled():
                raise ValueError("VAE adversarial distillation supports DDP or FSDP with sequence parallel disabled.")
            latent_channels = self.model.latent_channels
            self.adversarial = VAEAdversarialObjective(
                adversarial_config,
                device=self.model.device,
                latent_channels=latent_channels,
                gradient_accumulation_iters=self.gradient_accumulation_iters,
            )
        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)
        if self.stage_iteration_offset:
            logger.info(
                "[train] VAE stage schedule uses iteration offset={} (training iter {} maps to stage iter {})",
                self.stage_iteration_offset,
                self.current_train_iteration,
                self.current_train_iteration + self.stage_iteration_offset,
            )

    def compute_loss_on_sample(self, sample):
        result = self.vae_distillation.compute_loss(
            sample,
            VAEDistillationStepContext(
                running_dtype=self.running_dtype,
                iteration=self.current_train_iteration + self.stage_iteration_offset,
                micro_step=self._micro_step,
                adversarial_objective=self.adversarial,
            ),
        )
        self._micro_step += 1
        return result

    def _set_gradient_sync(self, enabled):
        super()._set_gradient_sync(enabled)
        if self.adversarial is not None:
            self.adversarial.set_gradient_sync(enabled)

    def _after_backward(self):
        if self.adversarial is not None:
            self.adversarial.step()
        self._micro_step = 0

    def _extra_checkpoint_state(self):
        state = {
            "vae_distillation_schedule": self.schedule_fingerprint,
            "vae_distillation_stage_iteration_offset": self.stage_iteration_offset,
        }
        if self.adversarial is not None:
            state["vae_adversarial"] = self.adversarial.state_dict()
        return state

    def _load_extra_checkpoint_state(self, state):
        if state.get("vae_distillation_schedule") != self.schedule_fingerprint:
            raise RuntimeError("The checkpoint was created with a different VAE distillation schedule.")
        checkpoint_offset = state.get("vae_distillation_stage_iteration_offset")
        if checkpoint_offset is not None and checkpoint_offset != self.stage_iteration_offset:
            raise RuntimeError(
                "The checkpoint was created with a different VAE distillation stage iteration offset."
            )
        if checkpoint_offset is None and self.stage_iteration_offset:
            logger.info(
                "[train] Applying VAE stage iteration offset={} while resuming a checkpoint created before stage offsets were enabled",
                self.stage_iteration_offset,
            )
        if self.adversarial is None:
            return
        if "vae_adversarial" not in state:
            raise RuntimeError("The checkpoint does not contain the enabled VAE discriminator state.")
        self.adversarial.load_state_dict(state["vae_adversarial"])
