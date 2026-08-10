from lightx2v_train.model_capabilities import (
    TeacherForcingCapability,
    TeacherForcingStepContext,
)
from lightx2v_train.runtime.sequence_parallel import (
    broadcast_sequence_parallel_value,
)
from lightx2v_train.schedulers.flow_matching import (
    CausalForcingFlowMatchScheduler,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .base import BaseTrainer
from .flow_matching import FlowMatchingTrainer


@TRAINER_REGISTER("teacher_forcing")
class TeacherForcingTrainer(FlowMatchingTrainer):
    trainer_name = "teacher_forcing"
    required_capabilities = (
        *BaseTrainer.required_capabilities,
        TeacherForcingCapability,
    )

    def __init__(self, config):
        super().__init__(config)
        if self.train_type != "full":
            raise ValueError("Teacher forcing only supports training.train_type='full'.")

        teacher_forcing = self.training_config.get("teacher_forcing", {})
        mode = teacher_forcing.get("mode", "chunkwise")
        if mode != "chunkwise":
            raise ValueError(f"Unsupported teacher_forcing.mode={mode!r}; expected 'chunkwise'.")
        self.num_frame_per_chunk = int(teacher_forcing["num_frame_per_chunk"])
        self.noise_augmentation_max_timestep = int(teacher_forcing.get("noise_augmentation_max_timestep", 0))
        scheduler_config = self.config["scheduler"]
        self.teacher_forcing_scheduler = CausalForcingFlowMatchScheduler(
            num_train_timesteps=scheduler_config.get(
                "num_train_timesteps",
                1000,
            ),
            time_shift_settings=scheduler_config.get(
                "time_shift_settings",
                {},
            ),
        )

    def set_model(self, model):
        BaseTrainer.set_model(self, model)
        self.teacher_forcing = model.capabilities.require(TeacherForcingCapability)

    def compute_loss_on_sample(self, sample):
        return self.teacher_forcing.compute_loss(
            sample,
            TeacherForcingStepContext(
                scheduler=self.teacher_forcing_scheduler,
                running_dtype=self.running_dtype,
                num_frame_per_chunk=self.num_frame_per_chunk,
                noise_augmentation_max_timestep=(self.noise_augmentation_max_timestep),
                broadcast=broadcast_sequence_parallel_value,
            ),
        )
