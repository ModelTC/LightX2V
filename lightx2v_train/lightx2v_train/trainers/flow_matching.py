from lightx2v_train.model_capabilities import (
    FlowMatchingSFTCapability,
    SFTStepContext,
)
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value, sync_sequence_parallel_gradients
from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .base import BaseTrainer
from .optimizer import OptimizerTrainer


@TRAINER_REGISTER("flow_matching")
class FlowMatchingTrainer(OptimizerTrainer):
    trainer_name = "flow_matching"
    required_capabilities = (
        *BaseTrainer.required_capabilities,
        FlowMatchingSFTCapability,
    )

    def __init__(self, config):
        super().__init__(config)
        self.noise_scheduler = RectifiedFlowMatchingScheduler(config)

    def set_model(self, model):
        super().set_model(model)
        self.sft = model.capabilities.require(FlowMatchingSFTCapability)

    def compute_loss_on_sample(self, sample):
        return self.sft.compute_loss(
            sample,
            SFTStepContext(
                noise_scheduler=self.noise_scheduler,
                running_dtype=self.running_dtype,
                broadcast=broadcast_sequence_parallel_value,
            ),
        )

    def _after_backward(self):
        sync_sequence_parallel_gradients(self.trainable_params)
