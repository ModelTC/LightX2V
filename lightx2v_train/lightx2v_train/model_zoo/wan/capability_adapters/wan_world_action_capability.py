"""World-action training capability for Wan FastWAM."""

from lightx2v_train.model_capabilities import (
    BoundCapability,
    LossResult,
    WorldActionTrainingCapability,
)


class WanWorldActionCapability(
    BoundCapability,
    WorldActionTrainingCapability,
):
    def configure(self) -> None:
        self.model.set_dit_only_trainable()
        self.model.log_model_structure()

    def parameters(self):
        return self.model.trainable_parameters()

    def module(self):
        return self.model.unwrap_module()

    def compute_loss(self, batch, module=None):
        if module is None:
            module = self.module()
        with self.model.autocast_context():
            loss, metrics = module(batch)
        return LossResult(loss=loss, metrics=metrics)

    def evaluation_loss(self, batch):
        with self.model.autocast_context():
            loss, metrics = self.module().training_loss(batch)
        return LossResult(loss=loss, metrics=metrics)

    def load_checkpoint(self, path) -> None:
        self.model.load_checkpoint(path)

    def save_checkpoint(self, path, step=None) -> None:
        self.model.save_checkpoint(path, step=step)
