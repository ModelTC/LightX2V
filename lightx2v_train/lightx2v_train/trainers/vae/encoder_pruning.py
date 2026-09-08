"""Fixed-budget residual-branch search for the H3 video encoder."""

from lightx2v_train.utils.registry import TRAINER_REGISTER

from .pruning import VAEPruningTrainer


@TRAINER_REGISTER("vae_encoder_pruning")
class VAEEncoderPruningTrainer(VAEPruningTrainer):
    trainer_name = "vae_encoder_pruning"
    pruning_component_name = "encoder"

    def _export_pruned_model(self, output_dir):
        self.model.export_pruned_encoder(output_dir, self.gate_ema.cpu())
