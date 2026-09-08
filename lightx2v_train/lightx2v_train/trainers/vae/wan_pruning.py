"""Wan VAE search shares the optimizer, EMA and export lifecycle with H3."""

from lightx2v_train.runtime.ddp import unwrap_ddp_module
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .pruning import VAEPruningTrainer


@TRAINER_REGISTER("wan21_decoder_pruning")
class WanDecoderPruningTrainer(VAEPruningTrainer):
    trainer_name = "wan21_decoder_pruning"
    pruning_component_name = "decoder"

    def _pruning_component(self):
        return unwrap_ddp_module(self.model.denoiser_module())

    def _export_pruned_model(self, output_dir):
        self.model.export_pruned_component(output_dir, self.gate_ema.cpu())


@TRAINER_REGISTER("wan21_encoder_pruning")
class WanEncoderPruningTrainer(WanDecoderPruningTrainer):
    trainer_name = "wan21_encoder_pruning"
    pruning_component_name = "encoder"
