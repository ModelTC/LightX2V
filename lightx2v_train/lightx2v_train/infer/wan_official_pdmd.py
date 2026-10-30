"""FP32 previews using exactly the official PDMD training Euler rollout."""

from lightx2v_train.trainers.dmd.official_pdmd_core import full_rollout, rollout_grid
from lightx2v_train.utils.registry import INFERENCER_REGISTER

from .turbo_video import WanTurboInferencer


@INFERENCER_REGISTER("wan_official_pdmd_infer")
class WanOfficialPdmdInferencer(WanTurboInferencer):
    preview_title = "Wan2.1 1.3B dCM -> official PDMD preview"
    sampler_name = "official_pdmd_full_euler"
    trained_weights = "student_live"

    def _shift(self):
        return float(self.config["training"]["dmd"].get("official_rollout_video_shift", 1.0))

    def _sample(self, capability, initial, positive, steps, shift):
        grid = rollout_grid(steps, shift, initial.device)
        return full_rollout(capability, initial, positive, grid, grid).states[-1]

    def _weight_label(self, directory):
        if directory.name == "iter-000000000" and self.config["training"]["student"].get("checkpoint_path"):
            return "dcm100_student_ema_warmstart"
        return self.trained_weights
