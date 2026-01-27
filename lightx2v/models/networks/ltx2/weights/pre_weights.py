from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    MM_WEIGHT_REGISTER,
)


class LTX2PreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Video weights
        self.add_module(
            "patchify_proj",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.patchify_proj.weight",
                "model.diffusion_model.patchify_proj.bias",
            ),
        )

        self.add_module(
            "adaln_single_emb_timestep_embedder_linear_1",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight",
                "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias",
            ),
        )
        self.add_module(
            "adaln_single_emb_timestep_embedder_linear_2",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.weight",
                "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.bias",
            ),
        )

        self.add_module(
            "adaln_single_linear",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.adaln_single.linear.weight",
                "model.diffusion_model.adaln_single.linear.bias",
            ),
        )

        self.add_module(
            "caption_projection_linear_1",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.caption_projection.linear_1.weight",
                "model.diffusion_model.caption_projection.linear_1.bias",
                lora_prefix="diffusion_model.caption_projection",
            ),
        )
        self.add_module(
            "caption_projection_linear_2",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.caption_projection.linear_2.weight",
                "model.diffusion_model.caption_projection.linear_2.bias",
                lora_prefix="diffusion_model.caption_projection",
            ),
        )

        # Audio weights
        self.add_module(
            "audio_patchify_proj",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.audio_patchify_proj.weight",
                "model.diffusion_model.audio_patchify_proj.bias",
            ),
        )

        self.add_module(
            "audio_adaln_single_emb_timestep_embedder_linear_1",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_1.weight",
                "model.diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_1.bias",
            ),
        )
        self.add_module(
            "audio_adaln_single_emb_timestep_embedder_linear_2",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_2.weight",
                "model.diffusion_model.audio_adaln_single.emb.timestep_embedder.linear_2.bias",
            ),
        )
        self.add_module(
            "audio_adaln_single_linear",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.audio_adaln_single.linear.weight",
                "model.diffusion_model.audio_adaln_single.linear.bias",
            ),
        )

        self.add_module(
            "audio_caption_projection_linear_1",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.audio_caption_projection.linear_1.weight",
                "model.diffusion_model.audio_caption_projection.linear_1.bias",
            ),
        )
        self.add_module(
            "audio_caption_projection_linear_2",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.audio_caption_projection.linear_2.weight",
                "model.diffusion_model.audio_caption_projection.linear_2.bias",
            ),
        )

        self.add_module(
            "av_ca_video_scale_shift_adaln_single_emb_linear_1",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_1.weight",
                "model.diffusion_model.av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_1.bias",
            ),
        )
        self.add_module(
            "av_ca_video_scale_shift_adaln_single_emb_linear_2",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_2.weight",
                "model.diffusion_model.av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_2.bias",
            ),
        )
        self.add_module(
            "av_ca_video_scale_shift_adaln_single_linear",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_video_scale_shift_adaln_single.linear.weight",
                "model.diffusion_model.av_ca_video_scale_shift_adaln_single.linear.bias",
            ),
        )

        # AV CA Audio scale-shift AdaLN
        self.add_module(
            "av_ca_audio_scale_shift_adaln_single_emb_linear_1",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_1.weight",
                "model.diffusion_model.av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_1.bias",
            ),
        )
        self.add_module(
            "av_ca_audio_scale_shift_adaln_single_emb_linear_2",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_2.weight",
                "model.diffusion_model.av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_2.bias",
            ),
        )
        self.add_module(
            "av_ca_audio_scale_shift_adaln_single_linear",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_audio_scale_shift_adaln_single.linear.weight",
                "model.diffusion_model.av_ca_audio_scale_shift_adaln_single.linear.bias",
            ),
        )

        # AV CA A2V gate AdaLN
        self.add_module(
            "av_ca_a2v_gate_adaln_single_emb_linear_1",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.weight",
                "model.diffusion_model.av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.bias",
            ),
        )
        self.add_module(
            "av_ca_a2v_gate_adaln_single_emb_linear_2",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_2.weight",
                "model.diffusion_model.av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_2.bias",
            ),
        )
        self.add_module(
            "av_ca_a2v_gate_adaln_single_linear",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_a2v_gate_adaln_single.linear.weight",
                "model.diffusion_model.av_ca_a2v_gate_adaln_single.linear.bias",
            ),
        )

        # AV CA V2A gate AdaLN
        self.add_module(
            "av_ca_v2a_gate_adaln_single_emb_linear_1",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_1.weight",
                "model.diffusion_model.av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_1.bias",
            ),
        )
        self.add_module(
            "av_ca_v2a_gate_adaln_single_emb_linear_2",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_2.weight",
                "model.diffusion_model.av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_2.bias",
            ),
        )
        self.add_module(
            "av_ca_v2a_gate_adaln_single_linear",
            MM_WEIGHT_REGISTER["Default"](
                "model.diffusion_model.av_ca_v2a_gate_adaln_single.linear.weight",
                "model.diffusion_model.av_ca_v2a_gate_adaln_single.linear.bias",
            ),
        )
