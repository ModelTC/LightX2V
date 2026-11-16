"""Model registration with import paths relative to this package."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ModelEntry:
    runner: str
    config: str = "config.load_config"


RUNNER_MODULES = {
    "bagel": ModelEntry("bagel.bagel_runner"),
    "cosmos3": ModelEntry("cosmos3.cosmos3_runner", config="cosmos3.config.load_config"),
    "depth_anything3": ModelEntry("depth_anything3.depth_anything3_runner"),
    "dreamzero": ModelEntry("wan.wan_dreamzero_runner", config="wan.dreamzero_config.load_config"),
    "ernie_image": ModelEntry("ernie_image.ernie_image_runner", config="ernie_image.config.load_config"),
    "fastwam": ModelEntry("wan.fastwam_runner", config="wan.config.load_config"),
    "flux2": ModelEntry("flux2.flux2_runner"),
    "hidream_o1_image": ModelEntry("hidream_o1_image.hidream_o1_image_runner", config="config.load_vae_only_config"),
    "hunyuan3d": ModelEntry("hunyuan3d.hunyuan3d_shape_runner", config="config.load_vae_only_config"),
    "hunyuan_image3": ModelEntry("hunyuan_image3.hunyuan_image3_runner", config="hunyuan_image3.config.load_config"),
    "hunyuan_video_1.5": ModelEntry("hunyuan_video.hunyuan_video_15_runner", config="hunyuan_video.config.load_config"),
    "infinitetalk": ModelEntry("wan.wan_infinitetalk_runner", config="wan.config.load_config"),
    "lingbot_va": ModelEntry("wan.wan_lingbot_va_runner", config="wan.lingbot_va_config.load_config"),
    "lingbot_video": ModelEntry("lingbot_video.lingbot_video_runner", config="lingbot_video.config.load_config"),
    "lingbot_world": ModelEntry("wan.wan_runner", config="wan.config.load_config"),
    "lingbot_world_fast": ModelEntry("wan.wan_lingbot_fast_runner", config="wan.config.load_config"),
    "matrix_game4": ModelEntry("wan.wan_matrix_game4_runner", config="wan.matrix_game4_config.load_config"),
    "longcat_image": ModelEntry("longcat_image.longcat_image_runner", config="longcat_image.config.load_config"),
    "ltx2": ModelEntry("ltx2.ltx2_runner", config="ltx2.config.load_config"),
    "ltx2_5": ModelEntry("ltx2.ltx25_runner", config="ltx2.config.load_config"),
    "minimax_h3": ModelEntry("minimax_h3.minimax_h3_runner", config="minimax_h3.config.load_config"),
    "minimax_h3_image": ModelEntry("minimax_h3_image.minimax_h3_image_runner", config="minimax_h3_image.config.load_config"),
    "minimax_h3_causal": ModelEntry("minimax_h3_causal.minimax_h3_causal_runner", config="minimax_h3.config.load_config"),
    "motus": ModelEntry("motus.motus_runner", config="motus.config.load_config"),
    "neopp": ModelEntry("neopp.neopp_runner"),
    "qwen_image": ModelEntry("qwen_image.qwen_image_runner"),
    "qwen_image_21": ModelEntry("qwen_image_21.qwen_image_21_runner"),
    "realtimewam": ModelEntry("wan.realtimewam_runner", config="wan.config.load_config"),
    "relayvsr_x2": ModelEntry("relayvsr.relayvsr_runner"),
    "relayvsr_x4": ModelEntry("relayvsr.relayvsr_runner"),
    "seedvr2": ModelEntry("seedvr.seedvr_runner"),
    "seko_talk": ModelEntry("wan.wan_audio_runner", config="wan.config.load_config"),
    "seko_talk_ar": ModelEntry("wan.wan_audio_runner", config="wan.config.load_config"),
    "taomate_h3": ModelEntry("taomate_h3.runner", config="minimax_h3.config.load_config"),
    "streamrender_h3": ModelEntry("streamrender_h3.runner", config="h3.config.load_config"),
    "swiftvr": ModelEntry("swiftvr.swiftvr_runner"),
    "tbdub": ModelEntry("wan.wan_tbdub_runner", config="wan.tbdub_config.load_config"),
    "wan2.1": ModelEntry("wan.wan_runner", config="wan.config.load_config"),
    "wan2.1_sf": ModelEntry("wan.wan_sf_runner", config="wan.config.load_config"),
    "wan2.1_realtime": ModelEntry("wan.wan_realtime_video_runner", config="wan.config.load_config"),
    "wan2.1_vace": ModelEntry("wan.wan_vace_runner", config="wan.config.load_config"),
    "wan2.2": ModelEntry("wan.wan_runner", config="wan.config.load_config"),
    "wan2.2_animate": ModelEntry("wan.wan_animate_runner", config="wan.config.load_config"),
    "wan2.2_animate2_distilled": ModelEntry("wan.wan_animate2_runner", config="wan.config.load_config"),
    "wan2.2_moe": ModelEntry("wan.wan_runner", config="wan.config.load_config"),
    "wan2.2_moe_vace": ModelEntry("wan.wan_vace_runner", config="wan.config.load_config"),
    "wan2.2_s2v": ModelEntry("wan.wan_s2v_runner", config="wan.s2v_config.load_config"),
    "wan21_edenlive": ModelEntry("wan.edenlive_runner", config="wan.edenlive_config.load_config"),
    "wan21_edenlive_ar": ModelEntry("wan.edenlive_runner", config="wan.edenlive_config.load_config"),
    "wan22_edenlive": ModelEntry("wan.edenlive_runner", config="wan.edenlive_config.load_config"),
    "wan22_edenlive_ar": ModelEntry("wan.edenlive_runner", config="wan.edenlive_config.load_config"),
    "wan_dancer": ModelEntry("wan.wan_dancer_runner", config="wan.config.load_config"),
    "worldmirror": ModelEntry("worldmirror.worldmirror_runner", config="worldmirror.config.load_config"),
    "worldplay_ar": ModelEntry("worldplay.worldplay_ar_runner", config="hunyuan_video.config.load_config"),
    "worldplay_bi": ModelEntry("worldplay.worldplay_bi_runner", config="hunyuan_video.config.load_config"),
    "worldplay_distill": ModelEntry("worldplay.worldplay_distill_runner", config="hunyuan_video.config.load_config"),
    "z_image": ModelEntry("z_image.z_image_runner", config="z_image.config.load_config"),
}


def get_model_spec(model_cls):
    module = RUNNER_MODULES[model_cls].runner.rsplit(".", 1)[0]
    return import_module(f".{module}.request_spec", __package__).SPECS[model_cls]


def get_model_config_loader(model_cls):
    module, name = RUNNER_MODULES[model_cls].config.rsplit(".", 1)
    return getattr(import_module(f".{module}", __package__), name)
