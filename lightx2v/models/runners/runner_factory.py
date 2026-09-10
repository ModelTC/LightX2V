from importlib import import_module

import torch

from lightx2v.utils.registry_factory import RUNNER_REGISTER

RUNNER_MODULES = {
    "bagel": "lightx2v.models.runners.bagel.bagel_runner",
    "cosmos3": "lightx2v.models.runners.cosmos3.cosmos3_runner",
    "dreamzero": "lightx2v.models.runners.wan.wan_dreamzero_runner",
    "ernie_image": "lightx2v.models.runners.ernie_image.ernie_image_runner",
    "fastwam": "lightx2v.models.runners.wan.fastwam_runner",
    "flux2": "lightx2v.models.runners.flux2.flux2_runner",
    "hidream_o1_image": "lightx2v.models.runners.hidream_o1_image.hidream_o1_image_runner",
    "hunyuan3d": "lightx2v.models.runners.hunyuan3d.hunyuan3d_shape_runner",
    "hunyuan_image3": "lightx2v.models.runners.hunyuan_image3.hunyuan_image3_runner",
    "hunyuan_video_1.5": "lightx2v.models.runners.hunyuan_video.hunyuan_video_15_runner",
    "infinitetalk": "lightx2v.models.runners.wan.wan_infinitetalk_runner",
    "lingbot_va": "lightx2v.models.runners.wan.wan_lingbot_va_runner",
    "lingbot_video": "lightx2v.models.runners.lingbot_video.lingbot_video_runner",
    "lingbot_world": "lightx2v.models.runners.wan.wan_runner",
    "lingbot_world_fast": "lightx2v.models.runners.wan.wan_lingbot_fast_runner",
    "longcat_image": "lightx2v.models.runners.longcat_image.longcat_image_runner",
    "ltx2": "lightx2v.models.runners.ltx2.ltx2_runner",
    "ltx2_5": "lightx2v.models.runners.ltx2.ltx25_runner",
    "ltx2_ar": "lightx2v.models.runners.ltx2.ltx2_runner",
    "minimax_h3": "lightx2v.models.runners.minimax_h3.minimax_h3_runner",
    "motus": "lightx2v.models.runners.motus.motus_runner",
    "neopp": "lightx2v.models.runners.neopp.neopp_runner",
    "openpi": "lightx2v.models.runners.openpi.openpi_runner",
    "qwen_image": "lightx2v.models.runners.qwen_image.qwen_image_runner",
    "seedvr2": "lightx2v.models.runners.seedvr.seedvr_runner",
    "seko_talk": "lightx2v.models.runners.wan.wan_audio_runner",
    "seko_talk_ar": "lightx2v.models.runners.wan.wan_audio_runner",
    "sensenova_vision": "lightx2v.models.runners.bagel.sensenova_vision_runner",
    "swiftvr": "lightx2v.models.runners.swiftvr.swiftvr_runner",
    "wan2.1": "lightx2v.models.runners.wan.wan_runner",
    "wan2.1_sf": "lightx2v.models.runners.wan.wan_sf_runner",
    "wan2.1_sf_mtxg2": "lightx2v.models.runners.wan.wan_matrix_game2_runner",
    "wan2.1_vace": "lightx2v.models.runners.wan.wan_vace_runner",
    "wan2.2": "lightx2v.models.runners.wan.wan_runner",
    "wan2.2_animate": "lightx2v.models.runners.wan.wan_animate_runner",
    "wan2.2_animate2_distilled": "lightx2v.models.runners.wan.wan_animate2_runner",
    "wan2.2_audio": "lightx2v.models.runners.wan.wan_audio_runner",
    "wan2.2_matrix_game3": "lightx2v.models.runners.wan.wan_matrix_game3_runner",
    "wan2.2_moe": "lightx2v.models.runners.wan.wan_runner",
    "wan2.2_moe_vace": "lightx2v.models.runners.wan.wan_vace_runner",
    "wan2.2_s2v": "lightx2v.models.runners.wan.wan_s2v_runner",
    "wan_dancer": "lightx2v.models.runners.wan.wan_dancer_runner",
    "worldmirror": "lightx2v.models.runners.worldmirror.worldmirror_runner",
    "worldplay_ar": "lightx2v.models.runners.worldplay.worldplay_ar_runner",
    "worldplay_bi": "lightx2v.models.runners.worldplay.worldplay_bi_runner",
    "worldplay_distill": "lightx2v.models.runners.worldplay.worldplay_distill_runner",
    "z_image": "lightx2v.models.runners.z_image.z_image_runner",
}


def build_runner(config):
    """Instantiate a runner and initialize its model modules."""
    model_cls = config["model_cls"]
    import_module("lightx2v.common.ops")
    import_module(RUNNER_MODULES[model_cls])

    torch.set_grad_enabled(False)
    runner = RUNNER_REGISTER[model_cls](config)
    runner.init_modules()
    return runner
