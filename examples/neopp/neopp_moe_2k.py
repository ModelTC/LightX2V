from lightx2v import LightX2VPipeline

# -------------------------------------------------
# Initialize pipeline for NeoPP
# -------------------------------------------------

pipe = LightX2VPipeline(
    model_path="/data/nvme1/yongyang/FL/moe_new/mt_32experts_norm_resume_50khf",
    model_cls="neopp",
    support_tasks=["t2i", "i2i"],
)

pipe.create_generator(config_json="../../configs/neopp/neopp_moe.json")
pipe.modify_config({"load_kv_cache_in_pipeline_for_debug": False, "save_result_for_debug": True})


# -------------------------------------------------
# Load KV cache and generate
# -------------------------------------------------

# -------------------------------------------------
# TURN 0
# -------------------------------------------------
pipe.runner.load_kvcache(
    "/data/nvme1/yongyang/FL/moe_new/vlm_tensor/to_x2v_cond_kv_325.pt",
    "/data/nvme1/yongyang/FL/moe_new/vlm_tensor/to_x2v_uncond_kv_9.pt",
)
pipe.runner.set_inference_params(
    index_offset_cond=325,
    index_offset_uncond=9,
    cfg_interval=(0, 1),
    cfg_scale=4.0,
    cfg_norm="none",
    timestep_shift=3.0,
)

pipe.generate(
    seed=200,
    save_result_path="/data/nvme1/yongyang/FL/LightX2V/save_results/output_lightx2v_neopp_moe_2k_0.png",
    target_shape=[2048, 2048],  # Height, Width
)
