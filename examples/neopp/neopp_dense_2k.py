from lightx2v import LightX2VPipeline

# -------------------------------------------------
# Initialize pipeline for NeoPP
# -------------------------------------------------

pipe = LightX2VPipeline(
    model_path="/data/nvme1/yongyang/FL/neo_9b_new/hf_step44000_ema_x2v_part",
    model_cls="neopp",
    support_tasks=["t2i", "i2i"],
)

pipe.create_generator(config_json="../../configs/neopp/neopp_dense.json")
pipe.modify_config({"load_kv_cache_in_pipeline_for_debug": False, "save_result_for_debug": True})


# -------------------------------------------------
# Load KV cache and generate
# -------------------------------------------------

# -------------------------------------------------
# TURN 0
# -------------------------------------------------
pipe.runner.load_kvcache(
    "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor_44000_ema_2k/to_x2v_cond_kv_0_298.pt",
    "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor_44000_ema_2k/to_x2v_uncond_kv_0_9.pt",
)
pipe.runner.set_inference_params(
    index_offset_cond=298,
    index_offset_uncond=9,
    cfg_interval=(-1, 2),
    cfg_scale=4.0,
    cfg_norm="none",
    timestep_shift=3.0,
)

pipe.generate(
    seed=200,
    save_result_path="/data/nvme1/yongyang/FL/LightX2V/save_results/output_lightx2v_neopp_dense_2k_0.png",
    target_shape=[2048, 2048],  # Height, Width
)


# -------------------------------------------------
# TURN 1
# -------------------------------------------------
pipe.runner.load_kvcache(
    "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor_44000_ema_2k/to_x2v_cond_kv_1_360.pt",
    "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor_44000_ema_2k/to_x2v_uncond_kv_1_12.pt",
)
pipe.runner.set_inference_params(
    index_offset_cond=366,
    index_offset_uncond=12,
    cfg_interval=(-1, 2),
    cfg_scale=4.0,
    cfg_norm="none",
    timestep_shift=3.0,
)

pipe.generate(
    seed=201,
    save_result_path="/data/nvme1/yongyang/FL/LightX2V/save_results/output_lightx2v_neopp_dense_2k_1.png",
    target_shape=[2048, 2048],  # Height, Width
)


# -------------------------------------------------
# TURN 2
# -------------------------------------------------
pipe.runner.load_kvcache(
    "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor_44000_ema_2k/to_x2v_cond_kv_2_439.pt",
    "/data/nvme1/yongyang/FL/neo_9b_new/vlm_tensor_44000_ema_2k/to_x2v_uncond_kv_2_15.pt",
)
pipe.runner.set_inference_params(
    index_offset_cond=441,
    index_offset_uncond=15,
    cfg_interval=(-1, 2),
    cfg_scale=4.0,
    cfg_norm="none",
    timestep_shift=3.0,
)

pipe.generate(
    seed=202,
    save_result_path="/data/nvme1/yongyang/FL/LightX2V/save_results/output_lightx2v_neopp_dense_2k_2.png",
    target_shape=[2048, 2048],  # Height, Width
)
