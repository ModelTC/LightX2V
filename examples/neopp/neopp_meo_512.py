from lightx2v import LightX2VPipeline

# -------------------------------------------------
# Initialize pipeline for NeoPP
# -------------------------------------------------

pipe = LightX2VPipeline(
    model_path="/data/nvme1/yongyang/FL/neo_gen_30b_moe/neo_gen_30b_moe",
    model_cls="neopp",
    task="t2i",
)

pipe.create_generator(config_json="../../configs/neopp/neopp_moe.json")
pipe.modify_config({"load_kv_cache_in_pipeline_for_debug": False})


# -------------------------------------------------
# Load KV cache and generate
# -------------------------------------------------

pipe.runner.load_kvcache_t2i(
    "/data/nvme1/yongyang/FL/neo_test/vlm_tensor/to_x2v_cond_kv.pt",
    "/data/nvme1/yongyang/FL/neo_test/vlm_tensor/to_x2v_uncond_kv.pt",
)

pipe.generate(
    seed=200,
    save_result_path="/data/nvme1/yongyang/FL/LightX2V/save_results/output_lightx2v_neopp_moe_t2i_512.png",
    target_shape=[512, 512],  # Height, Width
)
