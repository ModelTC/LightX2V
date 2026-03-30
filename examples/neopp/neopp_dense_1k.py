from lightx2v import LightX2VPipeline

# -------------------------------------------------
# Initialize pipeline for NeoPP
# -------------------------------------------------

pipe = LightX2VPipeline(
    model_path="/data/nvme1/yongyang/FL/neo9b/neo9b",
    model_cls="neopp",
    task="t2i",
)

pipe.create_generator(config_json="../../configs/neopp/neopp_dense_t2i.json")
pipe.modify_config({"load_kv_cache_in_pipeline_for_debug": False})


# -------------------------------------------------
# Load KV cache and generate
# -------------------------------------------------

pipe.runner.load_kvcache(
    "/data/nvme1/yongyang/FL/neo_test9b/vlm_tensor/to_x2v_cond_kv.pt",
    "/data/nvme1/yongyang/FL/neo_test9b/vlm_tensor/to_x2v_uncond_kv.pt",
)

pipe.generate(
    seed=200,
    save_result_path="/data/nvme1/yongyang/FL/LightX2V/save_results/output_lightx2v_neopp_dense_t2i_1k.png",
    target_shape=[1024, 1024],  # Height, Width
)
