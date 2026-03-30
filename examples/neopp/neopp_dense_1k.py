"""
NeoPP Dense text-to-image generation example (1024x1024).
This example demonstrates how to use LightX2V with NeoPP Dense model for T2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for NeoPP Dense T2I task
pipe = LightX2VPipeline(
    model_path="/data/nvme1/yongyang/FL/neo9b/neo9b",
    model_cls="neopp",
    task="t2i",
)

# Create generator from config JSON file
pipe.create_generator(config_json="../../configs/neopp/neopp_dense_t2i.json")

seed = 200
prompt = "a photo of two trucks"
negative_prompt = ""
save_result_path = "/data/nvme1/yongyang/FL/LightX2V/save_results/output_lightx2v_neopp_dense_t2i_1k.png"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
    target_shape=[1024, 1024],
)
