"""Checkpoint key mappings shared by offline converters."""


def get_key_mapping_rules(direction, model_type):
    if model_type == "wan_dit":
        unified_rules = [
            {
                "forward": (r"^head\.head$", "proj_out"),
                "backward": (r"^proj_out$", "head.head"),
            },
            {
                "forward": (r"^head\.modulation$", "scale_shift_table"),
                "backward": (r"^scale_shift_table$", "head.modulation"),
            },
            {
                "forward": (
                    r"^text_embedding\.0\.",
                    "condition_embedder.text_embedder.linear_1.",
                ),
                "backward": (
                    r"^condition_embedder.text_embedder.linear_1\.",
                    "text_embedding.0.",
                ),
            },
            {
                "forward": (
                    r"^text_embedding\.2\.",
                    "condition_embedder.text_embedder.linear_2.",
                ),
                "backward": (
                    r"^condition_embedder.text_embedder.linear_2\.",
                    "text_embedding.2.",
                ),
            },
            {
                "forward": (
                    r"^time_embedding\.0\.",
                    "condition_embedder.time_embedder.linear_1.",
                ),
                "backward": (
                    r"^condition_embedder.time_embedder.linear_1\.",
                    "time_embedding.0.",
                ),
            },
            {
                "forward": (
                    r"^time_embedding\.2\.",
                    "condition_embedder.time_embedder.linear_2.",
                ),
                "backward": (
                    r"^condition_embedder.time_embedder.linear_2\.",
                    "time_embedding.2.",
                ),
            },
            {
                "forward": (r"^time_projection\.1\.", "condition_embedder.time_proj."),
                "backward": (r"^condition_embedder.time_proj\.", "time_projection.1."),
            },
            {
                "forward": (r"blocks\.(\d+)\.self_attn\.q\.", r"blocks.\1.attn1.to_q."),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.to_q\.",
                    r"blocks.\1.self_attn.q.",
                ),
            },
            {
                "forward": (r"blocks\.(\d+)\.self_attn\.k\.", r"blocks.\1.attn1.to_k."),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.to_k\.",
                    r"blocks.\1.self_attn.k.",
                ),
            },
            {
                "forward": (r"blocks\.(\d+)\.self_attn\.v\.", r"blocks.\1.attn1.to_v."),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.to_v\.",
                    r"blocks.\1.self_attn.v.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.self_attn\.o\.",
                    r"blocks.\1.attn1.to_out.0.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.to_out\.0\.",
                    r"blocks.\1.self_attn.o.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.q\.",
                    r"blocks.\1.attn2.to_q.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.to_q\.",
                    r"blocks.\1.cross_attn.q.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.k\.",
                    r"blocks.\1.attn2.to_k.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.to_k\.",
                    r"blocks.\1.cross_attn.k.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.v\.",
                    r"blocks.\1.attn2.to_v.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.to_v\.",
                    r"blocks.\1.cross_attn.v.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.o\.",
                    r"blocks.\1.attn2.to_out.0.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.to_out\.0\.",
                    r"blocks.\1.cross_attn.o.",
                ),
            },
            {
                "forward": (r"blocks\.(\d+)\.norm3\.", r"blocks.\1.norm2."),
                "backward": (r"blocks\.(\d+)\.norm2\.", r"blocks.\1.norm3."),
            },
            {
                "forward": (r"blocks\.(\d+)\.ffn\.0\.", r"blocks.\1.ffn.net.0.proj."),
                "backward": (
                    r"blocks\.(\d+)\.ffn\.net\.0\.proj\.",
                    r"blocks.\1.ffn.0.",
                ),
            },
            {
                "forward": (r"blocks\.(\d+)\.ffn\.2\.", r"blocks.\1.ffn.net.2."),
                "backward": (r"blocks\.(\d+)\.ffn\.net\.2\.", r"blocks.\1.ffn.2."),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.modulation\.",
                    r"blocks.\1.scale_shift_table.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.scale_shift_table(?=\.|$)",
                    r"blocks.\1.modulation",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.k_img\.",
                    r"blocks.\1.attn2.add_k_proj.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.add_k_proj\.",
                    r"blocks.\1.cross_attn.k_img.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.v_img\.",
                    r"blocks.\1.attn2.add_v_proj.",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.add_v_proj\.",
                    r"blocks.\1.cross_attn.v_img.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.norm_k_img\.weight",
                    r"blocks.\1.attn2.norm_added_k.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.norm_added_k\.weight",
                    r"blocks.\1.cross_attn.norm_k_img.weight",
                ),
            },
            {
                "forward": (
                    r"img_emb\.proj\.0\.",
                    r"condition_embedder.image_embedder.norm1.",
                ),
                "backward": (
                    r"condition_embedder\.image_embedder\.norm1\.",
                    r"img_emb.proj.0.",
                ),
            },
            {
                "forward": (
                    r"img_emb\.proj\.1\.",
                    r"condition_embedder.image_embedder.ff.net.0.proj.",
                ),
                "backward": (
                    r"condition_embedder\.image_embedder\.ff\.net\.0\.proj\.",
                    r"img_emb.proj.1.",
                ),
            },
            {
                "forward": (
                    r"img_emb\.proj\.3\.",
                    r"condition_embedder.image_embedder.ff.net.2.",
                ),
                "backward": (
                    r"condition_embedder\.image_embedder\.ff\.net\.2\.",
                    r"img_emb.proj.3.",
                ),
            },
            {
                "forward": (
                    r"img_emb\.proj\.4\.",
                    r"condition_embedder.image_embedder.norm2.",
                ),
                "backward": (
                    r"condition_embedder\.image_embedder\.norm2\.",
                    r"img_emb.proj.4.",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.self_attn\.norm_q\.weight",
                    r"blocks.\1.attn1.norm_q.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.norm_q\.weight",
                    r"blocks.\1.self_attn.norm_q.weight",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.self_attn\.norm_k\.weight",
                    r"blocks.\1.attn1.norm_k.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn1\.norm_k\.weight",
                    r"blocks.\1.self_attn.norm_k.weight",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.norm_q\.weight",
                    r"blocks.\1.attn2.norm_q.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.norm_q\.weight",
                    r"blocks.\1.cross_attn.norm_q.weight",
                ),
            },
            {
                "forward": (
                    r"blocks\.(\d+)\.cross_attn\.norm_k\.weight",
                    r"blocks.\1.attn2.norm_k.weight",
                ),
                "backward": (
                    r"blocks\.(\d+)\.attn2\.norm_k\.weight",
                    r"blocks.\1.cross_attn.norm_k.weight",
                ),
            },
            # head projection mapping
            {
                "forward": (r"^head\.head\.", "proj_out."),
                "backward": (r"^proj_out\.", "head.head."),
            },
        ]

        if direction == "forward":
            return [rule["forward"] for rule in unified_rules]
        elif direction == "backward":
            return [rule["backward"] for rule in unified_rules]
        else:
            raise ValueError(f"Invalid direction: {direction}")
    elif model_type in {"h3", "h3_video_vae_decoder"}:
        # MiniMax-H3 transformer and Video VAE decoder checkpoints already use
        # LightX2V's runtime keys.
        return []
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
