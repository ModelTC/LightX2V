"""Shared instructions for setting up the MiniMax-H3 AdaLN cache."""

TOOL_DIR = "tools/cache_minimax_h3_adaln"
MODEL_GUIDES = {
    "minimax_h3": (
        f"{TOOL_DIR}/run_cache_minimax_h3_adaln.sh",
        "Set --model-variant fl2av for t2av/i2av/l2av/fl2av, or ref2av for ref2av.",
    ),
    "minimax_h3_causal": (
        f"{TOOL_DIR}/run_cache_minimax_h3_causal_adaln.sh",
        "Set --task refa2v or refa2av to match inference.",
    ),
    "minimax_h3_image": (
        f"{TOOL_DIR}/run_cache_minimax_h3_image_adaln.sh",
        "This script generates both T2I and I2I cache profiles.",
    ),
}


def get_adaln_cache_guide(model_cls: str) -> str:
    guide = MODEL_GUIDES.get(model_cls)
    if guide is None:
        return f"Refer to the .sh files under {TOOL_DIR}/ for AdaLN cache generation."

    script, note = guide
    return f"""1. Configure the cache in the inference JSON config:

     "use_adaln_cache": true,
     "adaln_cache_dir": "~/.cache/lightx2v/adaln"

   You can use a custom cache root or point to an existing cache root.
   Cache generation and inference must use the same model and JSON config.

2. Edit the cache generation script for {model_cls}:

     {script}

   Set lightx2v_path, model_path, and --config_json to match inference.
   {note}

3. Generate the cache if no matching cache exists.
   If the target cache directory exists but is invalid, move it aside first.
   Cache generation will not overwrite an existing path.

   Run the corresponding script from the repository root:

     bash {script}

4. Retry inference after the cache is ready."""
