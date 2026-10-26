"""Shared instructions for setting up the MiniMax-H3 AdaLN cache."""

ADALN_CACHE_GUIDE = """1. Configure the cache in the inference JSON config:

     "use_adaln_cache": true,
     "adaln_cache_dir": "~/.cache/lightx2v/adaln"

   You can use a custom cache root or point to an existing cache root.
   Cache generation and inference must use the same model and JSON config.

2. Edit the cache generation script for your model:

   MiniMax-H3:
     tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
   Causal RefA2V:
     scripts/minimax_h3_causal/cache_adaln.sh

   Set lightx2v_path, model_path, and --config_json to match inference.
   For MiniMax-H3, set --model-variant fl2av for t2av/i2av/l2av/fl2av,
   or ref2av for ref2av. The causal script already selects minimax_h3_causal.

3. Generate the cache if no matching cache exists.
   If the target cache directory exists but is invalid, move it aside first.
   Cache generation will not overwrite an existing path.

   Run the corresponding script from the repository root:

     # MiniMax-H3
     bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh
     # Causal RefA2V
     bash scripts/minimax_h3_causal/cache_adaln.sh

4. Retry inference after the cache is ready."""
