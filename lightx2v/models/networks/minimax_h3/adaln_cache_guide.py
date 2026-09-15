"""Shared instructions for setting up the MiniMax-H3 AdaLN cache."""

ADALN_CACHE_GUIDE = """1. Configure the cache in the inference JSON config:

     "use_adaln_cache": true,
     "adaln_cache_dir": "~/.cache/lightx2v/adaln"

   You can use a custom cache root or point to an existing cache root.
   Cache generation and inference must use the same model and JSON config.

2. Edit the cache generation script:

     tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh

   Set lightx2v_path, model_path, and --config_json.
   Set --model-variant fl2av for t2av/i2av/l2av/fl2av, or --model-variant ref2av for ref2av.

3. Generate the cache if no matching cache exists.
   If the target cache directory exists but is invalid, move it aside first.
   Cache generation will not overwrite an existing path.

   Run from the repository root:

     bash tools/cache_minimax_h3_adaln/run_cache_minimax_h3_adaln.sh

4. Retry inference after the cache is ready."""
