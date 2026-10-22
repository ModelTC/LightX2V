# MiniMax-H3 causal precision profiles

Use the BF16 configurations for quality-sensitive inference. `minimax_h3_causal_fp8_f16_accum_sm120.json` is an opt-in experiment for SM120; its native FP8-F16 accumulation kernel has not yet been tested on SM120.

The converted checkpoint uses per-channel FP8 weights with `weight_qmax=14`. On H200, the FP8-F16 kernel is unavailable and the runtime falls back to FP8-SGL. A real 56-frame TP2 run with fused QKV, KV and block-weight offload, compile, Graph, and warmup completed two requests with identical outputs, but its decoded video was visibly different from BF16. Against the BF16 reference, float-video range PSNR was 16.37 dB and the latent relative RMS error was 0.471. Reconstructed Q/K/V and FFN weights from blocks 0, 25, and 49 differed from BF16 by about 2.64% relative L2. These weight errors remain relevant on SM120, so native kernel availability alone does not establish quality parity.
