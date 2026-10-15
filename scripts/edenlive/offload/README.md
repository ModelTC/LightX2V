# EdenLive block offload, quantization and warmup

Edit the repository, model and input paths in `run_edenlive_block_offload.sh`, then run it from the repository root. The default entry is Wan2.1 14B AR I2V with a cover image, recording video and text. `model_path` points to the complete EdenLive-x2v release; T5, CLIP, DINO and VAE retain their default component paths.

`configs/edenlive/offload/` contains BF16 profiles for all four variants. `configs/edenlive/quantized/` contains FP8 profiles for each variant. Update `--model_cls` together with the config: `few_step` selects an `_ar` runner, while `bidir` selects full bidirectional inference.

| Profile | Weight storage and compute |
| --- | --- |
| BF16 block | CPU-resident DiT blocks with two alternating GPU staging blocks |
| `fp8-sgl` | Single-file FP8 weights and the existing SGL FP8 MM |
| `int8-convrot` | Single-file INT8 weights, scales and rotation group sizes using comfy-kitchen |

The examples use private CPU weights and do not enable `shared_cpu_weights`. Encoder and VAE offload follow the Wan implementation. INT8 ConvRot conversion and inference require `comfy-kitchen`; the existing converter also uses `qtorch`.

## Conversion

Convert the release to native EdenLive-x2v first. Quantization always writes one safetensors file:

```bash
python tools/convert/convert_edenlive.py \
  --source /data/nvme3/gushiqiao/models/EdenLive-x2v/dit-14b \
  --output /data/nvme2/gushiqiao/models/dit-quant \
  --quantize fp8 --output_name dit-14b-fp8 --device cuda:0

python tools/convert/convert_edenlive.py \
  --source /data/nvme3/gushiqiao/models/EdenLive-x2v/dit-14b \
  --output /data/nvme2/gushiqiao/models/dit-quant \
  --quantize int8-convrot --output_name dit-14b-int8-convrot --device cuda:0
```

`dit-14b` and `dit-5b` contain AR weights. Bidirectional models require `dit-14b-bidir` and `dit-5b-bidir`. Change both source and output name for another variant. Existing outputs are kept unless `--overwrite` is explicit. Set `dit_quantized_ckpt` to the absolute output file; architecture metadata still comes from the DiT config under `model_path`.

## Runtime options

- `cpu_offload=true`, `offload_granularity="block"`: existing private Wan block offload. Model offload remains available separately.
- `ar_config.async_vae_decode=true`: submit cached VAE decode after each AR chunk. The VAE stays on GPU during the loop and its previous offload setting is restored afterward. The disabled path retains full-video decode.
- `use_compile=true`: public block compilation with transfers outside the graph. Bidirectional block offload caches by staging object; AR caches by logical layer. KV bookkeeping and AR attention remain eager boundaries.
- `warmup=true`: one startup warmup using real encoders, Step 0, AR clean-cache commits and decode. The fixed resolutions are 512×512 and 704×1280 for 5B, and 480×480 and 480×832 for 14B. Frame count follows the config; an unspecified AR length covers the attention window, another chunk and a tail frame.

Warmup clears generators, solver history, KV, prompt and temporary inputs while preserving compiled callables and kernel caches. Requests use their own seeds. New resolutions, chunk lengths or text layouts can require additional compilation. Warmup does not remove model transfers or block weight copies.

Phase offload is unsupported. The common warmup gate still rejects disaggregated inference, unload_modules and feature caching. Example profiles enable compile and warmup, with async decode for AR. Disable options individually for comparisons. `scripts/base/base.sh` defaults to BF16 and preserves explicit environment overrides.

## Validation scope

| Path | Functional validation |
| --- | --- |
| BF16 AR block + compile + warmup + async VAE | 5B at 704×1280 and 14B at 480×832, 121 frames, including KV eviction and the tail chunk |
| BF16 AR normal / model offload + compile + warmup | 5B at 704×1280, 25 frames |
| BF16 bidirectional block + compile + warmup | 5B at 704×1280 and 14B at 480×832, 25 frames each |
| Synchronous / asynchronous VAE output | Pixel-identical decoded videos for both 5B and 14B, 25 frames each |
| Eager block output with / without warmup | Pixel-identical decoded videos for 5B at 704×1280, 25 frames |
| Single-file FP8 + normal / block + eager / compile | Small native DiT conversion and real-kernel inference tests passed; full model weights were not quantized |
| INT8 ConvRot | Configs and converter entry are provided; dependencies are not installed and runtime is unverified |
| Lazy block / SP / TP / LoRA combinations | Not validated in this change |

The warmup and quantization test files have 13 passing tests and 4 ConvRot cases skipped for missing dependencies. Video tests use the repository's `cryo-crossing` input. Compilation and component offload paths allow BF16 numerical differences. Comparing eager and compile in the same block mode for 5B, 25 frames, gives a decoded RGB mean absolute difference of approximately 4/255 and PSNR of 30.94 dB. These results establish functionality.
