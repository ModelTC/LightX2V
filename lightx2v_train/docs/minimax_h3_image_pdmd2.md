# MiniMax-H3-Image: six steps to two-step PDMD

This experiment uses the released H3-Image checkpoint, 10,000 streamed
Fine-T2I captions, and the released H3-Image-PE model. Run the launcher once
on **one ACP worker with eight GPUs**. The full fake score model keeps FP32
master parameters and Adam states; the default configuration targets H200
memory capacity. Eight 80-GB GPUs have not been validated.

## Model and sampling adaptation

H3-Image is an image model with six PDD output heads. Its released physical
noise schedule is `[1, 0.9675237, 0.8875485, 0.75, 0.5327739, 0.2320089, 0]`.
The two-step student uses **`[1, 0.75, 0]`**, selecting heads 0 and 3.
There is no additional schedule shift, audio branch, CFG, or DeCFG adapter.

Official PDMD queries teacher and fake scores at continuous noise levels.
For those queries this implementation selects the head at the start of the
enclosing released interval (`released_interval_start`). **This extension of
an already distilled, discrete six-head teacher is experimental**; MiniMax
does not provide a validated six-to-two-step PDMD recipe. The PDMD projection,
critic objective, trajectory queries, and fake/student update order use our
existing implementation of the official PDMD recipe.

The native transformer's timestep is `1 - sigma` and its output is
`clean - noise`; the trainer adapter changes the sign for the single-modality
PDMD convention. Image packing, RoPE coordinates, and text embeddings follow
the released implementation. Input/time/output projections remain FP32,
while attention and FFN computation use BF16. All PDD heads share one FSDP
unit, including zero gradients for unselected heads, so ranks can query
different noise intervals without mismatching gradient reductions.

## Environment and weights

Use an environment containing the existing LightX2V training dependencies
and PyTorch 2.8 or newer. The following package versions were used for the
local CPU tests, real PE/text-cache smoke test, and two-GPU model tests:

```bash
python -m pip install \
  'git+https://github.com/huggingface/diffusers.git@875373ac7f5856071e345a3e394c74efccc6b9e6' \
  'transformers==5.17.0' 'datasets==5.0.0' 'xgrammar==0.1.25' accelerate
```

For faster Qwen3.5 prompt expansion, install `flash-linear-attention` and
`causal-conv1d` compatible with that environment. Without them Transformers
uses its slower PyTorch reference kernels; that path passed the local smoke
test. No SGLang server is required for the default `transformers` backend.

Download on the training machine if the weights have not been copied there:

```bash
hf download MiniMaxAI/MiniMax-H3-Image \
  --revision fbde899fffb5c5225ee8dd7027680113746af42e \
  --local-dir /mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-Image
hf download MiniMaxAI/MiniMax-H3-Image-PE \
  --revision 35603d4b63949afa12cb098227fffefedf081afa \
  --local-dir /mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-Image-PE
```

The current image repository includes `text_encoder/`; it is the default
encoder location. `H3_IMAGE_TEXT_ENCODER_PATH` can override it. Training uses
`transformer_diffusers/`, not the AV H3 transformer or `transformer/`.

The public PE checkpoint uses ModelOpt NVFP4. On Hopper the local loader
dequantizes its weights to BF16 (about 55 GB per GPU) and uses the released
system/task prompts and JSON schema. Its weight dequantization was compared
against NVIDIA Model Optimizer on actual checkpoint tensors, with exact
agreement. BF16 activations are used: this is not a bitwise emulation of
native FP4 activation quantization. An existing official SGLang endpoint can
instead be selected through `H3_IMAGE_PE_BACKEND=http`,
`H3_IMAGE_PE_ENDPOINT`, and `H3_IMAGE_PE_SERVED_MODEL`.

## Eight-GPU ACP command

Copy the selected `prompts_raw.jsonl` into the dataset directory below to
reuse the exact local 10,000 captions. Keep its `.selection.json` receipt for
provenance. If the prompt file is absent, the launcher selects 10,000 unique
captions itself from `ma-xu/fine-t2i`, subset
`synthetic_enhanced_prompt_random_resolution`, using streaming and seed 2026.
Tar streams include JPEG bytes while advancing; JPEGs are not decoded or
saved and the complete dataset is not downloaded.

```bash
cd /mnt/lm_data_afs/gushiqiao/codes/news/LightX2V
source /mnt/lm_data_afs/gushiqiao/envs/lightx2v_h3/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
H3_CODE_ROOT="$PWD" \
H3_PYTHON="$VIRTUAL_ENV/bin/python" \
H3_IMAGE_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-Image \
H3_IMAGE_PE_MODEL_PATH=/mnt/lm_data_afs/gushiqiao/models/MiniMax-H3-Image-PE \
H3_IMAGE_DATASET_DIR=/mnt/lm_data_afs/gushiqiao/datasets/minimax_h3_image_finet2i_10k_2048 \
H3_IMAGE_PDMD_OUTPUT=/mnt/devsft_afs_2/gushiqiao/outputs/h3_image_pdmd2_10k_2048 \
H3_IMAGE_PROMPT_COUNT=10000 \
H3_IMAGE_MAX_ITERS=15000 \
KERNELS_CACHE=/mnt/lm_data_afs/gushiqiao/cache/kernels \
bash lightx2v_train/scripts/run_minimax_h3_image_pdmd2_fsdp8_acp.sh
```

Append `--dry-run` to print the commands without loading models or writing
files. The real run validates model shards and exactly eight visible GPUs.
The stages are sequential, and each process exits before the next model is
loaded:

1. Reuse or stream-select `prompts_raw.jsonl`.
2. Eight independent PE shards write `pe/prompts_expanded.jsonl`.
3. Eight text-encoder shards write `condition_cache/conditions/*.pt` and
   `condition_cache/metadata.jsonl`.
4. Eight-way FSDP2 starts PDMD from the released image model.

Only text conditions and latent geometry are cached. Pure PDMD does not need
paired clean images: student image latents are generated online. The cached
text features are Qwen3-VL `hidden_states[50]`, BF16 `[1,L,5120]`, at most
4096 tokens, with text token tags equal to 1. Image resolution is always
2048 x 2048, latent shape `[24,1,128,128]`. The PE-recommended aspect ratio is
retained as metadata but does not override the requested square resolution.

## Training and resumption

Config: `configs/train/dmd/minimax_h3_image_pdmd2_2048_r128_a128_fsdp8.yaml`.

| Setting | Value |
| --- | --- |
| Student | LoRA, rank/alpha 128/128; AdaLN and `norm_out.linear` 16/16 |
| Teacher | Frozen released image model |
| Fake score | Full model, initialized from the released image model |
| Student / fake LR | 2e-5 / 4e-6 |
| Update order | Five fake updates, then one student update |
| Budget | 15,000 role updates = 2,500 student + 12,500 fake |
| Batch | One prompt per rank, eight ranks, no gradient accumulation |
| Optimizer | Adam, betas 0/0.9, epsilon 1e-8, weight decay 0 |
| Precision | BF16 student/teacher base, FP32 LoRA/fake masters, FP32 gradient reduction |
| Memory | FSDP2 streamed weight loading, gradient checkpointing |
| Checkpoints | Every 100 role updates, retain two, auto-resume enabled |
| EMA / GAN | Disabled |

Rerun the same command to reuse completed PE/cache items and resume a complete
PDMD checkpoint. Recipe hashes reject stale preprocessing instead of mixing
different prompt or model settings. Interrupted items are regenerated.
Student exports include `h3_image_sampling.json`; inference must retain the
two-step sigma grid and PDD head selection, in addition to the adapter's
per-layer rank/alpha configuration.

Logs are under `$H3_IMAGE_PDMD_OUTPUT/logs/launch.*.log` for the whole pipeline
and `$H3_IMAGE_PDMD_OUTPUT/train/train.log` for training. The locally verified
path includes real PE, real text encoding, and full 33B checkpoint streamed
loading with LoRA forward/backward on two GPUs at small latent size. A full
2048-square, three-role, eight-GPU training run has not been performed locally.

The FSDP head-selection regression can be run manually from the repository
root (it uses tiny models and compares gradients and Adam updates with an
unsharded reference):

```bash
PYTHONPATH=lightx2v_train CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  -m lightx2v_train.model_zoo.minimax_h3_image.tests.verify_fsdp
```

References: [H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image),
[H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE),
[Fine-T2I](https://huggingface.co/datasets/ma-xu/fine-t2i),
[official PDMD](https://github.com/ZeamoxWang/pdmd).
