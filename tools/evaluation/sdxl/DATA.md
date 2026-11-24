# Frozen inputs and assets

GenEval: all 553 prompts. PartiPrompts: all 1,632 prompts. COCO val2014: the locked project fixture of 5,000 sorted image IDs and first caption by caption ID. One 1024×1024 image per prompt, Euler 30 steps, CFG 5, seed 42.

The checked-in JSONL manifests under configs/common freeze IDs, prompts and selection. They contain no generated results or local workspace paths. Manifests and external assets have SHA256 locks. `--paths` maps relative paths to your provisioned roots; it is not a dataset download switch. Do not regenerate a different fixture under the same name. Preserve failed/missing inputs in the denominator.



The manifests embed all GenEval/Parti/COCO text and metadata needed for generation. GenEval comes from the pinned official repository evaluation_metadata.jsonl; Parti source is the official PartiPrompts TSV; COCO captions_val2014.json provides the project-locked 5000 IDs/captions. COCO reference images are unnecessary for CLIPScore. Do not substitute another random 5000-image subset.

Model repository and revision are in `configs/common/weight-lock.json`. Use `huggingface_hub.snapshot_download(repo_id=..., revision=..., local_dir=...)`, then verify every listed file with check-inputs. Inference PR requirements/installers document accelerator and model dependencies; preserve licenses and obtain access to gated model weights yourself. This PR distributes no weights. Weight hashes include tokenizers/configs used by the inference and metric networks.



GenEval requires a separate legacy metric environment (Python 3.10, torch2.1.0+cu121, torchvision0.16.0, mmcv-full1.7.2, mmdet2.28.2, open-clip-torch2.20.0, clip-benchmark1.6.2, numpy1.26.4, scipy1.15.3). Fetch the official Mask2Former swin-s COCO checkpoint and the OpenAI ViT-L-14.pt CLIP checkpoint; locks pin their hashes. Install mmdetection configs for v2.28.2, and set geneval_config to the locked Mask2Former configuration; the five-file inheritance closure is locked. `geneval_clip` points at the directory containing ViT-L-14.pt; set CLIP’s cache via the metric subprocess environment as documented in configs/README.md. CLIPScore uses the separately locked Hugging Face openai/clip-vit-large-patch14 safetensors tree.

Quality protocol: GenEval: six published dimension accuracies and their unweighted mean, using the official Mask2Former/CLIP evaluator. Parti and COCO: OpenAI CLIP ViT-L/14 FP32 normalized cosine ×100 and 2.5×max(cosine,0). Native and adapted images are scored with the same metric runtime and weights. This is not FID.

GenEval also reads its external `evaluation/object_names.txt` vocabulary. Run `python prepare_assets.py --source-root /absolute/eval-sources/geneval` after source preparation. This downloads only that small file from the same pinned commit and verifies configs/common/evaluator-assets.json. For an existing installation use `--check-only --report /absolute/results/evaluator-assets.json` before scoring. The vocabulary is not copied into the PR.
