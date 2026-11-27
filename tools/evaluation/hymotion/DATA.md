# Frozen inputs and assets

HumanML3D official test split: 4,384 motions, first caption per motion; 4,198 references satisfy the official 40 <= frames < 200 feature-evaluator interval. All 4,384 are generated and timed. Original 20–360-frame model clamping is retained; requested-length completeness is reported separately.

The checked-in JSONL manifests under configs/common freeze IDs, prompts and selection. They contain no generated results or local workspace paths. Manifests and external assets have SHA256 locks. `--paths` maps relative paths to your provisioned roots; it is not a dataset download switch. Do not regenerate a different fixture under the same name. Preserve failed/missing inputs in the denominator.

HumanML3D mirror: `TeoGchx/HumanML3D@9265d4c3669b95d848607954fa653884ff7715ab`, data/test-00000-of-00002.parquet and test-00001-of-00002.parquet, laid out as `data_root/bench/humanml3d/TeoGchx/data/`. Test order matches the official test split; first caption tokens are frozen in the manifest. Original HY-Motion geometry/stats must be materialized with git-lfs in the pinned native checkout. The complete native LFS assets are used, not the pruned inference-only assets.



Model repository and revision are in `configs/common/weight-lock.json`. Use `huggingface_hub.snapshot_download(repo_id=..., revision=..., local_dir=...)`, then verify every listed file with check-inputs. Inference PR requirements/installers document accelerator and model dependencies; preserve licenses and obtain access to gated model weights yourself. This PR distributes no weights. Weight hashes include tokenizers/configs used by the inference and metric networks.

Qwen3-8B and CLIP ViT-L/14 need complete tokenizer/config/safetensors trees. HumanML author assets: Google Drive evaluator archive `19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM` and GloVe archive `1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n`, referenced by the pinned Momask prepare/download_evaluator.sh and download_glove.sh. Extract to metric_assets; expose evaluator checkpoint as metric_checkpoints/t2m/text_mot_match/model/finest.tar. The asset lock verifies the actual files. prepare-sources downloads just the locked author evaluator modules and applies two explicit compatibility fixes: removed np.float→np.float64 and clip negative distance roundoff before sqrt. Both original and patched hashes are locked.



Quality protocol: Authors’ Momask/HumanML3D evaluator: R-Precision@1/2/3, matching score, FID, diversity. Native fixed-skeleton FK, linear 30→20 FPS resampling and authors’ canonical 263-D extraction. Paired central crop to min(prediction, reference, 196), multiple of 4; original evaluator normalization; retrieval batch size 32 including last batch; diversity seed 42, 300 pairs. Length exclusions are explicit, never removed from generation coverage.

Duration-to-frame conversion follows the original Python expression `int(round(duration_s * 30))`, followed by its 20–360 clamp. Do not substitute a theoretically rounded precomputed frame field: binary floating-point values such as 4.15 and 2.05 can differ by one frame. The reference environment follows the original requirements without optional ftfy; the inference configuration explicitly selects its matching `clip_text_cleanup: basic`, reusing Transformers BasicTokenizer.
