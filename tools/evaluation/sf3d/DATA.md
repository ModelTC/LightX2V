# Frozen inputs and assets

GSO: exactly 1,023 objects, only thumbnail_0.jpg, one inference each. No GT-rendered input subset and no Objaverse campaign. The original object OBJ is used solely by the offline quality scorer.

The checked-in JSONL manifests under configs/common freeze IDs, prompts and selection. They contain no generated results or local workspace paths. Manifests and external assets have SHA256 locks. `--paths` maps relative paths to your provisioned roots; it is not a dataset download switch. Do not regenerate a different fixture under the same name. Preserve failed/missing inputs in the denominator.


GSO source: `hugging-3d/google-scanned-objects@b944bf0e56d93c82652b8b7243187be237434ed4`. Put the locked `<object>.thumbnail_0.jpg` and `<object>.obj` files under `data_root/GSO_google-scanned-objects/objects/`. Use existing downloaded data; the package never downloads GSO automatically. Each row pins both hashes. The 1023-object fixture is authoritative, not the old 1030×5 list.


Model repository and revision are in `configs/common/weight-lock.json`. Use `huggingface_hub.snapshot_download(repo_id=..., revision=..., local_dir=...)`, then verify every listed file with check-inputs. Inference PR requirements/installers document accelerator and model dependencies; preserve licenses and obtain access to gated model weights yourself. This PR distributes no weights. Weight hashes include tokenizers/configs used by the inference and metric networks.


SF3D needs the pinned native load/tets/160_tets.npz, DINOv2-large, LAION ViT-B/32 initializer and rembg u2net.onnx. The original checkpoint overwrites initializer weights, but their local offline paths must still resolve. Run `python prepare_runtime.py --paths paths.local.json --output /absolute/weight-overlays`; use its printed paths.local.json. It creates separate native and LightX2V config/weight overlays with symlinked checkpoints and preserves the original source config. Native changes only pretrained-model paths; the evaluator rejects neural/config changes. Set tets to the locked native grid; set rembg to a directory holding locked u2net.onnx. The upstream tree.png is a separate warmup input, outside GSO.


Quality protocol: Chamfer-L1/L2, F-score at 1/2/5% and surface-point voxel IoU32. Independently center each mesh at its bounding-box center and normalize its longest bounding-box edge to 1; no rotation fitting or ICP. Sample 20,000 points per surface with seed 42. Chamfer is the half-sum of the two directional means (squared for L2). IoU32 uses sampled surface occupancy, not solid-volume IoU. Texture quality is not measured by these geometry metrics.
