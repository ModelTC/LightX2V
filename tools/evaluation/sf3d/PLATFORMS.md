# Platform validation

Only NVIDIA A100 has an implementation in this package. For MLU/MetaX/Ascend NPU, provide native-project runtime parity, device synchronization/seed/memory APIs, a host-visible resource attribution mechanism, dependency/source locks and single-device complete-dataset evidence before enabling a launcher. Copying CUDA JSON values is not validation. Placeholder launchers fail immediately with exit code 2.

Submit only this evaluation directory. Exclude history/, artifacts/, cache/, runtime/, sources/, data/, WORK_LOG.md and all local paths. Before submission inspect the actual git diff and GitHub PR file list. Put sanitized proven summaries under results/nvidia; record source/evaluator hashes, full coverage, failures, quality metrics, end-to-end distribution, native ratios, GPU UUID and evidence archive hashes. Never describe a partial smoke test as a complete campaign.
