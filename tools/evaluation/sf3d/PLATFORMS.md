# Platform validation

The NVIDIA A100 campaign includes native comparison and process/resource evidence. MLU590 has a single-card LightX2V campaign that calls pinned inference PR #1753 across all 1,023 GSO objects and computes geometry quality and latency. The legacy entry records the assigned physical card and MLU peak allocated memory; it does not provide the observed-host entry guarantees described below. MLU results must be reported as MLU-only measurements. MetaX C500 provides a pinned LightX2V campaign with complete dataset quality/latency collection and host PID resource evidence. Its inference adapter retains complete model outputs, including SF3D textures. Ascend 910B provides a BF16 LightX2V campaign with the original FP32 scopes and quality evaluators, complete dataset coverage, end-to-end latency and host PID resource evidence. See README.md for the pinned source overlay and host runtime. Target-device full-run status is reported only by actual completed results.

Submit only this evaluation directory. Exclude history/, artifacts/, cache/, runtime/, sources/, data/, WORK_LOG.md and all local paths. Before submission inspect the actual git diff and GitHub PR file list. Put sanitized proven summaries under results/nvidia, results/metax or results/ascend_npu for the corresponding platform; record source/evaluator hashes, full coverage, failures, quality metrics, end-to-end distribution, native ratios, GPU UUID and evidence archive hashes. Never describe a partial smoke test as a complete campaign.


MLU observed-host entry: `scripts/platform/mlu/run_observed.sh` now supports
explicit LightX2V and independent native backends with host cnmon UUID/PCI/PID
ancestry, bounded workers and verified cleanup. Native models retain A100's
original native precision; PR inference retains BF16/FP32 boundaries. Use
`compare-mlu` for full-coverage mean-latency/quality reports without a
cross-precision exact-equality claim. The historical legacy MLU campaign lacks
these resource/native guarantees; its timings must stay separately labelled.
Current validation consists of real one-input PR/native generation smokes for
all three models and a one-input SF3D native/PR geometry score. This is not a
new full-dataset run or a cross-platform hardware regression result.
