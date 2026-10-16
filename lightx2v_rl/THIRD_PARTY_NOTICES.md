**Third-party provenance and notices**

This package is distributed under Apache License 2.0. The complete license text
is included in [LICENSE](LICENSE). The license file was copied without changes
from FewStepRL's `LICENSE.txt` at the source commit below, preserving its original
copyright notice.

**FewStepRL**

- Source repository: <https://github.com/GoatWu/FewStepRL>
- Source commit: `3dd8e4221f7fdfb408a294c418fdbf9f27d25c4b`
- License: Apache License 2.0
- Original notice: **Copyright 2025 Jintao Zhang, Haoxu Wang, Jianfei Chen**

The following FewStepRL source files supplied the migrated Wan algorithm:

```text
genrl/diffusers_patch/wan_distill_pipeline_with_logprob.py
genrl/trainer/wan_trainer.py
genrl/trainer/timestep_flow.py
```

Adaptations are in `lightx2v_rl/algorithms/fewstep/` and
`lightx2v_rl/rollout/trajectory.py`. The migration separates pure PyTorch bridge,
timestep and objective code from the original Accelerate trainer and Diffusers
pipeline; introduces explicit configuration and validation; supports generic
latent dimensions; and connects the shared math to external verl-omni adapters.
The source name `immb` is exposed as **Geo-FSRL / Geo / `geo`**, and `flow_cps`
as **CPS / `cps`**. The original Wan CPS fixed-denominator KL surrogate is
preserved and documented in the objective implementation.

This is a migration of the Wan policy mathematics and rollout behavior.
Reward model ports and weighted score aggregation are described below.
The NFT trainer and Qwen-specific objective variants are not migrated.

FewStepRL's `genrl/advantages.py` and `genrl/stat_tracking.py` supply the
per-reward advantage weighting and `max_group_std` normalization semantics,
under the same source commit and Apache License 2.0. The adaptation in
`lightx2v_rl/algorithms/fewstep/advantages.py` and its
`lightx2v_rl/backends/verl_omni/advantage.py` adapter uses the gathered rollout candidate
groups, explicit unweighted reward components, and configurable normalization.
It centers each component within its group, optionally divides by the selected
component-specific standard deviation, then sums the weighted advantages
without a second normalization. Existing scalar reward logging and validation
scores remain unchanged. Persistent prompt-history tracking is not migrated.

**HPSv3 and VideoAlign reward ports**

FewStepRL's `genrl/reward/hpsv3.py`, `genrl/reward/videoalign.py`, and
`genrl/rewards.py` at the commit above supply frame scoring, preprocessing,
normalization and weighted-mean aggregation behavior, under Apache License 2.0.
Adaptations in `lightx2v_rl/rewards/` use explicit local model paths, lazy loading,
shared model instances, and an independent HTTP inference service. The HPS
FP32 head now explicitly casts its input to FP32. Inference uses SDPA by default;
HPS frame chunking is configurable. Unknown metrics and inference failures raise
errors instead of producing a fallback zero score.

Minimal upstream inference classes and prompts are preserved separately:

- HPSv3: <https://github.com/ModelTC/HPSv3>, commit
  `a2eb2ef2c7b5d91a566347a5825cf6d872122149`; MIT license, Copyright 2024 HPSv3 Team.
  See `lightx2v_rl/rewards/_vendor/hpsv3/LICENSE`.
- VideoAlign: <https://github.com/KwaiVGI/VideoAlign>, FewStepRL submodule commit
  `aba26b658fec7d9fd30c295187b548ea673c8769`; MIT license.
  See `lightx2v_rl/rewards/_vendor/videoalign/LICENSE` and `PROVENANCE.md`.

These model code licenses do not replace the separate licenses of model weights
and their Qwen2-VL base models. Prepared assets retain available source/model
notices and a local `models/rewards/reward-assets.json` provenance and hash manifest.

**verl-omni**

- Source repository: <https://github.com/verl-project/verl-omni>
- Pinned submodule commit: `e09850c5abb919f9c1893dc72f9d623698742093`
- Submodule location: `../third_party/verl-omni`
- License: Apache License 2.0; see the submodule's [LICENSE](../third_party/verl-omni/LICENSE).

The integration uses verl-omni's model and loss registration interfaces and
subclasses its Wan rollout adapter. The latter's source notice is:

```text
Copyright 2026 Bytedance Ltd. and/or its affiliates
```

Tracked patches cover generic extension/data-transport interfaces, RGBA
validation images, offline engine defaults, rollout port reservation, custom
metric tracking, validation reward metadata collection, and a custom advantage
extension hook.
Upstream source-file notices remain in the submodule and patches. This package's
algorithm-specific implementations remain outside the submodule.

**Qwen-Image-2.1 and UnifiedReward-Edit**

The Qwen adapter calls the official Diffusers `QwenImage21Pipeline` and
`QwenImage21Transformer2DModel` rather than vendoring their implementation.
`requirements/qwen_image21.txt` pins Diffusers commit
`62cde2b0115d57492d2f5ecefab54994797df2e3`, under Apache License 2.0.
The Qwen checkpoint is from <https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo>,
revision `fc13ba1986fcb905854dc8de2f52eee395f49b5b`.

UnifiedReward T2I and editing prompts/score schemas follow the official
<https://github.com/CodeGoat24/UnifiedReward> implementation and
<https://huggingface.co/CodeGoat24/UnifiedReward-Edit-qwen35-9b>, revision
`31acd924c34fa3afdd31e787803a589a0f40c384`. The original model cards and licenses
are retained in the downloaded snapshots. The HTTP client provides explicit
task routing, image transport, strict score parsing and reward composition.

OpenGPT-4o-Image remains a separately supplied dataset. Preparation creates
relative manifests and extracts reference images without relicensing or
redistributing the original dataset.

**Runtime dependencies and model weights**

PyTorch, Diffusers, verl, vLLM, vLLM-Omni, downloaded model weights, datasets and
externally configured reward implementations retain their respective licenses.
This package does not relicense those artifacts. Local model weights and data
under the ignored `models/`, `weights/`, and `data/` resource entries are prepared
separately and are not included in the Python distribution or Git source.
