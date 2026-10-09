# DMAD source attribution and modification notice

The DMAD trainer, two-head adversarial objective, noise-band gap routing,
power-function student EMA, stochastic H3 rollout and the starting recipe
are adapted from the work of **Zhengming Yu and the DMAD authors**:

- Paper: [DMAD: Distribution Matching as Adversarial Distillation for Fast Visual Generation](https://arxiv.org/abs/2610.02188), 2026.
- Source: [Yzmblog/DMAD](https://github.com/Yzmblog/DMAD), specifically `train/h3/`.
- Revision: `8067c05f74a8cfc818d6e21c2b49405b49ba9cbc`.
- License: Apache License 2.0, as distributed in upstream `train/h3/LICENSE` and `LICENSE`. The same license text is available in this repository's [LICENSE](../../LICENSE).

The upstream H3 training code itself builds on LightX2V. This port integrates
the method into the current LightX2V capability-based architecture rather
than replacing the tree with the upstream fork. Material modifications
include Ref2AV conditioning, strict condition/real/teacher latent pairing,
32-rank FSDP launch support, FP32 master weights with BF16 compute, global
noise-band gap statistics, reference-cost data sampling, and checkpoint
integration. See [the recipe documentation](minimax_h3_ref2av_dmad.md) for
the comparison and limitations. Existing DMD/PDMD/HEAD objectives remain
separate.

The following is the upstream repository's `NOTICE` at the cited revision,
preserved for attribution. It describes upstream code and checkpoint
distribution; this port does not redistribute those checkpoint weights.

```text
DMAD inference code for MiniMax-H3
Copyright 2026 the DMAD authors. Licensed under the Apache License, Version 2.0 (see LICENSE).

The DMAD student checkpoints distributed with this code are Model Derivatives of MiniMax H3 and are
licensed under the MiniMax H3 Community License Agreement, not under Apache 2.0:

    "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement,
     Copyright (c) 2026 MiniMax. All Rights Reserved."

A copy of that agreement, including its territorial restrictions and Acceptable Use Policy, accompanies the
checkpoints (https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Videos produced with these
models are AI-generated.

Portions of dmad_h3/ (the packed-sequence geometry and the sampling / decoding helpers) were written for the
LightX2V-Train based training code of DMAD and are included here in modified form.
```

Model weights, derived checkpoints and source datasets retain their own
licenses; the Apache-2.0 code license does not replace them.
