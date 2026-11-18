# Codex skills

English | [简体中文](README_zh.md)

`skills/` contains 12 top-level groups and 28 skill entry points for model integration, performance tooling, and Intel GPU kernels. The two Intel XPU subskills and the kernel skills live in nested directories. Each entry point's directory name matches its frontmatter `name`. Detailed workflows, examples, and historical assets are available as needed in each skill's `references/` and `assets/` directories.

```text
skills/
├── support-model-warmup/
├── model-enable-intel-xpu/
├── lightx2v-kernel-skills/
├── support-new-model/
├── remove-ai-code-smell/
├── support-contiguous-block-offload/
├── support-cpu-block-offload/
├── support-static-block-residency/
├── benchmark-core-ops/
├── profile-model-module/
├── benchmark-module-parallelism/
└── support-model-compile/
```

The 24 upstream skill entries use `policy.allow_implicit_invocation: false` and can be invoked explicitly with `$skill-name`. `support-static-block-residency` and the three performance tools use the default policy, which allows automatic selection and explicit invocation.

In Codex CLI or the IDE extension, use `/skills` to browse skills or type `$skill-name` to invoke one explicitly. In the ChatGPT desktop app, open Skills in the sidebar. [Official invocation guide](https://learn.chatgpt.com/docs/build-skills#how-chatgpt-and-codex-use-skills)

Codex scans `.agents/skills` in each directory from the working directory up to the Git repository root. Set the working directory to the LightX2V repository root or one of its subdirectories to use these skills. If an update does not appear, restart Codex. [Official discovery rules](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)

## Performance tools

| Task | Skill |
|---|---|
| Compare core operator backends, including standalone SP attention tuning | [benchmark-core-ops](skills/benchmark-core-ops/SKILL.md) |
| Capture and analyze performance of real functions or modules, including multi-rank runs | [profile-model-module](skills/profile-model-module/SKILL.md) |
| Compare TP/SP strategies for the same complete module | [benchmark-module-parallelism](skills/benchmark-module-parallelism/SKILL.md) |

Choose a skill based on what you want to measure. Module parallelism comparisons can reuse SP operator screening and module profiling as needed. All three skills use the shared tool guide and existing report templates. Each task supplies the model-specific inputs and state handling. [Tool guide](../tools/AGENTS.md)

## Models and inference

| Task | Skill |
|---|---|
| Native integration of a new model | [support-new-model](skills/support-new-model/SKILL.md) |
| Intel XPU integration and troubleshooting | [model-enable-intel-xpu](skills/model-enable-intel-xpu/SKILL.md) |
| Intel XPU first-run diagnostics | [xpu-exec](skills/model-enable-intel-xpu/xpu-exec/SKILL.md) |
| Compatibility entry point for Intel XPU integration | [xpu-enable](skills/model-enable-intel-xpu/xpu-enable/SKILL.md) |
| Eager warmup | [support-model-warmup](skills/support-model-warmup/SKILL.md) |
| torch.compile | [support-model-compile](skills/support-model-compile/SKILL.md) |
| CPU block offload and shared weights | [support-cpu-block-offload](skills/support-cpu-block-offload/SKILL.md) |
| Static block weight residency and standalone estimates | [support-static-block-residency](skills/support-static-block-residency/SKILL.md) |
| Contiguous pinned weight layout | [support-contiguous-block-offload](skills/support-contiguous-block-offload/SKILL.md) |
| Remove redundant code and excessive defensive checks | [remove-ai-code-smell](skills/remove-ai-code-smell/SKILL.md) |

## Intel GPU kernels

- Hardware, ESIMD, memory, DPAS, barriers, and performance analysis. [Kernel fundamentals index](skills/lightx2v-kernel-skills/intel-xpu/kernel-basic-skills/README.md)
- GEMM, GEMV, Flash Attention, and quantization. [Kernel-specific skills index](skills/lightx2v-kernel-skills/intel-xpu/kernel-specific-skills/README.md)
- Python wheel integration covers the current repository build and an older Windows example project. [Integration skill](skills/lightx2v-kernel-skills/intel-xpu/integration-skills/sycl-esimd-to-python-wheel/SKILL.md) · [Windows example](skills/lightx2v-kernel-skills/intel-xpu/integration-skills/sycl-esimd-to-python-wheel/assets/lightx2v_kernel_xpu/README.md)

The `README.md` files in `kernel-basic-skills` and `kernel-specific-skills` serve as indexes. Codex reads skill instructions from `SKILL.md`. Some directories include `SKILL_ZH.md` as a Chinese reference or `SKILL_EN.md` as an English reference.

Codex reads invocation policy and optional UI metadata from `agents/openai.yaml`. `agents/openai_en.yaml` serves as an English template.
