---
name: contribute-code
description: Follow LightX2V contribution guidelines when adding or changing code, including new model integration, non-NVIDIA inference support, readable implementations, and pre-commit checks.
---

# Contribute Code to LightX2V

English | [简体中文](SKILL_ZH.md)

- When adding support for a model, see how LightX2V integrates similar models. Consult [support-new-model](../support-new-model/SKILL.md) as needed.
- For inference on non-NVIDIA platforms, keep backend-specific operators, dependencies, and device capability checks in `lightx2v_platform` wherever possible. Use the existing backend registration and operator interfaces. Keep common interfaces and shared inference logic in `lightx2v`.
- Reuse existing implementations and keep the normal path easy to follow. Follow the caller contract, remove redundant checks and fallbacks for cases callers cannot produce, and retain necessary boundary checks. See [remove-ai-code-smell](../remove-ai-code-smell/SKILL.md).

## Before Delivery

Delete all temporary test files you wrote for intermediate verification during the task.

Before delivering the changes, run `pre-commit run --all-files` from the repository root. If the tools are missing, install them with `pip install ruff pre-commit`.

Review any automatic fixes, resolve failures, and rerun the full check until it passes.
