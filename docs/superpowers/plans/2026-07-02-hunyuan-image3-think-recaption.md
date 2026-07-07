# Hunyuan Image3 Think Recaption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native LightX2V text-generation pre-stage for Hunyuan Image3 `recaption` and `think_recaption`, then feed generated COT text into the existing native image generation path.

**Architecture:** Keep the DiT path native. Add small runner-level helpers for text input preparation, autoregressive token generation without KV cache, stage transitions from `</think>` to `<recaption>`, and generated COT injection via tokenizer `batch_cot_text` for `gen_image`.

**Tech Stack:** Python, PyTorch, Hunyuan tokenizer/image processor helpers, LightX2V native HunyuanImage3 model/infer/weights.

---

### Task 1: Regression Tests

**Files:**
- Modify: `tests/test_hunyuan_image3_integration.py`

- [ ] Add tests for resolving bot task, generating COT before image inputs, and stage transition token forcing.
- [ ] Run targeted pytest and verify failures before implementation.

### Task 2: Runner Text Generation Helpers

**Files:**
- Modify: `lightx2v/models/runners/hunyuan_image3/hunyuan_image3_runner.py`

- [ ] Add helpers for system prompt resolution, text template creation, next-token selection, and COT text generation.
- [ ] Preserve existing `bot_task=image` behavior.
- [ ] For `think_recaption`, force `<recaption>` after `</think>` and stop after `</recaption>`.

### Task 3: Feed COT Into Image Generation

**Files:**
- Modify: `lightx2v/models/runners/hunyuan_image3/hunyuan_image3_runner.py`
- Modify: `configs/hunyuan_image3/hunyuan_image3_t2i.json`

- [ ] Pass `batch_cot_text` into `apply_chat_template(..., mode="gen_image")`.
- [ ] Add config keys for `bot_task`, `use_system_prompt`, `max_new_tokens`, and deterministic text generation.

### Task 4: Verification

**Files:**
- Test: `tests/test_hunyuan_image3_integration.py`

- [ ] Run full Hunyuan integration tests.
- [ ] Run ruff and py_compile on touched files.
