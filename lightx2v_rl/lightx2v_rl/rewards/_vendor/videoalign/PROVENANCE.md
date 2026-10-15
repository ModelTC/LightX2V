VideoAlign inference sources extracted from FewStepRL's VideoAlign submodule,
commit `aba26b658fec7d9fd30c295187b548ea673c8769`.
Upstream: https://github.com/KwaiVGI/VideoAlign

- `model.py`: only `Qwen2VLRewardModelBT` from `trainer.py`; training imports omitted.
- `prompt_template.py`: unchanged prompts and formatting helper.
- `LICENSE`: original MIT license.

The wrapper in `../../videoalign.py` ports FewStepRL's in-memory preprocessing,
grayscale MQ evaluation and checkpoint normalization without importing its
trainer or altering global Python/Accelerate state.
