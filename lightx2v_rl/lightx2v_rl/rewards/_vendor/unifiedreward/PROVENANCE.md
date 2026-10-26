# UnifiedReward pointwise prompts

Source: https://github.com/CodeGoat24/UnifiedReward (MIT license).
Retrieved 2026-10-15. Only the literal `problem` strings were extracted; inference
implementation is maintained locally. Their wording and image order are unchanged.
The official requests use JPEG quality 95 and disable sampling. This adapter uses
`temperature=0` and explicitly disables Qwen3.5 thinking.

- T2I: `inference_qwen/UnifiedReward-2.0-inference/point_score_ACS_image_generation.py`
- I2I: `UnifiedReward-Edit/edit_pointwise_instruction_following.py`

Source file SHA-256:

- `ur-t2i.py`: `fa67c436c6742084b43581d17bf3c98fa53aad5816ccd5ec39717c27c2a1dfa3`
- `ur-edit-if.py`: `fc27991f59a91eaf83cfa4630e4dcaefb901678c153ec9ae7507cd30280d4115`

Model: `CodeGoat24/UnifiedReward-Edit-qwen35-9b`, revision
`31acd924c34fa3afdd31e787803a589a0f40c384`.

Serving: pinned `vllm==0.20.2`, `transformers==5.5.4`. The native
Qwen3.5 model implementation and launch flags were checked against the vLLM
`v0.20.2` tag. Dependency resolution was validated for Python 3.12, Linux
x86_64/glibc 2.35. GPU model inference has not been run on the development host.
