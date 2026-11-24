# External code and weights

Model inference comes from the fixed ModelTC/LightX2V inference PR, and the original model project/package pinned in configs/nvidia/source-lock.json. No external model implementation is vendored in this evaluation PR. Check each source’s license and model/data access terms before downloading or redistribution.


The GenEval implementation is imported from djghosh13/geneval at its locked commit. Mask2Former/mmdetection and OpenAI CLIP run externally in a separate metric environment. No GenEval or detector source is copied here; local images are arranged in the evaluator’s public directory convention. COCO and Parti prompt fixtures identify their evaluation inputs, not training assets.
