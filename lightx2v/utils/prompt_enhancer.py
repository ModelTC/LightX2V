import argparse

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from lightx2v.utils.profiler import ProfilingContext, ProfilingContext4Debug

sys_prompt = """
Transform the short prompt into a detailed video-generation caption using this structure:
​​Opening shot type​​ (long/medium/close-up/extreme close-up/full shot)
​​Primary subject(s)​​ with vivid attributes (colors, textures, actions, interactions)
​​Dynamic elements​​ (movement, transitions, or changes over time, e.g., 'gradually lowers,' 'begins to climb,' 'camera moves toward...')
​​Scene composition​​ (background, environment, spatial relationships)
​​Lighting/atmosphere​​ (natural/artificial, time of day, mood)
​​Camera motion​​ (zooms, pans, static/handheld shots) if applicable.

Pattern Summary from Examples:
[Shot Type] of [Subject+Action] + [Detailed Subject Description] + [Environmental Context] + [Lighting Conditions] + [Camera Movement]

​One case:
Short prompt: a person is playing football
Long prompt: Medium shot of a young athlete in a red jersey sprinting across a muddy field, dribbling a soccer ball with precise footwork. The player glances toward the goalpost, adjusts their stance, and kicks the ball forcefully into the net. Raindrops fall lightly, creating reflections under stadium floodlights. The camera follows the ball’s trajectory in a smooth pan.

Note: If the subject is stationary, incorporate camera movement to ensure the generated video remains dynamic.

​​Now expand this short prompt:​​ [{}]. Please only output the final long prompt in English.
"""


class PromptEnhancer:
    def __init__(self, model_name="Qwen/Qwen2.5-32B-Instruct", device_map="cuda:0"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_device(self, device):
        self.model = self.model.to(device)

    @ProfilingContext("Run prompt enhancer")
    @torch.no_grad()
    def __call__(self, prompt):
        prompt = prompt.strip()
        prompt = sys_prompt.format(prompt)
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=8192,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        think_id = self.tokenizer.encode("</think>")
        if len(think_id) == 1:
            index = len(output_ids) - output_ids[::-1].index(think_id[0])
        else:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        logger.info(f"[Enhanced] thinking content: {thinking_content}")
        rewritten_prompt = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        logger.info(f"[Enhanced] rewritten prompt: {rewritten_prompt}")
        return rewritten_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="In a still frame, a stop sign")
    args = parser.parse_args()

    prompt_enhancer = PromptEnhancer()
    enhanced_prompt = prompt_enhancer(args.prompt)
    logger.info(f"Original prompt: {args.prompt}")
    logger.info(f"Enhanced prompt: {enhanced_prompt}")
