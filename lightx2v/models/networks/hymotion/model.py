from .infer.text import encode_text
from .weights.loader import load_hymotion


class HYMotionModel:
    def __init__(self, config):
        self.weights, self.text_weights, self.bound_operators = load_hymotion(config)

    def generate(self, prompt, seed, num_frames, steps, cfg_scale):
        if not isinstance(num_frames, int) or not 0 < num_frames < 5000:
            raise ValueError("HY-Motion requires an integer frame count in [1,4999]")
        # The released pipeline clamps short/long requests to 20–360 frames.
        # Keep that behavior instead of rejecting valid native inputs here.
        vtxt, ctxt, length = encode_text(self.text_weights, [prompt])
        self.weights.validation_steps = steps
        hidden = {"text_vec_raw": vtxt, "text_ctxt_raw": ctxt, "text_ctxt_raw_length": length}
        return self.weights.generate(prompt, [seed], num_frames / 30, cfg_scale=cfg_scale, use_special_game_feat=False, hidden_state_dict=hidden, length=num_frames)
