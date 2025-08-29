from lightx2v.utils.envs import *


class QwenImagePreInfer:
    def __init__(self, config, img_in, txt_norm, txt_in, time_text_embed, pos_embed):
        self.config = config
        self.img_in = img_in
        self.txt_norm = txt_norm
        self.txt_in = txt_in
        self.time_text_embed = time_text_embed
        self.pos_embed = pos_embed
        self.attention_kwargs = {}
        self.cpu_offload = config.get("cpu_offload", False)
        if self.cpu_offload:
            self.init_cpu_offload()

    def init_cpu_offload(self):
        self.img_in = self.img_in.to(torch.device("cuda"))
        self.txt_norm = self.txt_norm.to(torch.device("cuda"))
        self.txt_in = self.txt_in.to(torch.device("cuda"))
        self.time_text_embed = self.time_text_embed.to(torch.device("cuda"))
        self.pos_embed = self.pos_embed.to(torch.device("cuda"))

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, hidden_states, timestep, guidance, encoder_hidden_states_mask, encoder_hidden_states, img_shapes, txt_seq_lens, attention_kwargs):
        hidden_states_0 = hidden_states
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_text_embed(timestep, hidden_states) if guidance is None else self.time_text_embed(timestep, guidance, hidden_states)
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        return hidden_states, encoder_hidden_states, encoder_hidden_states_mask, (hidden_states_0, temb, image_rotary_emb)
