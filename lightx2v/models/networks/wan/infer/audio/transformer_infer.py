from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.models.networks.wan.infer.utils import compute_freqs_audio, compute_freqs_audio_dist


class WanAudioTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

    def compute_freqs(self, q, grid_sizes, freqs):
        if self.config["seq_parallel"]:
            freqs_i = compute_freqs_audio_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs, self.seq_p_group)
        else:
            freqs_i = compute_freqs_audio(q.size(2) // 2, grid_sizes, freqs)
        return freqs_i

    def post_process(self, x, y, c_gate_msa, pre_infer_out):
        x = super().post_process(x, y, c_gate_msa, pre_infer_out)

        # Apply audio_dit if available
        if pre_infer_out.audio_dit_blocks is not None and hasattr(self, "block_idx"):
            for ipa_out in pre_infer_out.audio_dit_blocks:
                if self.block_idx in ipa_out:
                    cur_modify = ipa_out[self.block_idx]
                    x = cur_modify["modify_func"](x, pre_infer_out.grid_sizes, **cur_modify["kwargs"])
        return x
