import torch
from einops import rearrange
from .utils_bf16 import apply_rotary_emb
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import *


class HunyuanTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.double_blocks_num = 20
        self.single_blocks_num = 40
        self.heads_num = 24
        self.hidden_size = 3072
        self.mlp_hidden_dim = 12288
        self.parallel_attention = None
        if self.config["cpu_offload"]:
            if "offload_ratio" in self.config:
                offload_ratio = self.config["offload_ratio"]
            else:
                offload_ratio = 1
            self.double_weights_stream_mgr = WeightAsyncStreamManager(blocks_num=self.double_blocks_num, offload_ratio=offload_ratio)
            self.single_weights_stream_mgr = WeightAsyncStreamManager(blocks_num=self.single_blocks_num, offload_ratio=offload_ratio)
            self.infer_func = self._infer_with_offload
        else:
            self.infer_func = self._infer_without_offload

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        return self.infer_func(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)

    def _infer_with_offload(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num):
        txt_seq_len = txt.shape[0]
        img_seq_len = img.shape[0]

        for double_block_idx in range(self.double_blocks_num):
            if double_block_idx == 0:
                self.double_weights_stream_mgr.active_weights[0] = weights.double_blocks[0]
                self.double_weights_stream_mgr.active_weights[0].to_cuda()

            with torch.cuda.stream(self.double_weights_stream_mgr.compute_stream):
                img, txt = self.infer_double_block(self.double_weights_stream_mgr.active_weights[0], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)

            if double_block_idx < self.double_blocks_num - 1:
                self.double_weights_stream_mgr.prefetch_weights(double_block_idx + 1, weights.double_blocks)
            self.double_weights_stream_mgr.swap_weights()

        x = torch.cat((img, txt), 0)

        img = img.cpu()
        txt = txt.cpu()
        del img, txt
        torch.cuda.empty_cache()

        for single_block_idx in range(self.single_blocks_num):
            if single_block_idx == 0:
                self.single_weights_stream_mgr.active_weights[0] = weights.single_blocks[0]
                self.single_weights_stream_mgr.active_weights[0].to_cuda()
            with torch.cuda.stream(self.single_weights_stream_mgr.compute_stream):
                x = self.infer_single_block(self.single_weights_stream_mgr.active_weights[0], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
            if single_block_idx < self.single_blocks_num - 1:
                self.single_weights_stream_mgr.prefetch_weights(single_block_idx + 1, weights.single_blocks)
            self.single_weights_stream_mgr.swap_weights()
            torch.cuda.empty_cache()

        img = x[:img_seq_len, ...]
        return img, vec

    def _infer_without_offload(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num):
        txt_seq_len = txt.shape[0]
        img_seq_len = img.shape[0]

        for i in range(self.double_blocks_num):
            img, txt = self.infer_double_block(weights.double_blocks[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)

        x = torch.cat((img, txt), 0)

        for i in range(self.single_blocks_num):
            x = self.infer_single_block(weights.single_blocks[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)

        img = x[:img_seq_len, ...]
        return img, vec

    def infer_double_block_phase_1(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num):
        vec_silu = torch.nn.functional.silu(vec)

        img_mod_out = weights.img_mod.apply(vec_silu)
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = img_mod_out.chunk(6, dim=-1)

        if token_replace_vec is not None:
            token_replace_vec_silu = torch.nn.functional.silu(token_replace_vec)
            token_replace_vec_img_mod_out = weights.img_mod.apply(token_replace_vec_silu)
            (tr_img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate) = token_replace_vec_img_mod_out.chunk(6, dim=-1)
        else:
            (tr_img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate) = None, None, None, None, None, None

        txt_mod_out = weights.txt_mod.apply(vec_silu)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = txt_mod_out.chunk(6, dim=-1)

        img_q, img_k, img_v = self.infer_double_block_img_pre_atten(weights, img, img_mod1_scale, img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_shift, frist_frame_token_num, freqs_cis)
        txt_q, txt_k, txt_v = self.infer_double_block_txt_pre_atten(weights, txt, txt_mod1_scale, txt_mod1_shift)

        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)
        v = torch.cat((img_v, txt_v), dim=0)

        if not self.parallel_attention:
            attn = weights.double_attn.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_qkv,
                cu_seqlens_kv=cu_seqlens_qkv,
                max_seqlen_q=max_seqlen_qkv,
                max_seqlen_kv=max_seqlen_qkv,
            )
        else:
            # world_size = dist.get_world_size()
            attn = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=img_q.shape[0],
                cu_seqlens_qkv=cu_seqlens_qkv,
                # cu_seqlens_qkv=cu_seqlens_qkv,
                # max_seqlen_qkv=max_seqlen_qkv,
            )

        img_attn, txt_attn = attn[: img.shape[0]], attn[img.shape[0] :]

        img_out = weights.img_attn_proj.apply(img_attn)
        txt_out = weights.txt_attn_proj.apply(txt_attn)

        return (
            img_out,
            txt_out,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
            tr_img_mod1_gate,
            tr_img_mod2_shift,
            tr_img_mod2_scale,
            tr_img_mod2_gate,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        )

    def infer_double_block_phase_2(
        self,
        weights,
        img,
        txt,
        vec,
        cu_seqlens_qkv,
        max_seqlen_qkv,
        freqs_cis,
        token_replace_vec,
        frist_frame_token_num,
        img_out,
        txt_out,
        img_mod1_gate,
        img_mod2_shift,
        img_mod2_scale,
        img_mod2_gate,
        tr_img_mod1_gate,
        tr_img_mod2_shift,
        tr_img_mod2_scale,
        tr_img_mod2_gate,
        txt_mod1_gate,
        txt_mod2_shift,
        txt_mod2_scale,
        txt_mod2_gate,
    ):
        if tr_img_mod1_gate is not None:
            x_zero = img_out[:frist_frame_token_num] * tr_img_mod1_gate
            x_orig = img_out[frist_frame_token_num:] * img_mod1_gate
            img_out = torch.concat((x_zero, x_orig), dim=0)
        else:
            img_out = img_out * img_mod1_gate
        img = img + img_out

        img_out = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        if tr_img_mod1_gate is not None:
            x_zero = img_out[:frist_frame_token_num] * (1 + tr_img_mod2_scale) + tr_img_mod2_shift
            x_orig = img_out[frist_frame_token_num:] * (1 + img_mod2_scale) + img_mod2_shift
            img_out = torch.concat((x_zero, x_orig), dim=0)
        else:
            img_out = img_out * (1 + img_mod2_scale) + img_mod2_shift
        img_out = weights.img_mlp_fc1.apply(img_out)
        img_out = torch.nn.functional.gelu(img_out, approximate="tanh")
        img_out = weights.img_mlp_fc2.apply(img_out)

        txt_out = txt_out * txt_mod1_gate
        txt = txt + txt_out
        txt_out = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)
        txt_out = txt_out * (1 + txt_mod2_scale) + txt_mod2_shift
        txt_out = weights.txt_mlp_fc1.apply(txt_out)
        txt_out = torch.nn.functional.gelu(txt_out, approximate="tanh")
        txt_out = weights.txt_mlp_fc2.apply(txt_out)

        return img, txt, img_out, txt_out, img_mod2_gate, txt_mod2_gate

    def infer_double_block_phase_3(self, img_out, img_mod2_gate, img, txt_out, txt_mod2_gate, txt):
        # img
        img_out = img_out * img_mod2_gate
        img = img + img_out

        # txt
        txt_out = txt_out * txt_mod2_gate
        txt = txt + txt_out

        return img, txt

    def infer_double_block(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num):
        (
            img_out,
            txt_out,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
            tr_img_mod1_gate,
            tr_img_mod2_shift,
            tr_img_mod2_scale,
            tr_img_mod2_gate,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.infer_double_block_phase_1(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
        img, txt, img_out, txt_out, img_mod2_gate, txt_mod2_gate = self.infer_double_block_phase_2(
            weights,
            img,
            txt,
            vec,
            cu_seqlens_qkv,
            max_seqlen_qkv,
            freqs_cis,
            token_replace_vec,
            frist_frame_token_num,
            img_out,
            txt_out,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
            tr_img_mod1_gate,
            tr_img_mod2_shift,
            tr_img_mod2_scale,
            tr_img_mod2_gate,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        )
        img, txt = self.infer_double_block_phase_3(img_out, img_mod2_gate, img, txt_out, txt_mod2_gate, txt)
        return img, txt

    def infer_double_block_img_pre_atten(self, weights, img, img_mod1_scale, img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_shift, frist_frame_token_num, freqs_cis):
        img_modulated = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        if tr_img_mod1_scale is not None:
            x_zero = img_modulated[:frist_frame_token_num] * (1 + tr_img_mod1_scale) + tr_img_mod1_shift
            x_orig = img_modulated[frist_frame_token_num:] * (1 + img_mod1_scale) + img_mod1_shift
            img_modulated = torch.concat((x_zero, x_orig), dim=0)
        else:
            img_modulated = img_modulated * (1 + img_mod1_scale) + img_mod1_shift
        img_qkv = weights.img_attn_qkv.apply(img_modulated)

        img_q, img_k, img_v = rearrange(img_qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)

        img_q = weights.img_attn_q_norm.apply(img_q)
        img_k = weights.img_attn_k_norm.apply(img_k)

        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)
        return img_q, img_k, img_v

    def infer_double_block_txt_pre_atten(self, weights, txt, txt_mod1_scale, txt_mod1_shift):
        txt_modulated = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)
        txt_modulated = txt_modulated * (1 + txt_mod1_scale) + txt_mod1_shift
        txt_qkv = weights.txt_attn_qkv.apply(txt_modulated)

        txt_q, txt_k, txt_v = rearrange(txt_qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)

        txt_q = weights.txt_attn_q_norm.apply(txt_q)
        txt_k = weights.txt_attn_k_norm.apply(txt_k)
        return txt_q, txt_k, txt_v

    def infer_single_block_phase_1(self, weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        out = torch.nn.functional.silu(vec)
        out = weights.modulation.apply(out)
        mod_shift, mod_scale, mod_gate = out.chunk(3, dim=-1)

        if token_replace_vec is not None:
            token_replace_vec_out = torch.nn.functional.silu(token_replace_vec)
            token_replace_vec_out = weights.modulation.apply(token_replace_vec_out)
            tr_mod_shift, tr_mod_scale, tr_mod_gate = token_replace_vec_out.chunk(3, dim=-1)
        else:
            tr_mod_shift, tr_mod_scale, tr_mod_gate = None, None, None

        out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        if token_replace_vec is not None:
            x_zero = out[:frist_frame_token_num] * (1 + tr_mod_scale) + tr_mod_shift
            x_orig = out[frist_frame_token_num:] * (1 + mod_scale) + mod_shift
            x_mod = torch.concat((x_zero, x_orig), dim=0)
        else:
            x_mod = out * (1 + mod_scale) + mod_shift

        x_mod = weights.linear1.apply(x_mod)

        qkv, mlp = torch.split(x_mod, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)

        q = weights.q_norm.apply(q)
        k = weights.k_norm.apply(k)

        img_q, txt_q = q[:-txt_seq_len, :, :], q[-txt_seq_len:, :, :]
        img_k, txt_k = k[:-txt_seq_len, :, :], k[-txt_seq_len:, :, :]
        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)

        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)

        if not self.parallel_attention:
            attn = weights.single_attn.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_qkv,
                cu_seqlens_kv=cu_seqlens_qkv,
                max_seqlen_q=max_seqlen_qkv,
                max_seqlen_kv=max_seqlen_qkv,
            )
        else:
            attn = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=img_q.shape[0],
                cu_seqlens_qkv=cu_seqlens_qkv,
                # cu_seqlens_qkv=cu_seqlens_qkv,
                # max_seqlen_qkv=max_seqlen_qkv,
            )

        out = torch.nn.functional.gelu(mlp, approximate="tanh")
        out = torch.cat((attn, out), 1)
        out = weights.linear2.apply(out)
        return out, mod_gate, tr_mod_gate

    def infer_single_block_phase_2(self, x, out, tr_mod_gate, mod_gate, token_replace_vec=None, frist_frame_token_num=None):
        if token_replace_vec is not None:
            x_zero = out[:frist_frame_token_num] * tr_mod_gate
            x_orig = out[frist_frame_token_num:] * mod_gate
            out = torch.concat((x_zero, x_orig), dim=0)
        else:
            out = out * mod_gate
        x = x + out
        return x

    def infer_single_block(self, weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        out, mod_gate, tr_mod_gate = self.infer_single_block_phase_1(weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
        x = self.infer_single_block_phase_2(x, out, tr_mod_gate, mod_gate, token_replace_vec, frist_frame_token_num)
        return x
