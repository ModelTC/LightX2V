import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import WanSFTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanLingbotFastTransformerInfer(WanSFTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

    def infer_block_with_kvcache(self, block, x, pre_infer_out):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )

        y_out = self.infer_self_attn_with_kvcache(
            block.compute_phases[0],
            pre_infer_out.grid_sizes.tensor,
            x,
            pre_infer_out.seq_lens,
            pre_infer_out.freqs,
            shift_msa,
            scale_msa,
        )

        x, attn_out = self.infer_cross_attn_with_kvcache(
            block.compute_phases[1],
            x,
            pre_infer_out.context,
            y_out,
            gate_msa,
            block=block,
            conditional_dict=pre_infer_out.conditional_dict,
        )

        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
        x = self.post_process(x, y, c_gate_msa, pre_infer_out)

        if self.has_post_adapter:
            x = self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)

        return x

    def infer_cross_attn_with_kvcache(self, phase, x, context, y_out, gate_msa, block=None, conditional_dict=None):
        num_frames = gate_msa.shape[0]
        frame_seqlen = x.shape[0] // num_frames
        seg_index = self.scheduler.seg_index

        x.add_((y_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) * gate_msa).flatten(0, 1))

        if conditional_dict and "c2ws_plucker_emb" in conditional_dict and block is not None:
            cam = conditional_dict["c2ws_plucker_emb"]
            if cam.dim() == 3:
                cam = cam.squeeze(0)
            if cam.shape[0] < x.shape[0]:
                cam = F.pad(cam, (0, 0, 0, x.shape[0] - cam.shape[0]))
            elif cam.shape[0] > x.shape[0]:
                cam = cam[: x.shape[0]]
            cam = cam.to(dtype=x.dtype, device=x.device)
            cam_hidden = block.cam_injector_layer2.apply(F.silu(block.cam_injector_layer1.apply(cam))) + cam
            x = (1.0 + block.cam_scale_layer.apply(cam_hidden)) * x + block.cam_shift_layer.apply(cam_hidden)

        norm3_out = phase.norm3.apply(x)

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if context_img is not None:
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)

        cross_kv_cache = self.kv_cache_manager.cross_attn_kv_cache

        if seg_index == 0:
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
            v = phase.cross_attn_v.apply(context).view(-1, n, d)
            cross_kv_cache.store_kv(k, v, self.block_idx)
            self._cross_kv_len = k.size(0)
        else:
            L = self._cross_kv_len
            k = cross_kv_cache.k_cache(self.block_idx)[:L]
            v = cross_kv_cache.v_cache(self.block_idx)[:L]

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
            q,
            k_lens=torch.tensor([k.size(0)], dtype=torch.int32),
        )
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )

        if context_img is not None:
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)
            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32),
            )
            attn_out.add_(
                phase.cross_attn_2.apply(
                    q=q,
                    k=k_img,
                    v=v_img,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_k,
                    max_seqlen_q=q.size(0),
                    max_seqlen_kv=k_img.size(0),
                )
            )

            if self.clean_cuda_cache:
                del k_img, v_img
                torch_device_module.empty_cache()

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch_device_module.empty_cache()
        return x, attn_out
