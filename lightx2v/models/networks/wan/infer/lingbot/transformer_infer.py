import torch

from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanLingbotTransformerInfer(WanTransformerInfer):
    def infer_block(self, block, x, pre_infer_out):
        if hasattr(block.compute_phases[0], "before_proj") and block.compute_phases[0].before_proj.weight is not None:
            x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        y_out = self.infer_self_attn(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
        )
        x, attn_out = self.infer_cross_attn(
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
        if hasattr(block.compute_phases[2], "after_proj"):
            pre_infer_out.adapter_args["hints"].append(block.compute_phases[2].after_proj.apply(x))

        if self.has_post_adapter:
            x = self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)

        return x

    def infer_cross_attn(self, phase, x, context, y_out, gate_msa, block, conditional_dict):
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y_out.to(self.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x.add_(y_out * gate_msa.squeeze())

        cam = conditional_dict["c2ws_plucker_emb"]
        if cam.dim() == 3:
            cam = cam.squeeze(0)
        if cam.shape[0] < x.shape[0]:
            cam = torch.nn.functional.pad(cam, (0, 0, 0, x.shape[0] - cam.shape[0]))
        elif cam.shape[0] > x.shape[0]:
            cam = cam[: x.shape[0]]
        cam = cam.to(dtype=x.dtype, device=x.device)
        cam_hidden = block.cam_injector_layer2.apply(torch.nn.functional.silu(block.cam_injector_layer1.apply(cam)))
        cam_hidden = cam_hidden + cam
        cam_scale = block.cam_scale_layer.apply(cam_hidden)
        cam_shift = block.cam_shift_layer.apply(cam_hidden)
        x = (1.0 + cam_scale) * x + cam_shift

        norm3_out = phase.norm3.apply(x)
        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
        v = phase.cross_attn_v.apply(context).view(-1, n, d)

        if self.cross_attn_cu_seqlens_q is None:
            self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)
        if self.cross_attn_cu_seqlens_kv is None:
            self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]]).cumsum(0, dtype=torch.int32)
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True) and context_img is not None:
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)

            if self.cross_attn_cu_seqlens_kv_img is None:
                self.cross_attn_cu_seqlens_kv_img = torch.tensor([0, k_img.shape[0]]).cumsum(0, dtype=torch.int32)

            img_attn_out = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=self.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=self.cross_attn_cu_seqlens_kv_img,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
            )
            attn_out.add_(img_attn_out)

            if self.clean_cuda_cache:
                del k_img, v_img, img_attn_out
                torch_device_module.empty_cache()

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch_device_module.empty_cache()
        return x, attn_out
