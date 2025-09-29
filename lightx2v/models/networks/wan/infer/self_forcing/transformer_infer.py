import torch

from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanSFTransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        sf_config = self.config.sf_config
        self.local_attn_size = sf_config.local_attn_size
        self.max_attention_size = 32760 if self.local_attn_size == -1 else self.local_attn_size * 1560
        self.num_frame_per_block = sf_config.num_frame_per_block
        self.num_transformer_blocks = sf_config.num_transformer_blocks
        self.frame_seq_length = sf_config.frame_seq_length
        self._initialize_kv_cache(self.device, self.dtype)
        self._initialize_crossattn_cache(self.device, self.dtype)
        
        self.infer_func = self.infer_with_kvcache
    
    def _initialize_kv_cache(self, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros((kv_cache_size, 12, 128)).to(dtype).to(device),
                "v": torch.zeros((kv_cache_size, 12, 128)).to(dtype).to(device),
                "global_end_index": torch.tensor([0], dtype=torch.long).to(device),
                "local_end_index": torch.tensor([0], dtype=torch.long).to(device)
            })

        self.kv_cache1_default = kv_cache1  # always store the clean cache
    
    def _initialize_crossattn_cache(self, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros((512, 12, 128)).to(dtype).to(device),
                "v": torch.zeros((512, 12, 128)).to(dtype).to(device),
                "is_init": False
            })
        self.crossattn_cache_default = crossattn_cache    

    def infer_with_kvcache(self, blocks, x, pre_infer_out):
        self.kv_cache1 = self.kv_cache1_default
        self.crossattn_cache = self.crossattn_cache_default
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_block_witch_kvcache(blocks[block_idx], x, pre_infer_out)
        return x
    
    def infer_self_attn_with_kvcache(self, phase, grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa):
        if hasattr(phase, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa.squeeze()) * phase.smooth_norm1_weight.tensor
            norm1_bias = shift_msa.squeeze() * phase.smooth_norm1_bias.tensor
        else:
            norm1_weight = 1 + scale_msa.squeeze()
            norm1_bias = shift_msa.squeeze()
        
        norm1_out = phase.norm1.apply(x)
        
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.sensitive_layer_dtype)
        
        norm1_out.mul_(norm1_weight[0:1, :]).add_(norm1_bias[0:1, :])
        
        if self.sensitive_layer_dtype != self.infer_dtype:  # False
            norm1_out = norm1_out.to(self.infer_dtype)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        
        q0 = phase.self_attn_q.apply(norm1_out)
        k0 = phase.self_attn_k.apply(norm1_out)
        
        q = phase.self_attn_norm_q.apply(q0).view(s, n, d)
        k = phase.self_attn_norm_k.apply(k0).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)

        seg_index = self.scheduler.seg_index
        
        current_start_frame = seg_index * self.num_frame_per_block

        q = causal_rope_apply(q.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]
        k = causal_rope_apply(k.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]
        
        # Assign new keys/values directly up to current_end 
        seg_seq_len = self.frame_seq_length * self.num_frame_per_block
        local_start_index = seg_index * seg_seq_len
        local_end_index = (seg_index + 1) * seg_seq_len

        self.kv_cache1[self.block_idx]['k'][local_start_index: local_end_index] = k
        self.kv_cache1[self.block_idx]['v'][local_start_index: local_end_index] = v
        
        attn_k = self.kv_cache1[self.block_idx]['k'][max(0, local_end_index - self.max_attention_size):local_end_index]
        attn_v = self.kv_cache1[self.block_idx]['v'][max(0, local_end_index - self.max_attention_size):local_end_index]
        
        k_lens = torch.empty_like(seq_lens).fill_(attn_k.size(0))
        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=k_lens)

        if self.clean_cuda_cache: 
            del freqs_i, norm1_out, norm1_weight, norm1_bias
            torch.cuda.empty_cache()
        
        if self.config["seq_parallel"]:
            attn_out = phase.self_attn_1_parallel.apply(
                q=q,
                k=attn_k,
                v=attn_v,
                img_qkv_len=q.shape[0],
                cu_seqlens_qkv=cu_seqlens_q,
                attention_module=phase.self_attn_1,
                seq_p_group=self.seq_p_group,
                model_cls=self.config["model_cls"],
            )
        else:
            attn_out = phase.self_attn_1.apply(
                q=q,
                k=attn_k,
                v=attn_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=attn_k.size(0),
                model_cls=self.config["model_cls"],
            ) 

        y = phase.self_attn_o.apply(attn_out) 

        if self.clean_cuda_cache:
            del q, k, v, attn_out
            torch.cuda.empty_cache()

        return y
    
    def infer_cross_attn_with_kvcache(self, phase, x, context, y_out, gate_msa):
        
        num_frames = gate_msa.shape[0]
        frame_seqlen = x.shape[0] // gate_msa.shape[0]
        seg_index = self.scheduler.seg_index

        x.add_((y_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) * gate_msa).flatten(0, 1))
        norm3_out = phase.norm3.apply(x)

        if self.task in ["i2v", "flf2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if self.task in ["i2v", "flf2v"] and self.config.get("use_image_encoder", True):
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim

        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)

        if seg_index == 0:
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
            v = phase.cross_attn_v.apply(context).view(-1, n, d)
            self.crossattn_cache[self.block_idx]["k"] = k 
            self.crossattn_cache[self.block_idx]["v"] = v
        else:
            k = self.crossattn_cache[self.block_idx]["k"]
            v = self.crossattn_cache[self.block_idx]["v"]
        
        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
            q,
            k_lens=torch.tensor([k.size(0)], dtype=torch.int32, device=k.device),
        )
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
            model_cls=self.config["model_cls"],
        )
     
        if self.task in ["i2v", "flf2v"] and self.config.get("use_image_encoder", True) and context_img is not None:
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)

            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32, device=k.device),
            )
            img_attn_out = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
                model_cls=self.config["model_cls"],
            )
            attn_out.add_(img_attn_out)

            if self.clean_cuda_cache:
                del k_img, v_img, img_attn_out
                torch.cuda.empty_cache()

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch.cuda.empty_cache()
        return x, attn_out
    
    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)
        
        if self.clean_cuda_cache:
            del attn_out
            torch.cuda.empty_cache()

        num_frames = c_shift_msa.shape[0]
        frame_seqlen = x.shape[0] // c_shift_msa.shape[0]
        
        norm2_weight = 1 + c_scale_msa
        norm2_bias = c_shift_msa

        norm2_out = phase.norm2.apply(x)
        norm2_out = norm2_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) 
        norm2_out.mul_(norm2_weight).add_(norm2_bias)
        norm2_out = norm2_out.flatten(0, 1)

        y = phase.ffn_0.apply(norm2_out)
        if self.clean_cuda_cache:
            del norm2_out, x, norm2_weight, norm2_bias
            torch.cuda.empty_cache()
        y = torch.nn.functional.gelu(y, approximate="tanh")
        if self.clean_cuda_cache:
            torch.cuda.empty_cache()
        y = phase.ffn_2.apply(y)

        return y
    
    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):

        num_frames = c_gate_msa.shape[0]
        frame_seqlen = x.shape[0] // c_gate_msa.shape[0]
        y = y.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) 
        x = x.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) 
        x.add_(y * c_gate_msa)
        x = x.flatten(0, 1)
        
        if self.clean_cuda_cache:
            del y, c_gate_msa
            torch.cuda.empty_cache()
        return x
  
    def infer_block_witch_kvcache(self, block, x, pre_infer_out):
        
        if hasattr(block.compute_phases[0], "before_proj"):
            x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x

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
        )

        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
        
        x = self.post_process(x, y, c_gate_msa, pre_infer_out)
        
        if hasattr(block.compute_phases[2], "after_proj"):
            pre_infer_out.adapter_output["hints"].append(block.compute_phases[2].after_proj.apply(x))

        if self.has_post_adapter:
            x = self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)

        return x
    
    def infer_non_blocks(self, weights, x, e):
        num_frames = e.shape[0]
        frame_seqlen = x.shape[0] // e.shape[0]
        
        x = weights.norm.apply(x)
        x = x.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        
        t = self.scheduler.timestep_input
        e = e.unflatten(dim=0, sizes=t.shape).unsqueeze(2)
        modulation = weights.head_modulation.tensor
        e = (modulation.unsqueeze(1) + e).chunk(2, dim=2)
        
        x.mul_(1 + e[1][0]).add_(e[0][0])
        x = x.flatten(0, 1)
        x = weights.head.apply(x)
        
        if self.clean_cuda_cache:
            del e
            torch.cuda.empty_cache()
        return x
