import torch
import xformers.ops
from einops import repeat, rearrange
import torch.nn.functional as F
from lightx2v.models.schedulers.mgcdr.scheduler import MagicDriverScheduler
from lightx2v.models.networks.mgcdr.weights.transformer_weights import MagicDriveTransformerWeight, MagicDriveTransformerAttnBlock
from lightx2v.models.networks.mgcdr.infer.funcs import construct_attn_input_from_map


class MagicDriveTransformerInfer:
    def __init__(self, config):
        self.config = config
        self.attention_type = config.get('attention_type', 'flash_attn2')
        self.control_blocks_num = config.get('control_depth', 13) # should be 13
        self.blocks_num = config.get('depth', 28) # should be 28
        self.num_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        
    def set_scheduler(self, scheduler: MagicDriverScheduler):
        self.scheduler = scheduler
        
    def t2i_modulate(self, x, shift, scale):
        return x * (1 + scale) + shift
    
    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x
    
    def _infer_self_attn(self, weights: MagicDriveTransformerAttnBlock, x, rope, temporal): 
        '''
        '''
        B, N, C = x.shape
        cond = x
        Bc, Nc, Cc = cond.shape
        assert B == Bc and C == Cc
        
        # bias = None if self.qkv.bias is None else self.qkv.bias[:self.dim]
        # q = F.linear(x, self.qkv.weight[:self.dim, :], bias)
        # q = q.view(B, N, self.num_heads, self.head_dim)

        # bias = None if self.qkv.bias is None else self.qkv.bias[self.dim:]
        # kv = F.linear(cond, self.qkv.weight[self.dim:, :], bias)
        # kv = kv.view(B, Nc, 2, self.num_heads, self.head_dim)
        
        # k, v = kv.unbind(2)
        
        # qkv = self.qkv(x)
        # x = x.reshape(-1, C)
        
        # q = weights.attn_q.apply(x)
        # q = q.reshape(B, N, -1)
        # q_shape = (B, N, self.num_heads, self.head_dim)
        # q = q.view(q_shape)
        
        # kv = weights.attn_kv.apply(x)
        # kv = kv.reshape(B, N, -1)
        # kv_shape = (B, N, 2, self.num_heads, self.head_dim)
        # kv = kv.view(kv_shape)
        # k, v = kv.unbind(2)
        
        
        qkv = weights.attn_qkv.apply(x)
        # qkv = qkv.reshape(B, N, -1)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        if temporal:
            qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            qkv = qkv.view(qkv_shape)
            q, k, v = qkv.unbind(2)
        
        # q, k = self.q_norm(q), self.k_norm(k)
        q = weights.attn_q_norm.apply(q)
        k = weights.attn_k_norm.apply(k)
        
        if rope:
            q = weights.rotary_emb.rotate_queries_or_keys(q)
            k = weights.rotary_emb.rotate_queries_or_keys(k)

        if temporal:
            x = weights.attn_naive.apply(
                q, k, v, softmax_scale=weights.softmax_scale, is_causal=weights.is_causal
            )
        else:
            x = weights.attn.apply(
                q, k, v, dropout_p=weights.attn_drop, softmax_scale=weights.softmax_scale, is_causal=weights.is_causal
            )
        
        x_output_shape = (B, N, C)
        
        x = x.reshape(x_output_shape)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # x = x.reshape(-1, C)
        x = weights.attn_proj.apply(x)
        # x = x.reshape(B, N, -1)
        return x
        
    def _infer_cross_attn(self, weights: MagicDriveTransformerAttnBlock, x, cond, mask):
        B, N, C = x.shape
        Bc, Nc, Cc = cond.shape  # [B, TS/p, C]
        assert Bc == B
        
        # x = x.reshape(-1, C)
        q = weights.cross_attn_q_linear.apply(x)
        # q = q.reshape(B, N, -1)
        
        # cond = cond.reshape(-1, Cc)
        kv = weights.cross_attn_kv_linear.apply(cond)
        # kv = kv.reshape(Bc, Nc, -1)
        
        if mask is None:  # for cond
            mask = [Nc] * B

        q = q.view(1, -1, self.num_heads, self.head_dim)
        kv = kv.view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([N] * B, mask)

        # x = xformers.ops.memory_efficient_attention(
        #     q, k, v, p=0.0, attn_bias=attn_bias, scale=self.scale
        # )
        # attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = weights.attn_xformers.apply(
            q, k, v, dropout_p=weights.attn_drop, softmax_scale=weights.softmax_scale, attn_bias=attn_bias
        )
        
        x = x.view(B, -1, C)
        # x = self.proj(x)
        # x = x.reshape(-1, C)
        x = weights.cross_attn_proj.apply(x)
        # x = x.reshape(B, N, -1)
        return x
    
    def _infer_cross_view_attn(self, weights: MagicDriveTransformerAttnBlock, x, cond):
        B, N, C = x.shape
        Bc, Nc, Cc = cond.shape
        assert B == Bc and C == Cc
        
        # x = x.reshape(-1, C)
        q = weights.cross_view_attn_q.apply(x)
        # q = q.reshape(B, N, -1)
        q = q.view(B, N, self.num_heads, self.head_dim)
        
        # cond = cond.reshape(-1, Cc)
        kv = weights.cross_view_attn_kv.apply(cond)
        # kv = kv.reshape(Bc, Nc, -1)
        kv = kv.view(B, Nc, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        
        q = weights.cross_view_attn_q_norm.apply(q)
        k = weights.cross_view_attn_k_norm.apply(k)
        # q = weights.rotary_emb.rotate_queries_or_keys(q)
        # k = weights.rotary_emb.rotate_queries_or_keys(k)
        
        x = weights.attn.apply(
            q, k, v, dropout_p=weights.attn_drop, softmax_scale=weights.softmax_scale, is_causal=weights.is_causal
        )
        
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = x.reshape(-1, C)
        x = weights.cross_view_attn_proj.apply(x)
        x = x.reshape(B, N, -1)
        return x

    def infer_block(self, weights: MagicDriveTransformerAttnBlock, x, y, t, mask, x_mask, t0, T, S, NC, mv_order_map, skip_cross_view, rope, temporal, is_control_block):
        # import pdb; pdb.set_trace()
        B, N, C = x.shape
        assert (N == T * S) and (B % NC == 0)
        b = B // NC
        # import pdb; pdb.set_trace()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = repeat(
            weights.scale_shift_table.tensor[None] + t.reshape(b, 6, -1),
            "b ... -> (b NC) ...", NC=NC,
        ).chunk(6, dim=1)
        
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = repeat(
                weights.scale_shift_table.tensor[None] + t0.reshape(b, 6, -1),
                "b ... -> (b NC) ...", NC=NC,
            ).chunk(6, dim=1)
        
        x_m = self.t2i_modulate(weights.norm1.apply(x), shift_msa, scale_msa)
        
        if x_mask is not None:
            x_m_zero = self.t2i_modulate(weights.norm1.apply(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)
        
        if temporal:
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = self._infer_self_attn(weights, x_m, rope=rope, temporal=temporal)
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = self._infer_self_attn(weights, x_m, rope=rope, temporal=temporal)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
                
        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        # drop_path = torch.nn.Identity()
        # x = x + drop_path(x_m_s)
        x = x + x_m_s

        ######################
        # cross attn
        ######################
        assert mask is None
        if y.shape[1] == 1:
            x_c = self._infer_cross_attn(weights, x, y[:, 0], mask)
        elif y.shape[1] == T:
            x_c = rearrange(x, "B (T S) C -> (B T) S C", T=T, S=S)
            y_c = rearrange(y, "B T L C -> (B T) L C", T=T)
            x_c = self._infer_cross_attn(weights, x_c, y_c, mask)
            x_c = rearrange(x_c, "(B T) S C -> B (T S) C", T=T, S=S)
        else:
            raise RuntimeError(f"unsupported y.shape[1] = {y.shape[1]}")

        # residual, we skip drop_path here
        x = x + x_c

        ######################
        # multi-view cross attention
        ######################
        if not skip_cross_view:
            assert mv_order_map is not None
            # here we re-use the first 3 parameters from t and t0
            shift_mva, scale_mva, gate_mva = repeat(
                weights.scale_shift_table_mva.tensor[None] + t[:, :3].reshape(b, 3, -1),
                "b ... -> (b NC) ...", NC=NC,
            ).chunk(3, dim=1)
            if x_mask is not None:
                shift_mva_zero, scale_mva_zero, gate_mva_zero = repeat(
                    weights.scale_shift_table_mva.tensor[None] + t0[:, :3].reshape(b, 3, -1),
                    "b ... -> (b NC) ...", NC=NC,
                ).chunk(3, dim=1)

            x_v = self.t2i_modulate(weights.norm3.apply(x), shift_mva, scale_mva)
            if x_mask is not None:
                x_v_zero = self.t2i_modulate(weights.norm3.apply(x), shift_mva_zero, scale_mva_zero)
                x_v = self.t_mask_select(x_mask, x_v, x_v_zero, T, S)

            # Prepare inputs for multiview cross attention
            x_mv = rearrange(x_v, "(B NC) (T S) C -> (B T) NC S C", NC=NC, T=T)
            x_targets, x_neighbors, cam_order = construct_attn_input_from_map(
                x_mv, mv_order_map, cat_seq=False)
            # multi-view cross attention forward with batched neighbors
            # import pdb; pdb.set_trace()
            cross_view_attn_output_raw = self._infer_cross_view_attn(
                weights, x_targets, x_neighbors)
            # arrange output tensor for sum over neighbors
            cross_view_attn_output = torch.zeros_like(x_mv)

            # cross_view_attn_output_raw [400, 350, 1152] t=20 b=1 ， c=1152
            for cam_i in range(NC):
                attn_out_mv = rearrange(
                    cross_view_attn_output_raw[cam_order == cam_i],
                    "(n_neighbors b) ... -> b n_neighbors ...",
                    b=B // NC * T,
                )
                cross_view_attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
            cross_view_attn_output = rearrange(
                cross_view_attn_output, "(B T) NC S C -> (B NC) (T S) C", T=T)

            # modulate (cross-view attention)
            x_v_s = gate_mva * cross_view_attn_output
            if x_mask is not None:
                x_v_s_zero = gate_mva_zero * cross_view_attn_output
                x_v_s = self.t_mask_select(x_mask, x_v_s, x_v_s_zero, T, S)

            # residual
            # x_v_s_shape = x_v_s.shape
            # x_v_s = x_v_s.reshape(-1, x_v_s_shape[-1])
            x_v_s = weights.mva_proj.apply(x_v_s)
            # x_v_s = x_v_s.reshape(x_v_s_shape[0], x_v_s_shape[1], -1)
            x = x + x_v_s

        ######################
        # MLP
        ######################
        x_m = self.t2i_modulate(weights.norm2.apply(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = self.t2i_modulate(weights.norm2.apply(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        # x_m = self.mlp(x_m)
        # x_m_shape = x_m.shape
        # x_m = x_m.reshape(-1, x_m_shape[-1])
        x_m = weights.mlp_fc1.apply(x_m)
        x_m = F.gelu(x_m, approximate='tanh')
        x_m = weights.mlp_fc2.apply(x_m)
        # x_m = x_m.reshape(x_m_shape[0], x_m_shape[1], -1)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + x_m_s

        if is_control_block:
            # x_shape = x.shape
            # x = x.reshape(-1, x_shape[-1])
            x_skip = weights.after_proj.apply(x)
            # x_skip = x_skip.reshape(x_shape[0], x_shape[1], -1)
            # x = x.reshape(x_shape[0], x_shape[1], -1)
            return x, x_skip
        else:
            return x
        
    def infer(self, weights: MagicDriveTransformerWeight, x, y, c, t_mlp, y_lens, x_mask, t0_mlp, T, S, NC, mv_order_map):
        for block_idx in range(0, self.control_blocks_num):
            # import pdb; pdb.set_trace()
            x = self.infer_block(
                weights.base_blocks_s_list[block_idx],
                x,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                T,
                S,
                NC,
                mv_order_map,
                skip_cross_view=False,
                rope=False,
                temporal=False,
                is_control_block=False
            )
            c, c_skip = self.infer_block(
                weights.control_blocks_s_list[block_idx],
                c,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                T,
                S,
                NC,
                mv_order_map,
                skip_cross_view=True,
                rope=False,
                temporal=False,
                is_control_block=True
            )
            x = x + c_skip
            # import pdb; pdb.set_trace()
            x = self.infer_block(
                weights.base_blocks_t_list[block_idx],
                x,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                T,
                S,
                NC,
                mv_order_map,
                skip_cross_view=True,
                rope=True,
                temporal=True,
                is_control_block=False
            )
            c, c_skip = self.infer_block(
                weights.control_blocks_t_list[block_idx],
                c,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                T,
                S,
                NC,
                mv_order_map,
                skip_cross_view=True,
                rope=True,
                temporal=True,
                is_control_block=True
            )
            x = x + c_skip
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        for block_idx in range(self.control_blocks_num, self.blocks_num):
            x = self.infer_block(
                weights.base_blocks_s_list[block_idx],
                x,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                T,
                S,
                NC,
                mv_order_map,
                skip_cross_view=False,
                rope=False,
                temporal=False,
                is_control_block=False
            )
            x = self.infer_block(
                weights.base_blocks_t_list[block_idx],
                x,
                y,
                t_mlp,
                y_lens,
                x_mask,
                t0_mlp,
                T,
                S,
                NC,
                mv_order_map,
                skip_cross_view=True,
                rope=True,
                temporal=True,
                is_control_block=False
            )
        # import pdb; pdb.set_trace()
        return x