import math
import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from lightx2v.models.schedulers.mgcdr.scheduler import MagicDriverScheduler
from lightx2v.models.networks.mgcdr.weights.pre_weights import MagicDrivePreWeights, MagicDriveSelfAttention


def cog_temp_down(x):
    batch_size, T, N, hidden_size = x.shape
    # B T N D -> B N D T
    x = x.permute(0, 2, 3, 1).reshape(batch_size * N, hidden_size, T)
    if x.shape[-1] % 2 == 1:
        x_first, x_rest = x[..., 0], x[..., 1:]
        if x_rest.shape[-1] > 0:
            # (batch_size * N, channels, frames - 1) -> (batch_size * N, channels, (frames - 1) // 2)
            # if DEVICE_TYPE == "npu":
            #     x_rest = avg_pool1d_using_conv2d(x_rest, kernel_size=2, stride=2)
            # else:
            x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

        x = torch.cat([x_first[..., None], x_rest], dim=-1)
        # (batch_size * N, channels, (frames // 2) + 1) -> (batch_size, N, channels, (frames // 2) + 1) -> (batch_size, (frames // 2) + 1, N, channels)
        x = x.reshape(batch_size, N, hidden_size, x.shape[-1]).permute(0, 3, 1, 2)
    else:
        # (batch_size * N, channels, frames) -> (batch_size * N, channels, frames // 2)
        # if DEVICE_TYPE == "npu":
        #     x = avg_pool1d_using_conv2d(x, kernel_size=2, stride=2)
        # else:
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        # (batch_size * N, channels, frames // 2) -> (batch_size, N, channels, frames // 2) -> (batch_size, frames // 2, N, channels)
        x = x.reshape(batch_size, N, hidden_size, x.shape[-1]).permute(0, 3, 1, 2)
    return x

def pad_at_dim(t, pad, dim=-1):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode="constant")


class MagicDrivePreInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = self.config.get('patch_size', (1,2,2))
        self.uncond_prob = self.config.get('uncond_prob', 0.1)
        self.frequency_embedding_size = self.config.get('frequency_embedding_size', 256)
        self.fps_outdim = self.config.get('fps_outdim', 1152)
        self.class_dropout_prob = self.config.get('class_dropout_prob', 0.1)
        self.micro_frame_size = self.config.get('micro_frame_size', 32)
        self.input_sq_size = self.config.get('input_sq_size', 512)
        self.frequency_embedding_size = self.config.get('frequency_embedding_size', 256)
        
        self.dtype = torch.bfloat16
        
        self.downsampler = lambda x: cog_temp_down(cog_temp_down(x))
        
    def set_scheduler(self, scheduler: MagicDriverScheduler):
        self.scheduler = scheduler
        
    def pad_input(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        return x
        
    def t2i_modulate(self, x, shift, scale):
        return x * (1 + scale) + shift
        
    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        embedding = embedding.to(self.dtype)
        return embedding
    
    def token_drop(self, weights: MagicDrivePreWeights, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], weights.y_embedder_y_embedding.tensor, caption)
        return caption
        
    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size() # self.patch_size (1, 2, 2)
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)
    
    def encode_text(self, weights: MagicDrivePreWeights, y, mask=None, drop_cond_mask=None):
        # NOTE: we do not use y mask, but keep the batch dim.
        # NOTE: we do not use drop in y_embedder
        
        '''
        self.y_embedder
        '''
        # import pdb; pdb.set_trace()
        if drop_cond_mask is not None:
            force_drop_ids = 1 - drop_cond_mask
            y = self.token_drop(weights, y, force_drop_ids)
        # import pdb; pdb.set_trace()
        # y = y.squeeze()
        y = weights.y_embedder_fc1.apply(y)
        y = F.gelu(y, approximate='tanh')
        y = weights.y_embedder_fc2.apply(y)
        # y = y.unsqueeze(0).unsqueeze(0)
        
        
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            y_lens = [i + 1 for i in mask.sum(dim=1).tolist()]
            max_len = int(min(max(y_lens), y.shape[2]))  # we need min because of +1
            if drop_cond_mask is not None and not drop_cond_mask.all():  # on any drop, this should be the max
                assert max_len == y.shape[2]
            # y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y = y.squeeze(1)[:, :max_len]
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1)
        return y, y_lens
    
    def _infer_self_attn(self, weights: MagicDriveSelfAttention, x, rope): 
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
        x = x.reshape(-1, C)
        qkv = weights.attn_qkv.apply(x)
        qkv = qkv.reshape(B, N, -1)
        qkv_shape = (B, N, 3, weights.num_heads, weights.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # q, k = self.q_norm(q), self.k_norm(k)
        # import pdb; pdb.set_trace()
        q = weights.attn_q_norm.apply(q)
        k = weights.attn_k_norm.apply(k)
        
        if rope:
            q = weights.rotary_emb.rotate_queries_or_keys(q)
            k = weights.rotary_emb.rotate_queries_or_keys(k)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        x = weights.attn.apply(
            q, k, v, dropout_p=weights.attn_drop, softmax_scale=weights.softmax_scale, is_causal=weights.is_causal
        )
        # import pdb; pdb.set_trace()
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = x.reshape(-1, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        x = weights.attn_proj.apply(x)
        x = x.reshape(B, N, -1)
        return x
    
    def _encode_box(self, weights: MagicDrivePreWeights, bboxes, classes, null_mask=None, mask=None, box_latent=None):
        # import pdb; pdb.set_trace()
        (B, N) = classes.shape
        bboxes = rearrange(bboxes, 'b n ... -> (b n) ...')
        
        def handle_none_mask(_mask):
            if _mask is None:
                _mask = torch.ones(len(bboxes))
            else:
                _mask = _mask.flatten()
            _mask = _mask.unsqueeze(-1).type_as(weights.bbox_embedder_null_pos_feature.tensor)
            return _mask
        
        mask = handle_none_mask(mask)
        null_mask = handle_none_mask(null_mask)
        
        # pos_emb = self.fourier_embedder(bboxes)
        pos_emb = weights.fourier_embedder(bboxes)
        pos_emb = pos_emb.reshape(
            pos_emb.shape[0], -1).type_as(weights.bbox_embedder_null_pos_feature.tensor)
        pos_emb = pos_emb * null_mask + weights.bbox_embedder_null_pos_feature.tensor[None] * (1 - null_mask)
        pos_emb = pos_emb * mask + weights.bbox_embedder_mask_pos_feature.tensor[None] * (1 - mask)
        
        # cls_emb = self.class_tokens[classes.flatten()]
        cls_emb = weights.bbox_embedder_class_tokens.tensor[classes.flatten()]
        # mean_var = self.mean_var[classes.flatten()]
        mean_var = weights.bbox_embedder_mean_var.tensor[classes.flatten()]
        mu, logvar = torch.split(mean_var, 1, dim=1)
        std = torch.exp(0.5 * logvar)
        if box_latent is None:
            box_latent = torch.randn_like(cls_emb)
        else:
            box_latent = rearrange(box_latent, 'b n ... -> (b n) ...')
        assert box_latent.shape == cls_emb.shape
        box_latent = box_latent * std + mu
        cls_emb = cls_emb + box_latent
        # cls_emb = cls_emb * null_mask + self.null_class_feature[None] * (1 - null_mask)
        cls_emb = cls_emb * null_mask + weights.bbox_embedder_null_class_feature.tensor[None] * (1 - null_mask)
        # cls_emb = cls_emb * mask + self.mask_class_feature[None] * (1 - mask)
        cls_emb = cls_emb * mask + weights.bbox_embedder_mask_class_feature.tensor[None] * (1 - mask)

        # combine
        # emb = self.forward_feature(pos_emb, cls_emb)
        # emb = self.bbox_proj(pos_emb)
        # import pdb; pdb.set_trace()
        emb = weights.bbox_embedder_bbox_proj.apply(pos_emb)
        emb = F.silu(emb)
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = weights.bbox_embedder_second_linear_0.apply(emb)
        emb = F.silu(emb)
        emb = weights.bbox_embedder_second_linear_2.apply(emb)
        emb = F.silu(emb)
        emb = weights.bbox_embedder_second_linear_4.apply(emb)
        emb = rearrange(emb, '(b n) ... -> b n ...', n=N)
        # emb = self.second_linear(emb)
        
        
        # emb = rearrange(emb, '(b n) ... -> b n ...', n=N)
        return emb
        
    
    def encode_box(self, weights: MagicDrivePreWeights, bboxes, drop_mask):  # changed
        # import pdb; pdb.set_trace()
        B, T, seq_len = bboxes['bboxes'].shape[:3]
        bbox_embedder_kwargs = {}
        for k, v in bboxes.items():
            bbox_embedder_kwargs[k] = v.clone()
        # each key should have dim like: (b, seq_len, dim...)
        # bbox_embedder_kwargs["masks"]: 0 -> null, -1 -> mask, 1 -> keep
        # drop_mask: 0 -> mask, 1 -> keep
        drop_mask = repeat(drop_mask, "B T -> B T S", S=seq_len)
        _null_mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _null_mask[bbox_embedder_kwargs["masks"] == 0] = 0
        _mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _mask[bbox_embedder_kwargs["masks"] == -1] = 0
        _mask[torch.logical_and(
            bbox_embedder_kwargs["masks"] == 1,
            drop_mask == 0,  # only drop those real boxes
        )] = 0
        
        bboxes=bbox_embedder_kwargs['bboxes']
        classes=bbox_embedder_kwargs["classes"].type(torch.int32)
        box_latent=bbox_embedder_kwargs.get('box_latent', None)
        null_mask = _null_mask
        mask = _mask
        
        B, T, N = classes.shape
        bboxes = rearrange(bboxes, 'b t n ... -> (b t) n ...')
        classes = rearrange(classes, 'b t n -> (b t) n')
        if box_latent is not None:
            box_latent = rearrange(box_latent, 'b t n ... -> (b t) n ...')
        if null_mask is not None:
            null_mask = rearrange(null_mask, 'b t n -> (b t) n')
        if mask is not None:
            mask = rearrange(mask, 'b t n -> (b t) n')
        
        bbox_emb = self._encode_box(weights, bboxes, classes, null_mask, mask, box_latent)
        bbox_emb = rearrange(bbox_emb, '(b t) n d -> (b n) t d', t=T)
        # import pdb; pdb.set_trace()
        if weights.bbox_embedder_scale_shift_table.tensor is not None:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = (
                weights.bbox_embedder_scale_shift_table.tensor[None]
            ).chunk(6, dim=1)
        else:
            shift_mha = shift_mlp = 0.
            scale_mha = scale_mlp = 0.
            gate_mha = gate_mlp = 1.

        x = bbox_emb
        # x_m = self.t2i_modulate(self.norm1(x), shift_mha, scale_mha)
        x_m = self.t2i_modulate(weights.bbox_embedder_norm1.apply(x), shift_mha, scale_mha)
        # x_m = self.attn(x_m)
        x_m = self._infer_self_attn(weights.bbox_embedder_attn, x_m, rope=True)
        x_m = gate_mha * x_m
        x = x + x_m

        x_m = self.t2i_modulate(weights.bbox_embedder_norm2.apply(x), shift_mlp, scale_mlp)
        # x_m = self.mlp(x_m)
        
        # x_m_shape = x_m.shape
        # x_m = x_m.reshape(-1, x_m_shape[-1])
        x_m = weights.bbox_embedder_mlp_fc1.apply(x_m)
        x_m = F.gelu(x_m, approximate='tanh')
        x_m = weights.bbox_embedder_mlp_fc2.apply(x_m)
        # x_m = x_m.reshape(x_m_shape[0], x_m_shape[1], -1)
        
        x_m = gate_mlp * x_m
        # import pdb; pdb.set_trace()
        x = x + x_m

        x = rearrange(x, '(b n) t d -> b t n d', b=B, n=N)
        # if self.final_proj:
        # x = self.final_proj(x)
        # x_shape = x.shape
        # x = x.reshape(-1, x_shape[-1])
        # import pdb; pdb.set_trace()
        x = weights.bbox_embedder_final_proj.apply(x)
        # x = x.reshape(x_shape[0], x_shape[1], x_shape[2], -1)
        x = self.downsampler(x)
        return x
    
    def encode_cam(self, weights: MagicDrivePreWeights, cam, drop_mask): # camera_embedder
        B, T, S = cam.shape[:3]
        NC = B // drop_mask.shape[0]
        mask = repeat(drop_mask, "b T -> (b NC T S)", NC=NC, S=S)
        cam = rearrange(cam, "B T S ... -> (B T S) ...")
        param = cam
        if param.shape[1] == 4:
            param = param[:, :-1]
        (bs, C_param, emb_num) = param.shape
        assert C_param == 3
        if mask is not None:
            param = torch.where((mask > 0)[:, None, None], param, weights.camera_embedder_uncond_cam.tensor[None])
        emb = weights.fourier_embedder(rearrange(param, "b d c -> (b c) d"))
        emb = rearrange(emb, "(b c) d -> b (c d)", b=bs)
        token = weights.camera_embedder_emb2token.apply(emb)
        token = weights.camera_embedder_after_proj.apply(token)
        cam_emb = token
        # cam_emb, _ = embedder.embed_cam(cam, mask, T=T, S=S)  # changed here
        # cam_emb = rearrange(cam_emb, "(B T S) ... -> B T S ...", B=B, T=T, S=S)
        return cam_emb
    
    def encode_frame(self, weights: MagicDrivePreWeights, frame, drop_mask): # frame_embedder
        B, T, S = frame.shape[:3]
        NC = B // drop_mask.shape[0]
        mask = repeat(drop_mask, "b T -> (b NC T S)", NC=NC, S=S)
        frame = rearrange(frame, "B T S ... -> (B T S) ...")
        param = frame
        
        # token, emb = super().embed_cam(param, mask)
        if param.shape[1] == 4:
            param = param[:, :-1]
        (bs, C_param, emb_num) = param.shape
        assert C_param == 3
        # apply mask
        if mask is not None:
            param = torch.where((mask > 0)[:, None, None], param, weights.frame_embedder_uncond_cam.tensor[None])
            
        emb = weights.fourier_embedder(rearrange(param, "b d c -> (b c) d"))
        emb = rearrange(emb, "(b c) d -> b (c d)", b=bs)
        # token = self.emb2token(emb)
        # token = self.after_proj(token)
        token = weights.frame_embedder_emb2token.apply(emb)
        # token = weights.frame_embedder_after_proj.apply(token)
        # import pdb; pdb.set_trace()
        token = rearrange(token, '(b T S) d -> (b S) T d', T=T, S=S)
        
        shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = (
                weights.frame_embedder_scale_shift_table.tensor[None]
            ).chunk(6, dim=1)
        
        x = token
        # x_m = self.t2i_modulate(self.norm1(x), shift_mha, scale_mha)
        x_m = self.t2i_modulate(weights.frame_embedder_norm1.apply(x), shift_mha, scale_mha)
        # x_m = self._infer(x_m)
        x_m = self._infer_self_attn(weights.frame_embedder_attn, x_m, rope=True)
        x_m = gate_mha * x_m
        x = x + x_m

        x_m = self.t2i_modulate(weights.frame_embedder_norm2.apply(x), shift_mlp, scale_mlp)
        # x_m = self.mlp(x_m)
        # import pdb; pdb.set_trace()
        x_m_shape = x_m.shape
        x_m = x_m.reshape(-1, x_m_shape[-1])
        x_m = weights.frame_embedder_mlp_fc1.apply(x_m)
        x_m = F.gelu(x_m, approximate='tanh')
        x_m = weights.frame_embedder_mlp_fc2.apply(x_m)
        x_m = x_m.reshape(x_m_shape[0], x_m_shape[1], -1)
        
        x_m = gate_mlp * x_m
        x = x + x_m

        x = rearrange(x, '(b S) T d -> b T S d', S=S, T=T)
        # if self.final_proj:
        #     x = self.final_proj(x)
        # import pdb; pdb.set_trace()
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        x = weights.frame_embedder_final_proj.apply(x)
        x = x.reshape(x_shape[0], x_shape[1], x_shape[2], -1)
        x = self.downsampler(x)
        frame_emb = x
        # frame_emb, _ = embedder.embed_cam(frame, mask, T=T, S=S)  # changed here
        # cam_emb = rearrange(cam_emb, "(B T S) ... -> B T S ...", B=B, T=T, S=S)
        return frame_emb
    
    def encode_map(self, weights: MagicDrivePreWeights, maps, NC, h_pad_size, x_shape):
        view_wise = False
        if len(maps.shape) == 6:  # B V T C H W:
            assert maps.shape[1] == NC
            maps = rearrange(maps, "V B ... -> (V B) ...")
            view_wise = True
        b, T = maps.shape[:2]  # [bs, T, 8, 400, 400]
        maps = rearrange(maps, "b T ... -> (b T) ...")
        # controlnet_cond = self.controlnet_cond_embedder(maps)
        
        controlnet_cond = maps
        # controlnet_cond = self._random_use_uncond_map(controlnet_cond)
        conditioning = controlnet_cond
        embedding = weights.controlnet_cond_embedder_conv_in.apply(conditioning)
        embedding = F.silu(embedding)
        embedding = weights.controlnet_cond_embedder_blocks_0.apply(embedding)
        embedding = F.silu(embedding)
        embedding = weights.controlnet_cond_embedder_blocks_1.apply(embedding)
        embedding = F.silu(embedding)
        embedding = weights.controlnet_cond_embedder_blocks_2.apply(embedding)
        embedding = F.silu(embedding)
        embedding = weights.controlnet_cond_embedder_blocks_3.apply(embedding)
        embedding = F.silu(embedding)
        embedding = weights.controlnet_cond_embedder_blocks_4.apply(embedding)
        embedding = F.silu(embedding)
        embedding = weights.controlnet_cond_embedder_blocks_5.apply(embedding)
        embedding = F.silu(embedding)
        embedding = weights.controlnet_cond_embedder_conv_out.apply(embedding)
        controlnet_cond = embedding
        
        controlnet_cond = rearrange(controlnet_cond, "(b T) C ... -> b C T ...", T=T)
        
        z_list = []
        for i in range(0, controlnet_cond.shape[2], self.micro_frame_size):
            x = controlnet_cond[:, :, i: i + self.micro_frame_size]
            # F.pad(input, self.padding, "constant", self.value)
            time_padding = 0
            x = pad_at_dim(x, (time_padding, 0), dim=2)
            x = F.pad(x, (1,0,1,0), "constant", value=0)
            batch_size, channels, frames, height, width = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(batch_size, frames, height * width, channels)
            x = cog_temp_down(x)
            x = x.reshape(batch_size, x.shape[1], height, width, channels).permute(0, 4, 1, 2, 3)
            x = F.pad(x, (0,1,0,1), mode="constant", value=0)
            batch_size, channels, frames, height, width = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
            x = weights.controlnet_cond_embedder_temp_conv_blocks_1_conv.apply(x)
            x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
            
            x = F.pad(x, (1,0,1,0), "constant", value=0)
            batch_size, channels, frames, height, width = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(batch_size, frames, height * width, channels)
            x = cog_temp_down(x)
            x = x.reshape(batch_size, x.shape[1], height, width, channels).permute(0, 4, 1, 2, 3)
            x = F.pad(x, (0,1,0,1), mode="constant", value=0)
            batch_size, channels, frames, height, width = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
            x = weights.controlnet_cond_embedder_temp_conv_blocks_3_conv.apply(x)
            x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
            z_list.append(x)
        controlnet_cond = torch.cat(z_list, dim=2)
        _controlnet_cond = []
        for ci in range(controlnet_cond.shape[0]):
            _controlnet_cond.append(
                F.interpolate(controlnet_cond[ci:ci + 1], x_shape[-3:])
            )
        controlnet_cond = torch.cat(_controlnet_cond, dim=0)
        if h_pad_size > 0:
            hx_pad_size = h_pad_size * self.patch_size[1]
            # pad c along the H dimension
            controlnet_cond = F.pad(controlnet_cond, (0, 0, 0, hx_pad_size))
        # controlnet_cond = self.controlnet_cond_patchifier(controlnet_cond)
        x = controlnet_cond
        x = self.pad_input(x)
        # _, _, D, H, W = x.size()
        # if W % self.patch_size[2] != 0:
        #     x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        # if H % self.patch_size[1] != 0:
        #     x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        # if D % self.patch_size[0] != 0:
        #     x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        # x = self.proj(x)  # (B C T H W)
        x = weights.controlnet_cond_patchifier_proj.apply(x)
        x = x.flatten(2).transpose(1, 2)
        controlnet_cond = x
        return controlnet_cond
    
    def encode_cond_sequence(self, weights, bbox, cams, rel_pos, y, mask, drop_cond_mask, drop_frame_mask):
        # import pdb; pdb.set_trace()
        b = len(y)
        NC, T = cams.shape[0] // b, cams.shape[1]
        cond = []

        # encode y
        y, _ = self.encode_text(weights, y, mask, drop_cond_mask)  # b, seq_len, dim
        # return y, None # change me!
        y = repeat(y, "b ... -> (b NC) ...", NC=NC)
        # cond.append(y)

        # encode box
        if bbox is not None:
            drop_box_mask = torch.logical_and(drop_cond_mask[:, None], drop_frame_mask)  # b, T
            drop_box_mask = repeat(drop_box_mask, "b ... -> (b NC) ...", NC=NC)
            bbox_emb = self.encode_box(weights, bbox, drop_mask=drop_box_mask)  # B, T, box_len, dim
            # bbox_emb = bbox_emb.mean(1)  # pooled token
            # zero proj on base token
            bbox_emb = weights.base_token.tensor[None, None, None] + bbox_emb
            cond.append(bbox_emb)

        # encode cam, just take from first frame
        cam_emb = self.encode_cam(
            weights, cams[:, 0:1], repeat(drop_cond_mask, "b -> b T", T=1)) # self.camera_embedder
        frame_emb = self.encode_frame(weights, rel_pos, drop_frame_mask) # self.frame_embedder
        # frame_emb = self.encode_cam(weights, rel_pos, self.frame_embedder, drop_frame_mask)
        cam_emb = rearrange(cam_emb, "(B 1 S) ... -> B 1 S ...", S=cams.shape[2])
        cam_emb = weights.base_token.tensor[None, None, None] + cam_emb
        frame_emb = weights.base_token.tensor[None, None, None] + frame_emb

        cam_emb = repeat(cam_emb, 'B 1 S ... -> B T S ...', T=frame_emb.shape[1])
        y = repeat(y, "B ... -> B T ...", T=frame_emb.shape[1])
        cond = [frame_emb, cam_emb, y] + cond
        cond = torch.cat(cond, dim=2)  # B, T, len, dim
        return cond, None
        
    def infer(self, weights: MagicDrivePreWeights, x, timestep, y, maps, bbox, cams, rel_pos, fps, height, width, drop_cond_mask=None, drop_frame_mask=None, mv_order_map=None, t_order_map=None, mask=None, x_mask=None, **kwargs):
        # import pdb; pdb.set_trace()
        
        B, real_T = x.size(0), rel_pos.size(1)
        NC = len(mv_order_map)
        x = x.to(self.dtype)
        x = rearrange(x, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        
        _, _, Tx, Hx, Wx = x.size()
        x_in_shape = x.shape  # before pad
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        
        h_pad_size = 0
        
        base_size = round(S ** 0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        
        pos_emb = weights.pos_embedding(x, H, W, scale, base_size)
        # t = weights.timestep_embedding.apply(timestep, dtype=x.dtype)
        t = self.timestep_embedding(timestep, self.frequency_embedding_size)
        # import pdb; pdb.set_trace()
        # t = t.to(self.dtype)
        t = weights.t_embedder_0.apply(t)
        t = F.silu(t)
        t = weights.t_embedder_2.apply(t)
        
        s = fps.unsqueeze(1)
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != B:
            s = s.repeat(B // s.shape[0], 1)
            assert s.shape[0] == B
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size)
        s_freq = s_freq.to(self.dtype)
        # import pdb; pdb.set_trace()
        fps = weights.fps_embedder_0.apply(s_freq)
        fps = F.silu(fps)
        fps = weights.fps_embedder_2.apply(fps)
        fps = rearrange(fps, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.fps_outdim)
        
        t = t + fps
        # t_block -> silu + linear
        t_mlp = F.silu(t) # silu
        t_mlp = weights.t_block_1.apply(t_mlp) # linear
        t0 = t0_mlp = None
        
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            # import pdb; pdb.set_trace()
            t0 = self.timestep_embedding(t0_timestep, self.frequency_embedding_size)
            t0 = weights.t_embedder_0.apply(t0)
            t0 = F.silu(t0)
            t0 = weights.t_embedder_2.apply(t0)
            
            t0 = t0 + fps
            # t_block -> silu + linear
            t0_mlp = F.silu(t0)
            t0_mlp = weights.t_block_1.apply(t0_mlp)
            
        # -------------------------------------
        
        y, y_lens = self.encode_cond_sequence(weights, 
            bbox, cams, rel_pos, y, mask, drop_cond_mask, drop_frame_mask)  # (B, L, D)
        # import pdb; pdb.set_trace()
        c = self.encode_map(weights, maps, NC, h_pad_size, x_in_shape)
        c = rearrange(c, "B (T S) C -> B T S C", T=T)  # [B, T, S, 1152]
        
        # x_b = weights.x_embedder(x)  # [B, N, C]
        # import pdb; pdb.set_trace()
        x_b = self.pad_input(x)
        
        # _, _, D, H, W = x_b.size()
        # if W % self.patch_size[2] != 0:
        #     x_b = F.pad(x_b, (0, self.patch_size[2] - W % self.patch_size[2]))
        # if H % self.patch_size[1] != 0:
        #     x_b = F.pad(x_b, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        # if D % self.patch_size[0] != 0:
        #     x_b = F.pad(x_b, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        # x = self.proj(x)
        x_b = weights.x_embedder_proj.apply(x_b)
        x_b = x_b.flatten(2).transpose(1, 2)
        
        
        x_b = rearrange(x_b, "B (T S) C -> B T S C", T=T, S=S)
        x_b = x_b + pos_emb
        
        x_c = self.pad_input(x)
        # import pdb; pdb.set_trace()
        # _, _, D, H, W = x_c.size()
        # if W % self.patch_size[2] != 0:
        #     x_c = F.pad(x_c, (0, self.patch_size[2] - W % self.patch_size[2]))
        # if H % self.patch_size[1] != 0:
        #     x_c = F.pad(x_c, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        # if D % self.patch_size[0] != 0:
        #     x_c = F.pad(x_c, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        # x = self.proj(x)
        # import pdb; pdb.set_trace()
        x_c = weights.x_control_embedder_proj.apply(x_c)
        x_c = x_c.flatten(2).transpose(1, 2)
        # x_c = weights.x_control_embedder(x)  # controlnet has another embedder!
        
        
        x_c = rearrange(x_c, "B (T S) C -> B T S C", T=T, S=S)
        x_c = x_c + pos_emb
        
        # c_shape = c.shape
        # c = c.reshape(-1, c_shape[-1])
        c = weights.before_proj.apply(c)
        # c = c.reshape(c_shape[0], c_shape[1], c_shape[2], -1)
        # import pdb; pdb.set_trace()
        c = x_c + c
        
        
        x = x_b
        
        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        c = rearrange(c, "B T S C -> B (T S) C", T=T, S=S)
        
        if x_mask is not None:
            x_mask = repeat(x_mask, "b ... -> (b NC) ...", NC=NC)
            
        return x, y, c, t, t_mlp, y_lens, x_mask, t0, t0_mlp, T, H, W, S, NC, Tx, Hx, Wx, mv_order_map
        