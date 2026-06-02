import torch

from lightx2v.models.networks.wan.infer.s2v.rope import rope_precompute


def apply_framepack_motion(frame_packer, motion_latents, freqs, num_heads, inner_dim, add_last_motion, drop_mode):
    zip_frame_buckets = torch.tensor([1, 2, 16], dtype=torch.long, device=motion_latents[0].device)
    mot = []
    mot_remb = []

    for m in motion_latents:
        lat_height, lat_width = m.shape[2], m.shape[3]
        padd_lat = torch.zeros(16, zip_frame_buckets.sum(), lat_height, lat_width, device=m.device, dtype=m.dtype)
        overlap_frame = min(padd_lat.shape[1], m.shape[1])
        if overlap_frame > 0:
            padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

        if add_last_motion < 2 and drop_mode != "drop":
            zero_end_frame = zip_frame_buckets[: zip_frame_buckets.numel() - add_last_motion - 1].sum()
            padd_lat[:, -zero_end_frame:] = 0

        padd_lat = padd_lat.unsqueeze(0)
        clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -zip_frame_buckets.sum() :, :, :].split(list(zip_frame_buckets)[::-1], dim=2)

        clean_latents_post = frame_packer.proj.apply(clean_latents_post).flatten(2).transpose(1, 2)
        clean_latents_2x = frame_packer.proj_2x.apply(clean_latents_2x).flatten(2).transpose(1, 2)
        clean_latents_4x = frame_packer.proj_4x.apply(clean_latents_4x).flatten(2).transpose(1, 2)

        if add_last_motion < 2 and drop_mode == "drop":
            clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
            clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

        motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

        start_time_id = -(zip_frame_buckets[:1].sum())
        end_time_id = start_time_id + zip_frame_buckets[0]
        grid_sizes = (
            []
            if add_last_motion < 2 and drop_mode == "drop"
            else [
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([zip_frame_buckets[0], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                ]
            ]
        )

        start_time_id = -(zip_frame_buckets[:2].sum())
        end_time_id = start_time_id + zip_frame_buckets[1] // 2
        grid_sizes_2x = (
            []
            if add_last_motion < 1 and drop_mode == "drop"
            else [
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([zip_frame_buckets[1], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                ]
            ]
        )

        start_time_id = -(zip_frame_buckets[:3].sum())
        end_time_id = start_time_id + zip_frame_buckets[2] // 4
        grid_sizes_4x = [
            [
                torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 8, lat_width // 8]).unsqueeze(0).repeat(1, 1),
                torch.tensor([zip_frame_buckets[2], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
            ]
        ]
        grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

        motion_rope_emb = rope_precompute(
            motion_lat.detach().view(1, motion_lat.shape[1], num_heads, inner_dim // num_heads),
            grid_sizes,
            freqs,
            start=None,
        )
        mot.append(motion_lat)
        mot_remb.append(motion_rope_emb)
    return mot, mot_remb


def inject_motion_tokens(x_list, seq_lens, rope_embs, mask_input, motion_latents, frame_packer, freqs, config, drop_motion_frames, add_last_motion):
    if config.get("enable_framepack", True):
        mot, mot_remb = apply_framepack_motion(
            frame_packer,
            motion_latents,
            freqs,
            config["num_heads"],
            config["dim"],
            add_last_motion,
            config.get("framepack_drop_mode", "padd"),
        )
    else:
        mot, mot_remb = [], []

    if drop_motion_frames:
        mot = [m[:, :0] for m in mot]
        mot_remb = [m[:, :0] for m in mot_remb]

    if len(mot) > 0:
        x_list = [torch.cat([u, m], dim=1) for u, m in zip(x_list, mot)]
        seq_lens = seq_lens + torch.tensor([r.size(1) for r in mot], dtype=torch.long, device=seq_lens.device)
        rope_embs = [torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)]
        mask_input = [torch.cat([m, 2 * torch.ones([1, u.shape[1] - m.shape[1]], device=m.device, dtype=m.dtype)], dim=1) for m, u in zip(mask_input, x_list)]
    return x_list, seq_lens, rope_embs, mask_input
