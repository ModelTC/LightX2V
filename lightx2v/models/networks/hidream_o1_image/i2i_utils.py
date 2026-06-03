import einops
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from loguru import logger

from lightx2v.models.networks.hidream_o1_image.utils import (
    PATCH_SIZE,
    TIMESTEP_TOKEN_NUM,
    calculate_dimensions,
    create_layout_reference_images,
    find_closest_resolution,
    get_rope_index_fix_point,
    load_layout_bboxes,
    resize_pilimage,
)

CONDITION_IMAGE_SIZE = 384
T_EPS = 0.001

TENSOR_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def _resolve_target_size(height, width, ref_image_paths, keep_original_aspect):
    preresized_ref_pil = None
    if keep_original_aspect and ref_image_paths and len(ref_image_paths) == 1:
        pil_orig = Image.open(ref_image_paths[0]).convert("RGB")
        preresized_ref_pil = resize_pilimage(pil_orig, 2048, PATCH_SIZE)
        width, height = preresized_ref_pil.size
        logger.info(f"keep_original_aspect: target size set to {width}x{height} from reference image")
    else:
        if keep_original_aspect:
            logger.warning("keep_original_aspect requires exactly one reference image; falling back to default resolution snapping.")
        snapped_w, snapped_h = find_closest_resolution(width, height)
        if snapped_w != width or snapped_h != height:
            logger.warning(f"Resolution snapped from {width}x{height} to {snapped_w}x{snapped_h}")
            width, height = snapped_w, snapped_h
    return height, width, preresized_ref_pil


def build_i2i_samples(
    prompt,
    ref_image_paths,
    height,
    width,
    keep_original_aspect,
    layout_bboxes,
    tokenizer,
    processor,
    model_config,
    device,
    dtype,
    guidance_scale,
):
    height, width, preresized_ref_pil = _resolve_target_size(height, width, ref_image_paths, keep_original_aspect)

    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    spatial_merge_size = model_config.vision_config.spatial_merge_size

    if preresized_ref_pil is not None:
        ref_pils = [preresized_ref_pil]
    else:
        ref_pils = [Image.open(path).convert("RGB") for path in ref_image_paths]

    k = len(ref_pils)
    layout_data = None
    if layout_bboxes and len(str(layout_bboxes).strip()) > 0 and preresized_ref_pil is None:
        try:
            layout_data = load_layout_bboxes(layout_bboxes)
            k += 1
        except Exception as exc:
            logger.warning(f"Incorrect layout_bboxes: {layout_bboxes}, {exc}")

    if k == 1:
        max_size = max(height, width)
    elif k == 2:
        max_size = max(height, width) * 48 // 64
    elif k <= 4:
        max_size = max(height, width) // 2
    elif k <= 8:
        max_size = max(height, width) * 24 // 64
    else:
        max_size = max(height, width) // 4

    if layout_data is not None:
        ref_pils = create_layout_reference_images(
            ref_pils=ref_pils,
            layout_bboxes=layout_data,
            image_width=width,
            image_height=height,
            ref_max_size=max_size,
            patch_size=PATCH_SIZE,
        )

    ref_pils_resized = []
    ref_images = []
    for pil in ref_pils:
        if preresized_ref_pil is not None and pil is preresized_ref_pil:
            pil_r = pil
        else:
            pil_r = resize_pilimage(pil, max_size, PATCH_SIZE)
        ref_pils_resized.append(pil_r)
        x = TENSOR_TRANSFORM(pil_r)
        x = einops.rearrange(x, "C (H p1) (W p2) -> (H W) (C p1 p2)", p1=PATCH_SIZE, p2=PATCH_SIZE)
        ref_images.append(x)

    ref_image_lens = [img.shape[0] for img in ref_images]
    total_ref_len = sum(ref_image_lens)
    ref_patches = torch.cat(ref_images, dim=0).unsqueeze(0).to(device, dtype)

    tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
    h_patches = height // PATCH_SIZE
    w_patches = width // PATCH_SIZE

    if k <= 4:
        cond_img_size = CONDITION_IMAGE_SIZE
    elif k <= 8:
        cond_img_size = CONDITION_IMAGE_SIZE * 48 // 64
    else:
        cond_img_size = CONDITION_IMAGE_SIZE // 2

    ref_pils_vlm = []
    for pil_r in ref_pils_resized:
        cond_w, cond_h = calculate_dimensions(cond_img_size, pil_r.width / pil_r.height)
        ref_pils_vlm.append(pil_r.resize((cond_w, cond_h), resample=Image.LANCZOS))

    image_grid_thw_tgt = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)
    image_grid_thw_ref = torch.zeros((k, 3), dtype=torch.int64)
    for i, pil_r in enumerate(ref_pils_resized):
        rw, rh = pil_r.size
        image_grid_thw_ref[i] = torch.tensor([1, rh // PATCH_SIZE, rw // PATCH_SIZE], dtype=torch.int64)

    samples = []
    captions = [prompt]
    if guidance_scale > 1.0:
        captions.append(" ")

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    for caption in captions:
        content = [{"type": "image"} for _ in range(k)]
        content.append({"type": "text", "text": caption})
        messages = [{"role": "user", "content": content}]
        template_caption = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc = processor(text=[template_caption], images=ref_pils_vlm, padding="longest", return_tensors="pt")
        input_ids_2 = tokenizer.encode(boi_token + tms_token * TIMESTEP_TOKEN_NUM, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([proc.input_ids, input_ids_2], dim=-1)

        igthw_cond = proc.image_grid_thw.clone()
        for i in range(k):
            igthw_cond[i, 1] //= spatial_merge_size
            igthw_cond[i, 2] //= spatial_merge_size
        igthw_all = torch.cat([igthw_cond, image_grid_thw_tgt, image_grid_thw_ref], dim=0)

        vision_tokens_list = []
        vt_tgt = torch.full((1, tgt_image_len), image_token_id, dtype=input_ids.dtype)
        vt_tgt[0, 0] = vision_start_token_id
        vision_tokens_list.append(vt_tgt)
        for rl in ref_image_lens:
            vt_ref = torch.full((1, rl), image_token_id, dtype=input_ids.dtype)
            vt_ref[0, 0] = vision_start_token_id
            vision_tokens_list.append(vt_ref)
        vision_tokens = torch.cat(vision_tokens_list, dim=1)
        input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

        position_ids, _ = get_rope_index_fix_point(
            1,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            input_ids=input_ids_pad,
            image_grid_thw=igthw_all,
            video_grid_thw=None,
            attention_mask=None,
            skip_vision_start_token=[0] * k + [1] + [1] * k,
        )
        txt_seq_len = input_ids.shape[-1]
        all_seq_len = position_ids.shape[-1]

        token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
        bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
        end = bgn + tgt_image_len + TIMESTEP_TOKEN_NUM
        token_types_raw[0, bgn:end] = 1
        token_types_raw[0, end : end + total_ref_len] = 2
        token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len] = 3

        vinput_mask = torch.logical_or(token_types_raw == 1, token_types_raw == 2)
        token_types_bin = (token_types_raw > 0).to(token_types_raw.dtype)

        samples.append(
            {
                "input_ids": input_ids.to(device),
                "position_ids": position_ids.to(device),
                "token_types": token_types_bin.to(device),
                "vinput_mask": vinput_mask.to(device),
                "pixel_values": proc.pixel_values.to(device, dtype),
                "image_grid_thw": proc.image_grid_thw.to(device),
            }
        )

    return {
        "samples": samples,
        "ref_patches": ref_patches,
        "height": height,
        "width": width,
        "h_patches": h_patches,
        "w_patches": w_patches,
        "tgt_image_len": tgt_image_len,
    }
