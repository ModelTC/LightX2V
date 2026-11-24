# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.hymotion.weights.motion import (
    Any,
    Dict,
    List,
    Optional,
    Tensor,
    Union,
    get_module_device,
    length_to_mask,
    matrix_to_quaternion,
    np,
    odeint,
    quaternion_fix_continuity,
    quaternion_to_matrix,
    rot6d_to_rotation_matrix,
    rotation_matrix_to_rot6d,
    savgol_filter,
    smooth_rotation,
    torch,
)


def MotionGeneration_decode_motion_from_latent(self, latent: Tensor, should_apply_smooothing: bool = True) -> Dict[str, Tensor]:
    std_zero = self.std < 0.001
    std = torch.where(std_zero, torch.zeros_like(self.std), self.std)
    latent_denorm = latent * std + self.mean
    return self._decode_o6dp(latent_denorm, num_joints=22, rel_trans=False, should_apply_smooothing=should_apply_smooothing)


def MotionGeneration__decode_o6dp(self, latent_denorm: torch.Tensor, num_joints: int, rel_trans: bool = False, should_apply_smooothing: bool = True) -> dict:
    device = get_module_device(self)
    B, L = latent_denorm.shape[:2]
    nj = num_joints
    body_n = nj - 1
    if not rel_trans:
        transl = latent_denorm[..., 0:3].clone()
    else:
        transl = torch.cumsum(latent_denorm[..., 0:3].clone(), dim=1) / self.output_mesh_fps
    root_rot6d = latent_denorm[..., 3:9].reshape(B, L, 1, 6).clone()
    body6d_start = 9
    body6d_end = body6d_start + body_n * 6
    body_rot6d_full = latent_denorm[..., body6d_start:body6d_end].clone().reshape(B, L, body_n, 6)
    left_hand_pose = right_hand_pose = None
    if nj == 52:
        body_rot6d = body_rot6d_full[:, :, :21, :].clone()
        left_hand_pose = body_rot6d_full[:, :, 21:36, :].clone()
        right_hand_pose = body_rot6d_full[:, :, 36:51, :].clone()
    else:
        body_rot6d = body_rot6d_full
    if left_hand_pose is not None and right_hand_pose is not None:
        body_full = torch.cat([body_rot6d, left_hand_pose, right_hand_pose], dim=2)
    else:
        body_full = body_rot6d
    rot6d = torch.cat([root_rot6d, body_full], dim=2)
    if should_apply_smooothing:
        rot6d_body = rot6d[:, :, :22, :]
        rot6d_fingers = rot6d[:, :, 22:, :]
        rot6d_body_smooth = self.smooth_with_slerp(rot6d_body, sigma=1.0)
        rot6d_smooth = torch.cat([rot6d_body_smooth, rot6d_fingers], dim=2)
    else:
        rot6d_smooth = rot6d
    root_rotmat_smooth = rot6d_to_rotation_matrix(rot6d_smooth[:, :, 0, :])
    transl_fixed = transl.detach()
    if should_apply_smooothing:
        transl_smooth = self.smooth_with_savgol(transl_fixed.detach(), window_length=11, polyorder=5)
    else:
        transl_smooth = transl_fixed
    if self.body_model is not None:
        print(f"{self.__class__.__name__} rot6d_smooth shape: {rot6d_smooth.shape}, transl_smooth shape: {transl_smooth.shape}")
        with torch.no_grad():
            vertices_all = []
            k3d_all = []
            for bs in range(rot6d_smooth.shape[0]):
                out = self.body_model.forward({"rot6d": rot6d_smooth[bs], "trans": transl_smooth[bs]})
                vertices_all.append(out["vertices"])
                k3d_all.append(out["keypoints3d"])
            vertices = torch.stack(vertices_all, dim=0)
            k3d = torch.stack(k3d_all, dim=0)
        print(f"{self.__class__.__name__} vertices shape: {vertices.shape}, k3d shape: {k3d.shape}")
        min_y = vertices[..., 1].amin(dim=(1, 2), keepdim=True)
        print(f"{self.__class__.__name__} min_y: {min_y}")
        k3d = k3d.clone()
        k3d[..., 1] -= min_y
        transl_smooth = transl_smooth.clone()
        transl_smooth[..., 1] -= min_y.squeeze(-1).to(device)
    else:
        k3d = torch.zeros(B, L, nj, 3, device=device)
    return {
        "latent_denorm": latent_denorm.cpu().detach(),
        "keypoints3d": k3d.cpu().detach(),
        "rot6d": rot6d_smooth.cpu().detach(),
        "transl": transl_smooth.cpu().detach(),
        "root_rotations_mat": root_rotmat_smooth.cpu().detach(),
    }


def MotionGeneration_smooth_with_savgol(input: torch.Tensor, window_length: int = 9, polyorder: int = 5) -> torch.Tensor:
    if len(input.shape) == 2:
        is_batch = False
        input = input.unsqueeze(0)
    else:
        is_batch = True
    input_np = input.cpu().numpy()
    input_smooth_np = np.empty_like(input_np, dtype=np.float32)
    for b in range(input_np.shape[0]):
        for j in range(input_np.shape[2]):
            input_smooth_np[b, :, j] = savgol_filter(input_np[b, :, j], window_length, polyorder)
    input_smooth = torch.from_numpy(input_smooth_np).to(input)
    if not is_batch:
        input_smooth = input_smooth.squeeze(0)
    return input_smooth


def MotionGeneration_smooth_with_slerp(input: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    def fix_time_continuity(q: Tensor, time_dim: int = -3):
        shape = q.shape
        qv = q.moveaxis(time_dim, 0).contiguous().view(shape[time_dim], -1, 4)
        qv = quaternion_fix_continuity(qv)
        return qv.view(shape[time_dim], *shape[:time_dim], *shape[time_dim + 1 :]).moveaxis(0, time_dim)

    num_joints = input.shape[2]
    RR = rot6d_to_rotation_matrix(input)
    qq = matrix_to_quaternion(RR)
    qq_np = fix_time_continuity(qq, time_dim=1).cpu().numpy()
    qq_s_np = smooth_rotation(qq_np, sigma=sigma)
    input_smooth = rotation_matrix_to_rot6d(quaternion_to_matrix(torch.from_numpy(qq_s_np)))
    return input_smooth.to(input.device)


def MotionGeneration_noise_from_seeds(latent: Tensor, seeds: Union[int, List[int]], seed_start: int = 0, random_generator_on_gpu: bool = True) -> Tensor:
    if isinstance(seeds, int):
        seeds = list(range(seeds))
    noise_list = []
    B = latent.shape[0]
    shape = (B, *latent.shape[1:])
    for seed in seeds:
        if random_generator_on_gpu:
            generator = torch.Generator(device=latent.device).manual_seed(seed + seed_start)
            noise_sample = torch.randn(shape, generator=generator, device=latent.device, dtype=latent.dtype)
        else:
            generator = torch.Generator().manual_seed(seed + seed_start)
            noise_sample = torch.randn(shape, generator=generator, device="cpu", dtype=latent.dtype).to(latent.device)
        noise_list.append(noise_sample)
    return torch.cat(noise_list, dim=0)


@torch.no_grad()
def MotionFlowMatching_generate(
    self, text: Union[str, List[str]], seed_input: List[int], duration_slider: int, cfg_scale: Optional[float] = None, hidden_state_dict=None, length=None
) -> Dict[str, Any]:
    device = get_module_device(self)
    if length is None:
        length = round(duration_slider * self.output_mesh_fps)
    assert 0 < length < 5000, f"input duration_slider must be in (0, {5000 / self.output_mesh_fps}] due to rope, but got {duration_slider}"
    if length > self.train_frames or length < min(self.train_frames, 20):
        print(f">>> given length is too long or too short, got {length}, will be truncated")
        length = min(length, self.train_frames)
        length = max(length, min(self.train_frames, 20))
    repeat = len(seed_input)
    if isinstance(text, list):
        assert len(text) == repeat, f"len(text) must equal len(seed_input), got {len(text)} vs {repeat}"
        text_list = text
    elif isinstance(text, str):
        text_list = [text] * repeat
    else:
        raise TypeError(f"Unsupported text type: {type(text)}")
    if not self.uncondition_mode:
        if hidden_state_dict is None:
            raise ValueError("Use HYMotionModel.generate to construct native text conditioning")
        vtxt_input = hidden_state_dict["text_vec_raw"]
        ctxt_input = hidden_state_dict["text_ctxt_raw"]
        ctxt_length = hidden_state_dict["text_ctxt_raw_length"]
        if len(vtxt_input.shape) == 2 and len(ctxt_input.shape) == 2:
            vtxt_input = vtxt_input[None].repeat(repeat, 1, 1)
            ctxt_input = ctxt_input[None].repeat(repeat, 1, 1)
            ctxt_length = ctxt_length.repeat(repeat)
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
    else:
        vtxt_input = self.null_vtxt_feat.expand(repeat, 1, -1)
        ctxt_input = self.null_ctxt_input.expand(repeat, 1, -1)
        ctxt_length = torch.tensor([1]).expand(repeat)
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1]).expand(repeat, -1)
    assert len(vtxt_input.shape) == 3, f"vtxt_input.shape: {vtxt_input.shape}, should be (B, 1, D)"
    assert len(ctxt_input.shape) == 3, f"ctxt_input.shape: {ctxt_input.shape}, should be (B, 1, D)"
    assert len(ctxt_length.shape) == 1, f"ctxt_length.shape: {ctxt_length.shape}, should be (B,)"
    ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
    x_length = torch.LongTensor([length] * repeat).to(device)
    x_mask_temporal = length_to_mask(x_length, self.train_frames)
    text_guidance_scale = cfg_scale if cfg_scale is not None else self.text_guidance_scale
    do_classifier_free_guidance = text_guidance_scale > 1.0 and (not self.uncondition_mode)
    if do_classifier_free_guidance is True:
        silent_text_feat = self.null_vtxt_feat.expand(*vtxt_input.shape)
        vtxt_input = torch.cat([silent_text_feat, vtxt_input], dim=0)
        if self.enable_ctxt_null_feat:
            silent_ctxt_input = self.null_ctxt_input.expand(*ctxt_input.shape)
        else:
            silent_ctxt_input = ctxt_input
        ctxt_input = torch.cat([silent_ctxt_input, ctxt_input], dim=0)
        ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
        x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)

    def fn(t: Tensor, x: Tensor) -> Tensor:
        x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
        x_pred = self.motion_transformer(
            x=x_input, ctxt_input=ctxt_input, vtxt_input=vtxt_input, timesteps=t.expand(x_input.shape[0]), x_mask_temporal=x_mask_temporal, ctxt_mask_temporal=ctxt_mask_temporal
        )
        if do_classifier_free_guidance:
            x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
            x_pred = x_pred_basic + text_guidance_scale * (x_pred_text - x_pred_basic)
        return x_pred

    t = torch.linspace(0, 1, self.validation_steps + 1, device=device)
    y0 = self.noise_from_seeds(torch.zeros(1, self.train_frames, self._network_module_args["input_dim"], device=device), seed_input, random_generator_on_gpu=self.random_generator_on_gpu)
    with torch.no_grad():
        trajectory = odeint(fn, y0, t, **self._noise_scheduler_cfg)
    sampled = trajectory[-1][:, :length, ...].clone()
    assert isinstance(sampled, Tensor), f"sampled must be a Tensor, but got {type(sampled)}"
    output_dict = self.decode_motion_from_latent(sampled, should_apply_smooothing=True)
    return {**output_dict, "text": text}
