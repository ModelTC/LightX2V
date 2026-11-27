# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.sf3d.weights.system import (
    Any,
    F,
    Float,
    Image,
    List,
    Mesh,
    Optional,
    Tensor,
    Tuple,
    Union,
    convert_data,
    create_intrinsic_from_fov_deg,
    default_cond_c2w,
    dilate_fill,
    float32_to_uint8_np,
    get_device,
    normalize,
    np,
    nullcontext,
    rearrange,
    scale_tensor,
    torch,
    trimesh,
)


def SF3D_triplane_to_meshes(self, triplanes: Float[Tensor, "B 3 Cp Hp Wp"]) -> list[Mesh]:
    meshes = []
    for i in range(triplanes.shape[0]):
        triplane = triplanes[i]
        grid_vertices = scale_tensor(self.isosurface_helper.grid_vertices.to(triplanes.device), self.isosurface_helper.points_range, self.bbox)
        values = self.query_triplane(grid_vertices, triplane)
        decoded = self.decoder(values, include=["vertex_offset", "density"])
        sdf = decoded["density"] - self.cfg.isosurface_threshold
        deform = decoded["vertex_offset"].squeeze(0)
        mesh: Mesh = self.isosurface_helper(sdf.view(-1, 1), deform.view(-1, 3) if deform is not None else None)
        mesh.v_pos = scale_tensor(mesh.v_pos, self.isosurface_helper.points_range, self.bbox)
        meshes.append(mesh)
    return meshes


def SF3D_query_triplane(self, positions: Float[Tensor, "*B N 3"], triplanes: Float[Tensor, "*B 3 Cp Hp Wp"]) -> Float[Tensor, "*B N F"]:
    batched = positions.ndim == 3
    if not batched:
        triplanes = triplanes[None, ...]
        positions = positions[None, ...]
    assert triplanes.ndim == 5 and positions.ndim == 3
    positions = scale_tensor(positions, (-self.cfg.radius, self.cfg.radius), (-1, 1))
    indices2D: Float[Tensor, "B 3 N 2"] = torch.stack((positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]), dim=-3).float()
    out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
        rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3).float(), rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3).float(), align_corners=True, mode="bilinear"
    )
    out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
    return out


def SF3D_get_scene_codes(self, batch) -> Float[Tensor, "B 3 C H W"]:
    if len(batch["rgb_cond"].shape) == 4:
        batch["rgb_cond"] = batch["rgb_cond"].unsqueeze(1)
        batch["mask_cond"] = batch["mask_cond"].unsqueeze(1)
        batch["c2w_cond"] = batch["c2w_cond"].unsqueeze(1)
        batch["intrinsic_cond"] = batch["intrinsic_cond"].unsqueeze(1)
        batch["intrinsic_normed_cond"] = batch["intrinsic_normed_cond"].unsqueeze(1)
    batch_size, n_input_views = batch["rgb_cond"].shape[:2]
    camera_embeds: Optional[Float[Tensor, "B Nv Cc"]]
    camera_embeds = self.camera_embedder(**batch)
    input_image_tokens: Float[Tensor, "B Nv Cit Nit"] = self.image_tokenizer(rearrange(batch["rgb_cond"], "B Nv H W C -> B Nv C H W"), modulation_cond=camera_embeds)
    input_image_tokens = rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=n_input_views)
    tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size)
    tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens, modulation_cond=None)
    direct_codes = self.tokenizer.detokenize(tokens)
    scene_codes = self.post_processor(direct_codes)
    return (scene_codes, direct_codes)


def SF3D_run_image(self, image: Union[Image.Image, List[Image.Image]], bake_resolution: int) -> Tuple[Union[trimesh.Trimesh, List[trimesh.Trimesh]], dict[str, Any]]:
    if isinstance(image, list):
        rgb_cond = []
        mask_cond = []
        for img in image:
            mask, rgb = self.prepare_image(img)
            mask_cond.append(mask)
            rgb_cond.append(rgb)
        rgb_cond = torch.stack(rgb_cond, 0)
        mask_cond = torch.stack(mask_cond, 0)
        batch_size = rgb_cond.shape[0]
    else:
        mask_cond, rgb_cond = self.prepare_image(image)
        batch_size = 1
    c2w_cond = default_cond_c2w(self.cfg.default_distance).to(self.device)
    intrinsic, intrinsic_normed_cond = create_intrinsic_from_fov_deg(self.cfg.default_fovy_deg, self.cfg.cond_image_size, self.cfg.cond_image_size)
    batch = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1),
        "intrinsic_cond": intrinsic.to(self.device).view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
        "intrinsic_normed_cond": intrinsic_normed_cond.to(self.device).view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
    }
    meshes, global_dict = self.generate_mesh(batch, bake_resolution)
    if batch_size == 1:
        return (meshes[0], global_dict)
    else:
        return (meshes, global_dict)


def SF3D_prepare_image(self, image):
    if image.mode != "RGBA":
        raise ValueError("Image must be in RGBA mode")
    img_cond = torch.from_numpy(np.asarray(image.resize((self.cfg.cond_image_size, self.cfg.cond_image_size))).astype(np.float32) / 255.0).float().clip(0, 1).to(self.device)
    mask_cond = img_cond[:, :, -1:]
    rgb_cond = torch.lerp(torch.tensor(self.cfg.background_color, device=self.device)[None, None, :], img_cond[:, :, :3], mask_cond)
    return (mask_cond, rgb_cond)


def SF3D_generate_mesh(self, batch, bake_resolution: int) -> Tuple[List[trimesh.Trimesh], dict[str, Any]]:
    batch["rgb_cond"] = self.image_processor(batch["rgb_cond"], self.cfg.cond_image_size)
    batch["mask_cond"] = self.image_processor(batch["mask_cond"], self.cfg.cond_image_size)
    scene_codes, non_postprocessed_codes = self.get_scene_codes(batch)
    global_dict = {}
    if self.image_estimator is not None:
        global_dict.update(self.image_estimator(batch["rgb_cond"] * batch["mask_cond"]))
    device = get_device()
    with torch.no_grad(), torch.autocast(device_type=device, enabled=False) if "cuda" in device else nullcontext():
        scene_codes = scene_codes.float()
        meshes = self.triplane_to_meshes(scene_codes)
        rets = []
        for i, mesh in enumerate(meshes):
            if mesh.v_pos.shape[0] == 0:
                rets.append(trimesh.Trimesh())
                continue
            mesh.unwrap_uv()
            rast = self.baker.rasterize(mesh.v_tex, mesh.t_pos_idx, bake_resolution)
            bake_mask = self.baker.get_mask(rast)
            pos_bake = self.baker.interpolate(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_pos = pos_bake[bake_mask]
            tri_query = self.query_triplane(gb_pos, scene_codes[i])[0]
            decoded = self.decoder(tri_query, exclude=["density", "vertex_offset"])
            nrm = self.baker.interpolate(mesh.v_nrm, rast, mesh.t_pos_idx)
            gb_nrm = F.normalize(nrm[bake_mask], dim=-1)
            decoded["normal"] = gb_nrm
            for k, v in global_dict.items():
                if k.startswith("decoder_"):
                    decoded[k.replace("decoder_", "")] = v[i]
            mat_out = {"albedo": decoded["features"], "roughness": decoded["roughness"], "metallic": decoded["metallic"], "normal": normalize(decoded["perturb_normal"]), "bump": None}
            for k, v in mat_out.items():
                if v is None:
                    continue
                if v.shape[0] == 1:
                    mat_out[k] = v[0]
                else:
                    f = torch.zeros(bake_resolution, bake_resolution, v.shape[-1], dtype=v.dtype, device=v.device)
                    if v.shape == f.shape:
                        continue
                    if k == "normal":
                        tng = self.baker.interpolate(mesh.v_tng, rast, mesh.t_pos_idx)
                        gb_tng = tng[bake_mask]
                        gb_tng = F.normalize(gb_tng, dim=-1)
                        gb_btng = F.normalize(torch.cross(gb_nrm, gb_tng, dim=-1), dim=-1)
                        normal = F.normalize(mat_out["normal"], dim=-1)
                        tangent_matrix = torch.stack([gb_tng, gb_btng, gb_nrm], dim=-1)
                        normal_tangent = torch.bmm(tangent_matrix.transpose(1, 2), normal.unsqueeze(-1)).squeeze(-1)
                        normal_tangent = (normal_tangent * 0.5 + 0.5).clamp(0, 1)
                        f[bake_mask] = normal_tangent.view(-1, 3)
                        mat_out["bump"] = f
                    else:
                        f[bake_mask] = v.view(-1, v.shape[-1])
                        mat_out[k] = f

            def uv_padding(arr):
                if arr.ndim == 1:
                    return arr
                return dilate_fill(arr.permute(2, 0, 1)[None, ...].contiguous(), bake_mask.unsqueeze(0).unsqueeze(0), iterations=bake_resolution // 150).squeeze(0).permute(1, 2, 0).contiguous()  # noqa: B023 - applied synchronously to the current mesh

            verts_np = convert_data(mesh.v_pos)
            faces = convert_data(mesh.t_pos_idx)
            uvs = convert_data(mesh.v_tex)
            basecolor_tex = Image.fromarray(float32_to_uint8_np(convert_data(uv_padding(mat_out["albedo"])))).convert("RGB")
            basecolor_tex.format = "JPEG"
            metallic = mat_out["metallic"].squeeze().cpu().item()
            roughness = mat_out["roughness"].squeeze().cpu().item()
            if "bump" in mat_out and mat_out["bump"] is not None:
                bump_np = convert_data(uv_padding(mat_out["bump"]))
                bump_up = np.ones_like(bump_np)
                bump_up[..., :2] = 0.5
                bump_up[..., 2:] = 1
                bump_tex = Image.fromarray(float32_to_uint8_np(bump_np, dither=True, dither_mask=np.all(bump_np == bump_up, axis=-1, keepdims=True).astype(np.float32))).convert("RGB")
                bump_tex.format = "JPEG"
            else:
                bump_tex = None
            material = trimesh.visual.material.PBRMaterial(baseColorTexture=basecolor_tex, roughnessFactor=roughness, metallicFactor=metallic, normalTexture=bump_tex)
            tmesh = trimesh.Trimesh(vertices=verts_np, faces=faces, visual=trimesh.visual.texture.TextureVisuals(uv=uvs, material=material))
            rot = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
            tmesh.apply_transform(rot)
            tmesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0]))
            tmesh.invert()
            rets.append(tmesh)
    return (rets, global_dict)
