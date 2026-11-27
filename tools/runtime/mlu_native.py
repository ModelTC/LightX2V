"""MLU device/geometry bindings for independent upstream native model adapters."""
from pathlib import Path
import os
import sys


def load_sf3d(paths):
    import numpy as np
    import torch
    import rembg
    from PIL import Image
    from omegaconf import OmegaConf
    sys.path.insert(0,paths['lightx2v'])
    sys.path.insert(0,paths['native'])
    os.chdir(paths['native'])
    from sf3d.system import SF3D
    from sf3d.utils import remove_background, resize_foreground
    cfg=OmegaConf.load(Path(paths['native_weights'])/'config.yaml')
    original=OmegaConf.load(Path(paths['weights'])/'config.upstream.yaml')
    cfg.image_tokenizer.pretrained_model_name_or_path=original.image_tokenizer.pretrained_model_name_or_path
    if 'pretrain' in original.image_estimator: cfg.image_estimator.pretrain=original.image_estimator.pretrain
    else: cfg.image_estimator.pop('pretrain',None)
    if OmegaConf.to_container(cfg)!=OmegaConf.to_container(original): raise ValueError('Native offline overlay changed model parameters')
    model=SF3D.from_pretrained(paths['native_weights'],config_name='config.yaml',weight_name='model.safetensors').to('mlu').eval()
    model.baker=MLUTextureBaker(model.baker)
    background=rembg.new_session(providers=['CPUExecutionProvider'])
    def infer(item,artifact):
        with Image.open(item['input']) as source: image=source.convert('RGBA')
        image=resize_foreground(remove_background(image,background),.85)
        with torch.autocast('mlu',dtype=torch.bfloat16):
            mesh,_=model.run_image([image],bake_resolution=1024,remesh='none',vertex_count=-1)
        if isinstance(mesh,(list,tuple)): mesh=mesh[0]
        if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all(): raise ValueError('Native SF3D returned empty/non-finite mesh')
        mesh.export(artifact)
    return infer


def load_hymotion(paths):
    import numpy as np
    import torch
    sys.path.insert(0,paths['lightx2v'])
    sys.path.insert(0,paths['native'])
    os.chdir(paths['native'])
    from hymotion.network.text_encoders import text_encoder
    from hymotion.pipeline.body_model import construct_smpl_data_dict
    from hymotion.utils.t2m_runtime import T2MRuntime
    text_encoder.LLM_ENCODER_LAYOUT['qwen3']['module_path']=paths['qwen']
    text_encoder.SENTENCE_EMB_LAYOUT['clipl']['module_path']=paths['clip']
    root=Path(paths['weights'])/'HY-Motion-1.0'
    # The upstream CUDA loader also prepares on CPU before moving its pipeline.
    # Select that same CPU preparation path, then move the unchanged native graph.
    runtime=T2MRuntime(config_path=str(root/'config.yml'),ckpt_name=str(root/'latest.ckpt'),device_ids=[0],force_cpu=True,disable_prompt_engineering=True)
    pipeline=runtime.pipelines[0].eval().to('mlu')
    def infer(item,artifact):
        result=pipeline.generate(item['prompt'],[42],item['duration_s'],cfg_scale=5.,use_special_game_feat=False)
        smpl=construct_smpl_data_dict(result['rot6d'][0].clone(),result['transl'][0].clone())
        arrays={key:value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value) for key,value in smpl.items()}
        arrays.update({key:value.detach().cpu().numpy() for key,value in result.items() if isinstance(value,torch.Tensor)})
        for value in arrays.values():
            if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all(): raise ValueError('Non-finite native motion')
        with artifact.open('wb') as stream: np.savez_compressed(stream,**arrays)
    return infer


from torch import nn


class MLUTextureBaker(nn.Module):
    def __init__(self, baker):
        super().__init__()
        self.baker = baker

    def rasterize(self, uv, face_indices, bake_resolution):
        return self.baker.rasterize(uv.cpu(), face_indices.cpu(), bake_resolution).to(uv.device)

    def get_mask(self, raster):
        return self.baker.get_mask(raster)

    def interpolate(self, attributes, raster, face_indices):
        return self.baker.interpolate(attributes.cpu(), raster.cpu(), face_indices.cpu()).to(attributes.device)


class MLUUnwrapper:
    def __init__(self, unwrapper):
        self.unwrapper = unwrapper

    def __call__(self, vertices, normals, faces, island_padding):
        uv, indices = self.unwrapper(vertices.cpu(), normals.cpu(), faces.cpu(), island_padding)
        return uv.to(vertices.device), indices.to(faces.device)
