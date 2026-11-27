#!/usr/bin/env python3
"""Complete the unified H3 JSON using independent GPU workers and local models.

Missing conditions are encoded from their recorded text and first image.
Missing teacher videos use the original dense FL2AV checkpoint: 30 sigma
points / 29 Euler evaluations, video/audio shift 12/3. Existing MP4s are only
VAE-encoded. Newly encoded training latents are derived from the saved video.
With --encode-only, absent videos remain missing and neither DiT nor the
condition encoder is loaded. Receipts are isolated from denoising runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ASSETS = ('condition', 'teacher_video', 'teacher_video_latent', 'teacher_audio_latent')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def present(row, key):
    value = row.get(key)
    if value is not None and not Path(value).is_file():
        raise FileNotFoundError(f"{row['sample_id']}: {key} does not exist: {value}")
    return value is not None


def work_needed(row, encode_only=False):
    if encode_only and not present(row, 'teacher_video'):
        return []
    keys = ASSETS[2:] if encode_only else ASSETS
    return [key for key in keys if not present(row, key)]


def condition_prompt(row):
    info = row.get('condition_prompt') or row.get('teacher_prompt')
    if not info or not isinstance(info.get('text'), str) or not info['text'].strip():
        raise ValueError(f"{row['sample_id']}: actual conditioning text is unknown")
    if info.get('sha256') != hashlib.sha256(info['text'].encode()).hexdigest():
        raise ValueError(f"{row['sample_id']}: prompt SHA256 does not match text")
    return info


def validate_rows(rows, encode_only=False):
    ids = set()
    for row in rows:
        sample_id = row['sample_id']
        if not sample_id or sample_id in ids or Path(sample_id).name != sample_id or sample_id in ('.', '..'):
            raise ValueError(f'Invalid or repeated sample_id: {sample_id}')
        ids.add(sample_id)
        if row['task'] not in ('t2av', 'i2av'):
            raise ValueError(f'{sample_id}: expected T2AV or I2AV')
        if min(row['width'], row['height'], row['num_frames']) <= 0 or row['width'] % 32 or row['height'] % 32 or row['num_frames'] % 17 != 5 or row['fps'] != 24:
            raise ValueError(f'{sample_id}: invalid H3 dimensions/fps')
        if work_needed(row, encode_only):
            condition_prompt(row)
            if not encode_only and not present(row, 'condition') and row['task'] == 'i2av' and not present(row, 'image'):
                raise ValueError(f'{sample_id}: missing I2AV image for condition encoding')


def sample_seed(sample_id, seed):
    return int(digest([sample_id, seed])[:15], 16)


def interrupt(signum, frame):
    raise KeyboardInterrupt


def artifact(root, row, category, suffix):
    path = root / 'data' / category / (row['sample_id'] + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def torch_save(path, value):
    import torch
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def completion_recipe(args):
    if args.encode_only:
        return {'mode': 'encode_only', 'model_path': str(args.model_path), 'posterior': 'mode',
                'latent_source': 'saved_teacher_video', 'version': 1}
    return {
        'model_path': str(args.model_path), 'condition_model_path': str(args.condition_model_path),
        'schedule_points': args.schedule_points, 'video_shift': 12.0, 'audio_shift': 3.0,
        'seed': args.seed, 'attention_backend': args.attention_backend, 'posterior': 'mode',
        'latent_source': 'saved_teacher_video', 'version': 1,
    }


class Worker:
    def __init__(self, args):
        self.args = args
        self.root = args.manifest.parent
        self.state = self.root / 'data' / ('vae_completion' if args.encode_only else 'completion')
        self.state.mkdir(parents=True, exist_ok=True)
        self.rows = json.loads(args.manifest.read_text())
        self.jobs, self.errors = [], []
        self.recipe = completion_recipe(args)
        for index, source in enumerate(self.rows):
            if index % args.workers != args.worker or not work_needed(source, args.encode_only):
                continue
            path = self.state / (source['sample_id'] + '.json')
            identity = digest({'input': source, 'recipe': self.recipe})
            if path.is_file():
                saved = json.loads(path.read_text())
                if saved['identity'] != identity and not (saved['record'] == source and saved['recipe'] == self.recipe):
                    raise ValueError(f'Pending record or generation recipe changed: {path}')
                row = saved['record']
            else:
                row = deepcopy(source)
            self.jobs.append({'row': row, 'identity': identity, 'receipt': path, 'source': source})

    def commit(self, job, stage):
        save_json(job['receipt'], {'identity': job['identity'], 'record': job['row'], 'recipe': self.recipe})
        print(json.dumps({'worker': self.args.worker, 'sample_id': job['row']['sample_id'], 'stage': stage}, ensure_ascii=False), flush=True)

    def failed(self, job, stage, error):
        job['failed'] = True
        self.errors.append({'sample_id': job['row']['sample_id'], 'stage': stage, 'error': str(error)})
        print(json.dumps(self.errors[-1], ensure_ascii=False), flush=True)
        traceback.print_exc()

    def active(self):
        return [job for job in self.jobs if not job.get('failed')]

    def conditions(self):
        import torch
        import build_minimax_h3_prompt_latents as helper
        from diffusers.models.attention_dispatch import attention_backend

        jobs = [j for j in self.active() if not present(j['row'], 'condition')]
        if not jobs:
            return
        encoder, tokenizer, processor = helper.load_conditioner(str(self.args.condition_model_path), 'cuda:0', torch.bfloat16)
        for job in jobs:
            row = job['row']
            try:
                info = condition_prompt(row)
                sample = {'first_frame': row['image']} if row['task'] == 'i2av' else {}
                images = helper.load_prepared_images(sample, row['height'], row['width'])
                positive = helper.encode_prompt(encoder, tokenizer, processor, info['text'], images, 'cuda:0', torch.bfloat16)
                positive.update(task=row['task'], target_height=row['height'], target_width=row['width'], target_num_frames=row['num_frames'], keyframe_anchors=torch.tensor([0] if images else [], dtype=torch.long))
                fingerprint = digest([row['sample_id'], info, row['width'], row['height'], row['num_frames'], str(self.args.model_path), str(self.args.condition_model_path)])
                payload = {'cache_schema_version': 1, 'cache_fingerprint': fingerprint, 'prompt': info['text'], 'task': row['task'], 'conditioning': {'positive': positive}}
                path = artifact(self.root, row, 'condition', '.pt')
                torch_save(path, payload)
                job['condition_path'] = path
                row['cache_fingerprint'] = fingerprint
                row['condition_prompt'] = info
                row['training_prompt'], row['prompt_variant'] = info['text'], info['variant']
                if row['task'] == 't2av':
                    row['condition'] = str(path)
                    self.commit(job, 'condition')
            except Exception as error:
                self.failed(job, 'text_condition', error)
        del encoder, tokenizer, processor
        helper.release_device_memory()
        image_jobs = [j for j in jobs if not j.get('failed') and j['row']['task'] == 'i2av']
        if image_jobs:
            vae = helper.load_video_vae(str(self.args.model_path), 'cuda:0', torch.float32)
            for job in image_jobs:
                row = job['row']
                try:
                    payload = torch.load(job['condition_path'], map_location='cpu', weights_only=True)
                    images = helper.load_prepared_images({'first_frame': row['image']}, row['height'], row['width'])
                    with attention_backend('native'):
                        payload['conditioning']['positive']['condition_video_latents'] = helper.encode_keyframes(vae, images, 'cuda:0')
                    torch_save(job['condition_path'], payload)
                    row['condition'] = str(job['condition_path'])
                    self.commit(job, 'condition')
                except Exception as error:
                    self.failed(job, 'image_condition', error)
            del vae
            helper.release_device_memory()

    def generate(self):
        import torch
        from diffusers import MiniMaxH3Scheduler
        from lightx2v_train.data.minimax_h3_cache_dataset import LatentDataset
        from lightx2v_train.model_capabilities import DistributionMatchingCapability
        from lightx2v_train.model_zoo import build_loaded_model
        from encode_minimax_h3_teacher_av_latents import release_device_memory

        jobs = [j for j in self.active() if not present(j['row'], 'teacher_video')]
        if not jobs:
            return
        config = {'model': {
            'name': 'minimax_h3_t2av', 'pretrained_model_name_or_path': str(self.args.model_path),
            'running_dtype': 'bf16', 'transformer_param_dtype': 'bf16', 'local_files_only': True,
            'use_autocast': False, 'attention_backend': self.args.attention_backend,
            'capabilities': {'distribution_matching': {'geometry_from_metadata': True, 'video_flow_shift': 12.0, 'audio_flow_shift': 3.0}},
        }}
        pending = [j for j in jobs if not artifact(self.root, j['row'], 'completion', '.packed.pt').is_file()
                   and not (present(j['row'], 'teacher_video_latent') and present(j['row'], 'teacher_audio_latent'))]
        if not pending:
            return
        if self.args.attention_kernel_path:
            from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY, AttentionBackendName, _resolve_kernel_attr
            from kernels import get_local_kernel
            kernel = get_local_kernel(self.args.attention_kernel_path)
            backend = _HUB_KERNELS_REGISTRY[AttentionBackendName(self.args.attention_backend)]
            backend.kernel_fn = _resolve_kernel_attr(kernel, backend.function_attr)
            backend.wrapped_forward_fn = _resolve_kernel_attr(kernel, backend.wrapped_forward_attr)
            backend.wrapped_backward_fn = _resolve_kernel_attr(kernel, backend.wrapped_backward_attr)
        model = build_loaded_model(config, load_transformer=True, load_vae=False, load_condition_encoder=False)
        capability = model.capabilities.require(DistributionMatchingCapability)
        capability.denoiser().requires_grad_(False)
        capability.set_training(False)
        loader = LatentDataset.__new__(LatentDataset)
        loader.prompt_column, loader.prompt_index, loader.defer_latent_loading, loader.negative_condition = 'caption', 0, True, None
        schedules = [MiniMaxH3Scheduler(shift=shift) for shift in (12.0, 3.0)]
        for schedule in schedules:
            schedule.set_timesteps(self.args.schedule_points, device=capability.device)
        for job in pending:
            row = job['row']
            try:
                torch.manual_seed(sample_seed(row['sample_id'], self.args.seed))
                info = condition_prompt(row)
                cached = torch.load(row['condition'], map_location='cpu', weights_only=True)
                if row.get('cache_fingerprint') and cached.get('cache_fingerprint') != row['cache_fingerprint']:
                    raise ValueError('condition cache fingerprint differs from JSON')
                if cached.get('prompt', info['text']) != info['text']:
                    raise ValueError('condition cache prompt differs from JSON')
                positive = cached.get('conditioning', cached).get('positive')
                if positive['task'] != row['task']:
                    raise ValueError('condition task differs from JSON')
                if row['task'] == 'i2av' and any(positive.get(a) != row[b] for a, b in [('target_height', 'height'), ('target_width', 'width'), ('target_num_frames', 'num_frames')]):
                    raise ValueError('I2AV cached image dimensions differ from JSON')
                legacy = dict(row, caption=info['text'], condition_path=row['condition'], target_height=row['height'], target_width=row['width'])
                sample = loader._load_metadata_sample(legacy, self.root)
                with torch.no_grad():
                    condition, _ = capability.encode_conditions(sample, '', 1.0, lambda value: value)
                    shape = capability.latent_shape(sample, None, lambda value: value)
                    xt = capability.initial_latents(shape, torch.float32, lambda value: value)
                    for step in range(len(schedules[0].timesteps)):
                        sigmas = torch.stack([s.sigmas[step] for s in schedules])
                        velocity = capability.predict_velocity_at_sigmas(xt, sigmas, condition)
                        deltas = [s.sigmas[step + 1] - s.sigmas[step] for s in schedules]
                        xt = type(xt)(xt.video + deltas[0] * velocity.video.float(), xt.audio + deltas[1] * velocity.audio.float(), shape)
                        if (step + 1) % 5 == 0 or step + 1 == len(schedules[0].timesteps):
                            print(json.dumps({'sample_id': row['sample_id'], 'denoise_step': step + 1, 'total': len(schedules[0].timesteps)}), flush=True)
                    packed = {'video': xt.video.cpu(), 'audio': xt.audio.cpu(), 'recipe': self.recipe}
                    if not all(bool(torch.isfinite(packed[k]).all()) for k in ('video', 'audio')):
                        raise FloatingPointError('nonfinite teacher output')
                    torch_save(artifact(self.root, row, 'completion', '.packed.pt'), packed)
                self.commit(job, 'denoise')
                del xt, velocity, packed, condition, sample, cached, positive
            except Exception as error:
                self.failed(job, 'denoise', error)
        del model, capability
        release_device_memory()

    def media_and_latents(self):
        import torch
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
        from diffusers.models.attention_dispatch import attention_backend
        from diffusers.modular_pipelines.minimax_h3.packing import unpatchify_video_tokens, unpack_audio_tokens
        from diffusers.utils.export_utils import encode_video
        import encode_minimax_h3_teacher_av_latents as helper

        jobs = [j for j in self.active() if work_needed(j['row'], self.args.encode_only)]
        if not jobs:
            return
        video_vae = AutoencoderKLMiniMaxH3.from_pretrained(str(self.args.model_path), subfolder='vae', torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to('cuda:0')
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(self.args.model_path), subfolder='audio_vae', torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to('cuda:0')
        for job in jobs:
            row = job['row']
            try:
                frames, height, width = row['num_frames'], row['height'], row['width']
                video_shape = [24, (frames - 5) // 17 * 5 + 2, height // 16, width // 16]
                audio_frames = round(frames / row['fps'] * 40)
                if not present(row, 'teacher_video'):
                    staged = artifact(self.root, row, 'completion', '.packed.pt')
                    generated = staged.is_file()
                    if generated:
                        packed = torch.load(staged, map_location='cuda:0', weights_only=True)
                        if packed['recipe'] != self.recipe:
                            raise ValueError('staged teacher recipe differs from current run')
                        video = unpatchify_video_tokens(packed['video'][0], video_shape[1], video_shape[2], video_shape[3], 24, (1, 2, 2))
                        audio = unpack_audio_tokens(packed['audio'][0], audio_frames)
                        del packed
                    else:
                        video = torch.load(row['teacher_video_latent'], map_location='cuda:0', weights_only=True)
                        video = (video['latents'] if isinstance(video, dict) else video).unsqueeze(0).float()
                        audio = torch.load(row['teacher_audio_latent'], map_location='cuda:0', weights_only=True).float()
                    mean = video.new_tensor(video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
                    std = video.new_tensor(video_vae.config.latents_std).view(1, -1, 1, 1, 1)
                    with torch.inference_mode(), attention_backend('native'), torch.autocast('cuda', dtype=torch.float16):
                        decoded = video_vae.decode(video * std + mean, return_dict=False)[0]
                    if not bool(torch.isfinite(decoded).all()):
                        raise FloatingPointError('nonfinite decoded video')
                    mean = decoded.new_tensor(helper.PIXEL_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1)
                    std = decoded.new_tensor(helper.PIXEL_STD, dtype=torch.float32).view(1, 3, 1, 1, 1)
                    pixels = (decoded.float() * std + mean).clamp_(0, 1)[0].permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).cpu()
                    del video, decoded
                    mean = audio.new_tensor(audio_vae.config.latents_mean).view(1, -1, 1)
                    std = audio.new_tensor(audio_vae.config.latents_std).view(1, -1, 1)
                    with torch.inference_mode(), attention_backend('native'):
                        waveform = audio_vae.decode(audio * std + mean, return_dict=False)[0][:, 0].float().cpu()
                    if not bool(torch.isfinite(waveform).all()):
                        raise FloatingPointError('nonfinite decoded audio')
                    destination = artifact(self.root, row, 'teacher_video', '.mp4')
                    temporary = destination.with_suffix('.tmp.mp4')
                    encode_video(pixels, fps=row['fps'], output_path=str(temporary), audio=waveform, audio_sample_rate=32000)
                    os.replace(temporary, destination)
                    row['teacher_video'] = str(destination)
                    if generated:
                        row.update(teacher_video_latent=None, teacher_audio_latent=None, teacher_origin='local_h3', teacher_model=str(self.args.model_path), teacher_task_id=None, teacher_exclusion_reason=None, teacher_prompt=condition_prompt(row), teacher_generation={**self.recipe, 'model_evaluations': self.args.schedule_points - 1, 'sample_seed': sample_seed(row['sample_id'], self.args.seed)})
                    else:
                        row['teacher_video_reconstructed_from_latents'] = True
                    self.commit(job, 'teacher_video')
                    del pixels, waveform, audio
                source = Path(row['teacher_video'])
                fingerprint = helper.latent_fingerprint(row, source, self.args.model_path, height, width, frames, 'mode')
                if not present(row, 'teacher_video_latent'):
                    pixels = helper.decode_video(source, height, width, frames, row['fps'])
                    with attention_backend('native'):
                        tensor = helper.encode_video_latent(video_vae, pixels, torch.device('cuda:0'), 'mode', self.args.seed).to(torch.bfloat16)
                    if list(tensor.shape) != video_shape or not bool(torch.isfinite(tensor).all()):
                        raise ValueError('invalid video latent shape or values')
                    destination = artifact(self.root, row, 'vae_teacher_video_latent' if self.args.encode_only else 'teacher_video_latent', '.pt')
                    torch_save(destination, {'latents': tensor, 'normalized': True, 'latent_fingerprint': fingerprint})
                    row['teacher_video_latent'] = str(destination)
                    row.setdefault('latent_metadata', {})['latent_fingerprint'] = fingerprint
                    self.commit(job, 'video_latent')
                    del pixels, tensor
                if not present(row, 'teacher_audio_latent'):
                    waveform = helper.decode_audio(source, audio_frames * 800)
                    with attention_backend('native'):
                        tensor = helper.encode_audio_latent(audio_vae, waveform, torch.device('cuda:0')).to(torch.bfloat16)
                    if list(tensor.shape) != [2, 32, audio_frames] or not bool(torch.isfinite(tensor).all()):
                        raise ValueError('invalid audio latent shape or values')
                    destination = artifact(self.root, row, 'vae_teacher_audio_latent' if self.args.encode_only else 'teacher_audio_latent', '.pt')
                    torch_save(destination, tensor)
                    row['teacher_audio_latent'] = str(destination)
                    self.commit(job, 'audio_latent')
                    del waveform, tensor
                row['teacher_latent_prompt'] = row['teacher_prompt']
                row['latent_metadata'] = {**row.get('latent_metadata', {}), 'video_latent_shape': video_shape, 'audio_latent_shape': [2, 32, audio_frames], 'cache_dtype': 'bf16', 'normalized': True}
                issues = set(row.get('validation', {}).get('issues', []))
                issues.difference_update('missing_' + key for key in ASSETS if present(row, key))
                if row.get('condition_prompt'):
                    issues.discard('condition_prompt_unverified')
                if row.get('condition_prompt') and row.get('teacher_prompt') and row['condition_prompt']['sha256'] == row['teacher_prompt']['sha256']:
                    issues.discard('condition_teacher_prompt_mismatch')
                row['validation'] = {**row.get('validation', {}), 'issues': sorted(issues)}
                self.commit(job, 'complete')
                if not self.args.encode_only:
                    artifact(self.root, row, 'completion', '.packed.pt').unlink(missing_ok=True)
            except Exception as error:
                self.failed(job, 'vae', error)
                helper.release_device_memory()

    def run(self):
        if not self.args.encode_only:
            self.conditions()
            self.generate()
        self.media_and_latents()
        save_json(self.state / f'errors.worker{self.args.worker}.json', self.errors)
        return bool(self.errors)


def launch(args):
    signal.signal(signal.SIGTERM, interrupt)
    rows = json.loads(args.manifest.read_text())
    validate_rows(rows, args.encode_only)
    summary = Counter(key for row in rows for key in work_needed(row, args.encode_only))
    print(json.dumps({'records': len(rows), 'mode': 'encode_only' if args.encode_only else 'complete', 'missing': dict(summary), 'skipped_missing_video': sum(not present(row, 'teacher_video') for row in rows) if args.encode_only else 0, 'model_evaluations': 0 if args.encode_only else args.schedule_points - 1}, ensure_ascii=False), flush=True)
    if args.plan or (not summary and not args.encode_only):
        return 0
    state = args.manifest.parent / 'data' / ('vae_completion' if args.encode_only else 'completion')
    state.mkdir(parents=True, exist_ok=True)
    with (state / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = args.manifest.read_bytes()
        backup = args.manifest.with_name('dataset.before_completion.' + hashlib.sha256(original).hexdigest()[:12] + '.json')
        if not backup.exists():
            backup.write_bytes(original)
        gpus = args.gpus.split(',') if summary else []
        processes, handles = [], []
        logdir = args.manifest.parent / 'logs' / ('vae_completion' if args.encode_only else 'completion')
        logdir.mkdir(parents=True, exist_ok=True)
        try:
            for rank, gpu in enumerate(gpus):
                handle = (logdir / f'gpu{gpu}.log').open('a')
                handles.append(handle)
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--manifest', str(args.manifest), '--model-path', str(args.model_path), '--worker', str(rank), '--workers', str(len(gpus)), '--schedule-points', str(args.schedule_points), '--seed', str(args.seed), '--attention-backend', args.attention_backend]
                command.extend(['--condition-model-path', str(args.condition_model_path)])
                if args.encode_only:
                    command.append('--encode-only')
                if args.attention_kernel_path:
                    command.extend(['--attention-kernel-path', str(args.attention_kernel_path)])
                processes.append(subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}, stdout=handle, stderr=subprocess.STDOUT))
            statuses = [process.wait() for process in processes]
        except BaseException:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()
            raise
        finally:
            for handle in handles:
                handle.close()
        if args.manifest.read_bytes() != original:
            raise RuntimeError('Input manifest changed during completion; worker receipts retained, merge aborted')
        recipe = completion_recipe(args)
        for index, row in enumerate(rows):
            receipt = state / (row['sample_id'] + '.json')
            if work_needed(row, args.encode_only) and receipt.is_file():
                saved = json.loads(receipt.read_text())
                if saved['identity'] == digest({'input': row, 'recipe': recipe}):
                    rows[index] = saved['record']
        save_json(args.manifest, rows)
        remaining = dict(Counter(key for row in rows for key in work_needed(row, args.encode_only)))
        report_name = 'vae_completion_report.json' if args.encode_only else 'completion_report.json'
        save_json(args.manifest.parent / report_name, {'records': len(rows), 'mode': 'encode_only' if args.encode_only else 'complete', 'remaining': remaining, 'skipped_missing_video': [row['sample_id'] for row in rows if args.encode_only and not present(row, 'teacher_video')], 'worker_exit_codes': statuses, 'logs': str(logdir)})
        print(json.dumps({'remaining': remaining, 'logs': str(logdir)}, ensure_ascii=False), flush=True)
        return int(any(statuses) or bool(remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--condition-model-path', type=Path, help='Model root containing text_encoder/tokenizer/processor; defaults to model-path')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--schedule-points', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attention-backend', default='_flash_3_hub')
    parser.add_argument('--attention-kernel-path', type=Path, default=os.environ.get('H3_FA3_KERNEL_PATH'), help='Local FA3 Hub kernel snapshot; avoids network loading')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--encode-only', action='store_true', help='Only VAE-encode existing teacher videos; skip absent videos, never load DiT or the condition encoder')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.manifest, args.model_path = args.manifest.resolve(), args.model_path.resolve()
    args.condition_model_path = (args.condition_model_path or args.model_path).resolve()
    gpu_ids = args.gpus.split(',')
    if any(not value.strip() for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error('gpus must name distinct devices')
    if args.schedule_points < 2:
        parser.error('schedule-points must be at least 2')
    if args.worker is not None:
        return Worker(args).run()
    return launch(args)


if __name__ == '__main__':
    raise SystemExit(main())
