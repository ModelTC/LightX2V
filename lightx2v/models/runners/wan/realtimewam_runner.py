import numpy as np
import torch
from loguru import logger

from lightx2v.models.networks.wan.realtimewam_model import RealtimeWAM
from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy, FastWAMRunner
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER


class RealtimeWAMCudaGraph:
    """Capture VAE and layerwise video/action execution on two CUDA streams."""

    def __init__(self, policy):
        self.policy = policy
        self.graph = None
        self.signature = None
        self.drop_masks = None

    @staticmethod
    def _can_drop_mask(mask):
        return mask is None or (mask.dtype == torch.bool and bool(mask.all().item()))

    def _finish_block(self, block, io, prepared, mask, kv=None):
        q, k, v = io[:3]
        if kv is not None:
            k, v = torch.cat((kv["k"], k)), torch.cat((kv["v"], v))
        mixed = block.self_attn.attn.apply(q, k, v, attn_mask=mask)
        return self.policy.model.transformer_infer._post_block(block, io[3], mixed, *io[4:], prepared.context, prepared.context_mask)

    def _run(self):
        image, context, mask, state, noise, timesteps, deltas = self.inputs
        model = self.policy.model
        infer, weights, scheduler = model.transformer_infer, model.transformer_weights, model.scheduler
        latents = self.policy.encode_image_latents(image)
        context, mask = model._append_robot_state_to_context(context, mask, state)
        video = model.pre_infer.infer_video(model.pre_weight, latents, context, mask)
        scheduler.latents = noise
        scheduler.timesteps, scheduler.deltas = timesteps, deltas
        scheduler.step_pre(0)
        action = model.pre_infer.infer_action(model.pre_weight, scheduler.latents, scheduler.current_timestep, context, mask)
        nv = video.tokens.shape[0]
        attention = infer.build_mot_attention_mask(nv, noise.shape[1], video.tokens_per_frame, noise.device)
        video_mask, action_mask = attention[:nv, :nv], attention[nv:]
        masks = (video_mask, action_mask, video.context_mask, action.context_mask)
        if self.drop_masks is None:  # Inspect once during warmup, outside graph capture.
            self.drop_masks = tuple(self._can_drop_mask(m) for m in masks)
        video_mask, action_mask, video.context_mask, action.context_mask = (None if drop else m for m, drop in zip(masks, self.drop_masks))
        caller = torch.cuda.current_stream(self.policy.device)
        self.video_stream.wait_stream(caller)
        self.action_stream.wait_stream(caller)
        vx, ax, cache = video.tokens, action.tokens, []
        for index in range(infer.num_layers):
            with torch.cuda.stream(self.video_stream):
                block = weights.video.blocks[index]
                vio = infer._build_self_attention_io(block, vx, video.freqs, video.t_mod)
                kv = {"k": vio[1], "v": vio[2]}
                cache.append(kv)
                self.ready[index].record(self.video_stream)
                vx = self._finish_block(block, vio, video, video_mask)
            with torch.cuda.stream(self.action_stream):
                block = weights.action.blocks[index]
                aio = infer._build_self_attention_io(block, ax, action.freqs, action.t_mod)
                self.action_stream.wait_event(self.ready[index])
                for tensor in kv.values():
                    tensor.record_stream(self.action_stream)
                ax = self._finish_block(block, aio, action, action_mask, kv)
        caller.wait_stream(self.video_stream)
        caller.wait_stream(self.action_stream)
        ax.record_stream(caller)
        scheduler.noise_pred = weights.action_head.apply(ax).unsqueeze(0)
        scheduler.step_post()
        if timesteps.numel() > 1:
            for kv in cache:
                for tensor in kv.values():
                    tensor.record_stream(caller)
            inputs = {"context": context, "context_mask": mask, "video_kv_cache": cache, "video_seq_len": nv, "attention_mask": attention}
            for index in range(1, timesteps.numel()):
                scheduler.step_pre(index)
                model.infer(inputs)
                scheduler.step_post()
        return scheduler.latents[0].float()

    @torch.no_grad()
    def __call__(self, image, context, mask, state, seed):
        policy = self.policy
        if policy.device.type != "cuda" or policy.vae_cpu_offload:
            raise ValueError("RealtimeWAM CUDA Graph requires the model and VAE resident on CUDA.")
        scheduler = policy.model.scheduler
        scheduler.prepare_loop(
            (1, policy.action_chunk_size, policy.action_dim),
            seed=seed,
            device=policy.device,
            dtype=GET_DTYPE(),
            infer_steps=policy.action_infer_steps,
        )
        state = torch.as_tensor(state, device=context.device, dtype=context.dtype)
        values = (image, context, mask, state, scheduler.latents, scheduler.timesteps, scheduler.deltas)
        signature = tuple((tuple(x.shape), x.dtype, x.device) for x in values)
        signature = (signature, self._can_drop_mask(mask))
        if self.graph is None or signature != self.signature:
            self.close()
            pre = policy.model.pre_infer
            pre.video_freqs = tuple(x.to(policy.device) for x in pre.video_freqs)
            pre.action_freqs = pre.action_freqs.to(policy.device)
            self.inputs = tuple(x.clone() for x in values)
            self.video_stream = torch.cuda.Stream(device=policy.device)
            self.action_stream = torch.cuda.Stream(device=policy.device)
            self.ready = [torch.cuda.Event() for _ in range(policy.model.transformer_infer.num_layers)]
            stream = torch.cuda.Stream(device=policy.device)
            stream.wait_stream(torch.cuda.current_stream(policy.device))
            with torch.cuda.device(policy.device), torch.cuda.stream(stream):
                for _ in range(3):
                    self._run()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(policy.device), torch.cuda.graph(graph, stream=stream):
                self.output = self._run()
            torch.cuda.current_stream(policy.device).wait_stream(stream)
            self.graph, self.signature = graph, signature
            logger.info("Captured RealtimeWAM CUDA Graph: VAE + video + {} action steps, two streams with per-block events", policy.action_infer_steps)
        else:
            for target, source in zip(self.inputs, values):
                target.copy_(source)
        with torch.cuda.device(policy.device):
            self.graph.replay()
        action = self.output.cpu()
        scheduler.clear()
        return action

    def close(self):
        if self.graph is not None:
            torch.cuda.synchronize(self.policy.device)
        self.graph = self.signature = None
        self.drop_masks = None
        self.inputs = self.output = None
        self.video_stream = self.action_stream = self.ready = None


class RealtimeWAMPolicy(FastWAMPolicy):
    model_class = RealtimeWAM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_graph = RealtimeWAMCudaGraph(self)

    def predict_action_chunk(self, images, state, task_description, seed=None):
        image = self.build_image_tensor(images)
        use_cuda_graph = self.config.get("cuda_graph", False)
        context, context_mask = self.encode_prompt(self.default_prompt.format(task_prompt=task_description))
        robot_state = self.state_normalizer.forward(np.asarray(state, dtype=np.float32))
        seed = self.seed if seed is None else seed

        with ProfilingContext4DebugL1("RealtimeWAM End-to-End Latency (Excluding Text Encoding)", time_unit="ms"):
            if use_cuda_graph:
                action = self.cuda_graph(image, context, context_mask, robot_state, seed)
            else:
                first_frame_latents = self.encode_image_latents(image)
                inputs, action_shape = self.model.prepare_action_inputs(
                    first_frame_latents=first_frame_latents,
                    context=context,
                    context_mask=context_mask,
                    action_chunk_size=self.action_chunk_size,
                    robot_state=robot_state,
                    seed=seed,
                )

                action = self._run_action_denoising(
                    inputs=inputs,
                    action_shape=action_shape,
                    action_infer_steps=self.action_infer_steps,
                    seed=seed,
                )

        return self._postprocess_actions(action)

    def close(self):
        self.cuda_graph.close()
        super().close()


@RUNNER_REGISTER("realtimewam")
class RealtimeWAMRunner(FastWAMRunner):
    policy_class = RealtimeWAMPolicy

    def warmup(self):
        if not self.config.get("warmup", False):
            return
        if self.policy.policy_profile != "libero" or not self.config.get("cuda_graph", False):
            raise NotImplementedError("RealtimeWAM startup warmup currently supports LIBERO with CUDA Graph only.")
        self.run_warmup()
        self._maybe_freeze_gc()

    @ProfilingContext4DebugL1("Warmup", time_unit="ms")
    def run_warmup(self):
        policy = self.policy
        image = np.zeros((policy.camera_size, policy.camera_size, 3), dtype=np.uint8)
        image_tensor = policy.build_image_tensor({"agentview": image, "wrist": image})
        context, mask = policy.encode_prompt(policy.default_prompt.format(task_prompt="warmup"))
        state = policy.state_normalizer.forward(np.zeros(policy.robot_state_dim, dtype=np.float32))
        # Build and retain the graph through its existing pre-capture warmup path.
        policy.cuda_graph(image_tensor, context, mask, state, seed=0)
        torch.cuda.synchronize(policy.device)
        logger.info("RealtimeWAM startup warmup completed; CUDA Graph is ready.")
