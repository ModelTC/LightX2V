"""Output and normalized block-feature KD (Dense2MoE equations 7–10)."""

import torch
from torch.nn import functional as F


class FeatureMSE(torch.autograd.Function):
    """Accumulate in fp32 without retaining fp32 copies of every block output."""

    @staticmethod
    def forward(ctx, student, teacher):
        ctx.save_for_backward(student, teacher)
        target = teacher.to(student.device, non_blocking=True)
        return (student.float() - target.float()).square().mean()

    @staticmethod
    def backward(ctx, gradient):
        student, teacher = ctx.saved_tensors
        target = teacher.to(student.device, non_blocking=True)
        delta = (student.float() - target.float()) * (2.0 / student.numel())
        return (delta * gradient).to(student.dtype), None


def modality_losses(student, teacher):
    return {name: F.mse_loss(value.float(), teacher[name].float()) for name, value in student.items()}


def feature_distillation_loss(output, student_features, teacher_features):
    errors = torch.stack([FeatureMSE.apply(s, t) for s, t in zip(student_features, teacher_features)])
    norms = torch.stack([t.float().norm().to(output.device) for t in teacher_features])
    # Weights are balancing coefficients, not trainable functions of the error.
    weights = (output.detach() / errors.detach().clamp_min(1e-12)) * (norms.mean() / norms.clamp_min(1e-12))
    return (weights * errors).sum()


class BlockFeatures:
    """Capture the ordinary forward only; checkpoint recomputation is ignored."""

    def __init__(self, transformer, layers, *, offload=False):
        self.values = []
        self.enabled = False
        self.offload = offload
        self.handles = [transformer.transformer_blocks[i].register_forward_hook(self.capture) for i in layers]

    def capture(self, module, inputs, output):
        if self.enabled:
            if self.offload:
                target = torch.empty_like(output, device="cpu", pin_memory=True)
                target.copy_(output, non_blocking=True)
                output = target
            self.values.append(output)

    def start(self):
        self.values.clear()
        self.enabled = True

    def stop(self):
        self.enabled = False
        if self.offload:
            # Finish D2H copies before CPU normalization or H2D reuse.
            torch.cuda.current_stream().synchronize()
        return self.values
