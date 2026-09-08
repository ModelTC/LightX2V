import copy
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from lightx2v_train.model_zoo.native.wan.fastwam.action_distill import CachedActionDenoiser, build_action_distill_condition
from lightx2v_train.model_zoo.native.wan.fastwam.mot import MoT

DEFAULT_LORA_TARGETS = ["q", "k", "v", "o", "ffn.0", "ffn.2"]


class JointConsistencyDenoiser(CachedActionDenoiser):
    """Build this role's video cache inside the action forward (and DDP)."""

    def __init__(self, expert, video_expert, model):
        mot = MoT(
            mixtures={"video": video_expert, "action": expert},
            mot_checkpoint_mixed_attn=model.mot.mot_checkpoint_mixed_attn,
        )
        super().__init__(expert, mot)
        self.mot = mot
        # Mask construction is weight-independent; VAE/text/proprio stay shared.
        self._build_mot_attention_mask = model._build_mot_attention_mask

    @property
    def video_expert(self):
        return self.mot.mixtures["video"]

    def build_condition(self, inputs):
        return build_action_distill_condition(self, inputs, requires_grad=torch.is_grad_enabled())

    def forward(self, action, timestep, condition):
        if isinstance(condition, dict):
            condition = self.build_condition(condition)
        return super().forward(action, timestep, condition)


def configure_student(expert, config):
    expert.requires_grad_(False)
    if config.train_type == "full":
        expert.requires_grad_(True)
        return expert.train()

    lora = config.lora
    expert = get_peft_model(
        expert,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora.get("alpha", lora["rank"])),
            lora_dropout=float(lora.get("dropout", 0.0)),
            init_lora_weights="gaussian",
            target_modules=list(lora.get("target_modules", DEFAULT_LORA_TARGETS)),
            modules_to_save=list(lora.get("modules_to_save", [])),
        ),
    )
    for parameter in expert.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    return expert.train()


def role_state_dict(expert, train_type):
    return get_peft_model_state_dict(expert) if train_type == "lora" else expert.state_dict()


def load_role_state_dict(expert, train_type, state_dict):
    if train_type == "full":
        expert.load_state_dict(state_dict, strict=True)
        return
    expected = set(get_peft_model_state_dict(expert))
    actual = set(state_dict)
    if expected != actual:
        raise RuntimeError(f"LoRA checkpoint structure mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
    incompatible = set_peft_model_state_dict(expert, state_dict)
    if incompatible and incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected LoRA checkpoint keys: {incompatible.unexpected_keys}")


@dataclass
class ConsistencyRoles:
    student: object
    target: object
    teacher: object
    ema_pairs: tuple

    @classmethod
    def build(cls, expert, config):
        teacher = copy.deepcopy(expert).eval().requires_grad_(False)
        student = configure_student(expert, config)
        trainable_names = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
        target = copy.deepcopy(student).eval().requires_grad_(False)
        target_parameters = dict(target.named_parameters())
        ema_pairs = tuple((target_parameters[name], dict(student.named_parameters())[name]) for name in trainable_names)
        return cls(student=student, target=target, teacher=teacher, ema_pairs=ema_pairs)

    @property
    def trainable_parameters(self):
        return [parameter for parameter in self.student.parameters() if parameter.requires_grad]

    @torch.no_grad()
    def copy_student_to_target(self):
        for target, student in self.ema_pairs:
            target.copy_(student.detach().to(dtype=target.dtype))

    @torch.no_grad()
    def update_target(self, decay):
        for target, student in self.ema_pairs:
            target.lerp_(student.detach().to(dtype=target.dtype), 1.0 - decay)
