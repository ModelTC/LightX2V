from dataclasses import dataclass


@dataclass(frozen=True)
class ActionStudentConfig:
    train_type: str
    optimizer: dict
    lora: dict | None

    @classmethod
    def from_mapping(cls, mapping):
        train_type = str(mapping.get("train_type", "lora")).lower()
        if train_type not in {"lora", "full"}:
            raise ValueError(f"training.student.train_type must be 'lora' or 'full', got {train_type!r}.")
        lora = mapping.get("lora")
        if train_type == "lora" and (not isinstance(lora, dict) or int(lora.get("rank", 0)) <= 0):
            raise ValueError("training.student.lora with a positive rank is required for LoRA training.")
        optimizer = mapping.get("optimizer")
        if not isinstance(optimizer, dict):
            raise TypeError("training.student.optimizer is required.")
        return cls(train_type=train_type, optimizer=optimizer, lora=lora)


@dataclass(frozen=True)
class FastWAMJointConsistencyConfig:
    student: ActionStudentConfig
    train_video: bool
    target_steps: int
    teacher_reference_steps: int
    ema_decay: float
    consistency_loss_weight: float
    flow_loss_weight: float
    huber_c: float

    @classmethod
    def from_mapping(cls, config):
        training = config["training"]
        train_video = training.get("train_video", False)
        if not isinstance(train_video, bool):
            raise TypeError("training.train_video must be a boolean.")
        consistency = training.get("action_consistency")
        if not isinstance(consistency, dict):
            raise TypeError("training.action_consistency is required.")

        target_steps = int(consistency.get("target_steps", 2))
        teacher_reference_steps = int(consistency.get("teacher_reference_steps", 20))
        ema_decay = float(consistency.get("ema_decay", 0.995))
        consistency_weight = float(consistency.get("consistency_loss_weight", 1.0))
        flow_weight = float(consistency.get("flow_loss_weight", 0.2))
        huber_c = float(consistency.get("huber_c", 0.001))
        if target_steps <= 0:
            raise ValueError("training.action_consistency.target_steps must be positive.")
        if teacher_reference_steps <= 0:
            raise ValueError("training.action_consistency.teacher_reference_steps must be positive.")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("training.action_consistency.ema_decay must be in [0, 1).")
        if consistency_weight < 0.0 or flow_weight < 0.0 or consistency_weight + flow_weight == 0.0:
            raise ValueError("Consistency/flow loss weights must be non-negative and not both zero.")
        if huber_c <= 0.0:
            raise ValueError("training.action_consistency.huber_c must be positive.")

        return cls(
            student=ActionStudentConfig.from_mapping(training["student"]),
            train_video=train_video,
            target_steps=target_steps,
            teacher_reference_steps=teacher_reference_steps,
            ema_decay=ema_decay,
            consistency_loss_weight=consistency_weight,
            flow_loss_weight=flow_weight,
            huber_c=huber_c,
        )
