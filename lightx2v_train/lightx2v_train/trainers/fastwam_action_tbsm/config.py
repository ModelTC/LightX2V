from dataclasses import dataclass

from lightx2v_train.trainers.fastwam_action_consistency.config import ActionStudentConfig


@dataclass(frozen=True)
class FastWAMActionTBSMConfig:
    student: ActionStudentConfig
    positive_source: str
    teacher_steps: int
    ema_decay: float

    @property
    def teacher_reference_steps(self):
        """Step count consumed by the shared action-policy evaluator."""
        return self.teacher_steps

    @classmethod
    def from_mapping(cls, config):
        training = config["training"]
        tbsm = training.get("action_tbsm")
        if not isinstance(tbsm, dict):
            raise TypeError("training.action_tbsm is required.")
        positive_source = str(tbsm.get("positive_source", "teacher"))
        teacher_steps = int(tbsm.get("teacher_steps", 20))
        ema_decay = float(tbsm.get("ema_decay", 0.995))
        if positive_source not in {"teacher", "data"}:
            raise ValueError("training.action_tbsm.positive_source must be 'teacher' or 'data'.")
        if teacher_steps <= 0:
            raise ValueError("training.action_tbsm.teacher_steps must be positive.")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("training.action_tbsm.ema_decay must be in [0, 1).")
        return cls(
            student=ActionStudentConfig.from_mapping(training["student"]),
            positive_source=positive_source,
            teacher_steps=teacher_steps,
            ema_decay=ema_decay,
        )
