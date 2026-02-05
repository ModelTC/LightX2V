

from enum import Enum


class PredictionType(str, Enum):
    x_0 = "x_0"
    x_T = "x_T"
    v_cos = "v_cos"
    v_lerp = "v_lerp"


class SamplingDirection(str, Enum):
    backward = "backward"
    forward = "forward"

    @staticmethod
    def reverse(direction):
        if direction == SamplingDirection.backward:
            return SamplingDirection.forward
        if direction == SamplingDirection.forward:
            return SamplingDirection.backward
        raise NotImplementedError
