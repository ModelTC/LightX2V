import math
from abc import ABC, abstractmethod


class BaseTransformerInfer(ABC):
    @abstractmethod
    def infer(self):
        pass

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.scheduler.transformer_infer = self


class BaseTaylorCachingTransformerInfer(BaseTransformerInfer):
    @abstractmethod
    def infer_calculating(self):
        pass

    @abstractmethod
    def infer_using_cache(self):
        pass

    @abstractmethod
    def get_taylor_step_diff(self):
        pass

    # 1. when fully calcualted, stored in cache
    def derivative_approximation(self, block_cache, module_name, out):
        if module_name not in block_cache:
            block_cache[module_name] = {0: out}
        else:
            step_diff = self.get_taylor_step_diff()

            previous_out = block_cache[module_name][0]
            block_cache[module_name][0] = out
            block_cache[module_name][1] = (out - previous_out) / step_diff

    def taylor_formula(self, tensor_dict):
        x = self.get_taylor_step_diff()

        output = 0
        for i in range(len(tensor_dict)):
            output += (1 / math.factorial(i)) * tensor_dict[i] * (x**i)

        return output
