# Torchvision compatibility fix for functional_tensor module.
# Copied from Hunyuan3D-2.1 for standalone postprocess usage.

import sys


def fix_torchvision_functional_tensor():
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401

        return True
    except ImportError:
        try:
            import torch
            import torchvision.transforms.functional as F

            class FunctionalTensorMock:
                @staticmethod
                def rgb_to_grayscale(img, num_output_channels=1):
                    if hasattr(F, "rgb_to_grayscale"):
                        return F.rgb_to_grayscale(img, num_output_channels)
                    if len(img.shape) == 4:
                        weights = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
                    else:
                        weights = torch.tensor([0.299, 0.587, 0.114], device=img.device, dtype=img.dtype).view(3, 1, 1)
                    grayscale = torch.sum(img * weights, dim=-3, keepdim=True)
                    if num_output_channels == 3:
                        if len(img.shape) == 4:
                            grayscale = grayscale.repeat(1, 3, 1, 1)
                        else:
                            grayscale = grayscale.repeat(3, 1, 1)
                    return grayscale

                @staticmethod
                def resize(img, size, interpolation=2, antialias=None):
                    try:
                        from torchvision.transforms.v2.functional import resize as v2_resize

                        return v2_resize(img, size, interpolation=interpolation, antialias=antialias)
                    except ImportError:
                        if hasattr(F, "resize"):
                            return F.resize(img, size, interpolation=interpolation)
                        import torch.nn.functional as torch_F

                        if isinstance(size, int):
                            size = (size, size)
                        return torch_F.interpolate(
                            img.unsqueeze(0) if len(img.shape) == 3 else img,
                            size=size,
                            mode="bilinear",
                            align_corners=False,
                        )

                def __getattr__(self, name):
                    if hasattr(F, name):
                        return getattr(F, name)
                    try:
                        import torchvision.transforms.v2.functional as v2_F

                        if hasattr(v2_F, name):
                            return getattr(v2_F, name)
                    except ImportError:
                        pass
                    raise AttributeError(f"'{name}' not found in functional_tensor mock")

            sys.modules["torchvision.transforms.functional_tensor"] = FunctionalTensorMock()
            return True
        except Exception:
            return False


def apply_fix():
    return fix_torchvision_functional_tensor()
