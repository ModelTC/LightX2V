import torch
from lightx2v_train.model_zoo.capability_adapters import PeftAdapterBankCapability


class FakeDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {
                "student": torch.nn.Linear(2, 1, bias=False),
                "teacher": torch.nn.Linear(2, 1, bias=False),
            }
        )
        self.active_adapter = None

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name


class FakeModel:
    def __init__(self):
        self.denoiser = FakeDenoiser()

    def denoiser_module(self):
        return self.denoiser

    def load_lora_weights_for_resume(self, *args, **kwargs):
        pass

    def save_lora_weights(self, *args, **kwargs):
        pass


def test_adapter_copy_ema_trainability_and_scoped_activation_are_strict():
    model = FakeModel()
    adapters = PeftAdapterBankCapability(model)
    student = model.denoiser.lora_A["student"].weight
    teacher = model.denoiser.lora_A["teacher"].weight
    student.data.fill_(2.0)
    teacher.data.zero_()

    adapters.copy("student", "teacher")
    assert torch.allclose(teacher, torch.full_like(teacher, 2.0))

    student.data.fill_(4.0)
    adapters.ema_update("student", "teacher", decay=0.5)
    assert torch.allclose(teacher, torch.full_like(teacher, 3.0))

    adapters.set_trainable("student")
    assert student.requires_grad
    assert not teacher.requires_grad
    assert model.denoiser.active_adapter == "student"

    with adapters.activate("teacher", training=False):
        assert model.denoiser.active_adapter == "teacher"
        assert not model.denoiser.training
    assert model.denoiser.active_adapter == "student"
    assert model.denoiser.training


def test_adapter_pairing_fails_instead_of_silently_skipping_missing_weights():
    model = FakeModel()
    del model.denoiser.lora_A["teacher"]
    adapters = PeftAdapterBankCapability(model)

    try:
        adapters.copy("student", "teacher")
    except RuntimeError as error:
        assert "No parameters found" in str(error)
    else:
        raise AssertionError("Expected strict adapter pairing to reject a missing teacher.")
