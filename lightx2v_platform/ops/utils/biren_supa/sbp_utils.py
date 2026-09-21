try:
    import torch_br
except ImportError:
    torch_br = None


def _tensor_layout(input_tensor, default="colmajor"):
    if torch_br is None:
        return default
    try:
        return torch_br.supa._debug.get_tensor_info(input_tensor)[0]["layout"].lower()
    except Exception:
        return default


def _tensor_sbp_axis(input_tensor):
    try:
        from torch_br.utils.tensor_methods import Sbp

        sbp_id = input_tensor.sbp().id
        if sbp_id == Sbp.bb():
            return "BB", 0
        if sbp_id == Sbp.pp():
            return "PP", 0
        if sbp_id == Sbp.sb(0):
            return "SB", 0
        if sbp_id == Sbp.sb(1):
            return "SB", 1
        if sbp_id == Sbp.sb(2):
            return "SB", 2
    except Exception:
        pass
    return None, None


def convBB(input_tensor, layout=None):
    if torch_br is None:
        return input_tensor
    out_layout = layout.lower() if layout is not None else _tensor_layout(input_tensor)
    if hasattr(input_tensor, "sbp") and callable(input_tensor.sbp) and input_tensor.sbp().is_bb:
        if layout is None or out_layout == _tensor_layout(input_tensor):
            return input_tensor
    input_sbp, _ = _tensor_sbp_axis(input_tensor)
    if input_sbp == "BB" and out_layout == _tensor_layout(input_tensor):
        return input_tensor
    out = torch_br._empty_ut_only(
        size=input_tensor.shape,
        dtype=input_tensor.dtype,
        is_numa=False,
        device=input_tensor.device,
        tensor_type=out_layout,
        sbp="BB",
    )
    out.copy_(input_tensor)
    return out

def convSB(input_tensor, axis=0, layout=None):
    if torch_br is None:
        return input_tensor
    if input_tensor.dim() <= axis or input_tensor.shape[axis] < 2:
        return input_tensor
    out_layout = layout.lower() if layout is not None else _tensor_layout(input_tensor)
    input_sbp, input_axis = _tensor_sbp_axis(input_tensor)
    if input_sbp == "SB" and input_axis == axis and out_layout == _tensor_layout(input_tensor):
        return input_tensor
    out = torch_br._empty_ut_only(
        size=input_tensor.shape,
        dtype=input_tensor.dtype,
        is_numa=False,
        device=input_tensor.device,
        tensor_type=out_layout,
        sbp="SB",
        axis=axis,
    )
    out.copy_(input_tensor)
    return out
