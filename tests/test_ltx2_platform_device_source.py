import ast
from pathlib import Path


def test_ltx2_tp_weight_distribution_does_not_hardcode_cuda_current_device():
    source_path = Path(__file__).resolve().parents[1] / "lightx2v/models/networks/ltx2/model.py"
    tree = ast.parse(source_path.read_text())
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_load_weights_from_rank0"
    )

    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_cuda_current_device = (
            isinstance(func, ast.Attribute)
            and func.attr == "current_device"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "cuda"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "torch"
        )
        assert not is_cuda_current_device
