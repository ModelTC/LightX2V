"""Full-output SP comparisons, independent from performance timing."""


def compare_output(actual, expected, *, scope, tolerance=None):
    import torch

    shape = list(actual.shape) if isinstance(actual, torch.Tensor) else None
    value = {
        "checked": True,
        "passed": None,
        "status": "pending_review",
        "scope": scope,
        "reference": "semantic reference; quality adoption is separate",
        "actual_output_shape": shape,
        "reference_output_shape": list(expected.shape),
        "shape_passed": shape == list(expected.shape),
        "finite": False,
        "allclose_passed": False if tolerance else None,
    }
    if not torch.isfinite(expected).all():
        raise ValueError("nonfinite reference output; correctness comparison unavailable")
    if not value["shape_passed"]:
        return {**value, "passed": False, "status": "failed", "reason": "output_shape_mismatch"}
    if not torch.isfinite(actual).all():
        return {**value, "passed": False, "status": "failed", "reason": "nonfinite_output"}
    actual, expected = actual.double(), expected.double()
    error = actual - expected
    rmse = error.square().mean().sqrt().item()
    reference_rms = expected.square().mean().sqrt().item()
    strict = torch.equal(actual, expected)
    value.update(
        finite=True,
        strict_equal=strict,
        compared_elements=actual.numel(),
        max_abs_error=error.abs().max().item(),
        rmse=rmse,
        reference_rms=reference_rms,
        normalized_rmse=rmse / reference_rms if reference_rms else (0.0 if rmse == 0 else None),
        allclose_passed=torch.allclose(actual, expected, atol=tolerance["atol"], rtol=tolerance["rtol"]) if tolerance else None,
    )
    if strict:
        value.update(passed=True, status="strict_equal")
    return value
