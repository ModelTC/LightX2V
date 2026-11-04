import json
import math

import pytest


def test_reference_reports_differences_for_review_and_rejects_invalid_output():
    torch = pytest.importorskip("torch")
    from tools.benchmarks.reference import compare_output

    expected = torch.tensor([1.0, 2.0])
    assert compare_output(expected.clone(), expected, scope="full")["passed"] is True
    different = compare_output(expected + 0.01, expected, scope="full")
    assert different["checked"] and different["passed"] is None
    assert different["status"] == "pending_review" and different["max_abs_error"] > 0
    assert compare_output(expected[:1], expected, scope="full")["passed"] is False
    assert compare_output(expected * float("nan"), expected, scope="full")["passed"] is False
    for actual in (expected, expected * float("nan")):
        with pytest.raises(ValueError, match="nonfinite reference"):
            compare_output(actual, expected * float("nan"), scope="full")


@pytest.mark.parametrize("value", [1e20, 1e-30])
def test_reference_metrics_preserve_finite_fp32_range(value):
    torch = pytest.importorskip("torch")
    from tools.benchmarks.reference import compare_output

    reference = torch.tensor([value], dtype=torch.float32)
    result = compare_output(reference * 2, reference, scope="full")
    assert result["strict_equal"] is False and result["passed"] is None
    assert math.isfinite(result["rmse"]) and result["rmse"] > 0
    assert result["reference_rms"] > 0 and result["normalized_rmse"] == pytest.approx(1.0)
    json.dumps(result, allow_nan=False)


def test_reference_metrics_do_not_round_away_double_output_differences():
    torch = pytest.importorskip("torch")
    from tools.benchmarks.reference import compare_output

    expected = torch.tensor([1.0], dtype=torch.float64)
    result = compare_output(expected + 1e-10, expected, scope="full")
    assert result["strict_equal"] is False and result["rmse"] > 0
