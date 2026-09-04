import numpy as np
import pytest

from modules.evaluations import evaluate_baseline_imputation


def test_evaluate_baseline_imputation_metric_smoke():
    ground_truth = np.array(
        [[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]], dtype=np.float32
    )
    target_mask = np.array(
        [[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], dtype=np.float32
    )
    # Identical draws exactly one unit above the target give deterministic
    # point-error and distributional metrics.
    imputation_samples = np.repeat(
        (ground_truth + 1.0)[:, np.newaxis, :, :], repeats=3, axis=1
    )

    results = evaluate_baseline_imputation(
        imp_samples=imputation_samples,
        ground_truth=ground_truth,
        target_mask=target_mask,
        method_name="LIFTS",
        alphas=[0.5, 0.9],
        print_results=False,
        verbose=False,
    )

    assert results["method"] == "LIFTS"
    assert results["n_samples"] == 1
    assert results["n_imputations"] == 3
    assert results["p"] == 2
    assert results["T"] == 3
    assert results["eval_rate"] == pytest.approx(4 / 6)
    assert results["MAE_raw"] == pytest.approx(1.0)
    assert results["RMSE_raw"] == pytest.approx(1.0)
    assert results["ES_raw"] == pytest.approx(1.0)

    scalar_metrics = [
        "MAE_raw",
        "RMSE_raw",
        "CRPS_raw",
        "MAE_norm",
        "RMSE_norm",
        "CRPS_norm",
        "ES_raw",
        "ES_norm",
        "denominator",
    ]
    assert all(np.isfinite(results[name]) for name in scalar_metrics)
    assert set(results["coverage"]) == {0.5, 0.9}
    assert set(results["ci_width"]) == {0.5, 0.9}
    assert all(value == pytest.approx(0.0) for value in results["coverage"].values())
    assert all(value == pytest.approx(0.0) for value in results["ci_width"].values())


def test_evaluate_baseline_imputation_supports_tp_layout():
    ground_truth_pt = np.arange(6, dtype=np.float32).reshape(1, 2, 3) + 1.0
    target_mask_pt = np.ones_like(ground_truth_pt)
    samples_pt = np.repeat(
        ground_truth_pt[:, np.newaxis, :, :], repeats=2, axis=1
    )

    results = evaluate_baseline_imputation(
        imp_samples=samples_pt.transpose(0, 1, 3, 2),
        ground_truth=ground_truth_pt.transpose(0, 2, 1),
        target_mask=target_mask_pt.transpose(0, 2, 1),
        input_format="Tp",
        alphas=[0.5],
        print_results=False,
        verbose=False,
    )

    assert results["p"] == 2
    assert results["T"] == 3
    assert results["MAE_raw"] == pytest.approx(0.0)
    assert results["RMSE_raw"] == pytest.approx(0.0)


def test_evaluate_baseline_imputation_rejects_non_4d_samples():
    invalid_samples = np.zeros((2, 3, 4), dtype=np.float32)
    ground_truth = np.zeros((2, 3, 4), dtype=np.float32)
    target_mask = np.ones_like(ground_truth)

    with pytest.raises(ValueError):
        evaluate_baseline_imputation(
            imp_samples=invalid_samples,
            ground_truth=ground_truth,
            target_mask=target_mask,
            print_results=False,
            verbose=False,
        )
