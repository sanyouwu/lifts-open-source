import numpy as np
import torch

from lifts import LIFTSImputer, make_case1_dataset


def test_case1_mcar_dataset_shapes_and_missing_value_contract():
    n_train, n_test, p, time_steps = 12, 5, 4, 8
    requested_obs_rate = 0.5

    data = make_case1_dataset(
        num_sample_train=n_train,
        num_sample_test=n_test,
        p=p,
        T=time_steps,
        d=2,
        obs_rate=requested_obs_rate,
        seed=17,
        mm="mcar",
    )

    expected_shapes = {
        "X_train": (n_train, p, time_steps),
        "mask_train": (n_train, p, time_steps),
        "X_train_full": (n_train, p, time_steps),
        "X_test": (n_test, p, time_steps),
        "mask_test": (n_test, p, time_steps),
        "X_test_full": (n_test, p, time_steps),
    }
    assert set(data) == set(expected_shapes)
    for name, shape in expected_shapes.items():
        assert data[name].shape == shape
        assert data[name].dtype == np.float32

    for split in ("train", "test"):
        observed = data[f"X_{split}"]
        mask = data[f"mask_{split}"]
        complete = data[f"X_{split}_full"]

        assert set(np.unique(mask)).issubset({0.0, 1.0})
        assert np.isfinite(complete).all()
        np.testing.assert_array_equal(np.isnan(observed), mask == 0)
        np.testing.assert_allclose(observed[mask == 1], complete[mask == 1])

    combined_mask = np.concatenate(
        [data["mask_train"].ravel(), data["mask_test"].ravel()]
    )
    assert abs(float(combined_mask.mean()) - requested_obs_rate) < 0.1


def test_lifts_imputer_one_loop_cpu_smoke_and_output_contract():
    complete = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            [[2.0, 3.0, 4.0], [1.0, 2.0, 5.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [[1, 0, 1], [0, 1, 1]],
            [[1, 1, 0], [1, 0, 1]],
        ],
        dtype=np.float32,
    )
    observed = complete.copy()
    observed[mask == 0] = np.nan

    model = LIFTSImputer(
        d=2,
        total_loops=1,
        batch_size=2,
        n_samples=3,
        lr=1e-3,
        lambda_selfmask=1.0,
        drop_rate=0.5,
        gen_sample_size=1,
        fx_type="linear",
        fo_type="constant",
        decoder_type="u",
        device=torch.device("cpu"),
        seed=11,
    )
    model.fit(observed, mask, verbose=False)

    assert model.trained
    assert next(model.model.parameters()).device.type == "cpu"

    imputation_samples = model.impute_samples(
        observed,
        mask,
        n_samples=3,
        batch_size=2,
        samples_per_forward=3,
    )
    forecast_samples = model.forecast_samples(
        observed,
        mask,
        n_samples=3,
        batch_size=2,
        samples_per_forward=3,
    )

    expected_shape = (2, 3, 2, 3)
    assert imputation_samples.shape == expected_shape
    assert forecast_samples.shape == expected_shape
    assert np.isfinite(imputation_samples).all()
    assert np.isfinite(forecast_samples).all()

    observed_locations = np.broadcast_to(
        mask[:, np.newaxis, :, :].astype(bool), expected_shape
    )
    expected_observed = np.broadcast_to(
        complete[:, np.newaxis, :, :], expected_shape
    )
    np.testing.assert_allclose(
        imputation_samples[observed_locations],
        expected_observed[observed_locations],
        rtol=0.0,
        atol=0.0,
    )
