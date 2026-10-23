# LIFTS

[![PyPI version](https://img.shields.io/pypi/v/lifts.svg)](https://pypi.org/project/lifts/)
[![Python](https://img.shields.io/pypi/pyversions/lifts.svg)](https://pypi.org/project/lifts/)
[![CI](https://github.com/sanyouwu/lifts-open-source/actions/workflows/ci.yml/badge.svg)](https://github.com/sanyouwu/lifts-open-source/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LIFTS is a Python package for probabilistic imputation of incomplete multivariate time series. It learns latent state-space dynamics and returns multiple imputations, so downstream analyses can retain uncertainty instead of relying on a single completed data set.

The current package version is `0.1.0`.

## Installation

Download the complete repository ZIP and extract it; downloading an individual
notebook is not sufficient. Open a terminal in the extracted repository root
(the directory containing `pyproject.toml`) and run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
jupyter lab
```

Open `examples/notebooks/case1_lifts.ipynb` from JupyterLab. Keep the extracted
repository in place while using the editable installation. The first install
requires an internet connection to download dependencies. A GPU is optional;
the example automatically uses the CPU when CUDA is unavailable.

For development and testing, install the development dependencies as well:

```bash
python -m pip install -e ".[dev,notebook]"
```

PyTorch automatically uses CUDA when it is available unless `device="cpu"` is passed to `LIFTSImputer`. A GPU is helpful for larger experiments but is not required for the small examples or the test suite.

## Case 1 quickstart

This example generates linear factor state-space data with MCAR missingness. `obs_rate=0.5` gives each entry a 50% observation probability, so the expected missing rate is 50%; a finite generated data set will generally differ slightly from exactly 50%.

```python
from lifts import LIFTSImputer, make_case1_dataset

data = make_case1_dataset(
    num_sample_train=80,
    num_sample_test=20,
    p=20,
    T=48,
    d=5,
    obs_rate=0.5,
    mm="mcar",
    seed=42,
)

imputer = LIFTSImputer(
    d=5,
    total_loops=5,
    batch_size=32,
    n_samples=20,
    lambda_selfmask=5.0,
    lambda_enc=1.0,
    enc_warmup=2,
    device="cpu",  # remove this line to use CUDA when available
    seed=42,
)

imputer.fit(data["X_train"], data["mask_train"], verbose=True)
imputation_samples = imputer.impute_samples(
    data["X_test"], data["mask_test"], n_samples=20
)

print(imputation_samples.shape)  # (20, 20, 20, 48): (N, draws, p, T)
completed = imputation_samples.mean(axis=1)  # (N, p, T)
```

Training a probabilistic neural model can take several minutes on CPU. Increase `total_loops`, the training-set size, and `n_samples` for substantive experiments.

The runnable [Case 1 notebook](examples/notebooks/case1_lifts.ipynb) uses the same linear state-space, MCAR, 50%-missing setting and includes evaluation plus uncertainty visualization.

## Visualizing imputations

`plot_probabilistic_imputation` plots the median and 95% interval across imputation draws. Observed values are shown separately from held-out values.

```python
import torch
from lifts import plot_probabilistic_imputation

sample_index = 0
n_features_show = 8

plot_probabilistic_imputation(
    torch.from_numpy(
        imputation_samples[sample_index, :, :n_features_show, :]
    ).float(),
    torch.from_numpy(
        data["X_test_full"][sample_index, :n_features_show, :]
    ).float(),
    p=n_features_show,
    T=data["X_test_full"].shape[-1],
    obs_mask=data["mask_test"][sample_index, :n_features_show, :],
    title="LIFTS probabilistic imputation",
)
```

## Data conventions

LIFTS uses NumPy arrays with feature-first time-series shape:

| Value | Shape | Meaning |
| --- | --- | --- |
| `X` | `(N, p, T)` | Incomplete input for `N` series, `p` features, and `T` time points |
| `mask` | `(N, p, T)` | Binary observation indicator: `1` observed, `0` missing |
| `impute_samples(...)` | `(N, m, p, T)` | `m` probabilistic completed-data draws |
| `impute(...)` | `(N, p, T)` | Mean across the `m` imputation draws |

Use `numpy.nan` for missing entries in `X` when relying on the default normalization in `fit`; the mask remains the authoritative observation indicator. Pre-normalized, zero-filled inputs are supported with `fit(..., skip_normalize=True)`. The built-in dataset helpers return incomplete inputs (`X_train`, `X_test`), masks, and complete simulation truth (`X_train_full`, `X_test_full`). The complete arrays are for simulation evaluation and should not be passed to training as observed data. Imputation outputs preserve the supplied observed entries exactly.

## Public API

The following names are exported from `lifts`:

| API | Purpose |
| --- | --- |
| `LIFTSImputer` | Fit the model and generate probabilistic imputations or in-sample forecasts |
| `make_case1_dataset` | Linear factor state-space simulation |
| `make_case3_dataset` | Nonlinear simulation variant |
| `make_case4_dataset` | Case 4 simulation (`version="v1"` or `"v2"`) |
| `make_default_config` | Build a lower-level configuration dictionary for advanced use |
| `plot_probabilistic_imputation` | Plot imputation draws, intervals, observed entries, and held-out truth |
| `plot_probabilistic_forecast` | Plot full-signal in-sample forecast draws |

Important estimator methods are:

- `fit(X, mask)`: train on incomplete arrays.
- `impute_samples(X, mask, n_samples=...)`: return probabilistic completed-data draws.
- `impute(X, mask, n_samples=...)`: return their mean.
- `transform(...)` and `predict(...)`: aliases for `impute_samples(...)`, so they return a draw dimension rather than a single array.
- `forecast_samples(...)` and `forecast(...)`: return full-signal in-sample predictions.
- `save(path)` and `load(path)`: persist and restore a fitted model.

`p` and `T` can be passed to the constructor or inferred from the first training array during `fit`.

### What “in-sample forecast” means

LIFTS forecasting is not an out-of-sample, future-horizon forecasting interface. `forecast_samples(X, mask)` uses the observed portions of the supplied length-`T` sequence to encode latent states, applies the learned transition and output models, and generates predictions over those same `T` positions. Unlike imputation, forecast output does not overwrite observed positions with their input values. Use `impute_samples` when the goal is to fill missing entries while retaining observations.

## Development

Install development dependencies and run the checks from the repository root:

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and [RELEASE.md](RELEASE.md) for the standalone-repository release process.

## Compatibility note

Version `0.1.0` retains the legacy import aliases `FBIImputer` and `get_fbi_config` for compatibility with earlier internal experiments. They are not the public-facing method name; new code, documentation, and issue reports should use `LIFTSImputer` and LIFTS terminology.

## License and acknowledgements

LIFTS is released under the [MIT License](LICENSE). Portions of the attention-layer implementation derive from the MIT-licensed SAITS project; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and citation details.
