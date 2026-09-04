import copy

import numpy as np
import torch

from baselines.fbi_wrapper import FBIImputer, get_fbi_config
from loader.case1 import generate_case1
from loader.case3 import generate_case3
from loader.case4 import generate_case4
from loader.case4_v1 import generate_case4 as generate_case4_v1


def _stack_case_outputs(X_list, mask_list, X_full_list):
    X_full = torch.stack(X_full_list).numpy().astype(np.float32)
    mask = torch.stack(mask_list).numpy().astype(np.float32)
    X_obs = X_full.copy()
    X_obs[mask == 0] = np.nan
    return X_obs, mask, X_full


def make_case1_dataset(
    num_sample_train=500,
    num_sample_test=100,
    p=20,
    T=48,
    d=5,
    obs_rate=0.5,
    coef_eps=0.1,
    seed=42,
    mm="mcar",
):
    [X_train_list, X_test_list], [mask_train_list, mask_test_list], _, [X_train_full_list, X_test_full_list], _, _, _ = generate_case1(
        num_sample_train=num_sample_train,
        num_sample_test=num_sample_test,
        p=p,
        T=T,
        d=d,
        obs_rate=obs_rate,
        coef_eps=coef_eps,
        seed=seed,
        mm=mm,
    )
    X_train, mask_train, X_train_full = _stack_case_outputs(X_train_list, mask_train_list, X_train_full_list)
    X_test, mask_test, X_test_full = _stack_case_outputs(X_test_list, mask_test_list, X_test_full_list)
    return {
        "X_train": X_train,
        "mask_train": mask_train,
        "X_train_full": X_train_full,
        "X_test": X_test,
        "mask_test": mask_test,
        "X_test_full": X_test_full,
    }


def make_case3_dataset(
    num_sample_train=500,
    num_sample_test=100,
    p=20,
    T=48,
    d=5,
    obs_rate=0.5,
    coef_eps=0.1,
    seed=42,
    mm="mcar",
):
    [X_train_list, X_test_list], [mask_train_list, mask_test_list], _, [X_train_full_list, X_test_full_list], _, _, _ = generate_case3(
        num_sample_train=num_sample_train,
        num_sample_test=num_sample_test,
        p=p,
        T=T,
        d=d,
        obs_rate=obs_rate,
        coef_eps=coef_eps,
        seed=seed,
        mm=mm,
    )
    X_train, mask_train, X_train_full = _stack_case_outputs(X_train_list, mask_train_list, X_train_full_list)
    X_test, mask_test, X_test_full = _stack_case_outputs(X_test_list, mask_test_list, X_test_full_list)
    return {
        "X_train": X_train,
        "mask_train": mask_train,
        "X_train_full": X_train_full,
        "X_test": X_test,
        "mask_test": mask_test,
        "X_test_full": X_test_full,
    }


def make_case4_dataset(
    num_sample_train=500,
    num_sample_test=100,
    p=20,
    T=48,
    d=5,
    obs_rate=0.5,
    coef_eps=1.0,
    seed=42,
    mm="mcar",
    version="v1",
):
    version_key = str(version).lower()
    if version_key in {"v1", "benchmark"}:
        generator = generate_case4_v1
    elif version_key in {"v2", "current"}:
        generator = generate_case4
    else:
        raise ValueError("version must be 'v1', 'benchmark', 'v2', or 'current'.")
    [X_train_list, X_test_list], [mask_train_list, mask_test_list], _, [X_train_full_list, X_test_full_list], _, _, _ = generator(
        num_sample_train=num_sample_train,
        num_sample_test=num_sample_test,
        p=p,
        T=T,
        d=d,
        obs_rate=obs_rate,
        coef_eps=coef_eps,
        seed=seed,
        mm=mm,
    )
    X_train, mask_train, X_train_full = _stack_case_outputs(X_train_list, mask_train_list, X_train_full_list)
    X_test, mask_test, X_test_full = _stack_case_outputs(X_test_list, mask_test_list, X_test_full_list)
    return {
        "X_train": X_train,
        "mask_train": mask_train,
        "X_train_full": X_train_full,
        "X_test": X_test,
        "mask_test": mask_test,
        "X_test_full": X_test_full,
    }


def make_default_config(
    p,
    T,
    d=None,
    obs_rate=0.5,
    total_loops=100,
    batch_size=128,
    lr=1e-3,
    lambda_selfmask=5.0,
    lambda_enc=0.0,
    enc_warmup=50,
    enc_syn_batch=None,
    enc_syn_mini_batch=None,
    enc_noise_scale=0.0,
    enc_ema_decay=0.999,
    drop_rate=0.2,
    gen_sample_size=1,
    fx_type="mlp",
    fo_type="mlp",
    encoder_type="v1",
    smoother_type="v1",
    decoder_type="v2",
    use_mask_decoder=False,
    **overrides,
):
    cfg = get_fbi_config(p=p, T=T, d=d, obs_rate=obs_rate)
    cfg["train"]["total_alternating_loops"] = total_loops
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["batch_size_fg"] = batch_size
    cfg["train"]["batch_size_huq"] = batch_size
    cfg["train"]["batch_size_u"] = batch_size
    cfg["train"]["lr_fg"] = lr
    cfg["train"]["lr_huq"] = lr
    cfg["train"]["lambda_selfmask"] = lambda_selfmask
    cfg["train"]["lambda_enc"] = lambda_enc
    cfg["train"]["enc_warmup"] = enc_warmup
    cfg["train"]["enc_syn_batch"] = enc_syn_batch
    cfg["train"]["enc_syn_mini_batch"] = enc_syn_mini_batch
    cfg["train"]["enc_noise_scale"] = enc_noise_scale
    cfg["train"]["enc_ema_decay"] = enc_ema_decay
    cfg["train"]["drop_rate"] = drop_rate
    cfg["train"]["gen_sample_size"] = gen_sample_size
    cfg["model"]["fx_type"] = fx_type
    cfg["model"]["fo_type"] = fo_type
    cfg["model"]["encoder_type"] = encoder_type
    cfg["model"]["smoother_type"] = smoother_type
    cfg["model"]["decoder_type"] = decoder_type
    cfg["model"]["use_mask_decoder"] = use_mask_decoder

    for section, section_overrides in overrides.items():
        if section not in cfg or not isinstance(section_overrides, dict):
            cfg[section] = section_overrides
        else:
            cfg[section].update(section_overrides)

    return cfg


class LIFTSImputer(FBIImputer):
    """Sklearn-style estimator facade for LIFTS imputation.

    Parameters may be supplied directly in the constructor. If ``p`` and ``T``
    are omitted, they are inferred from ``X`` during ``fit``.
    """

    def __init__(
        self,
        p=None,
        T=None,
        d=None,
        obs_rate=0.5,
        total_loops=100,
        batch_size=128,
        n_samples=100,
        lr=1e-3,
        lambda_selfmask=5.0,
        lambda_enc=0.0,
        enc_warmup=50,
        enc_syn_batch=None,
        enc_syn_mini_batch=None,
        enc_noise_scale=0.0,
        enc_ema_decay=0.999,
        drop_rate=0.2,
        gen_sample_size=1,
        fx_type="mlp",
        fo_type="mlp",
        encoder_type="v1",
        smoother_type="v1",
        decoder_type="v2",
        use_mask_decoder=False,
        device=None,
        seed=None,
        config=None,
        **overrides,
    ):
        self.p = p
        self.T = T
        self.d = d
        self.obs_rate = obs_rate
        self.total_loops = total_loops
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.lr = lr
        self.lambda_selfmask = lambda_selfmask
        self.lambda_enc = lambda_enc
        self.enc_warmup = enc_warmup
        self.enc_syn_batch = enc_syn_batch
        self.enc_syn_mini_batch = enc_syn_mini_batch
        self.enc_noise_scale = enc_noise_scale
        self.enc_ema_decay = enc_ema_decay
        self.drop_rate = drop_rate
        self.gen_sample_size = gen_sample_size
        self.fx_type = fx_type
        self.fo_type = fo_type
        self.encoder_type = encoder_type
        self.smoother_type = smoother_type
        self.decoder_type = decoder_type
        self.use_mask_decoder = use_mask_decoder
        self.overrides = copy.deepcopy(overrides)

        initial_config = copy.deepcopy(config) if config is not None else None
        if initial_config is None and p is not None and T is not None:
            initial_config = self._make_config(p=p, T=T)

        super().__init__(
            config=initial_config,
            device=device,
            n_samples=n_samples,
            seed=seed,
            total_loops=total_loops,
            batch_size=batch_size,
        )
        self.total_loops = total_loops
        self.batch_size = batch_size

    def _make_config(self, p, T):
        return make_default_config(
            p=p,
            T=T,
            d=self.d,
            obs_rate=self.obs_rate,
            total_loops=self.total_loops,
            batch_size=self.batch_size,
            lr=self.lr,
            lambda_selfmask=self.lambda_selfmask,
            lambda_enc=self.lambda_enc,
            enc_warmup=self.enc_warmup,
            enc_syn_batch=self.enc_syn_batch,
            enc_syn_mini_batch=self.enc_syn_mini_batch,
            enc_noise_scale=self.enc_noise_scale,
            enc_ema_decay=self.enc_ema_decay,
            drop_rate=self.drop_rate,
            gen_sample_size=self.gen_sample_size,
            fx_type=self.fx_type,
            fo_type=self.fo_type,
            encoder_type=self.encoder_type,
            smoother_type=self.smoother_type,
            decoder_type=self.decoder_type,
            use_mask_decoder=self.use_mask_decoder,
            **self.overrides,
        )

    def fit(self, X, mask, **kwargs):
        if self.config is None:
            if len(X.shape) != 3:
                raise ValueError("X must have shape (n_samples, n_features, n_timepoints).")
            _, inferred_p, inferred_T = X.shape
            self.p = inferred_p if self.p is None else self.p
            self.T = inferred_T if self.T is None else self.T
            self.config = self._make_config(p=self.p, T=self.T)
        return super().fit(X, mask, **kwargs)

    def transform(self, X, mask, n_samples=None):
        return self.impute_samples(X, mask, n_samples=n_samples)

    def predict(self, X, mask, n_samples=None):
        return self.transform(X, mask, n_samples=n_samples)

    def fit_transform(self, X, mask, n_samples=None, **fit_kwargs):
        self.fit(X, mask, **fit_kwargs)
        return self.transform(X, mask, n_samples=n_samples)

    def impute(self, X, mask, n_samples=None):
        samples = self.impute_samples(X, mask, n_samples=n_samples)
        return samples.mean(axis=1)

    def get_params(self, deep=True):
        return {
            "p": self.p,
            "T": self.T,
            "d": self.d,
            "obs_rate": self.obs_rate,
            "total_loops": self.total_loops,
            "batch_size": self.batch_size,
            "n_samples": self.n_samples,
            "lr": self.lr,
            "lambda_selfmask": self.lambda_selfmask,
            "lambda_enc": self.lambda_enc,
            "enc_warmup": self.enc_warmup,
            "enc_syn_batch": self.enc_syn_batch,
            "enc_syn_mini_batch": self.enc_syn_mini_batch,
            "enc_noise_scale": self.enc_noise_scale,
            "enc_ema_decay": self.enc_ema_decay,
            "drop_rate": self.drop_rate,
            "gen_sample_size": self.gen_sample_size,
            "fx_type": self.fx_type,
            "fo_type": self.fo_type,
            "encoder_type": self.encoder_type,
            "smoother_type": self.smoother_type,
            "decoder_type": self.decoder_type,
            "use_mask_decoder": self.use_mask_decoder,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter {key!r}.")
            setattr(self, key, value)
        if self.p is not None and self.T is not None:
            self.config = self._make_config(p=self.p, T=self.T)
        return self
