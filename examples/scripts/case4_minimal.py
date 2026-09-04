import numpy as np
import torch

from lifts import LIFTSImputer, make_case4_dataset


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    p = 20
    d = 5
    T = 48
    obs_rate = 0.5

    data = make_case4_dataset(
        num_sample_train=80,
        num_sample_test=20,
        p=p,
        d=d,
        T=T,
        obs_rate=obs_rate,
        seed=seed,
        mm="mcar",
        version="v1",
    )

    imputer = LIFTSImputer(
        p=p,
        T=T,
        d=d,
        obs_rate=obs_rate,
        total_loops=5,
        batch_size=32,
        n_samples=10,
        lambda_selfmask=5.0,
        lambda_enc=1.0,
        enc_warmup=2,
        seed=seed,
    )
    imputer.fit(data["X_train"], data["mask_train"], verbose=True)

    imp_samples = imputer.impute_samples(data["X_test"], data["mask_test"], n_samples=10)
    print("case4 imp_samples shape:", imp_samples.shape)


if __name__ == "__main__":
    main()
