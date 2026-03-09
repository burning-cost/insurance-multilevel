"""
Shared fixtures for insurance-multilevel tests.

We generate synthetic insurance data with a known group structure so tests
can verify that the model recovers the correct variance components and BLUPs.

The data-generating process:
  - 5 risk features (continuous + categorical)
  - 20 brokers with true random effects drawn from N(0, tau2_true)
  - Individual risk factor: linear in age + vehicle class
  - Noise: lognormal within group

Known parameters:
  SIGMA2_TRUE = 0.25  (within-group variance on log scale)
  TAU2_TRUE   = 0.10  (between-group variance on log scale)
  ICC_TRUE    = 0.10 / (0.10 + 0.25) = 0.286
"""

import numpy as np
import polars as pl
import pytest

RNG_SEED = 42
N_OBS = 2000
N_BROKERS = 20
SIGMA2_TRUE = 0.25
TAU2_TRUE = 0.10


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def synthetic_insurance_data(rng) -> dict:
    """
    Synthetic UK motor insurance dataset with broker random effects.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, weights_train,
                    weights_test, broker_effects (true values),
                    sigma2_true, tau2_true
    """
    n = N_OBS
    nb = N_BROKERS

    # True broker random effects
    broker_effects = rng.normal(0, np.sqrt(TAU2_TRUE), nb)
    broker_ids = rng.integers(0, nb, n)

    # Risk features
    age = rng.integers(18, 80, n).astype(float)
    vehicle_class = rng.choice(["A", "B", "C", "D"], n)
    ncb = rng.integers(0, 6, n).astype(float)  # no-claims bonus years
    region_risk = rng.uniform(0.8, 1.5, n)
    annual_mileage = rng.integers(5000, 30000, n).astype(float)

    # True fixed effect (log scale)
    vehicle_class_effect = {"A": 0.0, "B": 0.15, "C": 0.30, "D": 0.50}
    vc_num = np.array([vehicle_class_effect[v] for v in vehicle_class])
    log_base = (
        5.0  # intercept (~£148 base premium)
        + 0.01 * (age - 40)
        + vc_num
        - 0.05 * ncb
        + 0.3 * np.log(region_risk)
        + 0.0001 * annual_mileage
        + broker_effects[broker_ids]  # true group effect
    )

    # Response: lognormal with within-group noise
    noise = rng.normal(0, np.sqrt(SIGMA2_TRUE), n)
    y = np.exp(log_base + noise)

    # Weights (exposure in years, uniform 0.5-1.5)
    exposure = rng.uniform(0.5, 1.5, n)

    # Train/test split
    split = int(0.8 * n)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_idx, test_idx = idx[:split], idx[split:]

    def make_df(idx_: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({
            "age": age[idx_],
            "vehicle_class": [vehicle_class[i] for i in idx_],
            "ncb": ncb[idx_],
            "region_risk": region_risk[idx_],
            "annual_mileage": annual_mileage[idx_],
            "broker_id": [f"broker_{broker_ids[i]:02d}" for i in idx_],
        })

    return {
        "X_train": make_df(train_idx),
        "X_test": make_df(test_idx),
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "weights_train": exposure[train_idx],
        "weights_test": exposure[test_idx],
        "broker_effects": broker_effects,
        "sigma2_true": SIGMA2_TRUE,
        "tau2_true": TAU2_TRUE,
        "n_brokers": nb,
    }


@pytest.fixture(scope="session")
def small_data(rng) -> dict:
    """
    Tiny dataset for fast unit tests (n=200, 5 brokers).
    """
    n = 200
    nb = 5
    broker_effects = rng.normal(0, 0.2, nb)
    broker_ids = rng.integers(0, nb, n)

    age = rng.integers(18, 70, n).astype(float)
    y = np.exp(5.0 + 0.005 * age + broker_effects[broker_ids] + rng.normal(0, 0.3, n))
    weights = rng.uniform(0.5, 1.5, n)

    X = pl.DataFrame({
        "age": age,
        "broker_id": [f"broker_{broker_ids[i]}" for i in range(n)],
    })
    return {"X": X, "y": y, "weights": weights, "n_brokers": nb}
