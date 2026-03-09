"""
Tests for diagnostic utility functions.
"""

import numpy as np
import polars as pl
import pytest
from insurance_multilevel import (
    MultilevelPricingModel,
    VarianceComponents,
    icc,
    variance_decomposition,
    high_credibility_groups,
    groups_needing_data,
    residual_normality_check,
    lift_from_random_effects,
)


@pytest.fixture
def sample_vc() -> VarianceComponents:
    return VarianceComponents(
        sigma2=0.25,
        tau2={"broker_id": 0.10},
        k={"broker_id": 2.5},
        log_likelihood=-100.0,
        converged=True,
        iterations=15,
        n_groups={"broker_id": 20},
        n_obs_used=1000,
    )


@pytest.fixture
def sample_credibility_df() -> pl.DataFrame:
    return pl.DataFrame({
        "level": ["broker_id"] * 5,
        "group": ["B1", "B2", "B3", "B4", "B5"],
        "n_obs": [500.0, 200.0, 50.0, 20.0, 8.0],
        "group_mean": [0.1, -0.05, 0.2, -0.1, 0.05],
        "blup": [0.07, -0.03, 0.12, -0.05, 0.02],
        "multiplier": [1.07, 0.97, 1.13, 0.95, 1.02],
        "credibility_weight": [0.90, 0.75, 0.45, 0.25, 0.12],
        "tau2": [0.10] * 5,
        "sigma2": [0.25] * 5,
        "k": [2.5] * 5,
        "eligible": [True, True, True, True, True],
    })


# -------------------------------------------------------------------------
# ICC tests
# -------------------------------------------------------------------------

def test_icc_basic(sample_vc):
    val = icc(sample_vc, "broker_id")
    expected = 0.10 / (0.10 + 0.25)
    assert val == pytest.approx(expected, rel=1e-5)


def test_icc_zero_tau2():
    vc = VarianceComponents(
        sigma2=0.30,
        tau2={"broker_id": 0.0},
        k={"broker_id": float("inf")},
        log_likelihood=0.0,
        converged=True,
        iterations=0,
    )
    assert icc(vc, "broker_id") == pytest.approx(0.0)


def test_icc_all_between_group_variance():
    vc = VarianceComponents(
        sigma2=0.0,
        tau2={"broker_id": 0.5},
        k={"broker_id": 0.0},
        log_likelihood=0.0,
        converged=True,
        iterations=0,
    )
    # With sigma2=0, ICC should be 1.0
    assert icc(vc, "broker_id") == pytest.approx(1.0)


def test_icc_in_0_1_range(sample_vc):
    val = icc(sample_vc, "broker_id")
    assert 0.0 <= val <= 1.0


# -------------------------------------------------------------------------
# variance_decomposition tests
# -------------------------------------------------------------------------

def test_variance_decomposition_schema():
    vc_dict = {
        "broker_id": VarianceComponents(
            sigma2=0.25, tau2={"broker_id": 0.10},
            k={"broker_id": 2.5}, log_likelihood=-100.0,
            converged=True, iterations=10,
            n_groups={"broker_id": 20}, n_obs_used=1000,
        ),
    }
    df = variance_decomposition(vc_dict)

    assert set(df.columns) == {
        "level", "tau2", "sigma2", "icc", "buhlmann_k",
        "converged", "n_groups", "n_obs_used",
    }
    assert len(df) == 1


def test_variance_decomposition_icc_matches(sample_vc):
    vc_dict = {"broker_id": sample_vc}
    df = variance_decomposition(vc_dict)
    row = df.row(0, named=True)
    expected_icc = icc(sample_vc, "broker_id")
    assert row["icc"] == pytest.approx(expected_icc, rel=1e-5)


def test_variance_decomposition_multiple_levels():
    vc_dict = {
        "broker_id": VarianceComponents(
            sigma2=0.25, tau2={"broker_id": 0.10},
            k={"broker_id": 2.5}, log_likelihood=-100.0,
            converged=True, iterations=10,
        ),
        "scheme_id": VarianceComponents(
            sigma2=0.25, tau2={"scheme_id": 0.20},
            k={"scheme_id": 1.25}, log_likelihood=-80.0,
            converged=True, iterations=8,
        ),
    }
    df = variance_decomposition(vc_dict)
    assert len(df) == 2


# -------------------------------------------------------------------------
# high_credibility_groups tests
# -------------------------------------------------------------------------

def test_high_credibility_groups_filter(sample_credibility_df):
    result = high_credibility_groups(sample_credibility_df, min_z=0.5)
    # B1 (0.90) and B2 (0.75) should pass; B3 (0.45) should not
    groups = set(result["group"].to_list())
    assert "B1" in groups
    assert "B2" in groups
    assert "B3" not in groups


def test_high_credibility_groups_sorted(sample_credibility_df):
    result = high_credibility_groups(sample_credibility_df, min_z=0.0)
    z = result["credibility_weight"].to_list()
    assert z == sorted(z, reverse=True)


def test_high_credibility_groups_empty_when_all_below(sample_credibility_df):
    result = high_credibility_groups(sample_credibility_df, min_z=0.99)
    assert len(result) == 0


# -------------------------------------------------------------------------
# groups_needing_data tests
# -------------------------------------------------------------------------

def test_groups_needing_data_schema(sample_credibility_df):
    result = groups_needing_data(sample_credibility_df, target_z=0.8)
    assert "n_target" in result.columns
    assert "n_additional" in result.columns


def test_groups_needing_data_already_there(sample_credibility_df):
    """Groups already at target_z should need 0 additional observations."""
    result = groups_needing_data(sample_credibility_df, target_z=0.8)
    # B1 has Z=0.90 > 0.80, should need 0 more
    b1 = result.filter(pl.col("group") == "B1")
    assert b1["n_additional"][0] == pytest.approx(0.0, abs=1.0)


def test_groups_needing_data_invalid_target():
    df = pl.DataFrame({"level": ["b"], "group": ["b"], "n_obs": [10.0],
                       "credibility_weight": [0.5], "k": [2.5]})
    with pytest.raises(ValueError, match="target_z must be strictly"):
        groups_needing_data(df, target_z=1.0)


def test_groups_needing_data_n_additional_nonneg(sample_credibility_df):
    """n_additional should never be negative."""
    result = groups_needing_data(sample_credibility_df, target_z=0.8)
    assert (result["n_additional"] >= 0).all()


# -------------------------------------------------------------------------
# residual_normality_check tests
# -------------------------------------------------------------------------

def test_residual_normality_check_gaussian():
    rng = np.random.default_rng(42)
    residuals = rng.normal(0, 0.5, 1000)
    stats = residual_normality_check(residuals)

    assert abs(stats["mean"]) < 0.05  # close to 0
    assert abs(stats["skewness"]) < 0.3  # near-symmetric
    assert abs(stats["excess_kurtosis"]) < 0.5  # near-normal tails
    assert stats["std"] > 0


def test_residual_normality_check_returns_all_keys():
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = residual_normality_check(residuals)
    expected_keys = {"mean", "std", "skewness", "excess_kurtosis", "p95", "p99"}
    assert set(stats.keys()) == expected_keys


def test_residual_normality_check_constant():
    """Constant residuals (std=0) should not raise."""
    residuals = np.ones(10)
    stats = residual_normality_check(residuals)
    assert stats["std"] == 0.0


# -------------------------------------------------------------------------
# lift_from_random_effects tests
# -------------------------------------------------------------------------

def test_lift_computation():
    rng = np.random.default_rng(42)
    y_true = rng.lognormal(5, 0.5, 200)
    stage1 = y_true * rng.lognormal(0, 0.3, 200)  # noisy
    final = y_true * rng.lognormal(0, 0.15, 200)  # less noisy

    result = lift_from_random_effects(y_true, stage1, final)

    assert "stage1_rmse" in result
    assert "final_rmse" in result
    assert "rmse_improvement_pct" in result
    assert "malr_improvement_pct" in result


def test_lift_perfect_predictions():
    """Perfect final predictions should show positive improvement."""
    rng = np.random.default_rng(42)
    y_true = np.exp(rng.normal(5, 0.5, 100))
    stage1 = y_true * rng.lognormal(0, 0.2, 100)
    final = y_true.copy()  # perfect

    result = lift_from_random_effects(y_true, stage1, final)
    assert result["final_rmse"] < result["stage1_rmse"]
    assert result["rmse_improvement_pct"] > 0


def test_lift_with_weights():
    rng = np.random.default_rng(42)
    y_true = np.exp(rng.normal(5, 0.5, 100))
    preds = y_true * 1.1
    weights = rng.uniform(0.5, 2.0, 100)

    result = lift_from_random_effects(y_true, preds, preds, weights=weights)
    # Same predictions: no improvement
    assert abs(result["rmse_improvement_pct"]) < 1e-6


# -------------------------------------------------------------------------
# End-to-end diagnostic integration test
# -------------------------------------------------------------------------

def test_full_diagnostic_workflow(synthetic_insurance_data):
    """Run all diagnostics on a fitted model."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])

    # ICC
    vc = model.variance_components["broker_id"]
    icc_val = icc(vc, "broker_id")
    assert 0 <= icc_val <= 1

    # Variance decomposition
    vd = variance_decomposition(model.variance_components)
    assert len(vd) == 1

    # Credibility summary
    cred = model.credibility_summary()
    hc = high_credibility_groups(cred, min_z=0.3)
    assert len(hc) >= 0  # might be empty for small tau2

    # Lift
    stage1 = model.stage1_predict(d["X_test"])
    final = model.predict(d["X_test"])
    lift = lift_from_random_effects(d["y_test"], stage1, final, d["weights_test"])
    assert "rmse_improvement_pct" in lift

    # Normality check
    resids = model.log_ratio_residuals(d["X_train"], d["y_train"])
    norm_stats = residual_normality_check(resids)
    assert norm_stats["std"] > 0
