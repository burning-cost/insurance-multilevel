"""
Tests for MultilevelPricingModel.

Integration tests that verify the full two-stage pipeline:
1. fit() without errors on realistic data
2. predict() returns positive values, correct shape
3. credibility_summary() returns well-formed DataFrame
4. Stage 1 vs final predictions differ (random effects are contributing)
5. New groups handled gracefully
6. Error handling: unfitted model, zero/negative y, no features left
"""

import numpy as np
import polars as pl
import pytest
from insurance_multilevel import MultilevelPricingModel, VarianceComponents


def test_fit_and_predict_basic(synthetic_insurance_data):
    """Full pipeline runs without error on synthetic data."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 50, "verbose": 0},
        random_effects=["broker_id"],
        min_group_size=5,
    )
    model.fit(d["X_train"], d["y_train"], weights=d["weights_train"])
    preds = model.predict(d["X_test"], group_cols=["broker_id"])

    assert preds.shape == (len(d["X_test"]),)
    assert np.all(preds > 0)


def test_predict_positive(synthetic_insurance_data):
    """Predictions should always be positive (premium prices)."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])
    preds = model.predict(d["X_test"])

    assert np.all(preds > 0), "Found non-positive premium predictions"


def test_variance_components_stored(synthetic_insurance_data):
    """variance_components property returns dict with VarianceComponents."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])
    vc_dict = model.variance_components

    assert "broker_id" in vc_dict
    vc = vc_dict["broker_id"]
    assert isinstance(vc, VarianceComponents)
    assert vc.sigma2 > 0
    assert vc.tau2["broker_id"] >= 0


def test_credibility_summary_schema(synthetic_insurance_data):
    """credibility_summary() should return correct columns."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])
    summary = model.credibility_summary()

    expected_cols = {
        "level", "group", "exposure_sum", "n_obs", "group_mean", "blup", "multiplier",
        "credibility_weight", "tau2", "sigma2", "k", "eligible",
    }
    assert set(summary.columns) == expected_cols


def test_credibility_summary_n_rows(synthetic_insurance_data):
    """Summary should have one row per unique broker."""
    d = synthetic_insurance_data
    n_unique_brokers = d["X_train"]["broker_id"].n_unique()

    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])
    summary = model.credibility_summary()

    assert len(summary) == n_unique_brokers


def test_credibility_weight_in_01(synthetic_insurance_data):
    """Credibility weights Z should be in [0, 1]."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])
    summary = model.credibility_summary()

    z = summary["credibility_weight"].to_numpy()
    assert np.all(z >= -1e-9)
    assert np.all(z <= 1.0 + 1e-9)


def test_multiplier_is_exp_blup(synthetic_insurance_data):
    """multiplier column should equal exp(blup) to float precision."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])
    summary = model.credibility_summary()

    blup = summary["blup"].to_numpy()
    mult = summary["multiplier"].to_numpy()
    np.testing.assert_allclose(mult, np.exp(blup), rtol=1e-5)


def test_stage1_vs_final_predictions_differ(synthetic_insurance_data):
    """Final predictions should differ from Stage 1 when tau2 > 0."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 50, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])

    stage1 = model.stage1_predict(d["X_test"])
    final = model.predict(d["X_test"])

    # Some predictions should differ (random effects applied)
    assert not np.allclose(stage1, final), (
        "Stage 1 and final predictions are identical — random effects have no effect"
    )


def test_new_group_gets_unit_multiplier(synthetic_insurance_data):
    """Unseen groups at prediction time should get multiplier of 1.0."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])

    X_new = d["X_test"].head(5).with_columns(
        pl.lit("brand_new_broker_99").alias("broker_id")
    )
    preds_new = model.predict(X_new, allow_new_groups=True)
    stage1_new = model.stage1_predict(X_new)

    # With unseen group, predict == stage1 (exp(0) = 1.0 multiplier)
    np.testing.assert_allclose(preds_new, stage1_new, rtol=1e-5)


def test_new_group_raises_when_not_allowed(synthetic_insurance_data):
    """allow_new_groups=False should raise for unseen groups."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])

    X_new = d["X_test"].head(5).with_columns(
        pl.lit("ghost_broker").alias("broker_id")
    )
    with pytest.raises(KeyError, match="not seen during training"):
        model.predict(X_new, allow_new_groups=False)


def test_fit_negative_y_raises(small_data):
    """Negative response should raise ValueError."""
    d = small_data
    y_bad = d["y"].copy()
    y_bad[0] = -5.0

    model = MultilevelPricingModel(
        catboost_params={"iterations": 20, "verbose": 0},
        random_effects=["broker_id"],
    )
    with pytest.raises(ValueError, match="strictly positive"):
        model.fit(d["X"], y_bad, group_cols=["broker_id"])


def test_fit_zero_y_raises(small_data):
    """Zero response should raise ValueError."""
    d = small_data
    y_bad = d["y"].copy()
    y_bad[5] = 0.0

    model = MultilevelPricingModel(
        catboost_params={"iterations": 20, "verbose": 0},
        random_effects=["broker_id"],
    )
    with pytest.raises(ValueError, match="strictly positive"):
        model.fit(d["X"], y_bad, group_cols=["broker_id"])


def test_predict_before_fit_raises(small_data):
    """Calling predict before fit should raise RuntimeError."""
    d = small_data
    model = MultilevelPricingModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(d["X"])


def test_variance_components_before_fit_raises(small_data):
    """Accessing variance_components before fit should raise."""
    model = MultilevelPricingModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = model.variance_components


def test_no_group_cols_pure_catboost(small_data):
    """With no group cols, model should run as pure CatBoost."""
    d = small_data
    # Use only age column, no broker_id
    X_no_group = d["X"].select(["age"])

    model = MultilevelPricingModel(
        catboost_params={"iterations": 20, "verbose": 0},
        random_effects=[],
    )
    model.fit(X_no_group, d["y"], group_cols=[])
    preds = model.predict(X_no_group, group_cols=[])

    assert preds.shape == (len(d["X"]),)
    assert np.all(preds > 0)


def test_log_ratio_residuals(small_data):
    """log_ratio_residuals should return array of same length as input."""
    d = small_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 20, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X"], d["y"], group_cols=["broker_id"])
    residuals = model.log_ratio_residuals(d["X"], d["y"])

    assert residuals.shape == (len(d["X"]),)
    # Residuals should be finite
    assert np.all(np.isfinite(residuals))


def test_feature_importances(small_data):
    """feature_importances should return dict with one entry per Stage 1 feature."""
    d = small_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 20, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X"], d["y"], group_cols=["broker_id"])
    fi = model.feature_importances

    # Only non-group features
    assert "age" in fi
    assert "broker_id" not in fi


def test_catboost_model_property(small_data):
    """catboost_model property should return fitted CatBoostRegressor."""
    from catboost import CatBoostRegressor

    d = small_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 20, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X"], d["y"], group_cols=["broker_id"])

    cb = model.catboost_model
    assert isinstance(cb, CatBoostRegressor)


def test_weighted_fit_runs(synthetic_insurance_data):
    """Weighted fitting should run without error."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"], weights=d["weights_train"])
    preds = model.predict(d["X_test"])

    assert np.all(preds > 0)


def test_credibility_summary_group_filter(synthetic_insurance_data):
    """credibility_summary(group_col=...) should filter to that level."""
    d = synthetic_insurance_data
    model = MultilevelPricingModel(
        catboost_params={"iterations": 30, "verbose": 0},
        random_effects=["broker_id"],
    )
    model.fit(d["X_train"], d["y_train"])

    summary_all = model.credibility_summary()
    summary_broker = model.credibility_summary("broker_id")

    assert len(summary_broker) == len(summary_all.filter(pl.col("level") == "broker_id"))
