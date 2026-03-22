"""
Regression tests for P0/P1 bug fixes in v0.1.3 (batch 3 API audit).

B7: groups_needing_data() — n_obs is sum of exposure weights, not obs count.
    Fix: rename to exposure_sum, add actual n_obs observation count.
B8: MultilevelPricingModel.fit() — add docstring explaining positive y requirement.
    The ValueError for y<=0 was already in place; this test verifies it raises.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_multilevel import MultilevelPricingModel, groups_needing_data


RNG = np.random.default_rng(2025)


# ---------------------------------------------------------------------------
# B7 — groups_needing_data column names
# ---------------------------------------------------------------------------

def make_credibility_df(n_groups: int = 5, seed: int = 42) -> pl.DataFrame:
    """Make a minimal credibility summary DataFrame matching the new schema."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "level": ["broker_id"] * n_groups,
        "group": [f"B{i}" for i in range(n_groups)],
        "exposure_sum": rng.uniform(5.0, 500.0, size=n_groups).tolist(),
        "n_obs": rng.integers(5, 500, size=n_groups).tolist(),
        "group_mean": rng.normal(0, 0.1, size=n_groups).tolist(),
        "blup": rng.normal(0, 0.05, size=n_groups).tolist(),
        "multiplier": [1.0] * n_groups,
        "credibility_weight": rng.uniform(0.1, 0.95, size=n_groups).tolist(),
        "tau2": [0.10] * n_groups,
        "sigma2": [0.25] * n_groups,
        "k": [2.5] * n_groups,
        "eligible": [True] * n_groups,
    })


class TestGroupsNeedingDataColumns:
    """groups_needing_data must expose exposure_sum and n_obs as separate columns."""

    def test_output_has_exposure_sum_column(self):
        cred_df = make_credibility_df()
        result = groups_needing_data(cred_df, target_z=0.8)
        assert "exposure_sum" in result.columns, (
            "groups_needing_data must include 'exposure_sum' column (sum of weights)"
        )

    def test_output_has_n_obs_column(self):
        cred_df = make_credibility_df()
        result = groups_needing_data(cred_df, target_z=0.8)
        assert "n_obs" in result.columns, (
            "groups_needing_data must include 'n_obs' column (actual observation count)"
        )

    def test_n_obs_is_integer_dtype(self):
        cred_df = make_credibility_df()
        result = groups_needing_data(cred_df, target_z=0.8)
        assert result["n_obs"].dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64), (
            "n_obs should be an integer dtype (observation count)"
        )

    def test_exposure_sum_is_float_dtype(self):
        cred_df = make_credibility_df()
        result = groups_needing_data(cred_df, target_z=0.8)
        assert result["exposure_sum"].dtype == pl.Float64, (
            "exposure_sum should be Float64 (sum of exposure weights)"
        )

    def test_n_additional_non_negative(self):
        cred_df = make_credibility_df()
        result = groups_needing_data(cred_df, target_z=0.8)
        assert (result["n_additional"] >= 0).all()

    def test_output_has_required_columns(self):
        cred_df = make_credibility_df()
        result = groups_needing_data(cred_df, target_z=0.8)
        required = {"level", "group", "n_obs", "exposure_sum", "credibility_weight",
                    "n_target", "n_additional"}
        assert required.issubset(set(result.columns))


class TestCredibilitySummaryColumns:
    """credibility_summary() must return both exposure_sum and n_obs."""

    def _make_data(self, n: int = 300, n_groups: int = 8, seed: int = 99) -> tuple:
        rng = np.random.default_rng(seed)
        group_labels = [f"B{i}" for i in range(n_groups)]
        groups = rng.choice(group_labels, size=n)
        y = np.exp(rng.normal(0, 0.3, size=n) + rng.choice([0.1, -0.1, 0.2], size=n))
        y = np.abs(y) + 0.5  # ensure positive
        feature = rng.standard_normal(n)
        exposure = rng.uniform(0.5, 2.0, size=n)
        X = pl.DataFrame({
            "feature": feature.tolist(),
            "broker_id": groups.tolist(),
        })
        return X, y, exposure

    def test_credibility_summary_has_exposure_sum(self):
        X, y, exposure = self._make_data()
        model = MultilevelPricingModel(
            catboost_params={"iterations": 20},
            random_effects=["broker_id"],
            min_group_size=2,
        )
        model.fit(X, y, weights=exposure, group_cols=["broker_id"])
        summary = model.credibility_summary()
        assert "exposure_sum" in summary.columns, (
            "credibility_summary() must contain 'exposure_sum' column"
        )

    def test_credibility_summary_has_n_obs(self):
        X, y, exposure = self._make_data()
        model = MultilevelPricingModel(
            catboost_params={"iterations": 20},
            random_effects=["broker_id"],
            min_group_size=2,
        )
        model.fit(X, y, weights=exposure, group_cols=["broker_id"])
        summary = model.credibility_summary()
        assert "n_obs" in summary.columns, (
            "credibility_summary() must contain 'n_obs' column (observation count)"
        )

    def test_n_obs_le_total_observations(self):
        X, y, exposure = self._make_data(n=300)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 20},
            random_effects=["broker_id"],
            min_group_size=2,
        )
        model.fit(X, y, weights=exposure, group_cols=["broker_id"])
        summary = model.credibility_summary()
        # Each group's n_obs must be positive and sum <= total n
        assert (summary["n_obs"] > 0).all()
        assert summary["n_obs"].sum() <= 300


# ---------------------------------------------------------------------------
# B8 — MultilevelPricingModel.fit() raises ValueError for non-positive y
# ---------------------------------------------------------------------------

class TestFitPositiveY:
    """fit() must raise ValueError for y <= 0 with a clear message."""

    def _make_data(self, n: int = 50) -> tuple:
        rng = np.random.default_rng(1)
        X = pl.DataFrame({
            "feature": rng.standard_normal(n).tolist(),
            "group": rng.choice(["A", "B"], size=n).tolist(),
        })
        y = rng.uniform(0.5, 3.0, size=n)
        return X, y

    def test_y_with_zeros_raises(self):
        X, y = self._make_data()
        y[0] = 0.0  # inject a zero
        model = MultilevelPricingModel(catboost_params={"iterations": 5})
        with pytest.raises(ValueError, match="positive"):
            model.fit(X, y)

    def test_y_with_negatives_raises(self):
        X, y = self._make_data()
        y[5] = -1.0
        model = MultilevelPricingModel(catboost_params={"iterations": 5})
        with pytest.raises(ValueError, match="positive"):
            model.fit(X, y)

    def test_positive_y_does_not_raise(self):
        X, y = self._make_data()
        model = MultilevelPricingModel(catboost_params={"iterations": 5})
        model.fit(X, y)  # should not raise
        assert model._fitted
