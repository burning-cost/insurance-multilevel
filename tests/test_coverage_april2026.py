"""
Expanded test coverage sweep — April 2026.

Covers untested or under-tested code paths across all four modules:
  _types.py, _reml.py, _model.py, _diagnostics.py

Focus areas:
- Edge cases: empty inputs, single-element arrays, boundary values
- Validation: error messages, type checks, parameter constraints
- Utility functions: _to_numpy, _find_cat_feature_indices, _henderson_mom_init
- Properties accessed before fit (should raise RuntimeError)
- Credibility summary caching and multi-level filtering
- VarianceComponents repr with unusual values
- Normality check edge cases
- lift_from_random_effects numerical correctness
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

from insurance_multilevel import (
    MultilevelPricingModel,
    RandomEffectsEstimator,
    VarianceComponents,
    icc,
    variance_decomposition,
    high_credibility_groups,
    groups_needing_data,
    residual_normality_check,
    lift_from_random_effects,
)
from insurance_multilevel._reml import (
    _henderson_mom_init,
    _reml_neg_log_likelihood,
)
from insurance_multilevel._model import _to_numpy, _find_cat_feature_indices


# ---------------------------------------------------------------------------
# VarianceComponents — additional repr and edge-case tests
# ---------------------------------------------------------------------------


class TestVarianceComponentsRepr:
    def test_repr_multiple_tau2_keys(self):
        vc = VarianceComponents(
            sigma2=0.20,
            tau2={"broker_id": 0.08, "scheme_id": 0.15},
            k={"broker_id": 2.5, "scheme_id": 1.33},
            log_likelihood=-200.0,
            converged=True,
            iterations=20,
        )
        r = repr(vc)
        assert "broker_id" in r
        assert "scheme_id" in r
        assert "sigma2=0.2000" in r

    def test_repr_zero_iterations(self):
        vc = VarianceComponents(
            sigma2=0.1,
            tau2={"g": 0.0},
            k={"g": float("inf")},
            log_likelihood=0.0,
            converged=False,
            iterations=0,
        )
        r = repr(vc)
        assert "converged=False" in r
        assert "iterations=0" in r

    def test_repr_negative_log_likelihood(self):
        vc = VarianceComponents(
            sigma2=0.5,
            tau2={"g": 0.1},
            k={"g": 5.0},
            log_likelihood=-1234.5678,
            converged=True,
            iterations=7,
        )
        r = repr(vc)
        assert "log_likelihood=-1234.5678" in r

    def test_repr_large_k(self):
        vc = VarianceComponents(
            sigma2=0.001,
            tau2={"g": 0.0001},
            k={"g": 10.0},
            log_likelihood=-10.0,
            converged=True,
            iterations=5,
        )
        r = repr(vc)
        assert "k={g=10.0}" in r

    def test_variance_components_immutable_defaults(self):
        """Two separate instances must not share the same default dict."""
        vc1 = VarianceComponents(
            sigma2=0.1, tau2={}, k={}, log_likelihood=0.0,
            converged=False, iterations=0,
        )
        vc2 = VarianceComponents(
            sigma2=0.2, tau2={}, k={}, log_likelihood=0.0,
            converged=False, iterations=0,
        )
        vc1.n_groups["x"] = 5
        assert "x" not in vc2.n_groups, "default dict factory should not share state"


# ---------------------------------------------------------------------------
# _to_numpy utility
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_numpy_array_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_numpy(arr, "arr")
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float64

    def test_polars_series_conversion(self):
        s = pl.Series("x", [1.0, 2.0, 3.0])
        result = _to_numpy(s, "x")
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_integer_array_converted_to_float64(self):
        arr = np.array([1, 2, 3], dtype=int)
        result = _to_numpy(arr, "y")
        assert result.dtype == np.float64

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be None"):
            _to_numpy(None, "weights")

    def test_list_converted_via_asarray(self):
        """Lists are handled via np.asarray fallback."""
        lst = [1.0, 2.0, 3.0]
        result = _to_numpy(lst, "lst")
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_polars_integer_series(self):
        s = pl.Series("y", [10, 20, 30])
        result = _to_numpy(s, "y")
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# _find_cat_feature_indices utility
# ---------------------------------------------------------------------------


class TestFindCatFeatureIndices:
    def test_all_numeric_returns_empty(self):
        X = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        assert _find_cat_feature_indices(X) == []

    def test_string_column_detected(self):
        X = pl.DataFrame({"age": [30, 40], "vehicle": ["A", "B"]})
        indices = _find_cat_feature_indices(X)
        assert 1 in indices  # "vehicle" is at index 1

    def test_categorical_column_detected(self):
        X = pl.DataFrame({"cat": pl.Series(["x", "y"]).cast(pl.Categorical)})
        indices = _find_cat_feature_indices(X)
        assert 0 in indices

    def test_mixed_columns(self):
        X = pl.DataFrame({
            "age": [30.0, 40.0],
            "vehicle": ["A", "B"],
            "ncb": [1.0, 2.0],
            "region": ["N", "S"],
        })
        indices = _find_cat_feature_indices(X)
        # vehicle at index 1, region at index 3
        assert set(indices) == {1, 3}

    def test_empty_dataframe_returns_empty(self):
        X = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
        assert _find_cat_feature_indices(X) == []


# ---------------------------------------------------------------------------
# _henderson_mom_init edge cases
# ---------------------------------------------------------------------------


class TestHendersonMomInit:
    def test_single_group_does_not_crash(self):
        """With one group, m-1=0; the function should handle this."""
        residuals = np.array([0.1, -0.2, 0.3, 0.0, 0.1])
        group_ids = np.array(["g0"] * 5)
        weights = np.ones(5)
        # Should not raise, and sigma2 should be > 0
        sigma2, tau2 = _henderson_mom_init(residuals, group_ids, weights)
        assert sigma2 > 0
        assert tau2 >= 0

    def test_constant_residuals_within_group(self):
        """All residuals identical: within SS=0 -> sigma2 should be tiny."""
        residuals = np.zeros(20)
        group_ids = np.repeat(["A", "B"], 10)
        weights = np.ones(20)
        sigma2, tau2 = _henderson_mom_init(residuals, group_ids, weights)
        assert sigma2 >= 1e-6  # clamped minimum

    def test_large_between_group_variation(self):
        """When group means differ a lot, tau2 should be positive."""
        rng = np.random.default_rng(1)
        n_groups = 5
        # Group means: 0, 1, 2, 3, 4 — large between-group spread
        residuals = np.concatenate([
            rng.normal(i, 0.05, 20) for i in range(n_groups)
        ])
        group_ids = np.repeat(np.arange(n_groups).astype(str), 20)
        weights = np.ones(len(residuals))
        sigma2, tau2 = _henderson_mom_init(residuals, group_ids, weights)
        assert tau2 > 0, "Large between-group variation should give tau2 > 0"

    def test_none_weights_treated_as_uniform(self):
        """weights=None should give same result as uniform weights."""
        rng = np.random.default_rng(5)
        residuals = rng.normal(0, 0.5, 40)
        group_ids = np.repeat(["A", "B", "C", "D"], 10)
        s2_none, t2_none = _henderson_mom_init(residuals, group_ids, None)
        s2_ones, t2_ones = _henderson_mom_init(residuals, group_ids, np.ones(40))
        assert s2_none == pytest.approx(s2_ones, rel=1e-9)
        assert t2_none == pytest.approx(t2_ones, rel=1e-9)

    def test_two_groups_equal_size(self):
        """Two groups, balanced: n_bar = common group size."""
        rng = np.random.default_rng(3)
        residuals = np.concatenate([rng.normal(0, 0.3, 30), rng.normal(0.5, 0.3, 30)])
        group_ids = np.repeat(["A", "B"], 30)
        sigma2, tau2 = _henderson_mom_init(residuals, group_ids, None)
        assert sigma2 > 0
        assert tau2 >= 0


# ---------------------------------------------------------------------------
# _reml_neg_log_likelihood edge cases
# ---------------------------------------------------------------------------


class TestRemlNegLogLikelihood:
    def setup_method(self):
        rng = np.random.default_rng(99)
        self.n_groups = 4
        self.group_ids = np.repeat(np.arange(self.n_groups).astype(str), 15)
        self.residuals = rng.normal(0, 0.5, len(self.group_ids))
        self.weights = np.ones(len(self.group_ids))
        self.groups = np.unique(self.group_ids)

    def test_near_zero_sigma2_returns_large_value(self):
        """Very small sigma2 (via log_sigma2 << 0) should not crash."""
        params = np.array([-50.0, np.log(0.1)])  # sigma2 ~ exp(-50) ≈ 0
        val = _reml_neg_log_likelihood(
            params, self.residuals, self.group_ids, self.weights, self.groups, reml=True
        )
        assert val == 1e15

    def test_reml_vs_ml_flag_changes_value(self):
        """Same params but different reml flag must produce different values."""
        params = np.array([np.log(0.25), np.log(0.10)])
        ll_reml = _reml_neg_log_likelihood(
            params, self.residuals, self.group_ids, self.weights, self.groups, reml=True
        )
        ll_ml = _reml_neg_log_likelihood(
            params, self.residuals, self.group_ids, self.weights, self.groups, reml=False
        )
        assert ll_reml != ll_ml

    def test_output_is_finite_for_valid_params(self):
        params = np.array([np.log(0.30), np.log(0.05)])
        val = _reml_neg_log_likelihood(
            params, self.residuals, self.group_ids, self.weights, self.groups, reml=True
        )
        assert np.isfinite(val)

    def test_large_tau2_does_not_crash(self):
        """Very large tau2 should still return a finite value."""
        params = np.array([np.log(0.25), np.log(100.0)])
        val = _reml_neg_log_likelihood(
            params, self.residuals, self.group_ids, self.weights, self.groups, reml=True
        )
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# RandomEffectsEstimator — additional property and edge-case tests
# ---------------------------------------------------------------------------


class TestRandomEffectsEstimatorProperties:
    def test_blup_map_before_fit_returns_empty(self):
        """blup_map property returns empty dict before fit."""
        est = RandomEffectsEstimator()
        assert est.blup_map == {}

    def test_variance_components_before_fit_is_none(self):
        """variance_components property is None before fit."""
        est = RandomEffectsEstimator()
        assert est.variance_components is None

    def test_group_stats_before_fit_is_empty(self):
        """group_stats property is empty dict before fit."""
        est = RandomEffectsEstimator()
        assert est.group_stats == {}

    def test_blup_map_after_fit(self):
        rng = np.random.default_rng(10)
        residuals = rng.normal(0, 0.5, 100)
        group_ids = np.repeat(np.arange(5).astype(str), 20)
        est = RandomEffectsEstimator(min_group_size=5)
        est.fit(residuals, group_ids)
        blup_map = est.blup_map
        assert len(blup_map) == 5
        assert all(isinstance(v, float) for v in blup_map.values())

    def test_group_stats_keys(self):
        rng = np.random.default_rng(11)
        residuals = rng.normal(0, 0.5, 100)
        group_ids = np.repeat(np.arange(5).astype(str), 20)
        est = RandomEffectsEstimator(min_group_size=5)
        est.fit(residuals, group_ids)
        for gid, stats in est.group_stats.items():
            assert "n" in stats
            assert "mean" in stats
            assert "Z" in stats
            assert "n_obs" in stats

    def test_variance_components_after_fit(self):
        rng = np.random.default_rng(12)
        residuals = rng.normal(0, 0.5, 150)
        group_ids = np.repeat(np.arange(5).astype(str), 30)
        est = RandomEffectsEstimator(min_group_size=5)
        est.fit(residuals, group_ids)
        vc = est.variance_components
        assert vc is not None
        assert isinstance(vc, VarianceComponents)

    def test_fit_refits_cleanly(self):
        """Calling fit() twice should reset state and not accumulate."""
        rng = np.random.default_rng(13)
        residuals = rng.normal(0, 0.5, 100)
        group_ids = np.repeat(np.arange(5).astype(str), 20)
        est = RandomEffectsEstimator(min_group_size=5)
        vc1 = est.fit(residuals, group_ids, group_col="g1")
        vc2 = est.fit(residuals, group_ids, group_col="g2")
        # Only the last fit's results should remain
        assert "g2" in vc2.tau2
        assert "g1" not in vc2.tau2

    def test_string_group_ids_preserved(self):
        """String group IDs like 'broker_01' should be keys in blup_map."""
        rng = np.random.default_rng(14)
        names = ["alpha", "beta", "gamma", "delta", "epsilon"]
        residuals = rng.normal(0, 0.5, 100)
        group_ids = np.repeat(names, 20)
        est = RandomEffectsEstimator(min_group_size=5)
        est.fit(residuals, group_ids)
        for name in names:
            assert name in est.blup_map, f"Group '{name}' missing from blup_map"

    def test_predict_blup_all_training_groups(self):
        """predict_blup returns correct BLUPs for known training groups."""
        rng = np.random.default_rng(15)
        residuals = rng.normal(0, 0.5, 100)
        group_ids = np.repeat(["g0", "g1", "g2", "g3", "g4"], 20)
        est = RandomEffectsEstimator(min_group_size=5)
        est.fit(residuals, group_ids)
        blups = est.predict_blup(group_ids)
        assert len(blups) == len(group_ids)
        assert np.all(np.isfinite(blups))

    def test_predict_blup_mixed_known_and_new(self):
        """Mixed known/unknown groups should return 0 for unknown when allowed."""
        rng = np.random.default_rng(16)
        residuals = rng.normal(0, 0.5, 100)
        train_ids = np.repeat(["g0", "g1", "g2", "g3", "g4"], 20)
        est = RandomEffectsEstimator(min_group_size=5)
        est.fit(residuals, train_ids)
        pred_ids = np.array(["g0", "g1", "new_group"])
        blups = est.predict_blup(pred_ids, allow_new_groups=True)
        assert blups[2] == 0.0, "New group should have BLUP=0 with allow_new_groups=True"
        assert np.isfinite(blups[0])
        assert np.isfinite(blups[1])

    def test_min_group_size_1_all_groups_eligible(self):
        """With min_group_size=1, even singleton groups should be included."""
        rng = np.random.default_rng(17)
        residuals = np.concatenate([
            rng.normal(0, 0.3, 10),
            np.array([0.5]),  # single obs
        ])
        group_ids = np.concatenate([
            np.repeat(["big"], 10),
            np.array(["tiny"]),
        ])
        est = RandomEffectsEstimator(min_group_size=1)
        # Should not warn (both groups eligible)
        vc = est.fit(residuals, group_ids)
        # Both groups should have stats
        assert "big" in est.group_stats
        assert "tiny" in est.group_stats

    def test_ml_mode_estimator(self):
        """RandomEffectsEstimator with reml=False should run and return VarianceComponents."""
        rng = np.random.default_rng(18)
        residuals = rng.normal(0, 0.5, 100)
        group_ids = np.repeat(np.arange(5).astype(str), 20)
        est = RandomEffectsEstimator(reml=False, min_group_size=5)
        vc = est.fit(residuals, group_ids)
        assert isinstance(vc, VarianceComponents)
        assert vc.sigma2 > 0

    def test_tau2_zero_boundary_means_all_blups_zero(self):
        """When tau2=0 (boundary singularity), all BLUPs should be 0."""
        rng = np.random.default_rng(19)
        # Pure noise, no group structure — should drive tau2 to 0
        n_groups = 5
        residuals = rng.normal(0, 0.5, 500)  # large n to be confident
        group_ids = np.repeat(np.arange(n_groups).astype(str), 100)
        est = RandomEffectsEstimator(min_group_size=5)
        est.fit(residuals, group_ids)
        vc = est.variance_components
        gcol = list(vc.tau2.keys())[0]
        if vc.tau2[gcol] == 0.0:
            # All BLUPs should be zero
            for g, stats in est.group_stats.items():
                assert stats["Z"] == 0.0

    def test_zero_weight_observations_handled(self):
        """Observations with zero weight should not crash."""
        rng = np.random.default_rng(20)
        residuals = rng.normal(0, 0.5, 100)
        group_ids = np.repeat(np.arange(5).astype(str), 20)
        weights = np.ones(100)
        weights[0] = 0.0  # one zero-weight obs
        est = RandomEffectsEstimator(min_group_size=5)
        vc = est.fit(residuals, group_ids, weights=weights)
        assert vc.sigma2 > 0

    def test_fewer_than_2_groups_sets_tau2_zero(self):
        """With only 1 eligible group, tau2 must be 0.0."""
        rng = np.random.default_rng(21)
        residuals = rng.normal(0, 0.5, 30)
        group_ids = np.repeat(["only_group"], 30)
        est = RandomEffectsEstimator(min_group_size=5)
        with pytest.warns(UserWarning, match="Fewer than 2 groups"):
            vc = est.fit(residuals, group_ids)
        gcol = list(vc.tau2.keys()) if vc.tau2 else ["group"]
        tau2_val = vc.tau2.get("group", 0.0)
        assert tau2_val == 0.0

    def test_n_obs_used_matches_eligible(self):
        """n_obs_used should equal total observations in eligible groups."""
        rng = np.random.default_rng(22)
        big_residuals = rng.normal(0, 0.5, 100)  # 5 groups * 20 obs
        big_groups = np.repeat(np.arange(5).astype(str), 20)
        tiny_residuals = rng.normal(0, 0.5, 2)  # 2 obs, below min_group_size=5
        tiny_groups = np.array(["tiny"] * 2)
        residuals = np.concatenate([big_residuals, tiny_residuals])
        group_ids = np.concatenate([big_groups, tiny_groups])
        est = RandomEffectsEstimator(min_group_size=5)
        vc = est.fit(residuals, group_ids)
        assert vc.n_obs_used == 100  # only the 5 big groups

    def test_n_groups_in_variance_components(self):
        """n_groups field should count all unique groups, not just eligible ones."""
        rng = np.random.default_rng(23)
        residuals = np.concatenate([
            rng.normal(0, 0.5, 100),  # 5 big groups
            rng.normal(0, 0.5, 2),    # 1 tiny group
        ])
        group_ids = np.concatenate([
            np.repeat(np.arange(5).astype(str), 20),
            np.array(["tiny"] * 2),
        ])
        est = RandomEffectsEstimator(min_group_size=5)
        vc = est.fit(residuals, group_ids)
        gcol = list(vc.n_groups.keys())[0]
        assert vc.n_groups[gcol] == 6  # all 6 unique groups counted

    def test_blup_is_float_not_array(self):
        """Each value in blup_map must be a scalar float, not an array."""
        rng = np.random.default_rng(24)
        residuals = rng.normal(0, 0.5, 100)
        group_ids = np.repeat(np.arange(5).astype(str), 20)
        est = RandomEffectsEstimator(min_group_size=5)
        est.fit(residuals, group_ids)
        for g, blup in est.blup_map.items():
            assert isinstance(blup, float), f"blup for group {g} is {type(blup)}, expected float"


# ---------------------------------------------------------------------------
# MultilevelPricingModel — additional model tests
# ---------------------------------------------------------------------------


def _make_small_data(n: int = 200, n_groups: int = 5, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    group_labels = [f"G{i}" for i in range(n_groups)]
    broker_ids = rng.choice(group_labels, size=n)
    age = rng.uniform(18, 70, n)
    y = np.exp(5.0 + 0.01 * age + rng.normal(0, 0.3, n))
    X = pl.DataFrame({
        "age": age,
        "broker_id": broker_ids,
    })
    return {"X": X, "y": y, "n_groups": n_groups}


class TestModelUnfittedErrors:
    def test_catboost_model_before_fit_raises(self):
        model = MultilevelPricingModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.catboost_model

    def test_feature_importances_before_fit_raises(self):
        model = MultilevelPricingModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.feature_importances

    def test_stage1_predict_before_fit_raises(self):
        d = _make_small_data()
        model = MultilevelPricingModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.stage1_predict(d["X"])

    def test_log_ratio_residuals_before_fit_raises(self):
        d = _make_small_data()
        model = MultilevelPricingModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.log_ratio_residuals(d["X"], d["y"])

    def test_credibility_summary_before_fit_raises(self):
        model = MultilevelPricingModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.credibility_summary()

    def test_variance_components_before_fit_raises(self):
        model = MultilevelPricingModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.variance_components


class TestModelFitValidation:
    def test_all_group_cols_removes_no_features_raises(self):
        """If group_cols covers all columns, no features remain — should raise."""
        X = pl.DataFrame({"broker_id": ["A", "B", "C"] * 10})
        y = np.ones(30) * 100.0
        model = MultilevelPricingModel(
            catboost_params={"iterations": 5, "verbose": 0},
        )
        with pytest.raises(ValueError, match="No features remain"):
            model.fit(X, y, group_cols=["broker_id"])

    def test_polars_series_y_accepted(self):
        """y as pl.Series should work without error."""
        d = _make_small_data(n=100, seed=1)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 5, "verbose": 0},
            random_effects=["broker_id"],
        )
        y_series = pl.Series("y", d["y"])
        model.fit(d["X"], y_series, group_cols=["broker_id"])
        assert model._fitted

    def test_polars_series_weights_accepted(self):
        """weights as pl.Series should work without error."""
        d = _make_small_data(n=100, seed=2)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 5, "verbose": 0},
            random_effects=["broker_id"],
        )
        w_series = pl.Series("w", np.ones(100))
        model.fit(d["X"], d["y"], weights=w_series, group_cols=["broker_id"])
        assert model._fitted

    def test_group_cols_override_random_effects(self):
        """group_cols passed to fit() should override self.random_effects."""
        d = _make_small_data(n=150, n_groups=5, seed=3)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 5, "verbose": 0},
            random_effects=[],  # no random effects by default
        )
        model.fit(d["X"], d["y"], group_cols=["broker_id"])
        # Stage 2 should have been fitted for broker_id
        assert "broker_id" in model._re_estimators


class TestModelPredict:
    def test_predict_shape(self):
        d = _make_small_data(n=100, seed=4)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        preds = model.predict(d["X"])
        assert preds.shape == (100,)

    def test_predict_all_positive(self):
        d = _make_small_data(n=100, seed=5)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        preds = model.predict(d["X"])
        assert np.all(preds > 0)

    def test_predict_group_col_not_in_estimators_skipped(self):
        """If a group_col wasn't fitted, predict should silently skip it."""
        d = _make_small_data(n=100, seed=6)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        # Pass an extra group_col that was never fitted
        X_with_extra = d["X"].with_columns(pl.lit("X").alias("scheme_id"))
        preds = model.predict(X_with_extra, group_cols=["broker_id", "scheme_id"])
        assert np.all(preds > 0)

    def test_predict_no_group_cols_is_pure_catboost(self):
        """predict with group_cols=[] should equal stage1_predict."""
        d = _make_small_data(n=100, seed=7)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        preds_no_re = model.predict(d["X"], group_cols=[])
        preds_stage1 = model.stage1_predict(d["X"])
        np.testing.assert_allclose(preds_no_re, preds_stage1, rtol=1e-5)

    def test_stage1_predict_shape(self):
        d = _make_small_data(n=100, seed=8)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        stage1 = model.stage1_predict(d["X"])
        assert stage1.shape == (100,)
        assert np.all(stage1 > 0)


class TestModelCredibilitySummary:
    def _fit_model(self, seed: int = 42) -> tuple:
        d = _make_small_data(n=200, n_groups=5, seed=seed)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        return model, d

    def test_credibility_summary_cached(self):
        """Second call to credibility_summary() should return cached result."""
        model, _ = self._fit_model()
        df1 = model.credibility_summary()
        df2 = model.credibility_summary()
        # Same object (identity) — cache is returned
        assert df1 is df2

    def test_credibility_summary_group_col_filter(self):
        """credibility_summary(group_col=...) should return only that level."""
        model, _ = self._fit_model()
        df_all = model.credibility_summary()
        df_filtered = model.credibility_summary("broker_id")
        broker_rows = df_all.filter(pl.col("level") == "broker_id")
        assert len(df_filtered) == len(broker_rows)

    def test_credibility_summary_multiplier_positive(self):
        """All multipliers should be positive (they are exp(blup))."""
        model, _ = self._fit_model()
        summary = model.credibility_summary()
        assert (summary["multiplier"] > 0).all()

    def test_credibility_summary_eligible_flag(self):
        """eligible=True iff credibility_weight > 0."""
        model, _ = self._fit_model()
        summary = model.credibility_summary()
        for row in summary.iter_rows(named=True):
            if row["eligible"]:
                assert row["credibility_weight"] > 0
            else:
                assert row["credibility_weight"] == 0.0

    def test_credibility_summary_no_group_cols_empty(self):
        """Model with no group cols should return empty DataFrame from summary."""
        d = _make_small_data(n=100, seed=9)
        X_no_group = d["X"].select(["age"])
        model = MultilevelPricingModel(
            catboost_params={"iterations": 5, "verbose": 0},
            random_effects=[],
        )
        model.fit(X_no_group, d["y"], group_cols=[])
        summary = model.credibility_summary()
        assert len(summary) == 0

    def test_credibility_summary_cache_invalidated_after_refit(self):
        """Calling fit() again should invalidate the credibility_df cache."""
        d = _make_small_data(n=150, seed=10)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 5, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        df1 = model.credibility_summary()
        model.fit(d["X"], d["y"])  # refit
        # Cache should be None after refit
        assert model._credibility_df is None


class TestModelFeatureImportances:
    def test_feature_importances_sum_to_100(self):
        """CatBoost feature importances should sum to approximately 100."""
        d = _make_small_data(n=200, seed=11)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 20, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        fi = model.feature_importances
        total = sum(fi.values())
        assert total == pytest.approx(100.0, abs=1.0)

    def test_feature_importances_excludes_group_cols(self):
        """Group columns must not appear in feature importances."""
        d = _make_small_data(n=200, seed=12)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 20, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        fi = model.feature_importances
        assert "broker_id" not in fi
        assert "age" in fi

    def test_feature_importances_nonnegative(self):
        d = _make_small_data(n=200, seed=13)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 20, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        fi = model.feature_importances
        assert all(v >= 0 for v in fi.values())


class TestModelLogRatioResiduals:
    def test_log_ratio_residuals_finite(self):
        d = _make_small_data(n=150, seed=14)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        resids = model.log_ratio_residuals(d["X"], d["y"])
        assert np.all(np.isfinite(resids))

    def test_log_ratio_residuals_mean_near_zero(self):
        """For a well-calibrated model, mean log-ratio should be near 0."""
        rng = np.random.default_rng(15)
        n = 300
        age = rng.uniform(18, 70, n)
        y = np.exp(5.0 + 0.01 * age + rng.normal(0, 0.2, n))
        X = pl.DataFrame({"age": age, "broker_id": ["G0"] * n})
        model = MultilevelPricingModel(
            catboost_params={"iterations": 50, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(X, y)
        resids = model.log_ratio_residuals(X, y)
        # Mean should not be extreme — well-calibrated model should centre near 0
        assert abs(float(np.mean(resids))) < 2.0  # loose bound, not expecting perfection


class TestModelCatboostParams:
    def test_custom_catboost_params_merged(self):
        """Custom params should be merged with defaults."""
        d = _make_small_data(n=100, seed=16)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 7, "depth": 3},
            random_effects=["broker_id"],
        )
        # Check that our override is stored
        assert model._catboost_params["iterations"] == 7
        assert model._catboost_params["depth"] == 3
        # Default params also present
        assert "verbose" in model._catboost_params

    def test_random_seed_in_defaults(self):
        """Default catboost params should set random_seed for reproducibility."""
        model = MultilevelPricingModel()
        assert "random_seed" in model._catboost_params


# ---------------------------------------------------------------------------
# Diagnostic functions — additional edge-case tests
# ---------------------------------------------------------------------------


class TestICC:
    def test_icc_missing_group_col_returns_zero(self):
        """icc() with a group_col not in vc.tau2 should return 0."""
        vc = VarianceComponents(
            sigma2=0.25, tau2={"broker_id": 0.10},
            k={"broker_id": 2.5}, log_likelihood=-100.0,
            converged=True, iterations=5,
        )
        result = icc(vc, "nonexistent_col")
        assert result == pytest.approx(0.0)

    def test_icc_both_zero_returns_zero(self):
        """When both tau2 and sigma2 are zero (degenerate), icc returns 0."""
        vc = VarianceComponents(
            sigma2=0.0, tau2={"g": 0.0},
            k={"g": float("nan")}, log_likelihood=0.0,
            converged=False, iterations=0,
        )
        result = icc(vc, "g")
        assert result == 0.0

    def test_icc_symmetric(self):
        """ICC = tau2 / (tau2 + sigma2) algebraically."""
        tau2, sigma2 = 0.12, 0.36
        vc = VarianceComponents(
            sigma2=sigma2, tau2={"g": tau2},
            k={"g": 3.0}, log_likelihood=-50.0,
            converged=True, iterations=10,
        )
        expected = tau2 / (tau2 + sigma2)
        assert icc(vc, "g") == pytest.approx(expected, rel=1e-9)

    def test_icc_range(self):
        """ICC must always be in [0, 1]."""
        for tau2_val in [0.0, 0.01, 0.10, 0.5, 2.0]:
            vc = VarianceComponents(
                sigma2=0.25, tau2={"g": tau2_val},
                k={"g": 0.0}, log_likelihood=0.0,
                converged=True, iterations=0,
            )
            val = icc(vc, "g")
            assert 0.0 <= val <= 1.0


class TestVarianceDecomposition:
    def test_empty_dict_returns_empty_dataframe(self):
        """variance_decomposition({}) should return a DataFrame with correct schema."""
        df = variance_decomposition({})
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
        assert "level" in df.columns
        assert "icc" in df.columns

    def test_single_level_icc_matches(self):
        vc = VarianceComponents(
            sigma2=0.25, tau2={"broker_id": 0.10},
            k={"broker_id": 2.5}, log_likelihood=-100.0,
            converged=True, iterations=10,
            n_groups={"broker_id": 20}, n_obs_used=1000,
        )
        df = variance_decomposition({"broker_id": vc})
        row = df.row(0, named=True)
        expected_icc = 0.10 / (0.10 + 0.25)
        assert row["icc"] == pytest.approx(expected_icc, rel=1e-5)

    def test_buhlmann_k_inf_becomes_nan(self):
        """k=inf should be converted to NaN in the DataFrame."""
        vc = VarianceComponents(
            sigma2=0.25, tau2={"g": 0.0},
            k={"g": float("inf")}, log_likelihood=0.0,
            converged=False, iterations=0,
        )
        df = variance_decomposition({"g": vc})
        row = df.row(0, named=True)
        assert math.isnan(row["buhlmann_k"])

    def test_multiple_levels_row_count(self):
        vcs = {
            f"level_{i}": VarianceComponents(
                sigma2=0.25, tau2={f"level_{i}": 0.05 * i},
                k={f"level_{i}": float("inf") if i == 0 else 5.0 / i},
                log_likelihood=-50.0,
                converged=True, iterations=5,
            )
            for i in range(3)
        }
        df = variance_decomposition(vcs)
        assert len(df) == 3


class TestHighCredibilityGroups:
    def _make_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "level": ["broker_id"] * 4,
            "group": ["B1", "B2", "B3", "B4"],
            "exposure_sum": [500.0, 200.0, 50.0, 10.0],
            "n_obs": [480, 190, 48, 10],
            "group_mean": [0.1, -0.05, 0.2, -0.1],
            "blup": [0.07, -0.03, 0.12, -0.05],
            "multiplier": [1.07, 0.97, 1.13, 0.95],
            "credibility_weight": [0.90, 0.75, 0.45, 0.10],
            "tau2": [0.10] * 4,
            "sigma2": [0.25] * 4,
            "k": [2.5] * 4,
            "eligible": [True, True, True, False],
        })

    def test_min_z_zero_returns_all(self):
        df = self._make_df()
        result = high_credibility_groups(df, min_z=0.0)
        assert len(result) == 4

    def test_min_z_one_returns_empty(self):
        df = self._make_df()
        result = high_credibility_groups(df, min_z=1.0)
        assert len(result) == 0

    def test_exact_threshold_included(self):
        """Group with Z exactly equal to min_z should be included."""
        df = self._make_df()
        result = high_credibility_groups(df, min_z=0.90)
        groups = result["group"].to_list()
        assert "B1" in groups  # Z=0.90 >= 0.90

    def test_sorted_descending(self):
        df = self._make_df()
        result = high_credibility_groups(df, min_z=0.0)
        z_vals = result["credibility_weight"].to_list()
        assert z_vals == sorted(z_vals, reverse=True)


class TestGroupsNeedingData:
    def _make_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "level": ["broker_id"] * 3,
            "group": ["B1", "B2", "B3"],
            "exposure_sum": [500.0, 100.0, 10.0],
            "n_obs": [480, 95, 10],
            "group_mean": [0.1, -0.05, 0.2],
            "blup": [0.07, -0.03, 0.12],
            "multiplier": [1.07, 0.97, 1.13],
            "credibility_weight": [0.90, 0.65, 0.20],
            "tau2": [0.10] * 3,
            "sigma2": [0.25] * 3,
            "k": [2.5] * 3,
            "eligible": [True, True, True],
        })

    def test_target_z_zero_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError, match="target_z must be strictly"):
            groups_needing_data(df, target_z=0.0)

    def test_target_z_one_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError, match="target_z must be strictly"):
            groups_needing_data(df, target_z=1.0)

    def test_target_z_negative_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError, match="target_z must be strictly"):
            groups_needing_data(df, target_z=-0.1)

    def test_n_target_formula(self):
        """n_target = k * target_z / (1 - target_z)."""
        df = self._make_df()
        target_z = 0.8
        result = groups_needing_data(df, target_z=target_z)
        # k=2.5 for all groups, n_target = 2.5 * 0.8 / 0.2 = 10.0
        expected_n_target = 2.5 * target_z / (1 - target_z)
        n_targets = result["n_target"].to_list()
        for nt in n_targets:
            assert nt == pytest.approx(expected_n_target, rel=1e-6)

    def test_n_additional_clipped_at_zero(self):
        """Groups already at or past target_z should have n_additional=0."""
        df = self._make_df()
        result = groups_needing_data(df, target_z=0.5)
        assert (result["n_additional"] >= 0).all()

    def test_sorted_by_n_additional_descending(self):
        """Output should be sorted by n_additional descending within level."""
        df = self._make_df()
        result = groups_needing_data(df, target_z=0.8)
        n_add = result["n_additional"].to_list()
        assert n_add == sorted(n_add, reverse=True)


class TestResidualNormalityCheck:
    def test_single_element_no_crash(self):
        """Single-element residual should not crash."""
        result = residual_normality_check(np.array([1.0]))
        assert "mean" in result
        assert "std" in result

    def test_two_elements(self):
        result = residual_normality_check(np.array([0.0, 1.0]))
        assert result["mean"] == pytest.approx(0.5)
        assert result["std"] > 0

    def test_all_same_value_std_zero(self):
        """Constant array: std=0, skewness=0, kurtosis=0."""
        residuals = np.full(20, 3.14)
        result = residual_normality_check(residuals)
        assert result["std"] == pytest.approx(0.0)
        assert result["skewness"] == pytest.approx(0.0)
        assert result["excess_kurtosis"] == pytest.approx(0.0)
        assert result["mean"] == pytest.approx(3.14)
        assert result["p95"] == pytest.approx(3.14)
        assert result["p99"] == pytest.approx(3.14)

    def test_skewed_distribution(self):
        """Highly right-skewed data should have positive skewness."""
        rng = np.random.default_rng(42)
        residuals = rng.exponential(scale=1.0, size=500) - 1.0  # mean-centred exp
        result = residual_normality_check(residuals)
        assert result["skewness"] > 0.5, "Exponential distribution should be right-skewed"

    def test_p95_le_p99(self):
        rng = np.random.default_rng(1)
        residuals = rng.normal(0, 1, 200)
        result = residual_normality_check(residuals)
        assert result["p95"] <= result["p99"]

    def test_returns_all_keys(self):
        residuals = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        result = residual_normality_check(residuals)
        expected_keys = {"mean", "std", "skewness", "excess_kurtosis", "p95", "p99"}
        assert set(result.keys()) == expected_keys

    def test_normal_distribution_near_zero_moments(self):
        """Large Gaussian sample: skewness and kurtosis should be near 0."""
        rng = np.random.default_rng(99)
        residuals = rng.normal(0, 1, 5000)
        result = residual_normality_check(residuals)
        assert abs(result["skewness"]) < 0.15
        assert abs(result["excess_kurtosis"]) < 0.3


class TestLiftFromRandomEffects:
    def test_no_improvement_when_preds_equal(self):
        rng = np.random.default_rng(1)
        y = np.exp(rng.normal(5, 0.5, 100))
        preds = y * 1.05  # systematically off by 5%
        result = lift_from_random_effects(y, preds, preds)
        # Identical stage1 and final: improvement = 0
        assert result["rmse_improvement_pct"] == pytest.approx(0.0, abs=1e-6)
        assert result["malr_improvement_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_perfect_final_preds_positive_improvement(self):
        rng = np.random.default_rng(2)
        y = np.exp(rng.normal(5, 0.5, 100))
        stage1 = y * rng.lognormal(0, 0.3, 100)
        final = y.copy()
        result = lift_from_random_effects(y, stage1, final)
        assert result["rmse_improvement_pct"] > 0
        assert result["malr_improvement_pct"] > 0
        assert result["final_rmse"] < result["stage1_rmse"]

    def test_worse_final_preds_negative_improvement(self):
        rng = np.random.default_rng(3)
        y = np.exp(rng.normal(5, 0.5, 100))
        stage1 = y.copy()  # perfect stage1
        final = y * rng.lognormal(0, 0.3, 100)  # worse final
        result = lift_from_random_effects(y, stage1, final)
        assert result["rmse_improvement_pct"] < 0
        assert result["final_rmse"] > result["stage1_rmse"]

    def test_returns_all_six_keys(self):
        y = np.ones(10)
        result = lift_from_random_effects(y, y, y)
        expected_keys = {
            "stage1_rmse", "final_rmse", "stage1_malr", "final_malr",
            "rmse_improvement_pct", "malr_improvement_pct",
        }
        assert set(result.keys()) == expected_keys

    def test_uniform_weights_same_as_no_weights(self):
        rng = np.random.default_rng(4)
        y = np.exp(rng.normal(5, 0.5, 50))
        preds = y * 1.1
        r_no_w = lift_from_random_effects(y, preds, preds)
        r_w = lift_from_random_effects(y, preds, preds, weights=np.ones(50))
        assert r_no_w["rmse_improvement_pct"] == pytest.approx(r_w["rmse_improvement_pct"], abs=1e-6)

    def test_single_observation(self):
        """Single observation edge case should not crash."""
        y = np.array([100.0])
        stage1 = np.array([90.0])
        final = np.array([95.0])
        result = lift_from_random_effects(y, stage1, final)
        assert np.isfinite(result["stage1_rmse"])
        assert np.isfinite(result["final_rmse"])

    def test_rmse_nonnegative(self):
        rng = np.random.default_rng(5)
        y = np.exp(rng.normal(5, 0.5, 100))
        preds = y * rng.lognormal(0, 0.2, 100)
        result = lift_from_random_effects(y, preds, preds)
        assert result["stage1_rmse"] >= 0
        assert result["final_rmse"] >= 0

    def test_malr_nonnegative(self):
        rng = np.random.default_rng(6)
        y = np.exp(rng.normal(5, 0.5, 100))
        preds = y * rng.lognormal(0, 0.2, 100)
        result = lift_from_random_effects(y, preds, preds)
        assert result["stage1_malr"] >= 0
        assert result["final_malr"] >= 0


# ---------------------------------------------------------------------------
# Integration: two-level model with nested random effects
# ---------------------------------------------------------------------------


class TestTwoLevelModel:
    def _make_two_level_data(self, n: int = 400, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        n_l1, n_l2 = 8, 4
        l1_effects = rng.normal(0, 0.2, n_l1)
        l2_effects = rng.normal(0, 0.1, n_l2)
        l1_ids = rng.integers(0, n_l1, n)
        l2_ids = rng.integers(0, n_l2, n)
        age = rng.uniform(18, 70, n)
        log_y = 5.0 + 0.01 * age + l1_effects[l1_ids] + l2_effects[l2_ids] + rng.normal(0, 0.3, n)
        y = np.exp(log_y)
        X = pl.DataFrame({
            "age": age,
            "l1": [f"l1_{i}" for i in l1_ids],
            "l2": [f"l2_{i}" for i in l2_ids],
        })
        return {"X": X, "y": y}

    def test_two_level_fit_and_predict(self):
        d = self._make_two_level_data()
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["l1", "l2"],
            min_group_size=3,
        )
        model.fit(d["X"], d["y"])
        preds = model.predict(d["X"])
        assert np.all(preds > 0)
        assert preds.shape == (len(d["X"]),)

    def test_two_level_both_estimators_fitted(self):
        d = self._make_two_level_data()
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["l1", "l2"],
            min_group_size=3,
        )
        model.fit(d["X"], d["y"])
        assert "l1" in model._re_estimators
        assert "l2" in model._re_estimators

    def test_two_level_credibility_summary_rows(self):
        d = self._make_two_level_data()
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["l1", "l2"],
            min_group_size=3,
        )
        model.fit(d["X"], d["y"])
        summary = model.credibility_summary()
        levels = summary["level"].unique().to_list()
        assert "l1" in levels
        assert "l2" in levels

    def test_two_level_variance_components_both_keys(self):
        d = self._make_two_level_data()
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["l1", "l2"],
            min_group_size=3,
        )
        model.fit(d["X"], d["y"])
        vc_dict = model.variance_components
        assert "l1" in vc_dict
        assert "l2" in vc_dict

    def test_two_level_filter_by_level(self):
        d = self._make_two_level_data()
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["l1", "l2"],
            min_group_size=3,
        )
        model.fit(d["X"], d["y"])
        df_l1 = model.credibility_summary("l1")
        assert (df_l1["level"] == "l1").all()

    def test_two_level_variance_decomposition(self):
        d = self._make_two_level_data()
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["l1", "l2"],
            min_group_size=3,
        )
        model.fit(d["X"], d["y"])
        vd = variance_decomposition(model.variance_components)
        assert len(vd) == 2
        assert set(vd["level"].to_list()) == {"l1", "l2"}


# ---------------------------------------------------------------------------
# Numerical correctness: known DGP
# ---------------------------------------------------------------------------


class TestNumericalCorrectness:
    def test_buhlmann_k_formula(self):
        """k = sigma2 / tau2 must match the ratio in VarianceComponents."""
        rng = np.random.default_rng(50)
        n_groups, n_per_group = 10, 50
        sigma2_true, tau2_true = 0.25, 0.10
        effects = rng.normal(0, np.sqrt(tau2_true), n_groups)
        group_ids = np.repeat(np.arange(n_groups).astype(str), n_per_group)
        residuals = effects[np.repeat(np.arange(n_groups), n_per_group)] + rng.normal(0, np.sqrt(sigma2_true), n_groups * n_per_group)
        est = RandomEffectsEstimator(min_group_size=5)
        vc = est.fit(residuals, group_ids)
        gcol = list(vc.tau2.keys())[0]
        tau2 = vc.tau2[gcol]
        if tau2 > 1e-8:
            expected_k = vc.sigma2 / tau2
            assert vc.k[gcol] == pytest.approx(expected_k, rel=1e-4)

    def test_blup_is_z_times_deviation(self):
        """BLUP = Z_g * (group_mean - grand_mean) analytically."""
        rng = np.random.default_rng(51)
        n_groups, n_per_group = 5, 100
        residuals = rng.normal(0, 0.5, n_groups * n_per_group)
        # Inject a known group mean for group 0
        residuals[:n_per_group] += 0.5  # push group 0 up
        group_ids = np.repeat(np.arange(n_groups).astype(str), n_per_group)
        est = RandomEffectsEstimator(min_group_size=5)
        est.fit(residuals, group_ids)

        vc = est.variance_components
        gcol = list(vc.tau2.keys())[0]
        sigma2 = vc.sigma2
        tau2 = vc.tau2[gcol]

        if tau2 > 0:
            # For group "0": n_g=100, Z_g = tau2 / (tau2 + sigma2/100)
            n_g = 100.0
            Z_g = tau2 / (tau2 + sigma2 / n_g)
            # grand mean from REML
            mu_hat = est._mu_hat
            mu_0 = float(np.average(residuals[:n_per_group]))
            blup_expected = Z_g * (mu_0 - mu_hat)
            blup_actual = est.blup_map["0"]
            assert blup_actual == pytest.approx(blup_expected, rel=0.01)

    def test_icc_numerical_formula(self):
        """ICC at known sigma2=0.3, tau2=0.2 should be 0.2/0.5 = 0.4."""
        vc = VarianceComponents(
            sigma2=0.3, tau2={"g": 0.2},
            k={"g": 1.5}, log_likelihood=-50.0,
            converged=True, iterations=5,
        )
        assert icc(vc, "g") == pytest.approx(0.4, rel=1e-9)

    def test_predict_multiplies_stage1_by_exp_blup(self):
        """For known-group observations, predict = stage1 * exp(blup)."""
        d = _make_small_data(n=100, n_groups=5, seed=60)
        model = MultilevelPricingModel(
            catboost_params={"iterations": 10, "verbose": 0},
            random_effects=["broker_id"],
        )
        model.fit(d["X"], d["y"])
        stage1 = model.stage1_predict(d["X"])
        final = model.predict(d["X"])
        # Compute expected multiplier from blup_map
        log_adj = np.zeros(len(d["X"]))
        g_arr = d["X"]["broker_id"].to_numpy().astype(str)
        blups = model._re_estimators["broker_id"].predict_blup(g_arr)
        expected_final = stage1 * np.exp(blups)
        np.testing.assert_allclose(final, expected_final, rtol=1e-5)
