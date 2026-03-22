"""
MultilevelPricingModel: two-stage CatBoost + REML random effects.

Design decisions
----------------
1. Group columns are EXCLUDED from Stage 1 CatBoost features. This is the
   critical design choice that makes the two-stage approach identifiable.
   If broker_id were fed to CatBoost, the GBM would partially absorb the
   group signal — then REML would see only the residual group variation,
   underestimating tau2. See KB entry 655 for the full identifiability argument.

2. We model log-ratio residuals: r_i = log(y_i / f_hat_i). This transforms
   a multiplicative pricing problem into an additive random effects problem.
   The final premium is f_hat(x) * exp(b_hat_g), which is the standard
   multiplicative structure used throughout UK personal lines pricing.

3. We use a single RandomEffectsEstimator per group column. Nested hierarchies
   (e.g., policy within scheme within broker) are supported by fitting
   separate estimators per level and composing the BLUP adjustments. Crossed
   effects (broker x territory) are excluded from V1.
   When multiple group columns are fitted, each level's estimator sees residuals
   with the previous level's BLUPs subtracted. This prevents double-counting
   of group signal across levels (B2 fix).

4. min_group_size=5 is the default threshold below which a group gets Z=0.
   Groups with n=1 cannot separate their group effect from residual noise;
   shrinking them to the grand mean is the correct Bayesian answer.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostRegressor

from ._reml import RandomEffectsEstimator
from ._types import VarianceComponents


_TINY = 1e-9  # Guard against log(0)
_CATBOOST_DEFAULTS: dict[str, Any] = {
    "loss_function": "RMSE",
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "verbose": 0,
    "random_seed": 42,
    "allow_writing_files": False,
}


class MultilevelPricingModel:
    """
    Two-stage insurance pricing model combining CatBoost fixed effects
    with REML-estimated random effects for high-cardinality group factors.

    The model handles the fundamental tension in insurance pricing between:
    - Individual risk factors (age, vehicle, postcode sector) that CatBoost
      handles well given enough data
    - Group-level variation (broker, scheme, affinity partner) where the
      groups are too numerous for one-hot encoding but too few for CatBoost
      to learn stable effects

    The solution is sequential: fit CatBoost on all features EXCEPT the
    group columns (Stage 1), then fit random intercepts on the log-ratio
    residuals using REML (Stage 2). The final prediction multiplies the
    CatBoost score by the BLUP-derived group factor.

    Parameters
    ----------
    catboost_params : dict or None
        Parameters passed to CatBoostRegressor. Merged with sensible defaults.
        Do not pass loss_function if you want the default RMSE.
    random_effects : list[str]
        Column names to treat as random effects levels. Order matters for
        nested models: put the highest level first (e.g., ["broker", "scheme"]).
    min_group_size : int
        Groups whose total exposure weight (sum of the weights array passed to
        fit()) is below this threshold are excluded from REML and assigned
        credibility weight Z=0. Default 5. When weights=None every observation
        has weight 1, so this is equivalent to a minimum observation count.
    reml : bool
        Use REML (True, default) vs. ML (False). REML is almost always
        correct for variance component estimation.

    Examples
    --------
    >>> model = MultilevelPricingModel(
    ...     catboost_params={"iterations": 500, "loss_function": "Poisson"},
    ...     random_effects=["broker_id", "scheme_id"],
    ...     min_group_size=10,
    ... )
    >>> model.fit(X_train, y_train, weights=exposure, group_cols=["broker_id", "scheme_id"])
    >>> premiums = model.predict(X_test, group_cols=["broker_id", "scheme_id"])
    >>> summary = model.credibility_summary()
    """

    def __init__(
        self,
        catboost_params: dict[str, Any] | None = None,
        random_effects: list[str] | None = None,
        min_group_size: int = 5,
        reml: bool = True,
    ) -> None:
        self.random_effects: list[str] = random_effects or []
        self.min_group_size = min_group_size
        self.reml = reml

        # Merge catboost params with defaults
        merged = dict(_CATBOOST_DEFAULTS)
        if catboost_params:
            merged.update(catboost_params)
        self._catboost_params = merged

        # State set by fit()
        self._catboost: CatBoostRegressor | None = None
        self._re_estimators: dict[str, RandomEffectsEstimator] = {}
        self._feature_cols: list[str] | None = None
        self._cat_feature_indices: list[int] = []
        self._fitted = False

        # Summary storage
        self._variance_components: dict[str, VarianceComponents] = {}
        self._credibility_df: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series | np.ndarray,
        weights: pl.Series | np.ndarray | None = None,
        group_cols: list[str] | None = None,
    ) -> "MultilevelPricingModel":
        """
        Fit the two-stage model.

        Parameters
        ----------
        X : pl.DataFrame
            Feature matrix including group columns. Group columns will be
            excluded from Stage 1 CatBoost features automatically.
        y : pl.Series or np.ndarray
            Observed response (e.g., claims cost, frequency). Must be positive.
        weights : pl.Series or np.ndarray or None
            Observation weights (e.g., exposure in years). If None, all
            observations have equal weight.
        group_cols : list[str] or None
            Names of group columns in X. If None, uses self.random_effects.
            If neither is set, Stage 2 is skipped (pure CatBoost model).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If any value of y is <= 0. This model computes log(y / f_hat)
            as residuals, which is undefined for non-positive y. This means
            raw claim counts (including zeros) cannot be used directly — you
            must use a positive response such as claim frequency (claims/exposure),
            pure premium (loss/exposure), or severity (conditional on a claim).
            For zero-inflated count data, model frequency and severity separately
            and combine the predictions.
        """
        group_cols = self._resolve_group_cols(group_cols)
        y_arr = _to_numpy(y, "y")
        w_arr = _to_numpy(weights, "weights") if weights is not None else np.ones(len(y_arr))

        if np.any(y_arr <= 0):
            raise ValueError(
                "Response y must be strictly positive (all values > 0). "
                "This model uses log(y / f_hat) as residuals, which requires y > 0. "
                "For zero-inflated responses, consider modelling frequency and severity separately."
            )

        # Stage 1: CatBoost on non-group features
        self._feature_cols = [c for c in X.columns if c not in group_cols]
        if not self._feature_cols:
            raise ValueError(
                f"No features remain after excluding group_cols={group_cols}. "
                "CatBoost has nothing to learn from."
            )

        X_cb = X.select(self._feature_cols)
        cat_indices = _find_cat_feature_indices(X_cb)
        self._cat_feature_indices = cat_indices

        cb_params = dict(self._catboost_params)
        if cat_indices:
            cb_params["cat_features"] = cat_indices

        self._catboost = CatBoostRegressor(**cb_params)
        self._catboost.fit(
            X_cb.to_pandas(),
            y_arr,
            sample_weight=w_arr,
        )

        f_hat = self._catboost.predict(X_cb.to_pandas()).astype(float)

        # Protect against non-positive predictions (can happen with RMSE loss)
        f_hat = np.clip(f_hat, _TINY, None)

        # Stage 2: REML random effects on log-ratio residuals per group level.
        # FIX B2: When fitting multiple levels, subtract the previous level's
        # BLUPs from the residuals before fitting the next level. This prevents
        # double-counting overlap between levels. Each estimator therefore sees
        # only the residual variation not yet explained by higher levels.
        base_residuals = np.log(np.clip(y_arr, _TINY, None) / f_hat)

        self._re_estimators = {}
        self._variance_components = {}

        # running_blup tracks the cumulative log-scale BLUP adjustment from
        # all previously fitted levels. Each new level fits on what remains.
        running_blup = np.zeros(len(y_arr))

        for gcol in group_cols:
            g_arr = X[gcol].to_numpy().astype(str)
            level_residuals = base_residuals - running_blup
            est = RandomEffectsEstimator(
                reml=self.reml,
                min_group_size=self.min_group_size,
            )
            vc = est.fit(level_residuals, g_arr, w_arr, group_col=gcol)
            self._re_estimators[gcol] = est
            self._variance_components[gcol] = vc
            # Accumulate this level's BLUPs so the next level sees de-meaned residuals
            running_blup += est.predict_blup(g_arr, allow_new_groups=True)

        self._fitted = True
        self._credibility_df = None  # invalidate cache
        return self

    def predict(
        self,
        X: pl.DataFrame,
        group_cols: list[str] | None = None,
        allow_new_groups: bool = True,
    ) -> np.ndarray:
        """
        Predict premiums for new observations.

        Parameters
        ----------
        X : pl.DataFrame
            Feature matrix including group columns (same schema as training).
        group_cols : list[str] or None
            Group columns to apply random effects. If None, uses the columns
            fitted in Stage 2. If the model was fitted with no group cols,
            returns pure CatBoost predictions.
        allow_new_groups : bool
            If True, groups not seen at training time get a random effects
            multiplier of 1.0 (no adjustment). If False, raises KeyError.

        Returns
        -------
        np.ndarray of shape (n,) — predicted premiums.
        """
        self._check_fitted()
        group_cols = self._resolve_group_cols(group_cols)

        X_cb = X.select(self._feature_cols)
        f_hat = self._catboost.predict(X_cb.to_pandas()).astype(float)
        f_hat = np.clip(f_hat, _TINY, None)

        # Compose random effects adjustments (log scale, additive)
        log_adjustment = np.zeros(len(X))
        for gcol in group_cols:
            if gcol not in self._re_estimators:
                continue
            g_arr = X[gcol].to_numpy().astype(str)
            blups = self._re_estimators[gcol].predict_blup(
                g_arr, allow_new_groups=allow_new_groups
            )
            log_adjustment += blups

        return f_hat * np.exp(log_adjustment)

    def credibility_summary(self, group_col: str | None = None) -> pl.DataFrame:
        """
        Return a Bühlmann-Straub credibility summary per group.

        Parameters
        ----------
        group_col : str or None
            Which group level to summarise. If None and only one group was
            fitted, that group is used. If multiple groups, all are concatenated
            with a 'level' column added.

        Returns
        -------
        pl.DataFrame with columns:
            level      : str   — group column name
            group      : str   — group identifier
            exposure_sum : float — sum of exposure weights in group
            n_obs      : int   — actual observation count
            group_mean : float — weighted mean log-ratio residual
            blup       : float — BLUP adjustment (log scale)
            multiplier : float — exp(blup), multiplicative premium factor
            credibility_weight : float — Z_g (Bühlmann credibility weight)
            tau2       : float — between-group variance estimate
            sigma2     : float — within-group variance estimate
            k          : float — Bühlmann k = sigma2/tau2
            eligible   : bool  — whether group met min_group_size threshold
        """
        self._check_fitted()

        if self._credibility_df is not None:
            if group_col is None or group_col in [
                str(r) for r in self._credibility_df["level"].unique().to_list()
            ]:
                if group_col is None:
                    return self._credibility_df
                return self._credibility_df.filter(pl.col("level") == group_col)

        frames = []
        for gcol, est in self._re_estimators.items():
            vc = self._variance_components[gcol]
            tau2_val = vc.tau2.get(gcol, 0.0)
            sigma2_val = vc.sigma2
            k_val = vc.k.get(gcol, float("inf"))

            rows = []
            for gid, stats in est.group_stats.items():
                blup_val = est.blup_map.get(gid, 0.0)
                rows.append({
                    "level": gcol,
                    "group": str(gid),
                    "exposure_sum": stats["n"],
                    "n_obs": stats.get("n_obs", 0),
                    "group_mean": stats["mean"],
                    "blup": blup_val,
                    "multiplier": float(np.exp(blup_val)),
                    "credibility_weight": stats["Z"],
                    "tau2": tau2_val,
                    "sigma2": sigma2_val,
                    "k": k_val if k_val != float("inf") else float("nan"),
                    "eligible": stats["Z"] > 0,
                })

            if rows:
                frame = pl.DataFrame(rows, schema={
                    "level": pl.Utf8,
                    "group": pl.Utf8,
                    "exposure_sum": pl.Float64,
                    "n_obs": pl.Int64,
                    "group_mean": pl.Float64,
                    "blup": pl.Float64,
                    "multiplier": pl.Float64,
                    "credibility_weight": pl.Float64,
                    "tau2": pl.Float64,
                    "sigma2": pl.Float64,
                    "k": pl.Float64,
                    "eligible": pl.Boolean,
                })
                frames.append(frame)

        if not frames:
            return pl.DataFrame(schema={
                "level": pl.Utf8,
                "group": pl.Utf8,
                "exposure_sum": pl.Float64,
                "n_obs": pl.Int64,
                "group_mean": pl.Float64,
                "blup": pl.Float64,
                "multiplier": pl.Float64,
                "credibility_weight": pl.Float64,
                "tau2": pl.Float64,
                "sigma2": pl.Float64,
                "k": pl.Float64,
                "eligible": pl.Boolean,
            })

        self._credibility_df = pl.concat(frames).sort(["level", "credibility_weight"], descending=[False, True])

        if group_col is not None:
            return self._credibility_df.filter(pl.col("level") == group_col)
        return self._credibility_df

    @property
    def variance_components(self) -> dict[str, VarianceComponents]:
        """Variance components keyed by group column name."""
        self._check_fitted()
        return self._variance_components

    @property
    def catboost_model(self) -> CatBoostRegressor:
        """The fitted Stage 1 CatBoost model."""
        self._check_fitted()
        return self._catboost

    @property
    def feature_importances(self) -> dict[str, float]:
        """CatBoost feature importances (Stage 1 only)."""
        self._check_fitted()
        importances = self._catboost.get_feature_importance()
        return dict(zip(self._feature_cols, importances.tolist()))

    def stage1_predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Return Stage 1 CatBoost predictions without random effects adjustment.

        Useful for diagnosing how much of the residual variation is picked
        up by the random effects in Stage 2.
        """
        self._check_fitted()
        X_cb = X.select(self._feature_cols)
        f_hat = self._catboost.predict(X_cb.to_pandas()).astype(float)
        return np.clip(f_hat, _TINY, None)

    def log_ratio_residuals(
        self,
        X: pl.DataFrame,
        y: pl.Series | np.ndarray,
    ) -> np.ndarray:
        """
        Compute log-ratio residuals r_i = log(y_i / f_hat_i).

        These should be approximately N(mu_g, sigma2) within each group.
        Plotting these by group is a good diagnostic: if groups differ
        substantially, tau2 will be large and the random effects matter.
        """
        self._check_fitted()
        y_arr = _to_numpy(y, "y")
        f_hat = self.stage1_predict(X)
        return np.log(np.clip(y_arr, _TINY, None) / f_hat)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_group_cols(self, group_cols: list[str] | None) -> list[str]:
        if group_cols is not None:
            return group_cols
        if self.random_effects:
            return self.random_effects
        return []

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() before predict() or any diagnostic method."
            )


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def _to_numpy(x: pl.Series | np.ndarray | None, name: str) -> np.ndarray:
    """Convert Polars Series or ndarray to float64 ndarray."""
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(float)
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if x is None:
        raise ValueError(f"{name} cannot be None here.")
    return np.asarray(x, dtype=float)


def _find_cat_feature_indices(X: pl.DataFrame) -> list[int]:
    """Return indices of string/categorical columns for CatBoost."""
    indices = []
    for i, dtype in enumerate(X.dtypes):
        if dtype in (pl.Utf8, pl.Categorical, pl.String):
            indices.append(i)
    return indices
