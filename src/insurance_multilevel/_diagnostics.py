"""
Diagnostic utilities for MultilevelPricingModel.

These functions help pricing actuaries understand the model output:
- Is tau2 large enough to matter? (variance decomposition)
- Which groups have high credibility? (credibility summary)
- Are residuals well-behaved? (normality checks for REML validity)
- How much lift do random effects provide over pure CatBoost?
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._types import VarianceComponents


def icc(vc: VarianceComponents, group_col: str) -> float:
    """
    Intraclass Correlation Coefficient for a given group level.

    ICC = tau2 / (tau2 + sigma2)

    This is the proportion of total variance explained by group membership.
    An ICC of 0.10 means 10% of premium variation is attributable to which
    broker a policy came through, after controlling for risk factors.

    Typical values in UK insurance:
    - Broker effect: ICC 0.05-0.20 (brokers differ in risk appetite)
    - Scheme effect: ICC 0.10-0.30 (schemes selected by definition)
    - Territory (postcode sector): ICC 0.02-0.10

    Parameters
    ----------
    vc : VarianceComponents
        From model.variance_components[group_col]
    group_col : str
        Which group level to compute ICC for.

    Returns
    -------
    float in [0, 1]
    """
    tau2 = vc.tau2.get(group_col, 0.0)
    sigma2 = vc.sigma2
    total = tau2 + sigma2
    if total < 1e-12:
        return 0.0
    return tau2 / total


def variance_decomposition(
    variance_components_dict: dict[str, VarianceComponents],
) -> pl.DataFrame:
    """
    Decompose total variance into components across all group levels.

    Returns a summary showing how much of the residual variance is
    attributed to each group level vs. within-group noise.

    Parameters
    ----------
    variance_components_dict : dict[str, VarianceComponents]
        From model.variance_components

    Returns
    -------
    pl.DataFrame with columns: level, tau2, sigma2, icc, buhlmann_k
    """
    rows = []
    for gcol, vc in variance_components_dict.items():
        tau2 = vc.tau2.get(gcol, 0.0)
        sigma2 = vc.sigma2
        icc_val = icc(vc, gcol)
        k_val = vc.k.get(gcol, float("inf"))
        rows.append({
            "level": gcol,
            "tau2": tau2,
            "sigma2": sigma2,
            "icc": icc_val,
            "buhlmann_k": k_val if k_val != float("inf") else float("nan"),
            "converged": vc.converged,
            "n_groups": vc.n_groups.get(gcol, 0),
            "n_obs_used": vc.n_obs_used,
        })

    return pl.DataFrame(rows, schema={
        "level": pl.String,
        "tau2": pl.Float64,
        "sigma2": pl.Float64,
        "icc": pl.Float64,
        "buhlmann_k": pl.Float64,
        "converged": pl.Boolean,
        "n_groups": pl.Int64,
        "n_obs_used": pl.Int64,
    })


def high_credibility_groups(
    credibility_df: pl.DataFrame,
    min_z: float = 0.5,
) -> pl.DataFrame:
    """
    Return groups with credibility weight above threshold.

    Parameters
    ----------
    credibility_df : pl.DataFrame
        From model.credibility_summary()
    min_z : float
        Minimum credibility weight (default 0.5). Groups above this threshold
        have seen enough data that their own experience dominates the prior.

    Returns
    -------
    pl.DataFrame filtered and sorted by credibility_weight descending.
    """
    return (
        credibility_df
        .filter(pl.col("credibility_weight") >= min_z)
        .sort("credibility_weight", descending=True)
    )


def groups_needing_data(
    credibility_df: pl.DataFrame,
    target_z: float = 0.8,
) -> pl.DataFrame:
    """
    Compute how many more observations each group needs to reach target_z.

    Based on Z_g = tau2 / (tau2 + sigma2/n_g), solving for n_target:
        n_target = sigma2 * target_z / (tau2 * (1 - target_z))
               = k * target_z / (1 - target_z)

    Parameters
    ----------
    credibility_df : pl.DataFrame
        From model.credibility_summary()
    target_z : float
        Target credibility weight (default 0.8).

    Returns
    -------
    pl.DataFrame with added columns: n_target, n_additional
    """
    if target_z <= 0 or target_z >= 1:
        raise ValueError("target_z must be strictly between 0 and 1.")

    ratio = target_z / (1 - target_z)

    # n_target is in the same units as exposure_sum (weight sum), since k = sigma2/tau2
    # is estimated from the weighted data. The comparison tells you how many more
    # exposure-weight units a group needs to reach the target credibility level.
    return (
        credibility_df
        .with_columns([
            (pl.col("k") * ratio).alias("n_target"),
        ])
        .with_columns([
            (pl.col("n_target") - pl.col("exposure_sum")).clip(lower_bound=0).alias("n_additional"),
        ])
        .select([
            "level", "group", "n_obs", "exposure_sum", "credibility_weight", "n_target", "n_additional",
        ])
        .sort(["level", "n_additional"], descending=[False, True])
    )


def residual_normality_check(residuals: np.ndarray) -> dict[str, float]:
    """
    Basic normality checks on log-ratio residuals.

    REML inference is valid under Gaussian residuals. These checks flag
    severe departures that might indicate model misspecification.

    Returns
    -------
    dict with: mean, std, skewness, excess_kurtosis, p95, p99
    """
    mean = float(np.mean(residuals))
    std = float(np.std(residuals))
    if std < 1e-12:
        return {"mean": mean, "std": std, "skewness": 0.0,
                "excess_kurtosis": 0.0, "p95": mean, "p99": mean}

    centered = residuals - mean
    skewness = float(np.mean(centered ** 3) / std ** 3)
    excess_kurtosis = float(np.mean(centered ** 4) / std ** 4 - 3.0)

    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "p95": float(np.percentile(residuals, 95)),
        "p99": float(np.percentile(residuals, 99)),
    }


def lift_from_random_effects(
    y_true: np.ndarray,
    stage1_preds: np.ndarray,
    final_preds: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Quantify how much Stage 2 random effects improve on Stage 1 CatBoost.

    Measures: weighted RMSE and weighted mean absolute log ratio (MALR)
    for both Stage 1 and final predictions.

    Parameters
    ----------
    y_true : np.ndarray
    stage1_preds : np.ndarray — CatBoost only (no random effects)
    final_preds : np.ndarray — CatBoost + random effects
    weights : np.ndarray or None

    Returns
    -------
    dict with keys: stage1_rmse, final_rmse, stage1_malr, final_malr,
                    rmse_improvement_pct, malr_improvement_pct
    """
    if weights is None:
        weights = np.ones(len(y_true))
    w = weights / weights.sum()

    _tiny = 1e-9

    def w_rmse(pred: np.ndarray) -> float:
        return float(np.sqrt(np.sum(w * (y_true - pred) ** 2)))

    def w_malr(pred: np.ndarray) -> float:
        return float(np.sum(w * np.abs(np.log(np.clip(pred, _tiny, None) / np.clip(y_true, _tiny, None)))))

    s1_rmse = w_rmse(stage1_preds)
    f_rmse = w_rmse(final_preds)
    s1_malr = w_malr(stage1_preds)
    f_malr = w_malr(final_preds)

    rmse_imp = 100 * (s1_rmse - f_rmse) / max(s1_rmse, _tiny)
    malr_imp = 100 * (s1_malr - f_malr) / max(s1_malr, _tiny)

    return {
        "stage1_rmse": s1_rmse,
        "final_rmse": f_rmse,
        "stage1_malr": s1_malr,
        "final_malr": f_malr,
        "rmse_improvement_pct": rmse_imp,
        "malr_improvement_pct": malr_imp,
    }
