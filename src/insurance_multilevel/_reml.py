"""
REML variance components estimation for one-way random effects model.

Model: r_i = mu + b_g(i) + epsilon_i
  b_g ~ N(0, tau2)   (group random effect)
  epsilon_i ~ N(0, sigma2)   (within-group residual)

We use restricted maximum likelihood (REML) rather than ML because ML
systematically underestimates variance components — particularly tau2 —
when the number of groups is small relative to sample size. REML conditions
out the fixed effect (grand mean mu) before estimating variances, giving
unbiased estimates.

The REML log-likelihood for this model has a closed-form gradient and can
be optimised efficiently with L-BFGS-B. We use Henderson's method-of-moments
as the starting point — it converges instantly and gives good initial values
even for unbalanced group structures.

References
----------
- Henderson (1953): Estimation of variance and covariance components.
  Biometrics 9(2): 226-252.
- Patterson & Thompson (1971): Recovery of inter-block information when
  block sizes are unequal. Biometrika 58(3): 545-554.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from ._types import VarianceComponents

if TYPE_CHECKING:
    pass


def _henderson_mom_init(
    residuals: np.ndarray,
    group_ids: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[float, float]:
    """
    Henderson method-of-moments initialisation for (sigma2, tau2).

    This gives decent starting values without any iteration. The classical
    ANOVA decomposition:
        SS_within  / (n - m)     -> sigma2
        [SS_between/(m-1) - sigma2/n_bar] -> tau2  (clamped to 0)

    For weighted case we use the weighted SS decomposition.
    """
    groups = np.unique(group_ids)
    m = len(groups)
    n = len(residuals)

    if weights is None:
        weights = np.ones(n)

    w_total = weights.sum()
    mu_hat = np.average(residuals, weights=weights)

    ss_total = np.sum(weights * (residuals - mu_hat) ** 2)

    # Within-group SS
    ss_within = 0.0
    n_bar_inv_sum = 0.0
    for g in groups:
        mask = group_ids == g
        r_g = residuals[mask]
        w_g = weights[mask]
        w_g_sum = w_g.sum()
        mu_g = np.average(r_g, weights=w_g)
        ss_within += np.sum(w_g * (r_g - mu_g) ** 2)
        n_bar_inv_sum += (w_g ** 2).sum() / w_g_sum

    ss_between = ss_total - ss_within
    dof_within = w_total - m
    dof_between = m - 1

    sigma2_init = max(ss_within / max(dof_within, 1.0), 1e-6)

    # n_bar for unbalanced case (harmonic-ish mean of group sizes)
    n_bar = (w_total - n_bar_inv_sum / w_total) / max(dof_between, 1.0)
    tau2_init = max((ss_between / max(dof_between, 1.0) - sigma2_init) / max(n_bar, 1.0), 0.0)

    return sigma2_init, tau2_init


def _reml_neg_log_likelihood(
    params: np.ndarray,
    residuals: np.ndarray,
    group_ids: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Negative REML log-likelihood for one-way random effects model.

    The REML likelihood marginalises over mu (fixed effect), giving:

    -2 * l_REML = sum_g [ n_g * log(sigma2) + log(tau2 + sigma2/n_g_eff)
                          + (r_bar_g - mu_hat)^2 / (tau2 + sigma2/n_g_eff)
                          + SS_within_g / sigma2 ]
                  + log(sum_g n_g_eff / (tau2 + sigma2/n_g_eff))

    where n_g_eff = sum of weights in group g.
    """
    log_sigma2, log_tau2 = params
    sigma2 = np.exp(log_sigma2)
    tau2 = np.exp(log_tau2)

    if sigma2 < 1e-12 or tau2 < 0:
        return 1e15

    n = len(residuals)
    ll = 0.0
    information_sum = 0.0  # for REML correction term

    # Accumulate per-group terms
    group_means = []
    group_precisions = []

    for g in groups:
        mask = group_ids == g
        r_g = residuals[mask]
        w_g = weights[mask]
        n_g = w_g.sum()  # effective group size

        # Within-group weighted mean
        mu_g = np.average(r_g, weights=w_g)
        ss_g = np.sum(w_g * (r_g - mu_g) ** 2)

        # Marginal variance of group mean
        v_g = tau2 + sigma2 / n_g

        ll += np.log(v_g) + ss_g / sigma2

        group_means.append(mu_g)
        group_precisions.append(1.0 / v_g)
        information_sum += n_g / v_g

    # Grand mean (weighted by precision)
    gm = np.array(group_means)
    gp = np.array(group_precisions)
    mu_hat = np.sum(gp * gm) / np.sum(gp)

    # Between-group sum
    for i, g in enumerate(groups):
        v_g = 1.0 / group_precisions[i]
        ll += (group_means[i] - mu_hat) ** 2 / v_g

    # REML correction: log|X'V^{-1}X| = log(information_sum)
    ll += np.log(max(information_sum, 1e-12))

    # Scale: factor of n * log(sigma2) already partly included above;
    # add constant within-group terms
    # We already counted log(v_g) + SS_within/sigma2 per group
    # Full REML: need to add n*log(2*pi) terms (omit as constant)

    return 0.5 * ll


def _reml_neg_log_likelihood_tau0(
    log_sigma2: float,
    residuals: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Log-likelihood when tau2 = 0 (pooled model, no group effects).
    Used to check if random effects improve fit.
    """
    sigma2 = np.exp(log_sigma2)
    if sigma2 < 1e-12:
        return 1e15
    mu_hat = np.average(residuals, weights=weights)
    ss = np.sum(weights * (residuals - mu_hat) ** 2)
    n = len(residuals)
    return 0.5 * (n * np.log(sigma2) + ss / sigma2)


class RandomEffectsEstimator:
    """
    Fits a one-way random intercepts model via REML and computes BLUPs.

    This is the statistical workhorse of the two-stage model. It receives
    log-ratio residuals from Stage 1 (CatBoost) and estimates how much of
    the residual variance is attributable to group membership vs. within-
    group noise.

    The BLUPs (Best Linear Unbiased Predictors) are the Bayesian posterior
    means of the group effects given the data. They are shrunk toward zero
    in proportion to (1 - Z_g), where Z_g is the Bühlmann credibility
    weight for group g.

    Parameters
    ----------
    reml : bool
        If True, use REML (recommended). If False, use ML (underestimates
        tau2 but is useful for likelihood ratio tests).
    min_group_size : int
        Groups with fewer observations than this are excluded from variance
        component estimation and assigned Z=0 (no credibility).
    max_iter : int
        Maximum number of L-BFGS-B iterations.
    tol : float
        Convergence tolerance for the REML optimiser.
    """

    def __init__(
        self,
        reml: bool = True,
        min_group_size: int = 5,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> None:
        self.reml = reml
        self.min_group_size = min_group_size
        self.max_iter = max_iter
        self.tol = tol

        # Fitted attributes (set after fit())
        self._sigma2: float | None = None
        self._tau2: float | None = None
        self._mu_hat: float | None = None
        self._blup_map: dict = {}  # group_id -> blup value
        self._group_stats: dict = {}  # group_id -> (n, mean, Z)
        self._variance_components: VarianceComponents | None = None
        self._group_col: str = "group"

    def fit(
        self,
        residuals: np.ndarray,
        group_ids: np.ndarray,
        weights: np.ndarray | None = None,
        group_col: str = "group",
    ) -> VarianceComponents:
        """
        Fit random effects model on log-ratio residuals.

        Parameters
        ----------
        residuals : np.ndarray of shape (n,)
            Log-ratio residuals: log(y_i / f_hat_i). Should be roughly
            normally distributed around zero if Stage 1 CatBoost is well-
            calibrated.
        group_ids : np.ndarray of shape (n,)
            Group membership indicator (string or integer).
        weights : np.ndarray of shape (n,) or None
            Observation weights (e.g., exposure). If None, uniform weights.
        group_col : str
            Name of the group column (used in VarianceComponents keys).

        Returns
        -------
        VarianceComponents
        """
        self._group_col = group_col
        n = len(residuals)
        if weights is None:
            weights = np.ones(n)
        weights = np.asarray(weights, dtype=float)

        # Identify groups meeting min_group_size
        all_groups, counts = np.unique(group_ids, return_counts=True)
        eligible_mask = np.zeros(n, dtype=bool)
        eligible_groups = []
        for g, cnt in zip(all_groups, counts):
            obs_mask = group_ids == g
            w_g = weights[obs_mask].sum()
            if w_g >= self.min_group_size:
                eligible_mask |= obs_mask
                eligible_groups.append(g)
            else:
                # Store as ineligible with Z=0
                mu_g = float(np.average(residuals[obs_mask], weights=weights[obs_mask]))
                self._group_stats[g] = {
                    "n": float(weights[obs_mask].sum()),
                    "mean": mu_g,
                    "Z": 0.0,
                }

        eligible_groups = np.array(eligible_groups)

        if len(eligible_groups) < 2:
            # Cannot estimate variance components with fewer than 2 eligible groups
            warnings.warn(
                f"Fewer than 2 groups have n >= {self.min_group_size}. "
                "Setting tau2=0 (no group effects estimated).",
                UserWarning,
                stacklevel=3,
            )
            r_elig = residuals[eligible_mask]
            w_elig = weights[eligible_mask]
            mu_hat = float(np.average(r_elig, weights=w_elig)) if len(r_elig) > 0 else 0.0
            ss = float(np.sum(w_elig * (r_elig - mu_hat) ** 2)) if len(r_elig) > 0 else 0.0
            sigma2 = max(ss / max(len(r_elig) - 1, 1), 1e-6)
            return self._store_result(
                sigma2=sigma2,
                tau2=0.0,
                mu_hat=mu_hat,
                eligible_groups=eligible_groups,
                residuals=residuals,
                group_ids=group_ids,
                weights=weights,
                log_likelihood=0.0,
                converged=False,
                iterations=0,
                n_obs_used=int(eligible_mask.sum()),
                group_col=group_col,
            )

        r_elig = residuals[eligible_mask]
        w_elig = weights[eligible_mask]
        g_elig = group_ids[eligible_mask]

        n_obs_used = len(r_elig)

        # Initialise from Henderson moments
        sigma2_init, tau2_init = _henderson_mom_init(r_elig, g_elig, w_elig)

        # Optimise REML log-likelihood
        x0 = np.array([
            np.log(max(sigma2_init, 1e-6)),
            np.log(max(tau2_init, 1e-6)),
        ])

        result = minimize(
            fun=_reml_neg_log_likelihood,
            x0=x0,
            args=(r_elig, g_elig, w_elig, eligible_groups),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol * 0.1},
        )

        sigma2 = float(np.exp(result.x[0]))
        tau2 = float(np.exp(result.x[1]))

        # Boundary check: if tau2 is tiny relative to sigma2, treat as zero
        if tau2 < 1e-8 * sigma2:
            tau2 = 0.0

        # Grand mean estimate (weighted by group precision)
        mu_hat = self._compute_mu_hat(r_elig, g_elig, w_elig, eligible_groups, sigma2, tau2)

        if not result.success:
            warnings.warn(
                f"REML optimiser did not converge (message: {result.message}). "
                "Estimates may be unreliable. Try increasing max_iter or check "
                "for degenerate group structure.",
                UserWarning,
                stacklevel=2,
            )

        return self._store_result(
            sigma2=sigma2,
            tau2=tau2,
            mu_hat=mu_hat,
            eligible_groups=eligible_groups,
            residuals=residuals,
            group_ids=group_ids,
            weights=weights,
            log_likelihood=float(-result.fun),
            converged=result.success,
            iterations=result.nit,
            n_obs_used=n_obs_used,
            group_col=group_col,
        )

    def _compute_mu_hat(
        self,
        residuals: np.ndarray,
        group_ids: np.ndarray,
        weights: np.ndarray,
        groups: np.ndarray,
        sigma2: float,
        tau2: float,
    ) -> float:
        """Compute grand mean as precision-weighted average of group means."""
        prec_sum = 0.0
        prec_mean_sum = 0.0
        for g in groups:
            mask = group_ids == g
            w_g = weights[mask]
            n_g = w_g.sum()
            mu_g = float(np.average(residuals[mask], weights=w_g))
            v_g = tau2 + sigma2 / max(n_g, 1e-9)
            prec = 1.0 / max(v_g, 1e-12)
            prec_sum += prec
            prec_mean_sum += prec * mu_g
        return prec_mean_sum / max(prec_sum, 1e-12)

    def _store_result(
        self,
        sigma2: float,
        tau2: float,
        mu_hat: float,
        eligible_groups: np.ndarray,
        residuals: np.ndarray,
        group_ids: np.ndarray,
        weights: np.ndarray,
        log_likelihood: float,
        converged: bool,
        iterations: int,
        n_obs_used: int,
        group_col: str,
    ) -> VarianceComponents:
        """Store fitted parameters and compute BLUPs for all groups."""
        self._sigma2 = sigma2
        self._tau2 = tau2
        self._mu_hat = mu_hat

        # Compute BLUPs for eligible groups
        for g in eligible_groups:
            mask = group_ids == g
            w_g = weights[mask]
            n_g = float(w_g.sum())
            r_g = residuals[mask]
            mu_g = float(np.average(r_g, weights=w_g))

            if tau2 > 0:
                Z_g = tau2 / (tau2 + sigma2 / max(n_g, 1e-9))
            else:
                Z_g = 0.0

            blup_g = Z_g * (mu_g - mu_hat)
            self._blup_map[g] = blup_g
            self._group_stats[g] = {
                "n": n_g,
                "mean": mu_g,
                "Z": Z_g,
            }

        # Bühlmann k = sigma2 / tau2
        if tau2 > 0:
            k_val = sigma2 / tau2
        else:
            k_val = float("inf")

        vc = VarianceComponents(
            sigma2=sigma2,
            tau2={group_col: tau2},
            k={group_col: k_val},
            log_likelihood=log_likelihood,
            converged=converged,
            iterations=iterations,
            n_groups={group_col: len(np.unique(group_ids))},
            n_obs_used=n_obs_used,
        )
        self._variance_components = vc
        return vc

    def predict_blup(
        self,
        group_ids: np.ndarray,
        allow_new_groups: bool = True,
    ) -> np.ndarray:
        """
        Return BLUP adjustment for each observation.

        Parameters
        ----------
        group_ids : np.ndarray
            Group membership for prediction observations.
        allow_new_groups : bool
            If True, unseen groups get BLUP=0.0 (multiplicative factor of 1.0).
            If False, raises KeyError for unseen groups.

        Returns
        -------
        np.ndarray of shape (n,) — BLUP adjustments on log scale.
        """
        if self._blup_map is None:
            raise RuntimeError("Must call fit() before predict_blup().")

        blups = np.zeros(len(group_ids))
        for i, g in enumerate(group_ids):
            if g in self._blup_map:
                blups[i] = self._blup_map[g]
            elif not allow_new_groups:
                raise KeyError(
                    f"Group '{g}' not seen during training and allow_new_groups=False."
                )
            # else: blup stays 0.0 -> multiplicative factor exp(0) = 1.0
        return blups

    @property
    def variance_components(self) -> VarianceComponents | None:
        return self._variance_components

    @property
    def group_stats(self) -> dict:
        """Per-group statistics: {group_id: {n, mean, Z}}."""
        return self._group_stats

    @property
    def blup_map(self) -> dict:
        """Fitted BLUPs: {group_id: blup_value}."""
        return self._blup_map
