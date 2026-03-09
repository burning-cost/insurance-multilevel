"""
Type definitions for insurance-multilevel.

These types expose Bühlmann-Straub vocabulary deliberately.
The Z (credibility weight) is the same concept whether you derive it from
classical credibility theory or from a REML variance components model —
they're algebraically equivalent for the one-way random effects model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VarianceComponents:
    """
    Variance component estimates from REML fitting.

    Attributes
    ----------
    sigma2 : float
        Within-group variance (residual variance after removing group means).
    tau2 : dict[str, float]
        Between-group variance, keyed by group column name.
        tau2[g] = 0.0 when there is no detectable between-group variation —
        this is the boundary singularity case and signals that all groups
        behave identically on the log-residual scale.
    k : dict[str, float]
        Bühlmann k = sigma2 / tau2, one per level. This is the number of
        observations at which a group's credibility weight reaches 0.5.
        Larger k means more observations are needed before a group's
        experience dominates the grand mean.
    log_likelihood : float
        Restricted log-likelihood value at convergence (REML objective).
        Comparable across models with the same fixed effects but different
        random effects specifications.
    converged : bool
        Whether the REML optimiser converged within tolerance.
    iterations : int
        Number of optimiser iterations consumed.
    n_groups : dict[str, int]
        Number of unique groups per level (informational).
    n_obs_used : int
        Number of observations used in REML fitting (after min_group_size
        filter is applied).
    """

    sigma2: float
    tau2: dict[str, float]
    k: dict[str, float]
    log_likelihood: float
    converged: bool
    iterations: int
    n_groups: dict[str, int] = field(default_factory=dict)
    n_obs_used: int = 0

    def __repr__(self) -> str:
        tau_str = ", ".join(
            f"{col}={v:.4f}" for col, v in self.tau2.items()
        )
        k_str = ", ".join(
            f"{col}={v:.1f}" for col, v in self.k.items()
        )
        return (
            f"VarianceComponents("
            f"sigma2={self.sigma2:.4f}, "
            f"tau2={{{tau_str}}}, "
            f"k={{{k_str}}}, "
            f"log_likelihood={self.log_likelihood:.4f}, "
            f"converged={self.converged}, "
            f"iterations={self.iterations})"
        )
