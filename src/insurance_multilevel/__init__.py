"""
insurance-multilevel: two-stage CatBoost + REML random effects for insurance pricing.

Solves the high-cardinality group factor problem: how do you incorporate broker,
scheme, or territory effects when you have hundreds of groups, each with varying
amounts of data, and one-hot encoding would cause sparsity problems?

The answer is classical Bühlmann-Straub credibility theory, re-derived from a
Bayesian random effects perspective and combined with a CatBoost fixed-effects
model that handles the individual risk factors.

Quick start
-----------
>>> import polars as pl
>>> from insurance_multilevel import MultilevelPricingModel
>>>
>>> model = MultilevelPricingModel(
...     random_effects=["broker_id"],
...     min_group_size=5,
... )
>>> model.fit(X_train, y_train, weights=exposure, group_cols=["broker_id"])
>>> premiums = model.predict(X_test, group_cols=["broker_id"])
>>> cred_summary = model.credibility_summary()

The credibility_summary() output shows Bühlmann-Straub Z weights per broker —
the same quantity that classical credibility theory computes, but now derived
from a statistically principled REML variance components estimate.
"""

from ._model import MultilevelPricingModel
from ._reml import RandomEffectsEstimator
from ._types import VarianceComponents
from ._diagnostics import (
    icc,
    variance_decomposition,
    high_credibility_groups,
    groups_needing_data,
    residual_normality_check,
    lift_from_random_effects,
)

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-multilevel")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    "MultilevelPricingModel",
    "RandomEffectsEstimator",
    "VarianceComponents",
    "icc",
    "variance_decomposition",
    "high_credibility_groups",
    "groups_needing_data",
    "residual_normality_check",
    "lift_from_random_effects",
]
