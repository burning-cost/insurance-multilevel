"""Tests for VarianceComponents dataclass."""

import math
import pytest
from insurance_multilevel import VarianceComponents


def test_variance_components_basic():
    vc = VarianceComponents(
        sigma2=0.25,
        tau2={"broker_id": 0.10},
        k={"broker_id": 2.5},
        log_likelihood=-123.4,
        converged=True,
        iterations=12,
    )
    assert vc.sigma2 == pytest.approx(0.25)
    assert vc.tau2["broker_id"] == pytest.approx(0.10)
    assert vc.k["broker_id"] == pytest.approx(2.5)
    assert vc.converged is True
    assert vc.iterations == 12


def test_variance_components_repr():
    vc = VarianceComponents(
        sigma2=0.25,
        tau2={"broker_id": 0.10},
        k={"broker_id": 2.5},
        log_likelihood=-100.0,
        converged=True,
        iterations=5,
    )
    r = repr(vc)
    assert "sigma2=" in r
    assert "tau2=" in r
    assert "converged=True" in r


def test_variance_components_zero_tau2():
    """tau2=0 is a valid boundary case (no between-group variation)."""
    vc = VarianceComponents(
        sigma2=0.30,
        tau2={"broker_id": 0.0},
        k={"broker_id": float("inf")},
        log_likelihood=-50.0,
        converged=True,
        iterations=3,
    )
    assert vc.tau2["broker_id"] == 0.0
    assert math.isinf(vc.k["broker_id"])


def test_variance_components_multiple_levels():
    vc = VarianceComponents(
        sigma2=0.20,
        tau2={"broker_id": 0.08, "scheme_id": 0.15},
        k={"broker_id": 2.5, "scheme_id": 1.33},
        log_likelihood=-200.0,
        converged=True,
        iterations=20,
        n_groups={"broker_id": 50, "scheme_id": 30},
        n_obs_used=5000,
    )
    assert len(vc.tau2) == 2
    assert vc.n_groups["broker_id"] == 50
    assert vc.n_obs_used == 5000


def test_variance_components_defaults():
    vc = VarianceComponents(
        sigma2=0.1,
        tau2={},
        k={},
        log_likelihood=0.0,
        converged=False,
        iterations=0,
    )
    assert vc.n_groups == {}
    assert vc.n_obs_used == 0
