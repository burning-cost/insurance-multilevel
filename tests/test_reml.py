"""
Tests for RandomEffectsEstimator.

Focus on:
1. Variance component recovery: estimated sigma2, tau2 should be in the
   right ballpark when given data from a known DGP.
2. Boundary cases: tau2=0 (no group effects), n=1 groups, min_group_size filter.
3. BLUP direction: groups with high residuals should have positive BLUPs,
   groups with low residuals should have negative BLUPs.
4. New group handling: predict_blup returns 0 for unseen groups.
"""

import numpy as np
import pytest
from insurance_multilevel import RandomEffectsEstimator, VarianceComponents


RNG = np.random.default_rng(123)


def _make_residuals(
    n_groups: int = 15,
    n_per_group: int = 50,
    sigma2: float = 0.25,
    tau2: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Generate residuals from a known one-way random effects DGP."""
    rng = np.random.default_rng(seed)
    group_effects = rng.normal(0, np.sqrt(tau2), n_groups)
    group_ids = np.repeat(np.arange(n_groups), n_per_group).astype(str)
    noise = rng.normal(0, np.sqrt(sigma2), n_groups * n_per_group)
    residuals = group_effects[np.repeat(np.arange(n_groups), n_per_group)] + noise
    return residuals, group_ids, group_effects


def test_fit_returns_variance_components():
    residuals, group_ids, _ = _make_residuals()
    est = RandomEffectsEstimator(min_group_size=5)
    vc = est.fit(residuals, group_ids, group_col="broker")

    assert isinstance(vc, VarianceComponents)
    assert vc.sigma2 > 0
    assert vc.tau2["broker"] >= 0
    assert vc.log_likelihood != 0.0
    assert isinstance(vc.converged, bool)
    assert vc.iterations > 0


def test_sigma2_recovery():
    """sigma2 estimate should be within 50% of truth for n=750 observations."""
    sigma2_true = 0.25
    tau2_true = 0.10
    residuals, group_ids, _ = _make_residuals(
        n_groups=15, n_per_group=50, sigma2=sigma2_true, tau2=tau2_true
    )
    est = RandomEffectsEstimator(min_group_size=5)
    vc = est.fit(residuals, group_ids)

    assert vc.sigma2 == pytest.approx(sigma2_true, rel=0.5), (
        f"sigma2 estimate {vc.sigma2:.4f} is more than 50% off from {sigma2_true}"
    )


def test_tau2_recovery():
    """tau2 estimate should be positive and in the right ballpark."""
    sigma2_true = 0.25
    tau2_true = 0.15  # Use larger tau2 for easier recovery
    residuals, group_ids, _ = _make_residuals(
        n_groups=20, n_per_group=100, sigma2=sigma2_true, tau2=tau2_true, seed=99
    )
    est = RandomEffectsEstimator(min_group_size=5)
    vc = est.fit(residuals, group_ids)

    # tau2 should be positive (not clamped to zero)
    assert vc.tau2.get("group", vc.tau2.get(list(vc.tau2.keys())[0], 0)) > 0, (
        "tau2 should be positive when true tau2=0.15"
    )


def test_tau2_zero_when_no_group_variation():
    """When all groups have the same mean, tau2 should be near zero."""
    rng = np.random.default_rng(42)
    n_groups = 10
    n_per_group = 30
    # Pure within-group noise, no between-group variation
    residuals = rng.normal(0, 0.5, n_groups * n_per_group)
    group_ids = np.repeat(np.arange(n_groups), n_per_group).astype(str)

    est = RandomEffectsEstimator(min_group_size=5)
    vc = est.fit(residuals, group_ids)

    # tau2 should be small or zero
    tau2 = vc.tau2.get("group", vc.tau2.get(list(vc.tau2.keys())[0] if vc.tau2 else "group", 0))
    assert tau2 < 0.15, f"tau2={tau2:.4f} should be small when true tau2=0"


def test_blup_direction():
    """Groups with positive mean residuals should have positive BLUPs."""
    residuals, group_ids, group_effects = _make_residuals(
        n_groups=10, n_per_group=100, sigma2=0.10, tau2=0.20, seed=77
    )
    est = RandomEffectsEstimator(min_group_size=5)
    est.fit(residuals, group_ids)

    blups = est.blup_map
    # Check sign agreement with group effects (shrunk but same direction)
    groups = np.unique(group_ids)
    for g in groups:
        mask = group_ids == g
        group_mean = float(np.mean(residuals[mask]))
        blup = blups.get(str(g), blups.get(g, 0.0))
        if abs(group_mean) > 0.2:  # only check groups with clear signal
            assert np.sign(blup) == np.sign(group_mean), (
                f"Group {g}: BLUP {blup:.3f} has wrong sign vs mean {group_mean:.3f}"
            )


def test_blup_shrinkage():
    """BLUPs should be shrunk toward zero: |blup| <= |group_mean|."""
    residuals, group_ids, _ = _make_residuals(
        n_groups=10, n_per_group=20, sigma2=0.25, tau2=0.10
    )
    est = RandomEffectsEstimator(min_group_size=5)
    est.fit(residuals, group_ids)

    groups = np.unique(group_ids)
    for g in groups:
        if g not in est.blup_map:
            continue
        mask = group_ids == g
        group_mean = float(np.mean(residuals[mask]))
        blup = est.blup_map[g]
        assert abs(blup) <= abs(group_mean) + 1e-6, (
            f"Group {g}: BLUP {blup:.4f} exceeds group mean {group_mean:.4f} (no shrinkage)"
        )


def test_predict_blup_unseen_group_returns_zero():
    """New groups not seen at training time should get BLUP=0."""
    residuals, group_ids, _ = _make_residuals()
    est = RandomEffectsEstimator(min_group_size=5)
    est.fit(residuals, group_ids)

    new_ids = np.array(["unseen_broker_A", "unseen_broker_B"])
    blups = est.predict_blup(new_ids, allow_new_groups=True)
    assert np.all(blups == 0.0)


def test_predict_blup_new_group_raises_when_not_allowed():
    """allow_new_groups=False should raise KeyError for unseen groups."""
    residuals, group_ids, _ = _make_residuals()
    est = RandomEffectsEstimator(min_group_size=5)
    est.fit(residuals, group_ids)

    with pytest.raises(KeyError, match="not seen during training"):
        est.predict_blup(np.array(["ghost_broker"]), allow_new_groups=False)


def test_min_group_size_filter():
    """Groups below min_group_size should have Z=0 in group_stats."""
    rng = np.random.default_rng(42)
    # Groups: 0-9 have 50 obs, group 10 has only 2 obs
    residuals = np.concatenate([
        rng.normal(0, 0.5, 50 * 10),
        rng.normal(0.5, 0.5, 2),  # tiny group
    ])
    group_ids = np.concatenate([
        np.repeat(np.arange(10), 50).astype(str),
        np.array(["tiny_group", "tiny_group"]),
    ])

    est = RandomEffectsEstimator(min_group_size=5)
    est.fit(residuals, group_ids)

    tiny_stats = est.group_stats.get("tiny_group")
    assert tiny_stats is not None, "tiny_group should appear in group_stats"
    assert tiny_stats["Z"] == 0.0, (
        f"tiny_group Z={tiny_stats['Z']} should be 0.0 (below min_group_size)"
    )


def test_singleton_group_handled():
    """Single-observation groups should not cause errors."""
    rng = np.random.default_rng(42)
    residuals = np.concatenate([
        rng.normal(0, 0.5, 50 * 5),
        np.array([1.5]),  # n=1 group
    ])
    group_ids = np.concatenate([
        np.repeat(np.arange(5), 50).astype(str),
        np.array(["singleton"]),
    ])

    est = RandomEffectsEstimator(min_group_size=2)
    vc = est.fit(residuals, group_ids)

    assert vc.sigma2 > 0


def test_weighted_fit():
    """Weighted fitting should not raise and should return valid estimates."""
    residuals, group_ids, _ = _make_residuals(n_groups=10, n_per_group=40)
    rng = np.random.default_rng(42)
    weights = rng.uniform(0.5, 2.0, len(residuals))

    est = RandomEffectsEstimator(min_group_size=5)
    vc = est.fit(residuals, group_ids, weights=weights)

    assert vc.sigma2 > 0
    assert vc.tau2.get("group", 0.0) >= 0


def test_predict_blup_shape():
    """predict_blup should return same length as input."""
    residuals, group_ids, _ = _make_residuals(n_groups=10, n_per_group=30)
    est = RandomEffectsEstimator(min_group_size=5)
    est.fit(residuals, group_ids)

    test_ids = group_ids[:20]
    blups = est.predict_blup(test_ids)
    assert len(blups) == 20


def test_buhlmann_k_consistency():
    """k = sigma2 / tau2 should equal ratio of stored values."""
    residuals, group_ids, _ = _make_residuals(n_groups=15, n_per_group=50)
    est = RandomEffectsEstimator(min_group_size=5)
    vc = est.fit(residuals, group_ids)

    gcol = list(vc.tau2.keys())[0]
    tau2 = vc.tau2[gcol]
    if tau2 > 1e-8:
        expected_k = vc.sigma2 / tau2
        assert vc.k[gcol] == pytest.approx(expected_k, rel=1e-4)


def test_fit_before_predict_raises():
    """predict_blup before fit should raise RuntimeError."""
    est = RandomEffectsEstimator()
    with pytest.raises(RuntimeError, match="Must call fit"):
        est.predict_blup(np.array(["broker_0"]))


def test_fewer_than_2_eligible_groups_warning():
    """Should warn when fewer than 2 groups meet min_group_size."""
    rng = np.random.default_rng(42)
    residuals = np.concatenate([
        rng.normal(0, 0.5, 50),   # one eligible group
        rng.normal(0, 0.5, 2),    # below threshold
    ])
    group_ids = np.concatenate([
        np.repeat(["big_group"], 50),
        np.repeat(["small_group"], 2),
    ])

    est = RandomEffectsEstimator(min_group_size=5)
    with pytest.warns(UserWarning, match="Fewer than 2 groups"):
        vc = est.fit(residuals, group_ids)

    assert vc.tau2.get("group", 0.0) == 0.0
