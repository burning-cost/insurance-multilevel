# insurance-multilevel
[![Tests](https://github.com/burning-cost/insurance-multilevel/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-multilevel/actions/workflows/tests.yml)

Two-stage CatBoost + REML random effects for insurance pricing with high-cardinality group factors.

## The Problem

UK personal lines pricing teams face a specific structural problem: portfolios are distributed through hundreds of brokers, schemes, and affinity partners. These groups differ systematically — broker A has an older, lower-risk customer base; broker B has young drivers; scheme C operates in flood-prone postcodes. But you cannot capture this by throwing broker IDs into a GBM.

The reasons one-hot encoding fails at scale:
- 500 brokers means 500 extra features, most with sparse data
- A GBM with 300 trees will overfit to the largest brokers and ignore the rest
- New brokers at prediction time have no training data

What you actually need is **shrinkage**: for a new or low-volume broker, trust the book-wide average. For a high-volume broker with years of data, trust their own experience. The crossover point is determined by how variable brokers are relative to within-group noise.

This is classical Bühlmann-Straub credibility theory, reimplemented as a statistically principled REML random effects model. If you want the classical closed-form version — for scheme pricing or geographic rating without a GBM stage — see [insurance-credibility](https://github.com/burning-cost/insurance-credibility), which implements `BuhlmannStraub` and `HierarchicalBuhlmannStraub` directly.

## The Solution

Two stages, run sequentially:

**Stage 1: CatBoost on individual risk factors**
Group columns (broker, scheme) are deliberately excluded. CatBoost learns age bands, vehicle classes, postcode sectors, NCB — everything about the *individual* policy. Output: f̂ᵢ (CatBoost predicted premium).

**Stage 2: REML random intercepts on log-ratio residuals**
Compute rᵢ = log(yᵢ / f̂ᵢ) — the log-ratio of observed to CatBoost-predicted. Fit a one-way random effects model:

```
rᵢ = μ + b_g(i) + εᵢ
b_g ~ N(0, τ²)      (between-group variation)
εᵢ  ~ N(0, σ²)      (within-group noise)
```

REML estimates σ² and τ², then computes BLUPs for each group:

```
b̂_g = Z_g × (r̄_g - μ̂)         (shrunk group mean)
Z_g = τ² / (τ² + σ²/n_g)        (Bühlmann credibility weight)
```

**Final premium:** f̂(x) × exp(b̂_g)

## Installation

```bash
pip install insurance-multilevel
```

## Quick Start

```python
import polars as pl
from insurance_multilevel import MultilevelPricingModel

model = MultilevelPricingModel(
    catboost_params={"loss_function": "RMSE", "iterations": 500},
    random_effects=["broker_id", "scheme_id"],
    min_group_size=5,
)

model.fit(X_train, y_train, weights=exposure, group_cols=["broker_id", "scheme_id"])

premiums = model.predict(X_test, group_cols=["broker_id", "scheme_id"])
```

## Credibility Summary

```python
summary = model.credibility_summary()
print(summary)
```

```
shape: (47, 11)
┌───────────┬─────────────┬────────┬────────────┬────────┬────────────┬───────────────────┬───────┬────────┬──────┬──────────┐
│ level     ┆ group       ┆ n_obs  ┆ group_mean ┆ blup   ┆ multiplier ┆ credibility_weight┆ tau2  ┆ sigma2 ┆ k    ┆ eligible │
│ ---       ┆ ---         ┆ ---    ┆ ---        ┆ ---    ┆ ---        ┆ ---               ┆ ---   ┆ ---    ┆ ---  ┆ ---      │
│ str       ┆ str         ┆ f64    ┆ f64        ┆ f64    ┆ f64        ┆ f64               ┆ f64   ┆ f64    ┆ f64  ┆ bool     │
╞═══════════╪═════════════╪════════╪════════════╪════════╪════════════╪═══════════════════╪═══════╪════════╪══════╪══════════╡
│ broker_id ┆ broker_02   ┆ 342.0  ┆ 0.1823     ┆ 0.1691 ┆ 1.184      ┆ 0.928             ┆ 0.103 ┆ 0.248  ┆ 2.41 ┆ true     │
│ broker_id ┆ broker_07   ┆ 289.0  ┆ -0.1354    ┆-0.1231 ┆ 0.884      ┆ 0.919             ┆ 0.103 ┆ 0.248  ┆ 2.41 ┆ true     │
│ ...       ┆ ...         ┆ ...    ┆ ...        ┆ ...    ┆ ...        ┆ ...               ┆ ...   ┆ ...    ┆ ...  ┆ ...      │
```

The `multiplier` column is what pricing teams use. broker_02 has consistently worse-than-expected experience; apply a 1.184 loading on top of the base premium for policies written through that broker.

The column names — `tau2` (τ², between-group variance), `sigma2` (σ², within-group variance), `k` (Bühlmann's k), `credibility_weight` (Z) — deliberately mirror the notation in classical Bühlmann-Straub theory. See [insurance-credibility](https://github.com/burning-cost/insurance-credibility) for the closed-form version of the same model, where these parameters are explained in detail.

## Variance Components

```python
vc = model.variance_components["broker_id"]
print(vc)
# VarianceComponents(sigma2=0.2481, tau2={broker_id=0.1032},
#                   k={broker_id=2.41}, log_likelihood=-412.3,
#                   converged=True, iterations=23)
```

The **Bühlmann k** tells you the crossover point: a broker needs k=2.41 claim-years of data before their own experience gets more than 50% credibility. With σ²=0.25 and τ²=0.10, this is a reasonable portfolio — brokers do vary, but not outrageously.

## Diagnostics

```python
from insurance_multilevel import (
    icc,
    variance_decomposition,
    high_credibility_groups,
    groups_needing_data,
    lift_from_random_effects,
)

# Intraclass Correlation Coefficient
# "10% of total variance is between-broker"
print(icc(vc, "broker_id"))  # 0.294

# Which brokers have enough data to trust?
hc = high_credibility_groups(model.credibility_summary(), min_z=0.7)

# How much more data does each broker need to reach 80% credibility?
needs = groups_needing_data(model.credibility_summary(), target_z=0.8)

# Does Stage 2 actually help?
stage1 = model.stage1_predict(X_test)
final = model.predict(X_test)
lift = lift_from_random_effects(y_test, stage1, final, weights=exposure_test)
print(lift["malr_improvement_pct"])  # e.g., 4.2% improvement
```

## Design Choices

**Why two stages instead of joint estimation?**
Joint approaches (GPBoost, MERF) are mathematically cleaner but have identifiability problems when group IDs are high-cardinality. If broker_id is in the CatBoost feature set, the tree can absorb *some* of the group signal, leaving underestimated τ² in Stage 2. Two-stage with group exclusion is simpler and avoids this.

**Why REML instead of ML?**
Maximum likelihood underestimates variance components because it doesn't account for the degrees of freedom consumed by fixed effects (the grand mean μ). REML conditions out μ first. For small numbers of groups (m < 30), the difference is material. For m > 100, ML and REML converge.

**Why log-ratio residuals?**
Insurance premia are multiplicative. A broker loading of 1.15 applies regardless of whether the base premium is £300 or £1,200. By working on the log scale, we get additive random effects that translate cleanly to multiplicative adjustments.

**Why min_group_size=5?**
With n=1, you cannot separate the group random effect from within-group noise. The BLUP for a singleton group would be either 0 (correct: no information) or dominated by that single extreme observation (wrong: no shrinkage possible without group-level data). We exclude singletons from variance estimation and give them Z=0. This is conservative and actuarially correct.

## API Reference

### `MultilevelPricingModel`

```python
MultilevelPricingModel(
    catboost_params: dict | None = None,
    random_effects: list[str] | None = None,
    min_group_size: int = 5,
    reml: bool = True,
)
```

Methods:
- `fit(X, y, weights, group_cols)` — fit two-stage model
- `predict(X, group_cols, allow_new_groups)` — return premium predictions
- `credibility_summary(group_col)` — Bühlmann-Straub summary DataFrame
- `stage1_predict(X)` — CatBoost predictions only (no random effects)
- `log_ratio_residuals(X, y)` — log(y / f_hat) for diagnostics
- `variance_components` — dict of VarianceComponents per group level
- `feature_importances` — Stage 1 CatBoost feature importances

### `RandomEffectsEstimator`

Lower-level class if you want to use REML variance components without CatBoost.

```python
est = RandomEffectsEstimator(reml=True, min_group_size=5)
vc = est.fit(residuals, group_ids, weights)
blups = est.predict_blup(group_ids)
```

### `VarianceComponents`

Dataclass holding σ², τ², k (Bühlmann), log-likelihood, convergence info.

## Scope (V1)

- Random intercepts only (no random slopes)
- Gaussian residuals on log-transformed response
- Two-stage estimation (not joint)
- Nested hierarchy supported: fit separate estimators per group level
- Crossed effects excluded (broker × territory combinations)
- One-way random effects per group column

## Performance

Benchmarked on a synthetic UK motor portfolio: 5,000 policies across 30 brokers, true τ²=0.10, σ²=0.25, ICC=0.286. Full notebook: `notebooks/multilevel_demo.py`.

**Variance component recovery.** REML estimates on 4,000 training observations:

| Parameter | True | Estimated |
|-----------|------|-----------|
| σ² (within-broker) | 0.25 | typically 0.24–0.26 |
| τ² (between-broker) | 0.10 | typically 0.09–0.11 |
| ICC | 0.286 | typically 0.27–0.30 |
| Bühlmann k | 2.5 | typically 2.3–2.7 |

REML converges in under 30 iterations on this dataset. The estimates are close to the true values, confirming that the two-stage approach does not materially distort variance component estimation.

**Lift from Stage 2.** On the held-out test set (1,000 policies), adding broker random effects to the Stage 1 CatBoost predictions:

- RMSE improvement: 2–6% depending on broker distribution in the test set
- MALR improvement: 2–5%
- New brokers: multiplier = 1.000 exactly (correct fallback to Stage 1)

The lift is proportional to the ICC. On this synthetic portfolio with ICC≈0.29, the improvement is moderate. On a real UK portfolio where certain brokers systematically over- or under-perform by 20–30%, the lift is larger.

**Credibility weight distribution.** With k=2.41, a broker needs ~50 observations for Z=0.95, and ~3 observations for Z=0.5. On a 30-broker portfolio of 5,000 policies (mean ~167 obs/broker), most brokers reach high credibility and their own experience dominates.

**When to use:** Portfolios where group-level (broker, scheme, territory) effects are material but group membership cannot simply be included as a GBM feature due to cardinality, sparsity, or new groups at prediction time.

**When NOT to use:** When group effects are small (ICC < 0.05) or when you have fewer than 3–5 observations per group — in that case the REML estimate of τ² will be unreliable and BLUPs collapse to the grand mean regardless.

---

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub closed-form implementation — use this when you don't need a GBM Stage 1 |
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from the Stage 1 CatBoost model |
| [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 spatial random effects for territory — the geographic equivalent of broker random effects |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure, needed for validating the Stage 1 model |

[All Burning Cost libraries →](https://burning-cost.github.io)

## Read more

[Your Broker Adjustments Are Guesswork](https://burning-cost.github.io/blog/your-broker-adjustments-are-guesswork) — why ad hoc broker loadings fail and how REML random effects give you defensible, data-driven credibility weights.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility — the closed-form classical alternative when a GBM stage is not needed |
| [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) | GLM tooling including factor merging — use to reduce dimensionality of the group factor before adding random effects |

## Licence

MIT
