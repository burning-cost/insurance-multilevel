# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-multilevel: Two-Stage CatBoost + REML Random Effects
# MAGIC
# MAGIC This notebook demonstrates the full workflow for pricing with high-cardinality
# MAGIC group factors (brokers, schemes, territories) using Bühlmann-Straub credibility
# MAGIC theory implemented via REML variance components.
# MAGIC
# MAGIC **Dataset:** Synthetic UK motor insurance portfolio
# MAGIC - 5,000 policies distributed across 30 brokers
# MAGIC - True between-broker variance τ² = 0.10 (about 10% of residual variance)
# MAGIC - Individual risk factors: age, vehicle class, NCB, region, mileage

# COMMAND ----------

# MAGIC %pip install insurance-multilevel polars numpy scipy catboost --quiet

# COMMAND ----------

import numpy as np
import polars as pl

from insurance_multilevel import (
    MultilevelPricingModel,
    icc,
    variance_decomposition,
    high_credibility_groups,
    groups_needing_data,
    residual_normality_check,
    lift_from_random_effects,
)

print("insurance-multilevel imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Portfolio

# COMMAND ----------

RNG = np.random.default_rng(42)
N = 5000
N_BROKERS = 30
SIGMA2_TRUE = 0.25
TAU2_TRUE = 0.10

# True broker random effects
broker_effects = RNG.normal(0, np.sqrt(TAU2_TRUE), N_BROKERS)
broker_ids = RNG.integers(0, N_BROKERS, N)

# Individual risk features
age = RNG.integers(18, 80, N).astype(float)
vehicle_class = RNG.choice(["A", "B", "C", "D"], N)
ncb = RNG.integers(0, 6, N).astype(float)
region_risk = RNG.uniform(0.8, 1.5, N)
annual_mileage = RNG.integers(5000, 30000, N).astype(float)

vehicle_class_effect = {"A": 0.0, "B": 0.15, "C": 0.30, "D": 0.50}
vc_num = np.array([vehicle_class_effect[v] for v in vehicle_class])

log_base = (
    5.0
    + 0.01 * (age - 40)
    + vc_num
    - 0.05 * ncb
    + 0.3 * np.log(region_risk)
    + 0.0001 * annual_mileage
    + broker_effects[broker_ids]
)

noise = RNG.normal(0, np.sqrt(SIGMA2_TRUE), N)
y = np.exp(log_base + noise)
exposure = RNG.uniform(0.5, 1.5, N)

# Build Polars DataFrame
df = pl.DataFrame({
    "age": age,
    "vehicle_class": vehicle_class.tolist(),
    "ncb": ncb,
    "region_risk": region_risk,
    "annual_mileage": annual_mileage,
    "broker_id": [f"broker_{broker_ids[i]:02d}" for i in range(N)],
    "premium": y,
    "exposure": exposure,
})

print(f"Dataset: {N:,} policies across {N_BROKERS} brokers")
print(f"Mean premium: £{float(df['premium'].mean()):.0f}")
print(f"True tau2={TAU2_TRUE}, sigma2={SIGMA2_TRUE}")
print(f"True ICC={TAU2_TRUE/(TAU2_TRUE + SIGMA2_TRUE):.3f}")
print()
print(df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train/Test Split

# COMMAND ----------

split = int(0.8 * N)
idx = np.arange(N)
RNG.shuffle(idx)
train_idx, test_idx = idx[:split], idx[split:]

X_train = df[train_idx].select(["age", "vehicle_class", "ncb", "region_risk", "annual_mileage", "broker_id"])
X_test = df[test_idx].select(["age", "vehicle_class", "ncb", "region_risk", "annual_mileage", "broker_id"])
y_train = df[train_idx]["premium"].to_numpy()
y_test = df[test_idx]["premium"].to_numpy()
w_train = df[train_idx]["exposure"].to_numpy()
w_test = df[test_idx]["exposure"].to_numpy()

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit MultilevelPricingModel

# COMMAND ----------

model = MultilevelPricingModel(
    catboost_params={
        "loss_function": "RMSE",
        "iterations": 400,
        "learning_rate": 0.05,
        "depth": 6,
        "verbose": 0,
    },
    random_effects=["broker_id"],
    min_group_size=5,
    reml=True,
)

model.fit(X_train, y_train, weights=w_train)
print("Model fitted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Variance Components

# COMMAND ----------

vc = model.variance_components["broker_id"]
print(vc)
print()
print(f"Estimated sigma2: {vc.sigma2:.4f}  (true: {SIGMA2_TRUE})")
print(f"Estimated tau2:   {vc.tau2['broker_id']:.4f}  (true: {TAU2_TRUE})")
print(f"Bühlmann k:       {vc.k['broker_id']:.2f}  (obs needed for Z=0.5)")
print(f"REML converged:   {vc.converged} in {vc.iterations} iterations")

icc_est = icc(vc, "broker_id")
icc_true = TAU2_TRUE / (TAU2_TRUE + SIGMA2_TRUE)
print(f"\nEstimated ICC: {icc_est:.3f}  (true: {icc_true:.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Credibility Summary

# COMMAND ----------

cred_summary = model.credibility_summary()
print(f"Credibility summary: {len(cred_summary)} brokers")
print()
print(cred_summary.sort("credibility_weight", descending=True).head(10))

# COMMAND ----------

# Distribution of credibility weights
z_values = cred_summary["credibility_weight"].to_numpy()
print(f"Z weight distribution:")
print(f"  min:    {z_values.min():.3f}")
print(f"  median: {np.median(z_values):.3f}")
print(f"  max:    {z_values.max():.3f}")
print(f"  mean:   {z_values.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. High-Credibility Brokers

# COMMAND ----------

high_cred = high_credibility_groups(cred_summary, min_z=0.5)
print(f"{len(high_cred)} brokers have Z >= 0.5")
print()
print(high_cred[["group", "n_obs", "credibility_weight", "multiplier"]].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Data Needs Analysis

# COMMAND ----------

needs = groups_needing_data(cred_summary, target_z=0.8)
print("Brokers needing the most additional data to reach 80% credibility:")
print()
print(needs.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Residual Diagnostics

# COMMAND ----------

resids = model.log_ratio_residuals(X_train, y_train)
norm_stats = residual_normality_check(resids)
print("Log-ratio residual statistics:")
for k, v in norm_stats.items():
    print(f"  {k:20s}: {v:.4f}")

print()
print("Interpretation:")
print(f"  |skewness| = {abs(norm_stats['skewness']):.3f}  (< 0.5 is good for REML)")
print(f"  |excess kurtosis| = {abs(norm_stats['excess_kurtosis']):.3f}  (< 1.0 is acceptable)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Lift from Random Effects

# COMMAND ----------

stage1_preds = model.stage1_predict(X_test)
final_preds = model.predict(X_test)

lift = lift_from_random_effects(y_test, stage1_preds, final_preds, weights=w_test)
print("Lift from random effects (Stage 2 vs Stage 1):")
print(f"  Stage 1 RMSE:        {lift['stage1_rmse']:.2f}")
print(f"  Final RMSE:          {lift['final_rmse']:.2f}")
print(f"  RMSE improvement:    {lift['rmse_improvement_pct']:.1f}%")
print(f"  Stage 1 MALR:        {lift['stage1_malr']:.4f}")
print(f"  Final MALR:          {lift['final_malr']:.4f}")
print(f"  MALR improvement:    {lift['malr_improvement_pct']:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. New Broker at Prediction Time

# COMMAND ----------

# Simulate a brand-new broker joining mid-year
X_new_broker = X_test.head(20).with_columns(
    pl.lit("brand_new_broker_99").alias("broker_id")
)

preds_new = model.predict(X_new_broker, allow_new_groups=True)
preds_stage1 = model.stage1_predict(X_new_broker)

ratio = preds_new / preds_stage1
print("New broker predictions vs Stage 1 (ratio should be 1.0):")
print(f"  Mean ratio: {float(np.mean(ratio)):.6f}")
print(f"  Max deviation: {float(np.max(np.abs(ratio - 1.0))):.8f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Variance Decomposition

# COMMAND ----------

vd = variance_decomposition(model.variance_components)
print("Variance decomposition:")
print(vd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The model correctly:
# MAGIC - Estimates σ² and τ² close to the true generating values
# MAGIC - Computes Z weights proportional to group size
# MAGIC - Shrinks small-data brokers toward the grand mean
# MAGIC - Applies unit multiplier to new brokers
# MAGIC - Provides actionable credibility summaries for pricing teams
