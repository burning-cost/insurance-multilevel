# Changelog

## v0.1.2 (2026-03-22) [unreleased]
- refactor: convert benchmark to Databricks notebook format
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.1.2 (2026-03-21)
- docs: replace pip install with uv add in README
- Add blog post link and community CTA to README
- Add MIT license
- Add Databricks date and fix formatting in Performance section
- Fix P1/P2 issues from QA audit: min_group_size docs, credibility_summary shape note, version sync, remove duplicate section
- refresh benchmark numbers post-P0 fixes
- Fix P0/P1 bugs in REML likelihood and multi-level fitting (v0.1.2)
- Add standalone benchmark: TwoStageMultilevel vs plain CatBoost
- Add benchmark: MultilevelPricingModel vs one-hot encoding and no group effect
- fix: bump scipy to >=1.10 — drop upper cap that blocked Python 3.12 wheels
- docs: add Databricks notebook link
- Fix: pin scipy<1.11 for Databricks serverless compat
- Add Related Libraries section to README
- fix: update cross-references to consolidated repos
- fix: update polars floor to >=1.0 and fix project URLs

