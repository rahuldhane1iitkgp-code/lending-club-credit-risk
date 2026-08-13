---
title: Lending Club Credit Risk Model
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Lending Club Credit Risk Model

A deployment-scoped credit risk scoring model (XGBoost, calibrated, cost-sensitive threshold), built on 39 realistically-collectible application-time features. Retains 94.3% of the test PR-AUC of a fuller 151-feature research model while using only inputs a borrower or loan officer could realistically provide.

Includes a per-prediction SHAP explanation and a lending decision at a profit-optimized threshold (not accuracy/F1-optimized), derived from an explicit, labeled expected-loss framework.

This is a portfolio/demo project — assumptions (loss given default, profit margin) are explicitly stated, not derived from proprietary lender data.
