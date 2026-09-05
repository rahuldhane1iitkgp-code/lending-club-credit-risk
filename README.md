---
title: Lending Club Credit Risk Model
emoji: 💳
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# Lending Club Credit Risk Model

A credit risk scoring model that ends in a **lending decision**, not just a probability.

XGBoost, isotonically calibrated, with an approve/reject threshold chosen to maximise net profit under an explicit expected-loss framework — not to maximise accuracy or F1. Built on 39 application-time features that a borrower or loan officer could realistically supply, retaining **94.3% of the test PR-AUC** of a fuller 151-feature research model.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/rahuldhane/lending-club-credit-risk)

---

## The problem this is actually solving

Most credit-risk portfolio projects stop at "here is a model with AUC 0.7". That output is not a decision. Two questions remain unanswered:

1. **At what probability do you decline a loan?** Accuracy and F1 treat a false positive and a false negative as equally costly. In lending they are not remotely equal: rejecting a good borrower costs you the margin on one loan, while approving a defaulter costs you a large fraction of the principal.
2. **Can you actually collect these features at application time?** A model that leans on `int_rate`, `grade`, or `sub_grade` is using Lending Club's *own* risk assessment as an input. It scores well and is useless — those fields do not exist until after someone has already priced the risk.

This project answers both.

## Approach

**Two models, deliberately.**

| Model | Features | Test PR-AUC | Test ROC-AUC | Brier |
|---|---|---|---|---|
| A — research | 151 | **0.4132** | — | — |
| B1 — deployment, no region | 36 | 0.3873 | 0.6930 | 0.1633 |
| **B2 — deployment, shipped** | **39** | **0.3896** | 0.6946 | 0.1630 |

Model B2 retains **94.3%** of Model A's test PR-AUC using a quarter of the features, none of which require post-origination information. Quantifying that gap is the point: a 5.7% performance cost to become deployable is a tradeoff you can defend in a room; an unquantified one is not.

Adding US region to B1 was worth **+0.0023 PR-AUC** — a real but marginal gain, kept because it costs the applicant nothing to provide.

**Leakage control.** `int_rate`, `grade`, and `sub_grade` are excluded throughout — they encode the platform's existing risk judgement. `combined_fico_low`/`combined_fico_high` are collapsed into a single `fico_score`.

**Class imbalance.** `scale_pos_weight` with `aucpr` as the eval metric, and **PR-AUC as the headline metric rather than ROC-AUC** — with a minority default class, ROC-AUC is flattered by the large negative class.

**Calibration.** Isotonic regression over the raw XGBoost scores. Necessary because the threshold logic below operates on probabilities: an uncalibrated score of 0.15 is not a 15% default rate, and the expected-loss arithmetic silently breaks if you pretend otherwise. Reported alongside Brier score.

**The decision layer.** For a sweep of thresholds, over the loans the policy would approve:

```
net = Σ (principal × margin)   over approved loans that repaid
    − Σ (principal × LGD)      over approved loans that defaulted
```

with `LGD = 50%` and `profit margin = 10%` as stated assumptions.

| Threshold | Approval rate | Net |
|---|---|---|
| 0.05 | 3.7% | $6.0M |
| 0.10 | 15.5% | $18.9M |
| 0.12 | 22.6% | $22.7M |
| **0.15** | **31.0%** | **$23.8M** |
| 0.18 | 41.1% | $21.9M |
| 0.25 | 61.8% | $4.4M |
| 0.30 | 73.4% | **−$13.6M** |

**0.15 is the shipped threshold** — approve roughly the safest 31% of applicants. The curve is the argument: profit peaks and then collapses, turning negative by a 73% approval rate. A model tuned for accuracy or F1 would sit far out on the right-hand side of this table, approving loans that destroy money. The decision rule, not the classifier, is what makes this profitable.

**Explanation.** Per-prediction SHAP contributions, so the app returns a reason alongside a decision.

## Repo layout

| File | Purpose |
|---|---|
| `model_b_step1.py` | Feature construction — region mapping, FICO collapse, deployment feature set |
| `model_b_step2.py` | Trains B1/B2, calibrates, evaluates vs. Model A, picks the winner |
| `model_b_step3.py` | Expected-loss threshold sweep |
| `save_deployment_artifacts.py` | Bundles model, features, threshold, metrics into one artifact |
| `app.py` | Streamlit app — form → calibrated probability → decision → SHAP reasons |
| `test_app_logic.py` | Reproduces the app's feature assembly headlessly; includes a risky-applicant sanity check |
| `model_a_research.ipynb` | Model A research work (EDA, full feature set) |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app needs only `deployment_artifacts.joblib`, which is committed. Retraining from scratch additionally requires the [Lending Club accepted-loans dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (~1.6 GB, not committed).

## Known limitations

Stated rather than hidden, because they are the things a reviewer should ask about:

- **The threshold is selected on the test set.** The sweep in `model_b_step3.py` picks the profit-maximising threshold using test-set outcomes, so the reported net-profit figure is optimistically biased. The *ranking* metrics (PR-AUC, ROC-AUC) are unaffected — those are clean test estimates. Selecting the threshold on validation and reporting profit on test is the correct fix.
- **Calibration and early stopping share the validation set.** Tree count is chosen on validation, then isotonic calibration is fit on that same set, which makes calibration quality somewhat optimistic.
- **LGD and margin are assumed, not measured.** 50% and 10% are plausible industry figures, not values derived from lender data. The threshold moves if they do.
- **SHAP explains the uncalibrated model.** Isotonic calibration is monotonic, so the direction and ranking of contributions hold, but the magnitudes are on the raw score's scale.
- **No temporal validation.** Splits are random rather than out-of-time, so the estimates do not capture macroeconomic drift across the 2007–2018 window.

This is a portfolio project. The assumptions are documented so the numbers can be judged against them.
