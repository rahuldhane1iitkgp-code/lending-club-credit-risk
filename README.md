# Lending Club Credit Risk Model

A credit risk scoring model that ends in a **lending decision**, not just a probability.

XGBoost, isotonically calibrated, with an approve/reject threshold chosen to maximise net profit under an explicit expected-loss framework — not to maximise accuracy or F1. Built on 39 application-time features that a borrower or loan officer could realistically supply, retaining **94.3% of the test PR-AUC** of a fuller 151-feature research model.

**Live demo:** [lending-club-credit-risk.streamlit.app](https://lending-club-credit-risk-dyvypcroo68hgvdpjlgfpm.streamlit.app/)

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

**Out-of-time evaluation.** Splits are by origination year — train ≤2015 (829,355 loans), validate 2016 (293,105), test 2017 (169,321) — not random. A random split would let the model train on 2016 and be scored on 2010, which no deployed model ever gets to do.

**Leakage control.** `int_rate`, `grade`, and `sub_grade` are excluded throughout — they encode the platform's existing risk judgement. `combined_fico_low`/`combined_fico_high` are collapsed into a single `fico_score`.

**Class imbalance.** `scale_pos_weight` with `aucpr` as the eval metric, and **PR-AUC as the headline metric rather than ROC-AUC** — with a minority default class, ROC-AUC is flattered by the large negative class.

**Calibration.** Isotonic regression over the raw XGBoost scores. Necessary because the threshold logic below operates on probabilities: an uncalibrated score of 0.15 is not a 15% default rate, and the expected-loss arithmetic silently breaks if you pretend otherwise. Reported alongside Brier score.

**The decision layer.** For a sweep of thresholds, over the loans the policy would approve:

```
net = Σ (principal × margin)   over approved loans that repaid
    − Σ (principal × LGD)      over approved loans that defaulted
```

with `LGD = 50%` and `profit margin = 10%` as stated assumptions.

**The threshold is a fitted parameter, so it gets its own held-out data.** Selecting it on test and then reporting profit at that threshold reports the maximum of a noisy surface — a biased number. Validation is therefore split in two:

```
val_cal (146,552 rows)  ->  fit isotonic calibration
val_thr (146,553 rows)  ->  sweep and select the threshold
test                    ->  report profit at that fixed threshold
```

Selected threshold: **0.16**. Test performance at it, reported once:

| Threshold | Approval rate | Test net |
|---|---|---|
| 0.05 | 3.8% | $6.1M |
| 0.11 | 18.9% | $20.8M |
| 0.14 | 28.4% | $23.9M |
| **0.16** | **33.9%** | **$23.5M** |
| 0.20 | 41.7% | $21.8M |
| 0.26 | 66.2% | **−$1.7M** |
| 0.32 | 75.9% | **−$19.8M** |

**Approve roughly the safest 34% of applicants.** The shape of the curve is the argument: profit peaks and then collapses, going negative past a ~66% approval rate.

The F1-optimal threshold — the standard ML default — lands at **0.23**, approving 58.2% of applicants for a test net of **$8.9M**. It doesn't lose money, but it leaves **almost two-thirds of the achievable profit on the table** ($8.9M vs $23.5M). Both thresholds are computed on the same calibrated probability scale, which matters: comparing an F1 cutoff derived from raw XGBoost scores against a profit cutoff derived from calibrated ones compares two different units and produces a much more dramatic — and wrong — gap.

The decision rule, not the classifier, is what makes this profitable.

**How much did the honest split cost?** The test-optimal ("oracle") threshold was 0.14 at $23.94M. Selecting on validation and reporting on test gives $23.50M — a **$0.44M gap, 1.8%**. So the selection bias was real but small, which is itself the finding: the profit conclusion is robust to how the threshold was chosen. Measuring that gap is cheaper than arguing about it.

**Explanation.** Per-prediction SHAP contributions, so the app returns a reason alongside a decision.

## Repo layout

| File | Purpose |
|---|---|
| `model_b_step1.py` | Feature construction — region mapping, FICO collapse, deployment feature set |
| `model_b_step2.py` | Trains B1/B2, calibrates, evaluates vs. Model A, picks the winner |
| `model_b_step3.py` | Expected-loss threshold sweep — selects on validation, reports on test |
| `model_a_threshold.py` | Same procedure applied to the 151-feature research model |
| `save_deployment_artifacts.py` | Bundles model, features, threshold, metrics into one artifact |
| `app.py` | Streamlit app — form → calibrated probability → decision → SHAP reasons |
| `test_app_logic.py` | Reproduces the app's feature assembly headlessly; includes a risky-applicant sanity check |
| `model_a_research.ipynb` | Model A research work (EDA, full feature set) |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deployed on Streamlit Community Cloud, which rebuilds automatically on every push to `main`. The app needs only `deployment_artifacts.joblib`, which is committed. Retraining from scratch additionally requires the [Lending Club accepted-loans dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (~1.6 GB, not committed).

## Known limitations

Stated rather than hidden, because they are the things a reviewer should ask about:

- **Early stopping still uses the full validation set.** The number of trees is chosen on all of validation, which includes the `val_thr` rows the threshold is later selected on. Calibration and threshold selection are cleanly separated; tree count is not. The residual optimism is small relative to the 1.8% measured above, but it is not zero — a fully clean design would carve a fourth split for early stopping.
- **LGD and margin are assumed, not measured.** 50% and 10% are plausible industry figures, not values derived from lender data. The threshold moves if they do.
- **SHAP explains the uncalibrated model.** Isotonic calibration is monotonic, so the direction and ranking of contributions hold, but the magnitudes are on the raw score's scale.
- **The test set is a single vintage.** Evaluation is out-of-time by construction (train ≤2015, validate 2016, test 2017), but 2017 is one origination year — it measures one step of drift, not drift across a cycle. 2018 was excluded because loans that close that quickly are disproportionately early payoffs, which biases its apparent default rate downward.
- **No confidence interval on the profit figure.** $23.5M is a point estimate. A bootstrap over the test set would say how much of the gap to the F1 threshold is real.

This is a portfolio project. The assumptions are documented so the numbers can be judged against them.
