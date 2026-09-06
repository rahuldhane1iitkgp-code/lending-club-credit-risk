# Lending Club Credit Risk Model

A credit risk scoring model that ends in a **lending decision**, not just a probability.

XGBoost, isotonically calibrated, with an approve/reject threshold chosen to maximise net profit under an explicit expected-loss framework — not to maximise accuracy or F1. Built on 39 application-time features that a borrower or loan officer could realistically supply, retaining **89.5% of the test PR-AUC** of a fuller 151-feature research model.

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
| A — research | 151 | **0.4383** | 0.7265 | — |
| **B — deployment, shipped** | **39** | **0.3922** | 0.6987 | 0.1624 |

Model B retains **89.5%** of Model A's test PR-AUC using a quarter of the features, none of which require post-origination information. Quantifying that gap is the point: a 10.5% performance cost to become deployable is a tradeoff you can defend in a room; an unquantified one is not.

Both models train on loans originated through 2016 and are tested on 2017 — see [the training-window section](#does-a-wider-training-window-help-only-if-the-model-can-use-it) for why, and for the earlier ≤2015 figures (0.4132 / 0.3896, a 94.3% retention) those numbers replace.

Adding US region was worth **+0.0023 PR-AUC** (measured on the earlier ≤2015 models) — a real but marginal gain, kept because it costs the applicant nothing to provide.

**Out-of-time evaluation.** Splits are by origination year, not random — a random split would let the model train on 2016 and be scored on 2010, which no deployed model ever gets to do. 2017 (169,321 loans) is the test year and is read once. 2018 is excluded entirely: loans that close that fast are disproportionately early payoffs, which drags its apparent default rate down to 15.8%.

**Leakage control.** `int_rate`, `grade`, and `sub_grade` are excluded throughout — they encode the platform's existing risk judgement. `combined_fico_low`/`combined_fico_high` are collapsed into a single `fico_score`.

**Class imbalance.** `scale_pos_weight` with `aucpr` as the eval metric, and **PR-AUC as the headline metric rather than ROC-AUC** — with a minority default class, ROC-AUC is flattered by the large negative class.

**Calibration.** Isotonic regression over the raw XGBoost scores. Necessary because the threshold logic below operates on probabilities: an uncalibrated score of 0.15 is not a 15% default rate, and the expected-loss arithmetic silently breaks if you pretend otherwise. Reported alongside Brier score.

**The decision layer.** For a sweep of thresholds, over the loans the policy would approve:

```
net = Σ (principal × margin)   over approved loans that repaid
    − Σ (principal × LGD)      over approved loans that defaulted
```

with `LGD = 50%` and `profit margin = 10%` as stated assumptions.

**Every fitted quantity gets data nothing else touched.** A decision threshold is a fitted parameter like any hyperparameter: select it on test and the reported profit is the maximum of a noisy surface, not an estimate. Since 2016 is now training data, it is cut four ways so that tree count, calibration and threshold each get their own rows:

```
<=2015 + 60% of 2016 (1,005,218)  ->  train
           13% of 2016   (38,103)  ->  early stopping, tree count
           13% of 2016   (37,986)  ->  isotonic calibration
           14% of 2016   (41,153)  ->  threshold selection
2017                    (169,321)  ->  test, read once
```

Selected threshold: **0.16**. Test performance at it, reported once:

| Threshold | Approval rate | Test net |
|---|---|---|
| 0.05 | 4.6% | $7.3M |
| 0.11 | 17.6% | $20.8M |
| 0.14 | 24.0% | $24.1M |
| **0.16** | **35.6%** | **$25.2M** |
| 0.20 | 44.7% | $22.2M |
| 0.26 | 64.5% | $3.1M |
| 0.32 | 78.7% | **−$24.2M** |

**Approve roughly the safest 36% of applicants.** The shape of the curve is the argument: profit peaks and then collapses, going negative past a ~70% approval rate.

The F1-optimal threshold — the standard ML default — lands at **0.21**, approving **52.5%** of applicants for a test net of **$17.32M**. It does not lose money, but it forfeits **$7.85M, or 31%** of the achievable profit, because F1 treats a rejected good borrower and an approved defaulter as equally costly when they differ by an order of magnitude.

Both thresholds are computed on the **same calibrated probability scale**, which matters more than it sounds. An earlier version of this comparison optimised F1 against raw XGBoost scores (mean 0.452) and evaluated profit on calibrated ones (mean 0.233) — two different units. That cutoff rejected almost nobody and appeared to lose $88.7M. It was an arithmetic artifact, not a finding, and the corrected comparison above replaces it.

The decision rule, not the classifier, is what makes this profitable.

**How much did the honest split cost?** On this model, nothing measurable: the validation-selected threshold of 0.16 is also the test-optimal one, so the selection optimism is **0.0%**. On the earlier ≤2015 model the same procedure cost 1.8% ($23.50M against a test-selected $23.94M). Measuring that gap is cheaper than arguing about it.

**Explanation.** Per-prediction SHAP contributions, so the app returns a reason alongside a decision.

## Is 0.41 PR-AUC low? Benchmarked, then tested for significance

A single model's score says nothing about whether a better one exists. Five algorithm families were run on the identical temporal split, each early-stopped on 2016 and scored once on 2017:

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | vs XGBoost |
|---|---|---|---|---|
| Rank-avg ensemble (3 GBMs) | 0.4371 | **0.4193** | 0.7134 | +0.0021 |
| LightGBM | 0.4353 | 0.4180 | 0.7117 | +0.0008 |
| HistGradientBoosting | 0.4350 | 0.4176 | **0.7127** | +0.0004 |
| **XGBoost (shipped)** | 0.4352 | 0.4172 | 0.7114 | — |
| Logistic Regression | 0.4196 | 0.4018 | 0.7015 | −0.0154 |
| Random Forest | 0.4199 | 0.3989 | 0.6982 | −0.0182 |

No-skill baseline 0.2313 · best lift **1.81×** · full output in [`reports_model_benchmark.csv`](reports_model_benchmark.csv)

Point estimates cannot say whether that ordering is meaningful, so a **1000-resample paired bootstrap** on the test set follows — every model rescored on the same resample each time, so shared sampling noise cancels:

| Model | Δ PR-AUC | 95% CI | P(beats XGBoost) | Distinguishable |
|---|---|---|---|---|
| Ensemble (3 GBMs) | +0.0021 | [+0.0015, +0.0028] | 100% | **yes** |
| LightGBM | +0.0008 | [−0.0002, +0.0018] | 95.0% | no |
| HistGradientBoosting | +0.0004 | [−0.0008, +0.0017] | 75.5% | no |
| Logistic Regression | −0.0154 | [−0.0177, −0.0131] | 0% | **yes** |
| Random Forest | −0.0182 | [−0.0203, −0.0162] | 0% | **yes** |

XGBoost test PR-AUC 0.4172, bootstrap 95% CI [0.4126, 0.4227] · [`reports_bootstrap_benchmark.csv`](reports_bootstrap_benchmark.csv)

**The conclusion: 0.41 is the data's ceiling, not the model's.** Three independently written gradient-boosting implementations — different split-finding, different regularisation, different missing-value handling — agree to the third decimal on validation (0.4350–0.4353) and test (0.4172–0.4180). None is statistically distinguishable from the others.

The most informative row is **logistic regression at 0.4018**: a linear model with no interaction terms captures **96%** of what a 423-tree boosted ensemble achieves. If substantial non-linear structure remained, that gap would be far wider.

The ensemble's +0.0021 *is* statistically resolvable — 169k test rows and a paired design give enough power to detect it — but it is a **0.5% relative gain** for triple the inference cost and the loss of single-model SHAP attribution the app depends on. It is not shipped. Statistical significance and practical significance are different questions, and this is a clean case of the two disagreeing.

Default is driven substantially by post-origination events — job loss, illness, divorce — that are unknowable at application time. That is the ceiling. The only lever that would move it materially is re-admitting `grade`/`int_rate`, which means re-admitting the circularity those were dropped for.

> **A benchmark trap worth documenting.** LightGBM's sklearn wrapper initially scored 0.368 — apparently far behind XGBoost. In LightGBM 4.7 the wrapper's `eval_set` argument is deprecated and the validation set never reaches early stopping, so training halted at **iteration 7**. Via the native `lgb.train` API it runs to 523 iterations and scores 0.4180. A silently mistrained challenger is how benchmarks reach confident, wrong conclusions about which algorithm wins.

## Does a wider training window help? Only if the model can use it

The algorithm benchmark above pinned every learner near 0.417, which suggested the limit was the data rather than the model. It was — but not in the way "get more features" implies. The shipped model trained on loans through **2015** and was scored on **2017**: a two-year gap across which the default rate moves from 18.5% to 23.1%. It was being asked to extrapolate across a real shift in credit conditions.

Folding 2016 into training, holding the split design constant:

| Model | Features | Train ≤2015 | Train ≤2016 | Gain | Best iteration |
|---|---|---|---|---|---|
| A — research | 151 | 0.4172 | **0.4383** | **+0.0211** | 423 → **766** |
| B — deployment | 39 | 0.3896 | **0.3922** | **+0.0034** | 423 → **367** |

**The comparison is single-variable, and that was verified rather than assumed.** The ≤2015 and ≤2016 runs originally also differed in their early-stopping set — all of 2016 (293k rows, out-of-time) versus a 38k in-year slice. A control trained on ≤2015 while early-stopping on that same 38k slice scores **0.4172** with best_iter 421, against 0.4172 and best_iter 423 for the original. The early-stopping regime is worth **−0.0000**, so the entire +0.0211 is the training window.

Identical rows, identical split, identical seed — and a **6× difference in benefit**. The 151-feature model absorbs the extra vintage and nearly doubles its tree count before overfitting. The 39-feature model *shrinks*, because it has already extracted everything its features can express.

**More recent data only helps if the model has the capacity to use it.** That is why the deployability gap widened from 5.7% to 10.5%: the constraint that makes Model B shippable also caps how much it can learn.

For the shipped model the change is still worth real money, and both intervals clear zero on a 1000-resample paired bootstrap:

| Metric | ≤2015 | ≤2016 | Δ | 95% CI |
|---|---|---|---|---|
| Test PR-AUC | 0.3896 | 0.3922 | +0.0034 | [+0.0024, +0.0045] |
| Test ROC-AUC | 0.6946 | 0.6987 | +0.0042 | — |
| Approval rate | 33.9% | 35.6% | +1.7pp | — |
| **Test net profit** | **$23.50M** | **$25.17M** | **+$1.68M** | **[+$0.97M, +$2.37M]** |

**+7.1% profit from changing the training window, against +0.5% from ensembling three gradient boosters.** The lever was never the algorithm.

### Where the $1.68M actually comes from

Profit moved far more than PR-AUC, so the gain is decomposed rather than asserted. Holding the new model to the **identical number of approvals** as the old one isolates ranking from volume:

| | Approvals | Default rate among approved | Test net |
|---|---|---|---|
| Old model @ 0.16 | 57,348 (33.9%) | 10.42% | $23.50M |
| New model, same volume | 57,348 (33.9%) | **10.14%** | $24.98M |
| New model @ 0.16 | 60,224 (35.6%) | 10.38% | $25.17M |

| Effect | Δ Net | Share |
|---|---|---|
| **Ranking** — a better set of borrowers at the same volume | **+$1.49M** | **89%** |
| Volume — 2,876 extra approvals at a 15.23% default rate, against a 16.67% break-even | +$0.19M | 11% |

So it is overwhelmingly a **ranking** improvement, not extra volume. Same book size, 0.28pp fewer defaulters inside it.

**Why PR-AUC barely registered it.** PR-AUC averages over the whole ranking; the decision only touches the top third. The retrained model improved specifically near the cutoff — separating good from bad among the *safest* applicants — while leaving the middle and tail largely as they were. Spearman correlation between the two models' rankings is **0.9799**: they mostly agree, and the 2% where they disagree is where the money is. A global ranking metric averages that away. **Ranking quality and decision quality are not the same measurement**, and a 0.9% PR-AUC gain worth 7.1% of profit is the proof.

**A calibration side-effect worth knowing.** Isotonic regression is a step function: it maps 169,321 test loans onto **396 distinct probabilities**, the largest plateau holding 12,168 loans at exactly p=0.2048. Thousands of applicants therefore share an identical score, and moving the threshold can jump the approval rate by a whole plateau or not move it at all — which is why approval went 33.9% → 35.6% at the *same* 0.16 cutoff. Approval rate cannot be finely tuned with this model. Platt scaling would give a smooth curve at some cost in calibration accuracy; isotonic was chosen for accuracy, and this is the price.

## Repo layout

| File | Purpose |
|---|---|
| `model_b_step1.py` | Feature construction — region mapping, FICO collapse, deployment feature set |
| `model_b_step2.py` | Trains B1/B2, calibrates, evaluates vs. Model A, picks the winner |
| `model_b_step3.py` | Expected-loss threshold sweep — selects on validation, reports on test |
| `model_a_threshold.py` | Same procedure applied to the 151-feature research model |
| `model_benchmark.py` | Five algorithm families plus an ensemble on the identical split |
| `bootstrap_benchmark.py` | 1000-resample paired bootstrap on the PR-AUC differences |
| `experiment_recency.py` | Training-window experiment on the 151-feature model |
| `experiment_recency_control.py` | Same experiment under the deployment split, as a control |
| `experiment_recency_clean.py` | Isolates the training window by holding the early-stopping set fixed |
| `model_b_recency.py` | Rebuilds the shipped model end to end on the wider window |
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
- **Calibration and threshold rows are in-distribution, not out-of-time.** Once 2016 joined the training window, the calibration and threshold-selection sets became held-out *rows* from a year the model has otherwise seen, rather than a wholly unseen year. Test remains a genuine future vintage, so the headline PR-AUC and profit figures are clean, but the threshold may be slightly optimistic for 2017 in a way the earlier ≤2015 design was not.
- **The test set is a single vintage.** Evaluation is out-of-time by construction (train ≤2015, validate 2016, test 2017), but 2017 is one origination year — it measures one step of drift, not drift across a cycle. 2018 was excluded because loans that close that quickly are disproportionately early payoffs, which biases its apparent default rate downward.
- **No confidence interval on the profit figure.** $23.5M is a point estimate. A bootstrap over the test set would say how much of the gap to the F1 threshold is real.

This is a portfolio project. The assumptions are documented so the numbers can be judged against them.
