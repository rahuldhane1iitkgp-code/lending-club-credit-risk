# Credit Risk Modeling and Lending Decisions — one pager

**The ask:** predict default at application time, then turn that probability into an
approve/reject decision using expected loss rather than a classification metric.

**Data:** Lending Club public loans. 2,260,701 rows → **1,348,099** with a final outcome.
Train on loans originated ≤2016, test on **169,321** loans from 2017, read once.

---

## Every number, and where it comes from

| Claim | Value | Source |
|---|---|---|
| Loans modelled | 1,348,099 | 2.26M filtered to resolved outcomes |
| Features | 39 | application-time only, from 151 raw columns |
| Test set | 169,321 | all 2017 originations |
| ROC-AUC | **0.6990** | 2017 holdout |
| PR-AUC | 0.3922 | floor is the class rate, **0.2313** → 1.7× random |
| Recall for defaults | 0.8283 | at the 0.16 cutoff |
| Precision | 0.3053 | low by design — expected loss, not precision |
| Brier | 0.2170 → **0.1623** | before/after isotonic; a constant scores 0.1778 |
| Approval rate | **37.2%** | 63,037 of 169,321 |
| Default among approved | **10.67%** | vs **23.13%** approving everyone |
| Net profit | **$24.95M** | LGD 50%, margin 10%, stated assumptions |
| Profit from wider window | +4.5% / +9.1% | fixed cutoff / each re-tuning its own |
| Default rate drift | 18.5% → 23.1% | train vintage vs test vintage |
| Logistic regression | 0.3859 PR-AUC | vs XGBoost's 0.3930 — the baseline |

---

## The 60-second pitch

> Lending Club publishes about 2.3 million loans, 1.35 million with final outcomes. I built
> a model predicting default at application time — but the part I care about is the layer
> after that: converting a probability into an approve/reject decision using an
> expected-loss framework rather than F1. Along the way I found the training data was two
> years staler than the test year and the default rate had moved from 18.5% to 23.1%, so I
> widened the training window. The final policy approves 37% of applicants, and that
> approved book defaults at 10.7% against 23.1% for the full population.

---

## Six answers you must have

**Why XGBoost?** → Meaningful missingness. XGBoost learns a default split direction for
missing values; logistic regression needs imputation, which relabels "never delinquent" as
average. I measured the alternative: LR 0.3859 vs XGB 0.3930. Narrow — the relationship is
mostly additive, which is why WOE scorecards still dominate regulated credit.

**Why calibrate?** → Brier 0.2170 against 0.1778 for a constant. The model lost to a
constant while PR-AUC looked fine — ranking good, probabilities wrong. `scale_pos_weight`
4.42 creates a 50/50 world, so predictions centre at 0.46 instead of 0.23. Matters because
the approval rule multiplies PD by loss.

**Why a temporal split?** → A random split lets the model train on 2016 and be scored on
2010. It also caught a bug: six bureau columns are 100% null before 2015 and 0% from 2016.

**How did you pick 0.16?** → Argmax of net profit on 41,153 held-out 2016 rows, never test.
A threshold is a fitted parameter. Measured the bias avoided at 1.8%.

**Is 0.70 ROC-AUC low?** → Normal for unsecured credit at application time; default depends
on post-origination shocks not in the data. 0.9 would be the red flag — that's roughly what
leaving `grade` or `int_rate` in produces.

**Couldn't you reject everyone?** → Zero defaults, zero profit. That's why the approval rate
is quoted alongside: at 37% approved, the book defaults at under half the population rate.

---

## Four weaknesses to volunteer before you're asked

1. **Reject inference.** Every row passed Lending Club's existing screen, so the model is
   trained on default *conditional on approval*. Probabilities are only trustworthy over the
   region the incumbent policy covered.
2. **`region` is a fair-lending proxy.** Geography correlates with race; ECOA judges
   disparate impact on outcomes, not intent. It bought +0.0023 PR-AUC — a rounding error
   against that exposure. Should be dropped or impact-tested.
3. **2017 is ~38% resolved**, so the test set skews toward early resolvers. A fixed 24-month
   seasoning window would fix it.
4. **LGD and margin are assumed**, not derived — the columns giving real recovery rates are
   post-issuance and were dropped as leakage. No sensitivity grid.

---

## Two mistakes worth volunteering

**Threshold selected on test.** A threshold is a fitted parameter, so that inflates reported
profit and burns the holdout. Re-ran selecting on held-out validation rows; measured the
difference at **1.8%**.

**F1 comparison with mismatched units.** Claimed the F1 threshold lost money — but I'd
optimised F1 on raw scores averaging 0.45 and applied that cutoff to calibrated
probabilities averaging 0.23. Computed consistently, F1 is profitable, just far from
optimal. *A result that flatters your thesis is the one to re-derive first.*

---

## The three findings that aren't obvious

**Verified loans default more** — 23.9% vs 14.7%. Not causal: Lending Club demands documents
when a file already looks shaky. Reverse causality, still monotone, so ordinal encoding
holds.

**2018's default rate falls to 15.8%** — below 2013. Not better lending: only ~11% of that
vintage had resolved. Right-censoring biasing *downward*, the opposite of intuition.

**More data only helps with capacity.** Same rows, same split: the 151-feature model went
423 → 766 trees and gained 0.021 PR-AUC; the 39-feature model went *down* to 350 trees and
gained 0.003.

---

**Code:** `pipeline/credit_risk_pipeline.ipynb` — runs in 310s, outputs stored.
**Detail:** `pipeline/WALKTHROUGH.md` — cell-by-cell, plus ~30 interview questions.
