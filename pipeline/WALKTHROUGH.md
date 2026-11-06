# Reading this pipeline, cell by cell

A top-to-bottom account of `credit_risk_pipeline.ipynb` — what each cell does, why it does
it that way, what to look for in its output, and the question it answers if someone asks.

The notebook runs in **310 seconds** and prints every number this project claims.

---

## The problem, stated once

Most credit-risk projects stop at *"here is a model with AUC 0.7"*. That is not a decision.
Two questions stay open:

1. **At what probability do you decline a loan?** Accuracy and F1 treat a false positive and
   a false negative as equally costly. In lending they differ by roughly 5×: rejecting a good
   borrower costs the margin on one loan, approving a defaulter costs half the principal.
2. **Can you actually collect these features at application time?** A model leaning on
   `int_rate` or `grade` is consuming Lending Club's own risk assessment. It scores well and
   is useless — those fields don't exist until someone has already priced the risk.

This pipeline answers both, ending in an approval policy rather than a probability.

---

## Cell index

| Cells | Stage | Produces |
|---|---|---|
| 1–2 | Imports, constants | `LGD = 0.50`, `MARGIN = 0.10` |
| 3–4 | Load, target definition | 2.26M → **1,348,099** rows |
| 5–9 | Leakage, circularity, availability | 151 → **89** columns |
| 10–11 | Joint applications | the `1.0000` verification |
| 12–18 | Encoding, sanity checks | the `verification_status` finding |
| 19–22 | Vintage table, split, ratios | **39 features**, three frames |
| 23 | Logistic regression baseline | **0.3859** PR-AUC |
| 24 | XGBoost on ≤2015 | **0.3930** PR-AUC, 433 trees |
| 25–26 | Four-way split of 2016, refit | 350 trees on 1,005,218 rows |
| 27 | Calibration | Brier **0.2170 → 0.1623** |
| 28–30 | Expected-loss cutoff | **0.16** |
| 31–32 | Final report | ROC-AUC **0.6990**, **$24.95M** |

---

## Cells 3–4 · Load and define the target

```
raw: 2,260,701 rows, 151 columns
resolved loans: 1,348,099  default rate: 0.1998
```

Only loans with a final outcome can be labelled. `Current`, `Late`, `In Grace Period` are
still running — calling them 0 would mislabel every default that hasn't happened yet.

**That deletes 912,602 rows, 40% of the data.** The alternative is survival analysis, which
models time-to-default and uses censored rows properly. That is the honest answer if someone
asks whether dropping them was wasteful.

---

## Cells 5–9 · Three different reasons to drop a column

Not one rule. Four lists, each with its own diagnostic.

**Post-issuance leakage** — `total_pymnt`, `recoveries`, `last_fico_range_*`, every
`hardship_*` and `settlement_*`. These accumulate *during* the loan, so a model using them
reads the outcome instead of predicting it.

The diagnostic that found them was a **univariate AUC screen**: score every numeric column
alone against the target. `last_fico_range_low` came back at **0.9386**. A single feature
scoring 0.94 on credit default is impossible — that number *is* the alarm. The whole top of
that table was post-issuance columns.

> Note that nullity is a red herring. `hardship_type` is 99.5% null and `hardship_flag` is 0%
> null. Both are leakage, for the same reason: both are only populated after issuance.

**Circularity** — a different problem. `grade`, `sub_grade` and `int_rate` *are* known at
application time, so they pass the timing test. But they are Lending Club's own risk model's
output, not borrower facts. Keeping them means partly learning to decode someone else's
score. Measured rather than assumed: **sub-grade explains 95.5% of interest-rate variance**,
so they are one variable wearing two hats. `installment` goes too — it is algebraically
recoverable from `loan_amnt + term + int_rate`, so it smuggles the signal back in.

**Identifiers and redundant text** — `title` is 63,154 free-text values duplicating a clean
14-category `purpose`. `emp_title` is 512,694 unique job titles: real signal, but TF-IDF on
it is a separate project, and it is the strongest unused feature in this dataset.
`member_id` is 100% null in the public release.

**Available in training but not over time** — six bureau columns (`open_rv_24m`, `all_util`,
`il_util` …) are **100% null before 2015 and 0% null from 2016**. Training is ≤2016 and test
is 2017, so the model would barely see them in training then meet them fully populated at
scoring time. This was only visible because the split is temporal; on a random split the
nulls spread evenly and nothing looks wrong.

---

## Cells 10–11 · Missing, or not applicable?

```
joint rows: 25,806  joint income >= individual income: 1.0000
```

`annual_inc_joint` is 94.7% null across the raw file — right in the band where people drop
things. It is **0% null within joint applications**. Structurally null, not sparse.

So the hypothesis gets tested before the column gets used: if these really are household
incomes, every one should be at least the primary applicant's. It returns exactly **1.0000**.
Only then is it coalesced into `combined_annual_inc`, with `is_joint` kept alongside — two
loans with a strong versus terrible co-signer should not look identical.

> 25,806 rather than 120,710 because joint applications only became common on the platform
> around 2017, so most are still unresolved and get dropped with the `Current` loans.

---

## Cells 12–18 · Encoding by structure, not by one rule

**Ordinal where order is real.** `emp_length` 0–10, `term` as an integer.

**The counterintuitive finding, cell 13:**

```
Not Verified     0.1474   407,541
Source Verified  0.2096   521,579
Verified         0.2386   418,979
```

The most rigorously verified loans default the **most** — a 9-point spread in the wrong
direction. Not a bug: verification is not randomly assigned. Lending Club demands documents
when an application already looks shaky, so this measures the selection that triggers
verification, not the effect of it. **Reverse causality.**

Ordinal encoding is still correct, because ordinal requires *monotonicity*, not a particular
direction — and 0.1474 → 0.2096 → 0.2386 is strictly increasing.

**Collapse tiny categories.** `ANY` has 286 rows and `NONE` has 53. A dummy fitted on 53 rows
learns noise and carries it to production.

**Geography to four regions.** Fifty state dummies is a lot of width for one feature, and
`zip_code` was already dropped as a fair-lending proxy.

**Sanity checks, cell 18 — three different treatments, each following from *why* the value
is extreme:**

```
negative DTI (nulled): 1
revol_util > 100 (kept): 4714
```

Negative debt-to-income is impossible → null it, let XGBoost route it. Utilisation above
100% is unusual but **real** — you can exceed your credit limit, and being over-limit is
predictive — so it stays. An 83-year credit history is implausible but 0.09% of rows and this
is a tree model, so it stays too, documented.

---

## Cells 19–22 · Split by time, and what the vintage table shows

```
issue_year   default rate   loans
2009             0.1369      5,281
2013             0.1560    134,804
2015             0.2019    375,546
2016             0.2329    293,105
2017             0.2313    169,321
2018             0.1576     56,318
```

**Two things to read off this.**

The rate climbs steadily from 13.7% to 23.3%. That is the drift the training window later
addresses.

And **2018 drops to 15.76%** — below 2013. That is not better lending. Only ~11% of 2018
loans had resolved by the end of the data, and a loan can only close that fast through an
unusually early exit. Right-censoring, biasing **downward**, which is the opposite of the
intuition that unresolved loans hide defaults. 2018 is excluded.

```
train <=2015  829,355  default 0.1846
val   2016    293,105  default 0.2329
test  2017    169,321  default 0.2313
features: 39
```

A random split would let the model train on 2016 and be scored on 2010. No deployed model
sees the future.

> **The limitation worth raising before someone else does:** 2017 is only ~38% resolved, so
> the test set is itself skewed toward loans that resolved early. The clean fix is a fixed
> seasoning window — label every loan on whether it defaulted within 24 months — so every
> vintage is measured over the same exposure. That is the first thing to change.

**The six engineered ratios** (`loan_to_income`, `avg_revol_balance`, `open_to_total_acc`,
`accounts_per_credit_year`) are all **row-wise**, so building them before or after the split
gives identical values. There is no leakage either way. They are built after out of habit,
because the moment a transform *fits* something — an imputer, a scaler, a target encoding —
the ordering stops being cosmetic.

Be precise about this if asked. Claiming "I built features after the split to avoid leakage"
invites *"would `loan_amnt / annual_inc` have leaked?"* — and the answer is no.

---

## Cell 23 · The baseline, before anything clever

```
logistic regression  test ROC-AUC 0.6846  PR-AUC 0.3859
```

Worth knowing what the tree model is actually buying. Logistic regression needs median
imputation and scaling; XGBoost needs neither and handles the structural missingness natively
instead of relabelling "never delinquent" as average.

---

## Cell 24 · XGBoost on loans up to 2015

```
scale_pos_weight 4.4183  trees 433
xgboost (<=2015)     test ROC-AUC 0.6948  PR-AUC 0.3930
```

`scale_pos_weight = neg/pos` multiplies the gradient of every positive example by 4.42, so
the loss stops being dominated by the 81.5% majority class. **Remember that it makes the
effective class balance 50/50** — that is the entire explanation for cell 27.

`eval_metric='aucpr'` is the line that matters most. Early stopping halts when *this* metric
plateaus; stop on logloss instead and, with the probabilities already distorted, you optimise
something you do not care about and stop at a different tree.

`n_estimators=3000` is a **cap**, not a target. It stopped at 433, so the cap never bound.

**The gap against the baseline: +0.0102 ROC-AUC, +0.0071 PR-AUC.** Narrow. On the full
151-feature set it was +0.0154 PR-AUC. Once the ratios are hand-built, a linear model gets
most of the way — which is also why WOE scorecards remain standard in regulated credit.

---

## Cells 25–26 · Adding 2016, and why it costs three splits

The model above trains to 2015 and is scored on 2017, across a jump from 18.5% to 23.1%. It
is extrapolating on stale data. Adding 2016 should help — but 2016 was doing **three jobs**:
early stopping, calibration, and threshold selection. Once it becomes training data, each of
those needs its own rows.

```
2016 split -> train 175,863 | early-stop 38,103 | calibrate 37,986 | threshold 41,153
train rows 1,005,218  trees 350
```

Every fitted quantity gets rows nothing else touched. 2017 is still read exactly once.

> Note the tree count **falls** from 433 to 350 despite 21% more data. The 39-feature model
> has already extracted what its features can express. On the 151-feature version the same
> change pushed trees from 423 to 766 and gained +0.021 PR-AUC rather than +0.003 — **more
> data only helps if the model has the capacity to use it.**

**The honest limitation:** calibration and threshold rows are now held-out *rows* from a year
the model has otherwise seen, not a wholly unseen year. Test remains a genuine future
vintage, so the headline numbers are clean, but that asymmetry is real.

---

## Cell 27 · The calibration failure

```
uncalibrated: mean predicted 0.4585 vs actual 0.2313, Brier 0.2170
  (a constant at the base rate scores 0.1778 - the model is worse)
calibrated:   mean predicted 0.2326 vs actual 0.2313, Brier 0.1623
```

**The model loses to a constant.** Brier 0.2170 against 0.1778 for predicting the base rate
for every loan — while PR-AUC says the ranking is fine. That combination has exactly one
cause: the ranking is good and the probabilities are wrong.

And the cause of *that* is cell 24. `scale_pos_weight` created a 50/50 world, so predictions
centre near 0.46 instead of 0.23. **1.98× inflated.**

Ranking metrics cannot see this. PR-AUC and ROC-AUC are invariant to any monotone
transformation — multiply every prediction by two and neither moves. Only a proper scoring
rule notices, which is why Brier is in the evaluation block.

**Isotonic, not Platt.** Platt fits a sigmoid, assuming a specific parametric distortion.
This distortion is monotone with no reason to be sigmoidal. Isotonic assumes only
monotonicity. The usual objection — overfitting on small calibration sets — does not apply
with 37,986 rows.

**What it costs:** isotonic is a step function. It maps the test set onto a few hundred
distinct probabilities, so thousands of loans end up exactly tied. That trims a little
ranking resolution and means the approval rate cannot be finely tuned — moving the cutoff
jumps by whole plateaus. A real trade, not a free win.

**Why any of it matters:** the approval rule multiplies PD by loss. With PD inflated 2×,
every expected-loss figure is wrong and the optimal cutoff lands in the wrong place. For a
pure ranking task this would be ignorable; for a decision task it is fatal.

---

## Cells 28–30 · From probability to policy

```
net = Σ(principal × 10% margin) over approved loans that repaid
    − Σ(principal × 50% LGD)    over approved loans that defaulted
```

LGD and margin are **stated assumptions, not derived** — the columns that would give real
recovery rates (`recoveries`, `out_prncp`) are post-issuance and were dropped as leakage.

A threshold is a fitted parameter like any hyperparameter. Select it on test and the reported
profit is the maximum of a noisy surface, not an estimate. So it is chosen on the 41,153
held-out 2016 rows and test is read once.

```
cutoff  approve   test net profit
  0.05     4.8%   $    7.56M
  0.11    17.7%   $   20.99M
  0.14    26.0%   $   24.66M
  0.16    37.2%   $   24.95M  <-- chosen
  0.20    43.2%   $   23.23M
  0.26    64.6%   $    3.68M
  0.32    78.8%   $  -24.52M
```

**The shape is the argument.** Profit peaks around a third of applicants approved, then
collapses — negative past roughly 70%. A threshold tuned for accuracy or F1 sits far out on
the right of this table, approving loans that destroy money.

---

## Cells 31–32 · The final report

```
ROC-AUC                     0.6990
PR-AUC                      0.3922   (no-skill floor 0.2313)
Recall for defaults         0.8283
Precision                   0.3053

Approved                    63,037 of 169,321  (37.2%)
Default rate if we approved everyone   23.13%
Default rate among approved            10.67%

Net profit, trained <=2016                   $24.95M
  vs <=2015, each model re-tuning its cutoff $22.87M   (+9.1%)
  vs <=2015, both held at the same cutoff    $23.87M   (+4.5%)
```

**On 0.70 ROC-AUC.** Normal for unsecured consumer credit at application time. Default is
driven substantially by post-origination shocks — job loss, illness — that are not in the
data. A 0.9 here would be the red flag, not a triumph: that is roughly what leaving `grade`
or `int_rate` in produces.

**On PR-AUC 0.3922.** PR-AUC's no-skill floor is the **positive class rate**, not 0.5. At a
23.13% base rate, random scores 0.2313, so this is **1.7×** a random ranker. ROC-AUC is the
one with a 0.5 floor.

**On precision 0.3053.** Low by design. The cutoff optimises expected loss, not precision —
approving a defaulter costs half the principal, rejecting a good borrower costs a 10% margin,
so it deliberately accepts false positives to avoid false negatives.

**On the two profit figures.** They answer different questions and both are printed rather
than picking the flattering one. Fixed cutoff isolates the model: **+4.5%**. Each model
re-tuning its own cutoff is the realistic re-deployment: **+9.1%**. The second carries
threshold-selection noise — the old model's self-selected cutoff of 0.17 actually
*underperformed* 0.16 on test.

---

## Known limitations

Stated here rather than waiting to be found.

- **Reject inference.** Every row is an applicant Lending Club already approved. The model is
  trained on default conditional on passing someone else's screen, so its probabilities are
  only trustworthy over the region the incumbent policy covered. Push the cutoff much looser
  and it is extrapolating.
- **`region` is a fair-lending risk.** US geography proxies for race, and ECOA judges
  disparate impact on outcomes rather than intent. It bought +0.0023 PR-AUC. `zip_code` was
  dropped for exactly this reason; region is the same problem at coarser resolution and
  should get a disparate-impact test or be dropped.
- **Test vintage is partially censored.** 2017 is ~38% resolved, skewing toward early
  resolvers. A fixed 24-month seasoning window would fix it.
- **LGD and margin are assumed.** A sensitivity grid over both is missing.
- **No confidence interval on the profit figure.** It is a point estimate; a bootstrap over
  the test set would say how much of the gap to alternatives is real.
- **Repeat borrowers are unverifiable.** `member_id` is 100% null, so whether the same person
  appears in both training and test cannot be checked. A temporal split limits the exposure
  but does not eliminate it.
- **Early stopping still uses the full 2016 set** in the ≤2015 model, overlapping the
  calibration rows. Smaller effect than the above, but not zero.

---

## Running it

```bash
cd D:\Lending\pipeline
python credit_risk_pipeline.py        # ~310s
```

Needs `accepted_2007_to_2018Q4.csv` in `D:\Lending` (~1.6 GB, from Kaggle) and about 8 GB of
RAM for the load. The notebook alongside it holds the same code with outputs already stored.

---

**Interview questions** for this project live in [`QUESTION_BANK.md`](QUESTION_BANK.md) —
37 of them, grouped by theme, with the cell holding each answer's evidence.
Quick revision: [`ONE_PAGER.md`](ONE_PAGER.md).
