# Question bank

Thirty-seven questions on this project, grouped the way an interview moves. Answers are
written to be said out loud, not read. Cell references point at the evidence in
`credit_risk_pipeline.ipynb`.

Numbers all come from one run of that notebook — see `ONE_PAGER.md` for the summary table.

---

Grouped the way an interview actually moves. Cell references point at the evidence.

## Opening

**"Walk me through this project."**
> Lending Club publishes about 2.3 million loans, 1.35 million with final outcomes. I built
> a model predicting default at application time, but the part I care about is the layer
> after that: turning a probability into an approve/reject decision using expected loss
> rather than a classification metric. Along the way I found the training data was two years
> staler than the test year and the default rate had moved from 18.5% to 23.1%, so I widened
> the training window. The final policy approves 37% of applicants, and that approved book
> defaults at 10.7% against 23.1% for the full population.

**"What's the most interesting thing you found?"**
> Verified loans default *more* — 23.9% against 14.7% for unverified. It looks like a bug.
> It isn't: Lending Club demands income documents when an application already looks shaky,
> so verification is a proxy for underwriter suspicion. The label is causing the treatment,
> not the reverse. *(cell 13)*

## Data and leakage

**"How did you decide what to drop?"**
> Five reasons, each with its own diagnostic. Post-issuance leakage, caught by a univariate
> AUC screen — `last_fico_range_low` alone scored 0.9386, impossible on application-time
> data. Circularity, caught by correlation between features — sub-grade explains 95.5% of
> interest-rate variance. Availability mismatch, caught by null rate *by year*. Redundancy,
> caught by cardinality. And constants or intractables like `policy_code` and 512k unique
> job titles. *(cells 5–9)*

**"How do you catch leakage you didn't anticipate?"**
> Reasoning catches what you thought of; a univariate AUC screen catches what you didn't.
> Anything above about 0.75 alone on credit data gets investigated.

**"`hardship_type` is 99.5% null, `hardship_flag` is 0% null. Why are both leakage?"**
> Nullity isn't the test. Both are only *populated* after issuance. The rule is "would a
> lender know this at approval" — nullity is a symptom sometimes and a red herring other
> times.

**"You kept a 95%-null column and dropped a 0%-null one."**
> `annual_inc_joint` is 95% null overall but 0% null *within* joint applications —
> structurally null, not sparse. I tested it before using it: every joint row has joint
> income at least the primary's, which returns exactly 1.0. `emp_title` is 0% null but
> 512,694 unique strings, so it can't be one-hot encoded and TF-IDF on job titles is a
> separate project. *(cells 10–11)*

**"Why drop `int_rate` when it's known at application time?"**
> It's Lending Club's own risk model's output, not a borrower fact. Keeping it means partly
> learning to decode someone else's score. That's circularity, a different failure from
> leakage. I measured it rather than asserting it — 95.5% of its variance is sub-grade.

## Validation design

**"Why a temporal split?"**
> A random split lets the model train on 2016 and be scored on 2010. No deployed model sees
> the future. It also caught a bug: six bureau columns are 100% null before 2015 and 0% null
> from 2016. On a random split the nulls spread evenly and nothing looks wrong.

**"Why exclude 2018?"**
> Only ~11% of 2018 loans had resolved, and the apparent default rate falls to 15.8% — below
> 2013. A loan can only close that fast through an early exit. Right-censoring, and it
> biases *downward*, the opposite of the intuition that unresolved loans hide defaults.

**"Isn't 2017 censored too?"**
> Yes, ~38% resolved, so the test set skews toward early resolvers. The clean fix is a fixed
> seasoning window — did it default within 24 months — so every vintage gets the same
> exposure. I didn't do that; it's the first thing I'd change.

**"You built features after the split. Would building them before have leaked?"**
> No. They're row-wise ratios — `loan_amnt / annual_inc` uses nothing from any other row, so
> pre- or post-split gives identical values. I build them after as discipline, because the
> moment a transform *fits* something the ordering stops being cosmetic. The fitted things
> here — median imputation, `scale_pos_weight`, the calibrator — are all train-only.

## Modelling

**"Why XGBoost?"**
> Tabular data, mixed units, non-linear interactions, and the deciding factor: meaningful
> missingness. XGBoost learns a default split direction for missing values, so "never
> delinquent" stays distinguishable; logistic regression needs imputation, which relabels
> that group as average. I also measured the alternative — logistic regression gets 0.3859
> PR-AUC against XGBoost's 0.3930 on these 39 features. A narrow gap, which says the
> relationship is mostly additive. That's why WOE scorecards still dominate regulated
> credit, and if this went to a regulator I'd ship one. *(cells 23–24)*

**"Why `scale_pos_weight` instead of SMOTE?"**
> SMOTE interpolates between neighbours. Most of my features are one-hot after encoding, so
> interpolating 0 and 1 gives a "30% homeowner", which isn't a borrower. Reweighting the
> loss touches no data.

**"Why PR-AUC rather than accuracy?"**
> With a 20% positive class, predicting "no default" for everyone scores 80% accuracy and
> catches zero defaulters. At my operating cutoff accuracy is 51% — worse than that trivial
> baseline — because the policy deliberately flags a lot of applicants. Accuracy measures
> the wrong thing here.

**"0.70 ROC-AUC seems low."**
> Normal for unsecured consumer credit at application time. Default depends heavily on
> post-origination shocks — job loss, illness — that aren't in the data. I'd be more worried
> by a high number: 0.9 is roughly what you get leaving `grade` or `int_rate` in. And
> PR-AUC's floor is the class rate, not 0.5 — 0.3922 against a 0.2313 base rate is 1.7×
> random.

**"Why does early stopping use `aucpr`?"**
> Early stopping halts when *that* metric plateaus. Stop on logloss while `scale_pos_weight`
> is distorting probabilities and you optimise something you don't care about, stopping at a
> different tree. The stopping metric has to match the selection metric.

## Calibration

**"Why calibrate?"**
> Brier was 0.2170 against 0.1778 for predicting the base rate for every loan. The model
> lost to a constant while PR-AUC looked fine — ranking good, probabilities wrong. Cause is
> `scale_pos_weight` of 4.42, which makes the effective class balance 50/50, so predictions
> centre near 0.46 instead of 0.23. It mattered because the approval rule multiplies PD by
> loss-given-default. *(cell 27)*

**"Why didn't PR-AUC catch it?"**
> It's rank-invariant. Multiply every prediction by two and PR-AUC and ROC-AUC don't move.
> Only a proper scoring rule — Brier or log loss — notices.

**"Isotonic or Platt?"**
> Isotonic. Platt fits a sigmoid, assuming a specific parametric distortion; mine came from
> gradient reweighting, monotone but with no reason to be sigmoidal. The usual objection to
> isotonic is overfitting on small calibration sets and I had 38k rows. The cost is that
> it's a step function — it collapses the test set onto a few hundred distinct probabilities,
> so thousands of loans end up tied and the approval rate can't be finely tuned.

## The decision layer

**"How did you pick 0.16?"**
> Swept the cutoff on net profit and took the argmax — on 41,153 held-out 2016 rows, never
> on test. A threshold is a fitted parameter; pick it on test and you report the maximum of
> a noisy surface. I measured that bias on an earlier version: test-selected would have
> claimed $23.94M against an honest $23.50M, so 1.8%.

**"What exactly is the profit function?"**
> Net equals principal × 10% margin over approved loans that repaid, minus principal × 50%
> loss-given-default over approved loans that defaulted. Exposure is the actual loan amount.
> LGD and margin are stated assumptions — the columns that would give real recovery rates
> are post-issuance and I dropped them as leakage.

**"What if your LGD assumption is wrong?"**
> The cutoff moves, and I can say which way: break-even is roughly margin / (margin + LGD),
> so a lower LGD loosens it. What doesn't change under any plausible assumption is the
> ordering — a profit-optimised cutoff beats an F1-optimised one, because the asymmetry is
> structural. What's missing is a sensitivity grid over both parameters.

**"Couldn't you get 0% defaults by rejecting everyone?"**
> Yes, and zero profit with it. That's why the approval rate is quoted alongside: at 37%
> approved, the book defaults at 10.7% against 23.1% for the population.

**"Precision is 30%. Isn't that bad?"**
> Low by design, against a 23% base rate. I optimised expected loss, not precision.
> Approving a defaulter costs half the principal; rejecting a good borrower costs a 10%
> margin. The cutoff deliberately accepts false positives to avoid false negatives.

## The drift work

**"How did you find the drift?"**
> Default rate by origination year — it climbs from 13.7% in 2009 to 23.3% in 2016. My
> training window ended 2015 at 18.5% and I was scoring 2017 at 23.1%.

**"How much did widening the window buy?"**
> Depends how you define the comparison, so here's the range. Holding the cutoff fixed so
> the training window is the only change, 4.5% — that isolates the model. Letting each model
> re-tune its own cutoff, which is what happens on redeployment, 7 to 9% depending on the
> run. The larger figure carries threshold-selection noise: the old model's own cutoff of
> 0.17 actually underperformed 0.16 on test.

**"Adding data moved PR-AUC 0.003 here but 0.021 on your 151-feature model. Why?"**
> Capacity. Same rows, same split, same seed — the 151-feature model went from 423 to 766
> trees and gained 0.021; the 39-feature model went *down* to 350 trees and gained 0.003. It
> had already extracted what its features can express. More data only helps if the model has
> headroom to use it.

## Weaknesses

**"Your data is only loans Lending Club already approved."**
> Reject inference — the sharpest question about this project. Every row passed someone
> else's screen, so I'm modelling default conditional on approval. My probabilities are only
> trustworthy over the region the incumbent policy covered; push the cutoff much looser and
> I'm extrapolating. The rejected-loans file has few fields and no outcome label by
> definition. Standard treatments are parcelling or reweighting under a MAR assumption, none
> validatable from the data alone.

**"You have `region` in the model. Fair lending?"**
> A real problem and I'd treat it as a defect. US geography proxies for race, and ECOA
> judges disparate impact on outcomes, not intent — "I never included race" isn't a defence.
> Region bought +0.0023 PR-AUC, a rounding error against that exposure. I'd drop it or run a
> disparate-impact analysis first. I dropped `zip_code` for exactly this reason and kept
> region, which is the same problem at coarser resolution.

**"Could the same borrower be in both train and test?"**
> I can't verify it. `member_id` is 100% null in the public release, so the duplicate check
> returns nothing. A temporal split limits exposure — a repeat borrower's later loan is
> genuinely in the future — but doesn't eliminate it. With borrower IDs I'd have grouped the
> split.

**"Tell me about a mistake."**
> Two. The methodological one: I originally selected the decision threshold on the test set.
> A threshold is a fitted parameter, so that inflates reported profit and burns the holdout.
> I re-ran it selecting on held-out validation rows and measured the difference at 1.8%. The
> more embarrassing one produced my best-sounding result: I claimed the F1-optimal threshold
> lost money, but I'd optimised F1 on raw scores averaging 0.45 then applied that cutoff to
> calibrated probabilities averaging 0.23. A unit mismatch. Computed consistently, F1 is
> profitable, just far from optimal. The lesson: a result that flatters your thesis is the
> one to re-derive first.

## Production

**"How would you monitor this?"**
> Three layers. Input drift — PSI on each feature and on the score, which fires before any
> outcome data exists. Outcome lag — a 36-month loan doesn't report for years, so
> performance monitoring runs on early-warning proxies like first-payment default rather
> than waiting for the final label. And calibration drift — predicted versus actual by score
> band. Ranking degrades slowly; calibration degrades fast when the macro environment
> shifts, and my whole threshold depends on calibration.

**"How often would you retrain?"**
> Recalibrate far more often than retrain. Recalibration is cheap and needs no
> re-validation of the feature set, and this project is direct evidence that calibration is
> where the value sits — widening the window moved ranking by 0.003 but profit by several
> percent.

**"You surface SHAP reasons. Any problem with that?"**
> One real one. For a verified applicant SHAP reports that verification increased their
> risk. Under ECOA a declined applicant gets the principal reasons for denial, and "your
> income was verified" isn't a reason you can give someone — it's a proxy for an internal
> judgement I never documented. Production credit models use a reason-code mapping that
> excludes proxy features from adverse-action notices. Mine doesn't have one.

## Fundamentals they pivot to

**"Explain expected loss to someone non-technical."**
> Three questions about any loan. How likely is this person to stop paying? If they stop,
> how much do we lose after collections? And how much is at stake? Multiply the three and
> you get what the loan costs on average. It matters because it puts both kinds of mistake
> in the same unit — dollars — so you stop treating them as equally bad, which is exactly
> what accuracy and F1 do.

**"ROC-AUC versus PR-AUC?"**
> Both measure ranking, but ROC-AUC's false-positive rate divides by the entire negative
> class. With a large majority class that denominator dilutes the visible cost of false
> positives, so ROC-AUC flatters under imbalance. PR-AUC focuses on the rare positive class.
> The practical catch is the baseline: ROC-AUC's floor is always 0.5, PR-AUC's floor is the
> positive class rate — 0.2313 here.

**"Why do boosted trees beat a single deep tree?"**
> A single deep tree has low bias and high variance — it memorises. Boosting grows shallow
> trees sequentially, each fitting the residual error of the ensemble so far, so bias comes
> down in controlled steps while the learning rate holds variance in check. Bagging, like
> random forest, attacks variance instead by averaging independent trees. On this data
> boosting won, but only by about 0.02 PR-AUC over random forest.