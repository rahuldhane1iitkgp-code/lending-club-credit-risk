# Lending Club credit risk - full pipeline in one file
# Raw CSV -> leakage removal -> encoding -> temporal split -> model -> approval cutoff
#
# Run: python credit_risk_pipeline.py
# Takes ~15 min. Needs accepted_2007_to_2018Q4.csv in the same folder.

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             recall_score, precision_score, confusion_matrix)
from xgboost import XGBClassifier

SEED = 42
CSV = 'accepted_2007_to_2018Q4.csv'

# Economics for the approval decision. These are assumptions, not derived from the data -
# the columns that would give real recovery rates (recoveries, out_prncp) are post-issuance
# and got dropped as leakage.
LGD = 0.50            # we lose half the principal on a default
MARGIN = 0.10         # we make 10% on a loan that repays

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)


# --- 1. Load and define the target -------------------------------------------------

log("loading csv")
df = pd.read_csv(CSV, low_memory=False)
print(f"raw: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Only loans with a final outcome can be labelled. Current / Late / In Grace Period are
# still running, so calling them 0 would mislabel every default that hasn't happened yet.
closed = ['Fully Paid', 'Charged Off', 'Default',
          'Does not meet the credit policy. Status:Fully Paid',
          'Does not meet the credit policy. Status:Charged Off']
df = df[df['loan_status'].isin(closed)].copy()
df['target'] = df['loan_status'].str.contains('Charged Off|Default').astype(int)
print(f"resolved loans: {len(df):,}  default rate: {df['target'].mean():.4f}")


# --- 2. Leakage and circularity ----------------------------------------------------

# 2a. Post-issuance. These only exist because of what happened DURING the loan, so a model
# using them is reading the outcome rather than predicting it. Checked this with a
# univariate AUC screen - last_fico_range_low alone scores 0.94, which is impossible on
# application-time data and is exactly how I found them.
leakage = ['total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
           'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
           'out_prncp', 'out_prncp_inv', 'last_pymnt_d', 'last_pymnt_amnt',
           'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low',
           'next_pymnt_d', 'debt_settlement_flag', 'debt_settlement_flag_date',
           'hardship_flag', 'payment_plan_start_date', 'pymnt_plan',
           'orig_projected_additional_accrued_interest', 'deferral_term']
leakage += [c for c in df.columns if c.startswith(('hardship_', 'settlement_'))]

# 2b. Circularity - different problem. grade / sub_grade / int_rate ARE known at
# application time, but they're Lending Club's own risk score, not borrower facts. Keeping
# them means partly learning to decode someone else's model. Checked: sub_grade explains
# 95.5% of int_rate variance, so they're one variable. installment goes too because it's
# recoverable from loan_amnt + term + int_rate.
circular = ['grade', 'sub_grade', 'int_rate', 'installment']

# 2c. Identifiers and redundant text. title is 63k free-text values for the same thing as
# purpose (14 clean categories). emp_title is 512k unique - real signal, but TF-IDF on job
# titles is its own project. member_id is 100% null in the public release.
ids = ['id', 'member_id', 'url', 'zip_code', 'title', 'emp_title', 'desc', 'policy_code']

# 2d. Available in training but not consistently over time. These bureau columns don't
# exist before 2016 - 100% null up to 2014, 0% from 2016. Training is <=2016 and test is
# 2017, so the model would barely see them in training then meet them fully populated at
# scoring time. Only found this by checking null rate BY YEAR, which a random split hides.
not_available_early = ['open_rv_24m', 'open_rv_12m', 'open_acc_6m', 'inq_last_12m',
                       'all_util', 'il_util', 'max_bal_bc', 'total_cu_tl', 'inq_fi',
                       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il']

df = df.drop(columns=[c for c in leakage + circular + ids + not_available_early
                      if c in df.columns])
print(f"after leakage/circularity drops: {df.shape[1]} columns")


# --- 3. Joint applications ---------------------------------------------------------

# annual_inc_joint is 94.7% null, which looks droppable. It isn't - it's 0% null WITHIN
# joint applications, so it's structurally null, not sparse. Verified the hypothesis
# first: every joint row has annual_inc_joint >= annual_inc (comes out exactly 1.0).
jr = df[df['application_type'] == 'Joint App']
print(f"joint rows: {len(jr):,}  "
      f"joint income >= individual income: {(jr['annual_inc_joint'] >= jr['annual_inc']).mean():.4f}")

df['combined_annual_inc'] = df['annual_inc_joint'].fillna(df['annual_inc'])
df['combined_dti'] = df['dti_joint'].fillna(df['dti'])
df['fico_score'] = (df['sec_app_fico_range_low'].fillna(df['fico_range_low']) +
                    df['sec_app_fico_range_high'].fillna(df['fico_range_high'])) / 2
df['is_joint'] = (df['application_type'] == 'Joint App').astype(int)


# --- 4. Encoding -------------------------------------------------------------------

# Ordinal where the order is real. 10+ years caps at 10 (right-censored) - acceptable
# because the default rate is basically flat past year 1 anyway.
emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
           '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
           '10+ years': 10}
df['emp_length'] = df['emp_length'].map(emp_map)

df['term'] = df['term'].str.strip().str.replace(' months', '').astype(int)
df['disbursement_method'] = df['disbursement_method'].map({'Cash': 0, 'DirectPay': 1})

# verification_status is ordinal, but the direction is counterintuitive: Not Verified
# defaults at 14.7%, Verified at 23.9%. That's not verification causing default - LC
# demands documentation when an application already looks shaky, so it's a proxy for
# underwriter suspicion. Encoding ordinally is still valid because it's monotone.
print("\ndefault rate by verification_status:")
print(df.groupby('verification_status')['target'].agg(['mean', 'size']).round(4))
df['verification_status'] = df['verification_status'].map(
    {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2})

# ANY / NONE have 286 and 53 rows. A dummy fitted on 53 rows learns noise, so fold them in.
df['home_ownership'] = df['home_ownership'].replace(['ANY', 'NONE'], 'OTHER')

# 50 state dummies for one feature is a lot of width, and zip_code was already dropped as a
# fair-lending proxy. Grouping to 4 census regions keeps most of the geography signal.
regions = {'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
           'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast',
           'PA': 'Northeast',
           'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest',
           'WI': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest',
           'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
           'DE': 'South', 'DC': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South',
           'NC': 'South', 'SC': 'South', 'VA': 'South', 'WV': 'South', 'AL': 'South',
           'KY': 'South', 'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South',
           'OK': 'South', 'TX': 'South',
           'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
           'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
           'HI': 'West', 'OR': 'West', 'WA': 'West'}
df['region'] = df['addr_state'].map(regions).fillna('Other')

df = pd.get_dummies(df, columns=['home_ownership'], prefix='home', drop_first=True)
df = pd.get_dummies(df, columns=['purpose'], prefix='purpose', drop_first=True)
df = pd.get_dummies(df, columns=['region'], prefix='region', drop_first=True)

df['credit_history_months'] = ((pd.to_datetime(df['issue_d'], format='%b-%Y') -
                                pd.to_datetime(df['earliest_cr_line'], format='%b-%Y'))
                               .dt.days / 30.44)

# Sanity checks. Negative DTI is impossible -> null it and let XGBoost route it.
# revol_util > 100 is left alone on purpose: you CAN exceed your credit limit, and
# being over-limit is a real and predictive state, not a data error.
print(f"\nnegative DTI (nulled): {(df['combined_dti'] < 0).sum()}")
print(f"revol_util > 100 (kept): {(df['revol_util'] > 100).sum()}")
df.loc[df['combined_dti'] < 0, 'combined_dti'] = np.nan


# --- 5. Temporal split -------------------------------------------------------------

df['issue_year'] = pd.to_datetime(df['issue_d'], format='%b-%Y').dt.year

# Default rate by vintage. Two things to read off this: the rate climbs steadily (that's
# the drift I deal with in section 8), and 2018 DROPS - which isn't better lending, it's
# right-censoring. Only ~11% of 2018 loans had resolved by the end of the data, and a loan
# can only close that fast via an early exit. So 2018 comes out.
vint = df.groupby('issue_year')['target'].agg(['mean', 'size']).round(4)
print("\ndefault rate by origination year:")
print(vint.to_string())

df = df[df['issue_year'] != 2018].copy()

FEATURES = (['loan_amnt', 'term', 'emp_length', 'combined_annual_inc', 'verification_status',
             'combined_dti', 'fico_score', 'delinq_2yrs', 'pub_rec', 'open_acc', 'total_acc',
             'revol_util', 'mort_acc', 'credit_history_months', 'is_joint',
             'disbursement_method', 'loan_to_income', 'avg_revol_balance',
             'open_to_total_acc', 'accounts_per_credit_year']
            + [c for c in df.columns if c.startswith(('home_', 'purpose_', 'region_'))])

# Engineered ratios. All row-wise, so building them before or after the split gives
# identical values - no leakage either way. I build them after out of habit, because the
# moment a transform FITS something (an imputer, a scaler, a target encoding) the order
# stops being cosmetic.
def add_ratios(d):
    d = d.copy()
    d['loan_to_income'] = d['loan_amnt'] / d['combined_annual_inc'].replace(0, np.nan)
    d['avg_revol_balance'] = d['revol_bal'] / (d['open_acc'] + 1)
    d['open_to_total_acc'] = d['open_acc'] / (d['total_acc'] + 1)
    d['accounts_per_credit_year'] = d['total_acc'] / (d['credit_history_months'] / 12 + 1)
    return d

df = add_ratios(df)
train_old = df[df['issue_year'] <= 2015]
val_2016 = df[df['issue_year'] == 2016]
test = df[df['issue_year'] == 2017]

X_test, y_test = test[FEATURES].astype(float), test['target'].to_numpy()
ead_test = test['loan_amnt'].to_numpy(float)

print(f"\ntrain <=2015 {len(train_old):>8,}  default {train_old['target'].mean():.4f}")
print(f"val   2016   {len(val_2016):>8,}  default {val_2016['target'].mean():.4f}")
print(f"test  2017   {len(test):>8,}  default {test['target'].mean():.4f}")
print(f"features: {len(FEATURES)}")


# --- 6. Baseline before anything clever --------------------------------------------

# Worth knowing how much the tree model is actually buying. Logistic regression needs
# imputation and scaling; XGBoost needs neither, and it handles the structural missingness
# natively instead of pretending a missing value is the median.
log("fitting logistic regression baseline")
med = train_old[FEATURES].astype(float).median()
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(max_iter=1000, class_weight='balanced',
                                      random_state=SEED))
lr.fit(train_old[FEATURES].astype(float).fillna(med), train_old['target'])
lr_p = lr.predict_proba(X_test.fillna(med))[:, 1]
print(f"logistic regression  test ROC-AUC {roc_auc_score(y_test, lr_p):.4f}  "
      f"PR-AUC {average_precision_score(y_test, lr_p):.4f}")


# --- 7. XGBoost trained on loans up to 2015 ----------------------------------------

def fit_xgb(X, y, X_es, y_es):
    spw = (y == 0).sum() / (y == 1).sum()
    m = XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=6,
                      scale_pos_weight=spw,
                      eval_metric='aucpr',            # must match what I select on
                      early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
    m.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
    return m, spw

log("fitting xgboost on <=2015")
m_old, spw = fit_xgb(train_old[FEATURES].astype(float), train_old['target'],
                     val_2016[FEATURES].astype(float), val_2016['target'])
p_old = m_old.predict_proba(X_test)[:, 1]
print(f"scale_pos_weight {spw:.4f}  trees {m_old.best_iteration}")
print(f"xgboost (<=2015)     test ROC-AUC {roc_auc_score(y_test, p_old):.4f}  "
      f"PR-AUC {average_precision_score(y_test, p_old):.4f}")


# --- 8. The training window --------------------------------------------------------

# The model above trains on loans up to 2015 and is scored on 2017, across a jump in
# default rate from 18.5% to 23.1%. It's extrapolating on stale data. Adding 2016 should
# help - but 2016 was doing three jobs (early stopping, calibration, threshold), so once
# it becomes training data each of those needs its own rows.
idx = np.arange(len(val_2016))
y16 = val_2016['target'].to_numpy()
add_i, rest = train_test_split(idx, train_size=0.60, random_state=SEED, stratify=y16)
es_i, rest2 = train_test_split(rest, train_size=0.325, random_state=SEED, stratify=y16[rest])
cal_i, thr_i = train_test_split(rest2, train_size=0.48, random_state=SEED, stratify=y16[rest2])
print(f"\n2016 split -> train {len(add_i):,} | early-stop {len(es_i):,} | "
      f"calibrate {len(cal_i):,} | threshold {len(thr_i):,}")

X16 = val_2016[FEATURES].astype(float)
X_fit = pd.concat([train_old[FEATURES].astype(float), X16.iloc[add_i]])
y_fit = np.concatenate([train_old['target'].to_numpy(), y16[add_i]])

log("fitting xgboost on <=2016")
model, _ = fit_xgb(X_fit, y_fit, X16.iloc[es_i], y16[es_i])
print(f"train rows {len(y_fit):,}  trees {model.best_iteration}")


# --- 9. Calibration ----------------------------------------------------------------

# Raw XGBoost scores aren't probabilities here. scale_pos_weight of 4.4 makes the effective
# class balance 50/50, so predictions centre near 0.5 instead of 0.23. Ranking is fine -
# PR-AUC doesn't notice a monotone distortion - but the approval rule below multiplies
# PD by loss, so the arithmetic breaks unless the numbers are real.
raw = model.predict_proba(X_test)[:, 1]
print(f"\nuncalibrated: mean predicted {raw.mean():.4f} vs actual {y_test.mean():.4f}, "
      f"Brier {brier_score_loss(y_test, raw):.4f}")
print(f"  (a constant at the base rate scores "
      f"{brier_score_loss(y_test, np.full_like(raw, y_test.mean())):.4f} - the model is worse)")

# Isotonic, not Platt: the distortion is monotone but there's no reason for it to be
# sigmoidal, and with 38k calibration rows isotonic won't overfit.
cal = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
cal.fit(X16.iloc[cal_i], y16[cal_i])
p_test = cal.predict_proba(X_test)[:, 1]
print(f"calibrated:   mean predicted {p_test.mean():.4f} vs actual {y_test.mean():.4f}, "
      f"Brier {brier_score_loss(y_test, p_test):.4f}")


# --- 10. Approval cutoff from expected loss ----------------------------------------

# A probability isn't a decision. F1 treats a rejected good borrower and an approved
# defaulter as equally bad; in lending they differ by about 5x. So sweep the cutoff on
# net profit instead. The cutoff is a fitted parameter, so it gets picked on held-out 2016
# rows and test is read once at the end.
def net_profit(p, y, ead, cut):
    ok = p < cut
    return (ead[ok & (y == 0)].sum() * MARGIN) - (ead[ok & (y == 1)].sum() * LGD), ok.mean()

p_thr = cal.predict_proba(X16.iloc[thr_i])[:, 1]
y_thr, ead_thr = y16[thr_i], val_2016['loan_amnt'].to_numpy(float)[thr_i]

grid = np.round(np.arange(0.02, 0.61, 0.01), 2)
sweep = [(c,) + net_profit(p_thr, y_thr, ead_thr, c) for c in grid]
cutoff = max(sweep, key=lambda r: r[1])[0]
print(f"\ncutoff chosen on held-out 2016 rows: {cutoff:.2f}")

print("\ncutoff  approve   test net profit")
for c in [0.05, 0.11, 0.14, cutoff, 0.20, 0.26, 0.32]:
    n, ap = net_profit(p_test, y_test, ead_test, c)
    mark = '  <-- chosen' if c == cutoff else ''
    print(f"  {c:.2f}   {ap:6.1%}   ${n/1e6:8.2f}M{mark}")


# --- 11. Final report on 2017 ------------------------------------------------------

# Two ways to compare the old and new training windows, and they answer different
# questions - so I report both rather than picking the flattering one.
cal_old = CalibratedClassifierCV(m_old, method='isotonic', cv='prefit')
cal_old.fit(X16.iloc[cal_i], y16[cal_i])
p_old_cal = cal_old.predict_proba(X_test)[:, 1]

# (a) Hold the cutoff fixed. Only the training window differs, so this isolates the model.
fixed_old, _ = net_profit(p_old_cal, y_test, ead_test, cutoff)

# (b) Let the old model pick its own cutoff too, the way it would have in deployment.
# This is the realistic "retrain and re-tune" number, but it changes two things at once.
p_thr_old = cal_old.predict_proba(X16.iloc[thr_i])[:, 1]
cutoff_old = max([(c,) + net_profit(p_thr_old, y_thr, ead_thr, c) for c in grid],
                 key=lambda r: r[1])[0]
retuned_old, _ = net_profit(p_old_cal, y_test, ead_test, cutoff_old)

new_net, approval = net_profit(p_test, y_test, ead_test, cutoff)
print(f"\nold model's own cutoff: {cutoff_old:.2f}   (new model's: {cutoff:.2f})")

approved = p_test < cutoff
flagged = (p_test >= cutoff).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, flagged).ravel()

print("\n" + "=" * 60)
print("FINAL - 2017 holdout, read once")
print("=" * 60)
print(f"ROC-AUC                     {roc_auc_score(y_test, p_test):.4f}")
print(f"PR-AUC                      {average_precision_score(y_test, p_test):.4f}"
      f"   (no-skill floor {y_test.mean():.4f})")
print(f"Recall for defaults         {recall_score(y_test, flagged):.4f}")
print(f"Precision                   {precision_score(y_test, flagged):.4f}")
print()
print(f"Approved                    {approved.sum():,} of {len(y_test):,}  ({approval:.1%})")
print(f"Default rate if we approved everyone   {y_test.mean():.2%}")
print(f"Default rate among approved            {y_test[approved].mean():.2%}")
print()
print(f"Net profit, trained <=2016                   ${new_net/1e6:.2f}M")
print(f"  vs <=2015, each model re-tuning its cutoff ${retuned_old/1e6:.2f}M"
      f"   ({new_net/retuned_old - 1:+.1%})   <- full re-deployment")
print(f"  vs <=2015, both held at the same cutoff    ${fixed_old/1e6:.2f}M"
      f"   ({new_net/fixed_old - 1:+.1%})   <- model effect alone")
print(f"\n(LGD {LGD:.0%}, margin {MARGIN:.0%} - stated assumptions, not derived)")
log("done")
