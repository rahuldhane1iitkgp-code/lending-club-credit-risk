"""
Paired bootstrap on the test set: is any model's PR-AUC edge over XGBoost real?

The benchmark table shows four gradient-boosting implementations within 0.001 PR-AUC of
each other. A point estimate cannot say whether that ordering means anything. This
resamples the test set with replacement and recomputes every model's PR-AUC on the SAME
resample each time, so the differences are paired and the shared sampling noise cancels.

Reports the 95% percentile interval on each difference and the fraction of resamples in
which the challenger beats XGBoost. An interval spanning zero means the ordering in the
benchmark table is not distinguishable from chance.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SEED = 42
N_BOOT = 1000

state = joblib.load('D:/Lending/session_state_b.joblib')
y_test = np.asarray(state['y_test'])

bench = joblib.load('D:/Lending/model_benchmark.joblib')
scores = dict(bench['test_scores'])

# The sklearn-wrapper LightGBM runs early-stopped at iterations 7 and 17 because
# eval_set is deprecated in LightGBM 4.7 and the validation set was not wired through.
# Those rows measured a broken configuration, not the algorithm, so they are replaced
# by the native-API refit (523 iterations) and dropped from the comparison.
scores.pop('LightGBM (matched depth-6)', None)
scores.pop('LightGBM (leaf-wise, 127)', None)
scores.pop('Rank-avg ensemble (4 GBMs)', None)
scores['LightGBM'] = joblib.load('D:/Lending/lgbm_fixed.joblib')['lgbm_test']

from scipy.stats import rankdata
n = len(y_test)
scores['Ensemble (XGB+HGB+LGBM)'] = np.mean(
    [rankdata(scores[m]) / n for m in ['XGBoost (incumbent)', 'HistGradientBoosting', 'LightGBM']],
    axis=0)

REF = 'XGBoost (incumbent)'
names = [m for m in scores if m != REF]
print(f"Reference: {REF}")
print(f"Test rows: {n} | positives: {y_test.sum()} ({y_test.mean():.4f})")
print(f"Bootstrap resamples: {N_BOOT}\n")

rng = np.random.default_rng(SEED)
diffs = {m: np.empty(N_BOOT) for m in names}
ref_ap = np.empty(N_BOOT)

for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    yb = y_test[idx]
    if yb.sum() == 0:
        continue
    r = average_precision_score(yb, scores[REF][idx])
    ref_ap[b] = r
    for m in names:
        diffs[m][b] = average_precision_score(yb, scores[m][idx]) - r

rows = []
for m in names:
    d = diffs[m]
    lo, hi = np.percentile(d, [2.5, 97.5])
    rows.append({
        'model': m,
        'point_diff': average_precision_score(y_test, scores[m]) - average_precision_score(y_test, scores[REF]),
        'boot_mean_diff': d.mean(),
        'ci_low': lo,
        'ci_high': hi,
        'p_beats_xgb': (d > 0).mean(),
        'significant': 'yes' if (lo > 0 or hi < 0) else 'no',
    })

df = pd.DataFrame(rows).sort_values('point_diff', ascending=False)
pd.set_option('display.width', 200)
print("=== Paired bootstrap: PR-AUC difference vs XGBoost ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"\nXGBoost test PR-AUC: {average_precision_score(y_test, scores[REF]):.4f} "
      f"(bootstrap 95% CI {np.percentile(ref_ap, 2.5):.4f} to {np.percentile(ref_ap, 97.5):.4f})")

df.to_csv('D:/Lending/reports_bootstrap_benchmark.csv', index=False)
joblib.dump({'diffs': diffs, 'summary': df, 'n_boot': N_BOOT}, 'D:/Lending/bootstrap_benchmark.joblib')
print("\nSaved reports_bootstrap_benchmark.csv")
