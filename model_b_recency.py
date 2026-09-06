"""
Rebuild the deployment model (39 features) on the wider training window, end to end.

The recency experiment showed the 151-feature model gains +0.025 test PR-AUC when 2016
joins the training data. This applies the same change to the model that actually ships,
and carries it all the way through to a lending decision.

The split has to be redesigned. Previously 2016 was a clean out-of-time year, so it could
serve as early-stopping set, calibration set and threshold-selection set. Once 2016 is
training data none of those are free, and reusing training rows for calibration would
inflate exactly the probabilities the expected-loss arithmetic depends on. So 2016 is cut
four ways, and every fitted quantity gets rows nothing else touched:

    <=2015  + 60% of 2016   ->  train
              13% of 2016   ->  early stopping (tree count)
              13% of 2016   ->  isotonic calibration
              14% of 2016   ->  threshold selection
    2017 (169,321 rows)     ->  test, read once at the end

Honest limitation, recorded rather than buried: the calibration and threshold rows are
now held-out 2016 rows rather than a wholly unseen year. The model has seen other loans
from the same vintage, so those sets are in-distribution holdout, not out-of-time. Test
remains a genuine future year, which is what the headline numbers are quoted on.
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42
ASSUMED_LGD = 0.50
ASSUMED_PROFIT_MARGIN = 0.10
REGION_COLS = ['region_Northeast', 'region_South', 'region_West']

state = joblib.load('D:/Lending/session_state_b.joblib')
results_b = joblib.load('D:/Lending/model_b_results.joblib')
features = results_b['result_b2']['features']

train_df, val_df, test_df = state['train_df'], state['val_df'], state['test_df']
y_train = np.asarray(state['y_train'])
y_val = np.asarray(state['y_val'])
y_test = np.asarray(state['y_test'])

for d in (train_df, val_df, test_df):
    dummies = pd.get_dummies(d['addr_region'], prefix='region')
    for col in REGION_COLS:
        d[col] = dummies[col] if col in dummies.columns else 0

X_train, X_val, X_test = (d[features].astype(float) for d in (train_df, val_df, test_df))
print(f"features: {len(features)} | test 2017: {X_test.shape[0]:,} rows, "
      f"rate {y_test.mean():.4f}\n", flush=True)

# --- Cut 2016 four ways -------------------------------------------------------------
idx = np.arange(len(val_df))
add_idx, rest = train_test_split(idx, train_size=0.60, random_state=SEED, stratify=y_val)
es_idx, rest2 = train_test_split(rest, train_size=0.325, random_state=SEED, stratify=y_val[rest])
cal_idx, thr_idx = train_test_split(rest2, train_size=0.48, random_state=SEED, stratify=y_val[rest2])
print(f"2016 split -> train {len(add_idx):,} | early-stop {len(es_idx):,} | "
      f"calibrate {len(cal_idx):,} | threshold {len(thr_idx):,}", flush=True)

X_fit = pd.concat([X_train, X_val.iloc[add_idx]])
y_fit = np.concatenate([y_train, y_val[add_idx]])
print(f"train window: {len(y_fit):,} rows, rate {y_fit.mean():.4f} "
      f"(was {len(y_train):,} at {y_train.mean():.4f})\n", flush=True)

# --- Train --------------------------------------------------------------------------
t = time.time()
spw = (y_fit == 0).sum() / (y_fit == 1).sum()
model = XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=6,
                      scale_pos_weight=spw, eval_metric='aucpr',
                      early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
model.fit(X_fit, y_fit, eval_set=[(X_val.iloc[es_idx], y_val[es_idx])], verbose=False)
print(f"trained: best_iter={model.best_iteration} in {time.time()-t:.0f}s", flush=True)

# --- Calibrate ----------------------------------------------------------------------
calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calibrated.fit(X_val.iloc[cal_idx], y_val[cal_idx])

p_thr = calibrated.predict_proba(X_val.iloc[thr_idx])[:, 1]
p_test = calibrated.predict_proba(X_test)[:, 1]

pr, roc = average_precision_score(y_test, p_test), roc_auc_score(y_test, p_test)
brier = brier_score_loss(y_test, p_test)
print(f"\ntest PR-AUC {pr:.4f} | ROC-AUC {roc:.4f} | Brier {brier:.4f}", flush=True)
print(f"mean predicted {p_test.mean():.4f} vs actual {y_test.mean():.4f}", flush=True)


def sweep(probs, y, amnt, grid):
    y = np.asarray(y); ead = np.asarray(amnt, dtype=float)
    out = []
    for t_ in grid:
        ok = probs < t_
        out.append({'threshold': t_, 'approval_rate': ok.mean(),
                    'net': ead[ok & (y == 0)].sum() * ASSUMED_PROFIT_MARGIN
                           - ead[ok & (y == 1)].sum() * ASSUMED_LGD})
    return pd.DataFrame(out)


grid = np.round(np.arange(0.02, 0.61, 0.01), 2)
val_sweep = sweep(p_thr, y_val[thr_idx], val_df['loan_amnt'].to_numpy()[thr_idx], grid)
best_t = float(val_sweep.loc[val_sweep['net'].idxmax(), 'threshold'])
test_sweep = sweep(p_test, y_test, test_df['loan_amnt'], grid)
at = test_sweep.loc[test_sweep['threshold'] == best_t].iloc[0]

oracle = test_sweep.loc[test_sweep['net'].idxmax()]
print(f"\nthreshold {best_t:.2f} (selected on validation)", flush=True)
print(f"test net ${at['net']:,.0f} at {at['approval_rate']:.1%} approval", flush=True)
print(f"oracle {oracle['threshold']:.2f} -> ${oracle['net']:,.0f} "
      f"(optimism {oracle['net']-at['net']:,.0f}, "
      f"{(oracle['net']-at['net'])/oracle['net']*100:.1f}%)", flush=True)

# --- Compare against the shipped model ----------------------------------------------
old = joblib.load('D:/Lending/deployment_artifacts.joblib')
print("\n=== shipped vs retrained ===", flush=True)
print(f"{'':22s} {'shipped':>12s} {'retrained':>12s} {'delta':>10s}", flush=True)
for label, o, n in [('test PR-AUC', old['test_pr_auc'], pr),
                    ('test ROC-AUC', old['test_roc_auc'], roc),
                    ('threshold', old['threshold'], best_t),
                    ('approval rate', old['test_approval_rate'], at['approval_rate']),
                    ('test net ($M)', old['test_net_at_threshold']/1e6, at['net']/1e6)]:
    print(f"{label:22s} {o:12.4f} {n:12.4f} {n-o:+10.4f}", flush=True)

joblib.dump({
    'model': model, 'calibrated_model': calibrated, 'features': features,
    'threshold': best_t,
    'medians': X_fit.median(numeric_only=True).to_dict(),
    'test_pr_auc': pr, 'test_roc_auc': roc, 'test_brier': brier,
    'test_net_at_threshold': float(at['net']),
    'test_approval_rate': float(at['approval_rate']),
    'assumed_lgd': ASSUMED_LGD, 'assumed_profit_margin': ASSUMED_PROFIT_MARGIN,
    'train_rows': int(len(y_fit)), 'best_iter': int(model.best_iteration),
    'val_sweep': val_sweep, 'test_sweep': test_sweep,
}, 'D:/Lending/deployment_artifacts_recency.joblib')
np.save('D:/Lending/_recency_modelb_test.npy', p_test)
print("\nSaved deployment_artifacts_recency.joblib (not yet promoted)", flush=True)
