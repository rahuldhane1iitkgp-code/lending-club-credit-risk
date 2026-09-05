"""
Model A — threshold selection under the same corrected procedure as model_b_step3.py.

The original Model A sweep (in the research notebook) chose its threshold against test
outcomes. This re-runs it honestly: calibrate on one half of validation, select the
threshold on the other half, read test once at the fixed cutoff. Also recomputes the
F1-optimal threshold on validation so the "F1 loses money" comparison is like-for-like.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

SEED = 42
ASSUMED_LGD = 0.50
ASSUMED_PROFIT_MARGIN = 0.10

state = joblib.load('D:/Lending/session_state_b.joblib')
val_df, test_df = state['val_df'], state['test_df']
y_val, y_test = np.asarray(state['y_val']), np.asarray(state['y_test'])
model_a = state['model2']

features_a = list(model_a.get_booster().feature_names)
missing = [f for f in features_a if f not in val_df.columns]
if missing:
    raise SystemExit(f"Model A features absent from the saved frames: {missing}")
print(f"Model A: {len(features_a)} features")

X_val, X_test = val_df[features_a], test_df[features_a]

cal_idx, thr_idx = train_test_split(
    np.arange(len(val_df)), test_size=0.5, random_state=SEED, stratify=y_val
)
print(f"Validation split: {len(cal_idx)} calibrate / {len(thr_idx)} select")

calibrated = CalibratedClassifierCV(model_a, method='isotonic', cv='prefit')
calibrated.fit(X_val.iloc[cal_idx], y_val[cal_idx])

probs_thr = calibrated.predict_proba(X_val.iloc[thr_idx])[:, 1]
probs_test = calibrated.predict_proba(X_test)[:, 1]


def sweep(probs, y, loan_amnt, grid):
    ead = np.asarray(loan_amnt, dtype=float)
    rows = []
    for t in grid:
        approved = probs < t
        profit = ead[approved & (y == 0)].sum() * ASSUMED_PROFIT_MARGIN
        loss = ead[approved & (y == 1)].sum() * ASSUMED_LGD
        rows.append({'threshold': t, 'approval_rate': approved.mean(), 'net': profit - loss})
    return pd.DataFrame(rows)


grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
y_thr = y_val[thr_idx]
amnt_thr = val_df['loan_amnt'].to_numpy()[thr_idx]

# --- Profit-optimal, selected on validation ---------------------------------------
val_sweep = sweep(probs_thr, y_thr, amnt_thr, grid)
best_threshold = float(val_sweep.loc[val_sweep['net'].idxmax(), 'threshold'])

test_sweep = sweep(probs_test, y_test, test_df['loan_amnt'], grid)
at_selected = test_sweep.loc[test_sweep['threshold'] == best_threshold].iloc[0]

print(f"\nProfit-optimal threshold (validation): {best_threshold:.2f}")
print(f"  Test net: ${at_selected['net']:,.0f} at {at_selected['approval_rate']:.1%} approval")

# --- F1-optimal, also selected on validation ---------------------------------------
f1_scores = [(t, f1_score(y_thr, (probs_thr >= t).astype(int))) for t in grid]
f1_threshold = float(max(f1_scores, key=lambda r: r[1])[0])
at_f1 = test_sweep.loc[test_sweep['threshold'] == f1_threshold].iloc[0]
print(f"\nF1-optimal threshold (validation): {f1_threshold:.2f}")
print(f"  Test net: ${at_f1['net']:,.0f} at {at_f1['approval_rate']:.1%} approval")

# --- Optimism the old test-selected approach carried --------------------------------
oracle = test_sweep.loc[test_sweep['net'].idxmax()]
optimism = oracle['net'] - at_selected['net']
print(f"\nTest-selected 'oracle': {oracle['threshold']:.2f} -> ${oracle['net']:,.0f}")
print(f"Selection optimism: ${optimism:,.0f} ({optimism / oracle['net'] * 100:.1f}%)")

print("\n=== Test sweep ===")
show = test_sweep[test_sweep['threshold'].isin([0.05, 0.10, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50])]
print(show.to_string(index=False, formatters={'net': '${:,.0f}'.format,
                                              'approval_rate': '{:.1%}'.format}))

joblib.dump({
    'best_threshold': best_threshold,
    'test_net_at_selected': float(at_selected['net']),
    'test_approval_rate_at_selected': float(at_selected['approval_rate']),
    'f1_threshold': f1_threshold,
    'test_net_at_f1': float(at_f1['net']),
    'test_approval_rate_at_f1': float(at_f1['approval_rate']),
    'oracle_threshold': float(oracle['threshold']),
    'oracle_net': float(oracle['net']),
    'selection_optimism': float(optimism),
    'val_sweep': val_sweep,
    'test_sweep': test_sweep,
}, 'D:/Lending/model_a_threshold.joblib')
print("\nSaved model_a_threshold.joblib")
