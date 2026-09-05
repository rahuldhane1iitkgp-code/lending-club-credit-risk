"""
Step 3 — expected-loss threshold selection.

The decision threshold is a fitted parameter: choosing it on the test set and then
reporting profit at that threshold reports the maximum of a noisy surface, which is
optimistically biased. So the validation set is split in two:

    val_cal  -> fit isotonic calibration
    val_thr  -> sweep and select the profit-maximising threshold
    test     -> report profit at that (already-fixed) threshold, untouched

Splitting validation also stops the calibrator being fit on the same rows used to pick
the threshold. The test-selected threshold is still computed below, purely to quantify
how much optimism the old approach carried.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

SEED = 42
ASSUMED_LGD = 0.50
ASSUMED_PROFIT_MARGIN = 0.10
REGION_COLS = ['region_Northeast', 'region_South', 'region_West']

state = joblib.load('D:/Lending/session_state_b.joblib')
results = joblib.load('D:/Lending/model_b_results.joblib')

val_df, test_df = state['val_df'], state['test_df']
y_val, y_test = state['y_val'], state['y_test']
result_b2 = results['result_b2']
model = result_b2['model']           # uncalibrated XGBoost
features = result_b2['features']

# Region dummies, same construction as step 2
for d in (val_df, test_df):
    dummies = pd.get_dummies(d['addr_region'], prefix='region')
    for col in REGION_COLS:
        d[col] = dummies[col] if col in dummies.columns else 0

X_val, X_test = val_df[features], test_df[features]

# --- Split validation: half to calibrate, half to choose the threshold -------------
cal_idx, thr_idx = train_test_split(
    np.arange(len(val_df)), test_size=0.5, random_state=SEED, stratify=y_val
)
print(f"Validation split: {len(cal_idx)} rows to calibrate, {len(thr_idx)} to select threshold")

calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calibrated.fit(X_val.iloc[cal_idx], np.asarray(y_val)[cal_idx])

probs_thr = calibrated.predict_proba(X_val.iloc[thr_idx])[:, 1]
probs_test = calibrated.predict_proba(X_test)[:, 1]


def sweep(probs, y, loan_amnt, grid):
    """Net profit over the loans this policy would approve, at each threshold."""
    y = np.asarray(y)
    ead = np.asarray(loan_amnt, dtype=float)
    rows = []
    for t in grid:
        approved = probs < t
        profit = ead[approved & (y == 0)].sum() * ASSUMED_PROFIT_MARGIN
        loss = ead[approved & (y == 1)].sum() * ASSUMED_LGD
        rows.append({'threshold': t,
                     'approval_rate': approved.mean(),
                     'net': profit - loss})
    return pd.DataFrame(rows)


grid = np.round(np.arange(0.02, 0.61, 0.01), 2)

# --- Select on validation ----------------------------------------------------------
val_sweep = sweep(probs_thr, np.asarray(y_val)[thr_idx],
                  val_df['loan_amnt'].to_numpy()[thr_idx], grid)
best_threshold = float(val_sweep.loc[val_sweep['net'].idxmax(), 'threshold'])
print(f"\nSelected threshold (validation): {best_threshold:.2f}")

# --- Report on test at the fixed threshold -----------------------------------------
test_sweep = sweep(probs_test, y_test, test_df['loan_amnt'], grid)
test_at_selected = test_sweep.loc[test_sweep['threshold'] == best_threshold].iloc[0]

print(f"Test net profit at that threshold: ${test_at_selected['net']:,.0f} "
      f"(approval rate {test_at_selected['approval_rate']:.1%})")

# --- Quantify the optimism of the old (test-selected) approach ---------------------
oracle = test_sweep.loc[test_sweep['net'].idxmax()]
optimism = oracle['net'] - test_at_selected['net']
print(f"\nTest-selected 'oracle' threshold: {oracle['threshold']:.2f} -> ${oracle['net']:,.0f}")
print(f"Selection optimism avoided: ${optimism:,.0f} "
      f"({optimism / oracle['net'] * 100:.1f}% of the oracle figure)")

print("\n=== Test sweep around the selected threshold ===")
window = test_sweep[(test_sweep['threshold'] >= 0.05) & (test_sweep['threshold'] <= 0.35)]
print(window.iloc[::3].to_string(index=False,
                                 formatters={'net': '${:,.0f}'.format,
                                             'approval_rate': '{:.1%}'.format}))

joblib.dump({
    'best_threshold': best_threshold,
    'calibrated_model': calibrated,
    'val_sweep': val_sweep,
    'test_sweep': test_sweep,
    'test_net_at_selected': float(test_at_selected['net']),
    'test_approval_rate_at_selected': float(test_at_selected['approval_rate']),
    'oracle_threshold': float(oracle['threshold']),
    'oracle_net': float(oracle['net']),
    'selection_optimism': float(optimism),
    'assumed_lgd': ASSUMED_LGD,
    'assumed_profit_margin': ASSUMED_PROFIT_MARGIN,
}, 'D:/Lending/model_b_threshold.joblib')
print("\nSaved model_b_threshold.joblib")
