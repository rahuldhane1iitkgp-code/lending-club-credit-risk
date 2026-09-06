"""
Control: does the 151-feature recency gain survive the split used for the deployment model?

The first recency experiment added 80% of 2016 to training and early-stopped on the
remaining 20%, scoring +0.0250 test PR-AUC over the incumbent. The deployment rebuild
used a stricter four-way cut of 2016 (60% train / 13% early-stop / 13% calibrate /
14% threshold) and gained only +0.0034 on 39 features.

Those are two different training regimes, so the two results cannot be compared. This
runs the 151-feature model through the deployment split exactly, isolating the feature
set as the only difference. If the gain holds near +0.025 the earlier finding stands and
the 39-feature model is simply capacity-limited; if it collapses toward +0.003 then the
looser early-stopping regime was doing the work and the headline was overstated.
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42

state = joblib.load('D:/Lending/session_state_b.joblib')
train_df, val_df, test_df = state['train_df'], state['val_df'], state['test_df']
y_train = np.asarray(state['y_train'])
y_val = np.asarray(state['y_val'])
y_test = np.asarray(state['y_test'])

features = list(state['model2'].get_booster().feature_names)
X_train, X_val, X_test = (d[features].astype(float) for d in (train_df, val_df, test_df))

# Identical split to model_b_recency.py, same seed and proportions
idx = np.arange(len(val_df))
add_idx, rest = train_test_split(idx, train_size=0.60, random_state=SEED, stratify=y_val)
es_idx, _ = train_test_split(rest, train_size=0.325, random_state=SEED, stratify=y_val[rest])

X_fit = pd.concat([X_train, X_val.iloc[add_idx]])
y_fit = np.concatenate([y_train, y_val[add_idx]])
print(f"151 features | train {len(y_fit):,} rows | early-stop {len(es_idx):,} rows", flush=True)

t = time.time()
spw = (y_fit == 0).sum() / (y_fit == 1).sum()
m = XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=6, scale_pos_weight=spw,
                  eval_metric='aucpr', early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
m.fit(X_fit, y_fit, eval_set=[(X_val.iloc[es_idx], y_val[es_idx])], verbose=False)
p = m.predict_proba(X_test)[:, 1]
pr = average_precision_score(y_test, p)

print(f"best_iter={m.best_iteration} in {time.time()-t:.0f}s", flush=True)
print(f"\ntest PR-AUC {pr:.4f} | ROC {roc_auc_score(y_test, p):.4f}\n", flush=True)
print("comparison, 151 features:", flush=True)
print(f"  A  <=2015, early-stop on full 2016   0.4172   (incumbent, best_iter 423)", flush=True)
print(f"  B  <=2016 80%, early-stop on 20%     0.4410   (+0.0238, best_iter 859)", flush=True)
print(f"  D  <=2016 60%, deployment-split      {pr:.4f}   ({pr-0.4172:+.4f}, "
      f"best_iter {m.best_iteration})", flush=True)

np.save('D:/Lending/_recency_control.npy', p)
print("\nSaved _recency_control.npy", flush=True)
