"""
Isolate the training window as the ONLY difference between the two headline numbers.

The 0.4172 -> 0.4383 comparison currently changes two things at once:

    0.4172  train <=2015   early-stop on ALL of 2016 (293,105 rows, out-of-time)
    0.4383  train <=2016   early-stop on a 13% slice of 2016 (38,103 rows, in-year)

Early-stopping set size and composition are a confound. This adds the missing cell -
train <=2015 but early-stop on the same 38,103-row slice - so that variant and the
<=2016 run differ in training data and nothing else.

If the clean baseline lands near 0.4172, the headline comparison stands as stated. If it
lands materially higher, part of the reported gain was the early-stopping regime and the
CV claim has to be restated.
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

# Exactly the split used by model_b_recency.py and experiment_recency_control.py
idx = np.arange(len(val_df))
add_idx, rest = train_test_split(idx, train_size=0.60, random_state=SEED, stratify=y_val)
es_idx, _ = train_test_split(rest, train_size=0.325, random_state=SEED, stratify=y_val[rest])
X_es, y_es = X_val.iloc[es_idx], y_val[es_idx]
print(f"early-stopping slice: {len(es_idx):,} rows (identical to the <=2016 run)", flush=True)

spw = (y_train == 0).sum() / (y_train == 1).sum()
t = time.time()
m = XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=6, scale_pos_weight=spw,
                  eval_metric='aucpr', early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
m.fit(X_train, y_train, eval_set=[(X_es, y_es)], verbose=False)
p = m.predict_proba(X_test)[:, 1]
pr = average_precision_score(y_test, p)

print(f"\nA' <=2015, early-stop on the 38k in-year slice", flush=True)
print(f"   train rows {len(y_train):,} | best_iter {m.best_iteration} | {time.time()-t:.0f}s", flush=True)
print(f"   test PR-AUC {pr:.4f} | ROC {roc_auc_score(y_test, p):.4f}\n", flush=True)

print("clean comparison, training window as the only difference:", flush=True)
print(f"   A'  train <=2015   {pr:.4f}", flush=True)
print(f"   D   train <=2016   0.4383   ({0.4383-pr:+.4f})", flush=True)
print(f"\nfor reference, A (early-stop on all of 2016): 0.4172", flush=True)
print(f"early-stopping regime alone was worth {pr-0.4172:+.4f}", flush=True)

np.save('D:/Lending/_recency_clean_baseline.npy', p)
print("\nSaved _recency_clean_baseline.npy", flush=True)
