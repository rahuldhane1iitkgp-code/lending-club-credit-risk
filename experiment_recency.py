"""
Does training on more recent vintages beat a bigger algorithm?

The shipped model trains on <=2015 and is tested on 2017 - a two-year gap across which
the default rate moves from 18.5% to 23.1%. The algorithm benchmark showed every GBM
pinned at ~0.417 test PR-AUC, so if headroom exists it is in the data, not the learner.

Three training windows, one held-out test year (2017), identical XGBoost config:

  A  <=2015                     the incumbent - 829k rows, ends 2 years before test
  B  <=2016, early-stop on a    adds 293k rows and the vintage adjacent to test
     random 20% slice of 2016
  C  <=2016, recency-weighted   as B, but sample weights favour recent vintages

B and C cannot early-stop on the full 2016 set because 2016 is now training data, so a
random 20% of 2016 is carved out for that purpose. Test remains 2017, untouched.
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42
PARAMS = dict(n_estimators=2000, learning_rate=0.05, max_depth=6,
              eval_metric='aucpr', early_stopping_rounds=50,
              random_state=SEED, n_jobs=-1)

state = joblib.load('D:/Lending/session_state_b.joblib')
train_df, val_df, test_df = state['train_df'], state['val_df'], state['test_df']
y_train = np.asarray(state['y_train'])
y_val = np.asarray(state['y_val'])
y_test = np.asarray(state['y_test'])

features = list(state['model2'].get_booster().feature_names)
X_train, X_val, X_test = (d[features].astype(float) for d in (train_df, val_df, test_df))

print(f"train<=2015 {X_train.shape} rate {y_train.mean():.4f}", flush=True)
print(f"val 2016    {X_val.shape} rate {y_val.mean():.4f}", flush=True)
print(f"test 2017   {X_test.shape} rate {y_test.mean():.4f}\n", flush=True)

rows = []


def run(name, Xtr, ytr, Xes, yes_, weights=None, note=""):
    t = time.time()
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    m = XGBClassifier(scale_pos_weight=spw, **PARAMS)
    m.fit(Xtr, ytr, sample_weight=weights, eval_set=[(Xes, yes_)], verbose=False)
    p = m.predict_proba(X_test)[:, 1]
    r = {'variant': name, 'train_rows': len(ytr), 'train_rate': ytr.mean(),
         'test_pr_auc': average_precision_score(y_test, p),
         'test_roc_auc': roc_auc_score(y_test, p),
         'best_iter': m.best_iteration, 'secs': round(time.time() - t), 'note': note}
    rows.append(r)
    print(f"{name:34s} rows {r['train_rows']:>7,} | test PR-AUC {r['test_pr_auc']:.4f} "
          f"| ROC {r['test_roc_auc']:.4f} | iter {r['best_iter']} | {r['secs']}s", flush=True)
    np.save(f"D:/Lending/_recency_{name.split()[0]}.npy", p)
    return r


# A - incumbent
run('A <=2015 (incumbent)', X_train, y_train, X_val, y_val, note="early-stop on full 2016")

# B - add 2016, carve 20% of it out for early stopping
es_idx, keep_idx = train_test_split(np.arange(len(val_df)), test_size=0.8,
                                    random_state=SEED, stratify=y_val)
X_big = pd.concat([X_train, X_val.iloc[keep_idx]])
y_big = np.concatenate([y_train, y_val[keep_idx]])
run('B <=2016', X_big, y_big, X_val.iloc[es_idx], y_val[es_idx],
    note="80% of 2016 added to train, 20% for early stopping")

# C - same rows, recency weighting by origination year.
# issue_year was used to build the splits and then dropped from the saved frames, so it
# is recovered from the raw CSV by index - the same trick model_b_step1.py uses for
# addr_state. Only the date column is read, so the 1.6 GB file costs one column scan.
print("recovering issue_d from the raw CSV...", flush=True)
issue_d = pd.read_csv('D:/Lending/accepted_2007_to_2018Q4.csv', usecols=['issue_d'])['issue_d']
issue_year = pd.to_datetime(issue_d, format='%b-%Y', errors='coerce').dt.year

years = pd.concat([issue_year.loc[train_df.index],
                   issue_year.loc[val_df.index[keep_idx]]]).to_numpy()
assert not np.isnan(years).any(), "unparsed issue_d dates"
w = np.power(1.25, years - years.min()).astype(float)
w /= w.mean()
print(f"\nrecency weights: year {years.min()} -> {w[years == years.min()][0]:.3f}, "
      f"year {years.max()} -> {w[years == years.max()][0]:.3f}\n", flush=True)
run('C <=2016 + recency weights', X_big, y_big, X_val.iloc[es_idx], y_val[es_idx],
    weights=w, note="1.25^(year - min_year), mean-normalised")

df = pd.DataFrame(rows)
base = df.loc[df['variant'].str.startswith('A'), 'test_pr_auc'].iloc[0]
df['vs_incumbent'] = (df['test_pr_auc'] - base).map(lambda d: f"{d:+.4f}")
print("\n=== Training-window experiment ===", flush=True)
print(df[['variant', 'train_rows', 'train_rate', 'test_pr_auc', 'test_roc_auc',
          'vs_incumbent', 'best_iter']].to_string(index=False,
      float_format=lambda x: f"{x:.4f}"), flush=True)

df.to_csv('D:/Lending/reports_recency_experiment.csv', index=False)
print("\nSaved reports_recency_experiment.csv", flush=True)
