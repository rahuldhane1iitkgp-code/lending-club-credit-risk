"""
Algorithm benchmark on the Model A (151-feature) problem.

Same temporal split throughout: train <=2015, validate 2016, test 2017. Every model
early-stops on validation only; test is scored once per model. PR-AUC is the headline
because the positive class is the minority and the one that costs money.

Two fairness rules, so the comparison says something about algorithms rather than about
how much care each one got:
  - LightGBM is configured to XGBoost's depth-6 geometry rather than leaf-wise defaults.
  - Random Forest and Logistic Regression cannot take NaN, so they get train-median
    imputation. XGBoost and LightGBM keep their native missing handling, which is a
    genuine advantage of those algorithms rather than a thumb on the scale.

Ranking metrics are invariant to monotone calibration, so raw scores are compared;
calibration is a separate downstream concern.
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

SEED = 42

state = joblib.load('D:/Lending/session_state_b.joblib')
train_df, val_df, test_df = state['train_df'], state['val_df'], state['test_df']
y_train = np.asarray(state['y_train'])
y_val = np.asarray(state['y_val'])
y_test = np.asarray(state['y_test'])
spw = state['scale_pos_weight']

features = list(state['model2'].get_booster().feature_names)
X_train = train_df[features].astype(float)
X_val = val_df[features].astype(float)
X_test = test_df[features].astype(float)

med = X_train.median(numeric_only=True)
X_train_i, X_val_i, X_test_i = (d.fillna(med) for d in (X_train, X_val, X_test))

print(f"train {X_train.shape} | val {X_val.shape} | test {X_test.shape}", flush=True)
print(f"base rate  train {y_train.mean():.4f} | val {y_val.mean():.4f} | test {y_test.mean():.4f}", flush=True)
print(f"no-skill PR-AUC on test = {y_test.mean():.4f}\n", flush=True)

results, val_scores, test_scores = [], {}, {}


def record(name, s_val, s_test, secs, note=""):
    r = {
        'model': name,
        'val_pr_auc': average_precision_score(y_val, s_val),
        'test_pr_auc': average_precision_score(y_test, s_test),
        'test_roc_auc': roc_auc_score(y_test, s_test),
        'fit_seconds': round(secs),
        'note': note,
    }
    results.append(r)
    val_scores[name], test_scores[name] = s_val, s_test
    print(f"{name:30s} val PR-AUC {r['val_pr_auc']:.4f} | test PR-AUC {r['test_pr_auc']:.4f} "
          f"| test ROC-AUC {r['test_roc_auc']:.4f} | {r['fit_seconds']}s", flush=True)
    return r


# ---- 1. XGBoost (incumbent, retrained so the comparison is like-for-like)
from xgboost import XGBClassifier
t = time.time()
xgb = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6,
                    scale_pos_weight=spw, eval_metric='aucpr',
                    early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
record('XGBoost (incumbent)', xgb.predict_proba(X_val)[:, 1], xgb.predict_proba(X_test)[:, 1],
       time.time() - t, f"best_iter={xgb.best_iteration}")

# ---- 2. LightGBM, matched to XGBoost's depth-6 geometry
#
# Deliberately the native lgb.train API rather than the sklearn wrapper. In LightGBM 4.7
# the wrapper's `eval_set` argument is deprecated and the validation set is not passed
# through to early stopping, so LGBMClassifier stopped at iterations 7 and 17 and scored
# 0.368-0.387 test PR-AUC. That measured a broken configuration, not the algorithm; the
# native API trains to 523 iterations and lands within noise of XGBoost. Worth keeping as
# a comment because a silently-mistrained challenger is how benchmarks reach wrong
# conclusions about which algorithm is better.
import lightgbm as lgb
t = time.time()
dtrain = lgb.Dataset(X_train, y_train)
dval = lgb.Dataset(X_val, y_val, reference=dtrain)
lgb_params = dict(objective='binary', metric='average_precision', learning_rate=0.05,
                  max_depth=6, num_leaves=64, min_data_in_leaf=20,
                  scale_pos_weight=spw, seed=SEED, verbosity=-1)
booster = lgb.train(lgb_params, dtrain, num_boost_round=3000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
record('LightGBM', booster.predict(X_val), booster.predict(X_test),
       time.time() - t, f"best_iter={booster.best_iteration}")

# ---- 3. HistGradientBoosting (sklearn's own GBM, native NaN handling)
t = time.time()
hgb = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.05, max_depth=6,
                                     min_samples_leaf=100, early_stopping=True,
                                     validation_fraction=0.15, n_iter_no_change=50,
                                     class_weight='balanced', random_state=SEED)
hgb.fit(X_train, y_train)
record('HistGradientBoosting', hgb.predict_proba(X_val)[:, 1], hgb.predict_proba(X_test)[:, 1],
       time.time() - t, f"n_iter={hgb.n_iter_}")

# ---- 4. Random Forest (bagging rather than boosting)
t = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=100,
                            max_features='sqrt', class_weight='balanced_subsample',
                            random_state=SEED, n_jobs=-1)
rf.fit(X_train_i, y_train)
record('Random Forest', rf.predict_proba(X_val_i)[:, 1], rf.predict_proba(X_test_i)[:, 1],
       time.time() - t, "200 trees, depth 16, median-imputed")

# ---- 5. Logistic Regression (the linear baseline the trees must justify themselves against)
t = time.time()
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED))
lr.fit(X_train_i, y_train)
record('Logistic Regression', lr.predict_proba(X_val_i)[:, 1], lr.predict_proba(X_test_i)[:, 1],
       time.time() - t, "standardised, balanced, median-imputed")

# ---- 6. Rank-average ensemble of the GBMs (equal weights, nothing fitted -> no selection bias)
members = ['XGBoost (incumbent)', 'LightGBM', 'HistGradientBoosting']
record('Rank-avg ensemble (3 GBMs)',
       np.mean([rankdata(val_scores[m]) / len(y_val) for m in members], axis=0),
       np.mean([rankdata(test_scores[m]) / len(y_test) for m in members], axis=0),
       0, "equal weights, unfitted")

# ---- Summary
df = pd.DataFrame(results).sort_values('test_pr_auc', ascending=False)
baseline = next(r['test_pr_auc'] for r in results if r['model'] == 'XGBoost (incumbent)')
df['vs_incumbent'] = (df['test_pr_auc'] - baseline).map(lambda d: f"{d:+.4f}")
print("\n=== Ranked by test PR-AUC ===", flush=True)
print(df[['model', 'val_pr_auc', 'test_pr_auc', 'test_roc_auc', 'vs_incumbent', 'note']]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
print(f"\nNo-skill baseline: {y_test.mean():.4f} | best lift: "
      f"{df['test_pr_auc'].max() / y_test.mean():.2f}x", flush=True)

joblib.dump({'results': df, 'baseline_test_pr_auc': baseline, 'test_scores': test_scores},
            'D:/Lending/model_benchmark.joblib')
df.to_csv('D:/Lending/reports_model_benchmark.csv', index=False)
print("\nSaved model_benchmark.joblib and reports_model_benchmark.csv", flush=True)
