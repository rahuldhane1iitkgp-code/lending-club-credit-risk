import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

print("Loading state...")
state = joblib.load('D:/Lending/session_state_b.joblib')
train_df = state['train_df']
val_df = state['val_df']
test_df = state['test_df']
features_b1 = state['features_b1']
features_b2 = state['features_b2']
y_train = state['y_train']
y_val = state['y_val']
y_test = state['y_test']
scale_pos_weight = state['scale_pos_weight']

# Build region dummies once, consistently across splits (use train categories as the reference)
for d in [train_df, val_df, test_df]:
    region_dummies = pd.get_dummies(d['addr_region'], prefix='region')
    for col in ['region_Northeast', 'region_South', 'region_West']:
        d[col] = region_dummies[col] if col in region_dummies.columns else 0

def train_eval(features, name):
    print(f"\n=== Training {name} ({len(features)} features) ===")
    X_tr = train_df[features]
    X_v = val_df[features]
    X_te = test_df[features]

    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        scale_pos_weight=scale_pos_weight, eval_metric='aucpr',
        early_stopping_rounds=50, random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)

    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_v, y_val)

    test_probs = calibrated.predict_proba(X_te)[:, 1]
    pr_auc = average_precision_score(y_test, test_probs)
    roc_auc = roc_auc_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    print(f"Best iteration: {model.best_iteration}")
    print(f"Val PR-AUC: {model.best_score:.4f}")
    print(f"Test PR-AUC: {pr_auc:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test Brier: {brier:.4f}")
    print(f"Mean predicted prob: {test_probs.mean():.4f} | Actual rate: {y_test.mean():.4f}")

    return {
        'model': model, 'calibrated': calibrated, 'test_probs': test_probs,
        'pr_auc': pr_auc, 'roc_auc': roc_auc, 'brier': brier, 'features': features
    }

result_b1 = train_eval(features_b1, "Model B1 (no region)")
result_b2 = train_eval(features_b2, "Model B2 (with region)")

print("\n=== Comparison ===")
print(f"B1 test PR-AUC: {result_b1['pr_auc']:.4f}")
print(f"B2 test PR-AUC: {result_b2['pr_auc']:.4f}")
print(f"Lift from region: {result_b2['pr_auc'] - result_b1['pr_auc']:.4f}")

model_a_pr_auc = 0.4132072081921982
winner = result_b2 if result_b2['pr_auc'] > result_b1['pr_auc'] else result_b1
winner_name = "B2 (with region)" if winner is result_b2 else "B1 (no region)"
retention = winner['pr_auc'] / model_a_pr_auc

print(f"\nWinner: Model {winner_name}")
print(f"Model A test PR-AUC: {model_a_pr_auc:.4f}")
print(f"Model B ({winner_name}) test PR-AUC: {winner['pr_auc']:.4f}")
print(f"Performance retention: {retention*100:.1f}%")

joblib.dump({
    'result_b1': result_b1, 'result_b2': result_b2,
    'winner_name': winner_name, 'retention': retention,
    'model_a_pr_auc': model_a_pr_auc,
    'features_b1': features_b1, 'features_b2': features_b2,
}, 'D:/Lending/model_b_results.joblib')
print("\nSaved model_b_results.joblib")
