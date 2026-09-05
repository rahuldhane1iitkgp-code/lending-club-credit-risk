import joblib
import numpy as np

state = joblib.load('D:/Lending/session_state_b.joblib')
results = joblib.load('D:/Lending/model_b_results.joblib')
threshold_info = joblib.load('D:/Lending/model_b_threshold.joblib')

import pandas as pd

result_b2 = results['result_b2']
train_df = state['train_df']
features_b2 = results['features_b2']

region_dummies = pd.get_dummies(train_df['addr_region'], prefix='region')
for col in ['region_Northeast', 'region_South', 'region_West']:
    train_df[col] = region_dummies[col] if col in region_dummies.columns else 0

# Median values from training data, used only for fields NOT collected in the simplified UI
# (kept minimal since Model B was specifically designed so every feature IS collected;
# this is a safety net in case of any future field additions, not the median-fill-everything pattern we rejected for Model A)
medians = train_df[features_b2].median(numeric_only=True).to_dict()

joblib.dump({
    'model': result_b2['model'],
    'calibrated_model': result_b2['calibrated'],
    'features': features_b2,
    'threshold': threshold_info['best_threshold'],
    'medians': medians,
    'test_pr_auc': result_b2['pr_auc'],
    'test_roc_auc': result_b2['roc_auc'],
}, 'D:/Lending/deployment_artifacts.joblib')

print("Saved deployment_artifacts.joblib")
print(f"Features ({len(features_b2)}):", features_b2)
print(f"Threshold: {threshold_info['best_threshold']}")
