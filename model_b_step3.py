import joblib
import pandas as pd

state = joblib.load('D:/Lending/session_state_b.joblib')
results = joblib.load('D:/Lending/model_b_results.joblib')
test_df = state['test_df']
y_test = state['y_test']

result_b2 = results['result_b2']
test_probs = result_b2['test_probs']

ASSUMED_LGD = 0.50
ASSUMED_PROFIT_MARGIN = 0.10

thresholds_to_test = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
out = []
for t in thresholds_to_test:
    preds = (test_probs >= t).astype(int)
    approved_mask = (preds == 0)
    ead = test_df.loc[approved_mask, 'loan_amnt']
    actual_outcome = y_test[approved_mask]
    total_loss = (ead[actual_outcome == 1] * ASSUMED_LGD).sum()
    total_profit = (ead[actual_outcome == 0] * ASSUMED_PROFIT_MARGIN).sum()
    net = total_profit - total_loss
    out.append({'threshold': t, 'approval_rate': approved_mask.mean(), 'net': net})

results_df = pd.DataFrame(out)
print(results_df)

best_row = results_df.loc[results_df['net'].idxmax()]
print(f"\nBest threshold: {best_row['threshold']}, net: {best_row['net']:.1f}, approval_rate: {best_row['approval_rate']:.4f}")

joblib.dump({'threshold_sweep': results_df, 'best_threshold': best_row['threshold'], 'best_net': best_row['net']},
            'D:/Lending/model_b_threshold.joblib')
