import joblib
import pandas as pd
import shap

artifacts = joblib.load("D:/Lending/deployment_artifacts.joblib")
model = artifacts["model"]
calibrated_model = artifacts["calibrated_model"]
features = artifacts["features"]
threshold = artifacts["threshold"]

# Simulate a form submission exactly like app.py's button-click logic
loan_amnt, term, purpose = 15000, 36, "debt_consolidation"
disbursement_method, is_joint = "Cash", False
annual_inc, emp_length, home_ownership, region = 60000, 5, "RENT", "South"
verification_status_label, fico_score, credit_history_years = "Verified", 700, 10
dti, revol_util, open_acc, total_acc = 18.0, 40.0, 8, 20
mort_acc, revol_bal, delinq_2yrs, pub_rec = 1, 8000, 0, 0

verification_map = {"Not Verified": 0, "Source Verified": 1, "Verified": 2}
verification_status = verification_map[verification_status_label]
credit_history_months = credit_history_years * 12

row = {f: 0 for f in features}
row["loan_amnt"] = loan_amnt
row["term"] = term
row["emp_length"] = emp_length
row["combined_annual_inc"] = annual_inc
row["verification_status"] = verification_status
row["combined_dti"] = dti
row["fico_score"] = fico_score
row["delinq_2yrs"] = delinq_2yrs
row["pub_rec"] = pub_rec
row["open_acc"] = open_acc
row["total_acc"] = total_acc
row["revol_util"] = revol_util
row["mort_acc"] = mort_acc
row["credit_history_months"] = credit_history_months
row["is_joint"] = int(is_joint)
row["disbursement_method"] = 1 if disbursement_method == "DirectPay" else 0
row["loan_to_income"] = loan_amnt / annual_inc if annual_inc > 0 else 0
row["avg_revol_balance"] = revol_bal / (open_acc + 1)
row["open_to_total_acc"] = open_acc / (total_acc + 1)
row["accounts_per_credit_year"] = total_acc / (credit_history_months / 12 + 1)

home_col = f"home_{home_ownership}" if home_ownership != "MORTGAGE" else None
if home_col and home_col in row:
    row[home_col] = 1
purpose_col = f"purpose_{purpose}"
if purpose_col in row:
    row[purpose_col] = 1
region_col = f"region_{region}"
if region_col in row:
    row[region_col] = 1

X_input = pd.DataFrame([row])[features]
print("Input row assembled OK, shape:", X_input.shape)
print("Non-zero fields:", {k: v for k, v in row.items() if v != 0})

prob = calibrated_model.predict_proba(X_input)[:, 1][0]
decision = "REJECT" if prob >= threshold else "APPROVE"
print(f"\nPredicted default probability: {prob:.4f}")
print(f"Threshold: {threshold}")
print(f"Decision: {decision}")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_input)
explanation_df = pd.DataFrame({
    "feature": features, "value": X_input.iloc[0].values, "shap_contribution": shap_values[0]
}).sort_values("shap_contribution", key=abs, ascending=False).head(8)
print("\nTop SHAP contributors:")
print(explanation_df.to_string(index=False))

# Second test case: a clearly risky applicant, to confirm the model responds sensibly in the other direction
row2 = dict(row)
row2["combined_dti"] = 45.0
row2["fico_score"] = 580
row2["delinq_2yrs"] = 3
row2["revol_util"] = 95.0
row2["term"] = 60
row2["loan_to_income"] = loan_amnt / 25000
row2["combined_annual_inc"] = 25000
X_input2 = pd.DataFrame([row2])[features]
prob2 = calibrated_model.predict_proba(X_input2)[:, 1][0]
print(f"\n--- Risky applicant sanity check ---")
print(f"Predicted default probability: {prob2:.4f} (should be notably higher than {prob:.4f})")
print(f"Decision: {'REJECT' if prob2 >= threshold else 'APPROVE'}")
