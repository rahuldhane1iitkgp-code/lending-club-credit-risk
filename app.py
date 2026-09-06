import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="Lending Club Credit Risk Model", layout="centered")

@st.cache_resource
def load_artifacts():
    return joblib.load("deployment_artifacts.joblib")

artifacts = load_artifacts()
model = artifacts["model"]
calibrated_model = artifacts["calibrated_model"]
features = artifacts["features"]
threshold = artifacts["threshold"]

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

st.title("Lending Club Credit Risk Model")
st.caption(
    "Deployment model (39 features) — retains 89.5% of the full 151-feature research "
    "model's test PR-AUC while using only application-time information a borrower or "
    "loan officer can realistically provide. Trained on loans originated through 2016, "
    "tested on 2017."
)

st.header("Loan Details")
col1, col2 = st.columns(2)
with col1:
    loan_amnt = st.number_input("Loan amount ($)", min_value=1000, max_value=40000, value=15000, step=500)
    term = st.selectbox("Term (months)", [36, 60])
    purpose = st.selectbox("Loan purpose", [
        "debt_consolidation", "credit_card", "home_improvement", "other",
        "major_purchase", "medical", "small_business", "car", "moving",
        "vacation", "house", "wedding", "renewable_energy", "educational"
    ])
with col2:
    disbursement_method = st.selectbox("Disbursement method", ["Cash", "DirectPay"])
    is_joint = st.checkbox("Joint application (co-applicant)")

st.header("Borrower Profile")
col3, col4 = st.columns(2)
with col3:
    annual_inc = st.number_input("Annual income ($)", min_value=0, max_value=2000000, value=60000, step=1000)
    emp_length = st.slider("Years employed (10 = 10+ years)", 0, 10, 5)
    home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    region = st.selectbox("Region", ["Northeast", "South", "Midwest", "West"])
with col4:
    verification_status_label = st.selectbox("Income verification status", ["Not Verified", "Source Verified", "Verified"])
    fico_score = st.slider("Credit score (FICO)", 300, 850, 700)
    credit_history_years = st.slider("Years of credit history", 0, 60, 10)

st.header("Credit Profile")
col5, col6 = st.columns(2)
with col5:
    dti = st.number_input("Debt-to-income ratio (%)", min_value=0.0, max_value=100.0, value=18.0, step=0.5)
    revol_util = st.number_input("Revolving utilization (%)", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
    open_acc = st.number_input("Open credit accounts", min_value=0, max_value=100, value=8, step=1)
    total_acc = st.number_input("Total accounts ever opened", min_value=0, max_value=150, value=20, step=1)
with col6:
    mort_acc = st.number_input("Mortgage accounts", min_value=0, max_value=20, value=1, step=1)
    revol_bal = st.number_input("Revolving balance ($)", min_value=0, max_value=500000, value=8000, step=500)
    delinq_2yrs = st.number_input("Delinquencies in last 2 years", min_value=0, max_value=30, value=0, step=1)
    pub_rec = st.number_input("Public records (bankruptcies etc.)", min_value=0, max_value=20, value=0, step=1)

if st.button("Assess Risk", type="primary"):
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

    prob = calibrated_model.predict_proba(X_input)[:, 1][0]
    decision = "REJECT" if prob >= threshold else "APPROVE"

    st.divider()
    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Predicted default probability", f"{prob:.1%}")
    c2.metric("Decision", decision, help=f"Threshold: {threshold:.0%} — selected on held-out validation data to maximise net profit, not F1")

    if decision == "REJECT":
        st.error(f"This application exceeds the {threshold:.0%} risk threshold.")
    else:
        st.success(f"This application is within the {threshold:.0%} risk threshold.")

    st.subheader("Why this prediction — top contributing factors")
    shap_values = explainer.shap_values(X_input)
    explanation_df = pd.DataFrame({
        "feature": features,
        "value": X_input.iloc[0].values,
        "shap_contribution": shap_values[0]
    }).sort_values("shap_contribution", key=abs, ascending=False).head(8)

    for _, r in explanation_df.iterrows():
        direction = "increases risk" if r["shap_contribution"] > 0 else "decreases risk"
        st.write(f"**{r['feature']}** = {r['value']:.2f} → {direction} (impact: {r['shap_contribution']:+.3f})")

st.divider()
st.caption(
    f"Model: XGBoost, calibrated (isotonic). Test PR-AUC: {artifacts['test_pr_auc']:.3f}, "
    f"Test ROC-AUC: {artifacts['test_roc_auc']:.3f}. The {threshold:.0%} threshold was selected on a "
    f"held-out validation split — never on test — to maximize net profit under assumed economics "
    f"(LGD={artifacts['assumed_lgd']:.0%}, profit margin={artifacts['assumed_profit_margin']:.0%}), not F1. "
    f"On test it approves {artifacts['test_approval_rate']:.0%} of applicants for a net of "
    f"${artifacts['test_net_at_threshold']/1e6:.1f}M. "
    f"This is a portfolio/demo model — assumptions are documented, not derived from proprietary lender data."
)
