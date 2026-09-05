import joblib
import pandas as pd

print("Loading session state...")
state = joblib.load('D:/Lending/session_state.joblib')
train_df = state['train_df']
val_df = state['val_df']
test_df = state['test_df']
X_train = state['X_train']

print("Reloading addr_state from raw CSV...")
addr_state_raw = pd.read_csv('D:/Lending/accepted_2007_to_2018Q4.csv', usecols=['addr_state'])['addr_state']

region_map = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast", "RI": "Northeast",
    "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest", "WI": "Midwest",
    "IA": "Midwest", "KS": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "DC": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South",
    "SC": "South", "VA": "South", "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West", "NM": "West",
    "UT": "West", "WY": "West", "AK": "West", "CA": "West", "HI": "West", "OR": "West", "WA": "West"
}

for d in [train_df, val_df, test_df]:
    d['addr_state'] = addr_state_raw.loc[d.index]
    d['addr_region'] = d['addr_state'].map(region_map).fillna('Other')
    d['fico_score'] = (d['combined_fico_low'] + d['combined_fico_high']) / 2

home_cols = [c for c in train_df.columns if c.startswith('home_')]
purpose_cols = [c for c in train_df.columns if c.startswith('purpose_')]

base_b = [
    'loan_amnt', 'term', 'emp_length', 'combined_annual_inc', 'verification_status',
    'combined_dti', 'fico_score', 'delinq_2yrs', 'pub_rec', 'open_acc', 'total_acc',
    'revol_util', 'mort_acc', 'credit_history_months', 'is_joint', 'disbursement_method',
    'loan_to_income', 'avg_revol_balance', 'open_to_total_acc', 'accounts_per_credit_year'
] + home_cols + purpose_cols

region_dummies_train = pd.get_dummies(train_df['addr_region'], prefix='region', drop_first=True)
features_b1 = base_b
features_b2 = base_b + list(region_dummies_train.columns)

print(f"\nbase_b (before region): {len(base_b)} columns")
print(f"features_b2 (with region): {len(features_b2)} columns")
print(f"\nhome_cols: {home_cols}")
print(f"purpose_cols: {purpose_cols}")
print(f"region dummy cols: {list(region_dummies_train.columns)}")

audit = pd.DataFrame({'feature': X_train.columns})
audit['in_model_b1'] = audit['feature'].isin(features_b1)
print(f"\n{audit[audit['in_model_b1']].shape[0]} of Model A's {len(X_train.columns)} features reused in Model B1")
print("\nExclusion check (should show combined_fico_low/high excluded, sub_grade/int_rate/grade never in Model A anyway):")
check_cols = ['combined_fico_low', 'combined_fico_high']
print(audit[audit['feature'].isin(check_cols)])

print("\nAny features_b1 entries NOT found in train_df.columns (should be empty):")
missing = [f for f in features_b1 if f not in train_df.columns]
print(missing)

# Save updated dataframes + feature lists for step 2
joblib.dump({
    'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
    'features_b1': features_b1, 'features_b2': features_b2,
    'y_train': state['y_train'], 'y_val': state['y_val'], 'y_test': state['y_test'],
    'scale_pos_weight': state['scale_pos_weight'],
    'model2': state['model2'], 'calibrated_model2': state['calibrated_model2'],
}, 'D:/Lending/session_state_b.joblib')
print("\nSaved session_state_b.joblib for step 2")
