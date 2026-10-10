import pandas as pd
import numpy as np

# Set seed for reproducible synthetic dataset
np.random.seed(42)
n_samples = 2000

# 1. Feature Generation based on Indian Financial Demographics
cibil_score = np.random.randint(300, 901, size=n_samples)
monthly_income = np.random.randint(25000, 350000, size=n_samples)

# Existing EMIs typically scale relative to income (0% to ~60% of income)
existing_emi = (monthly_income * np.random.uniform(0.0, 0.6, size=n_samples)).astype(int)

# Requested Loan Amount (100k to 7.5M INR)
loan_amount = np.random.randint(100000, 7500001, size=n_samples)

# Tenure in Months (Standard Indian personal/home/car loan tenures)
tenure_months = np.random.choice([12, 24, 36, 48, 60, 120, 180, 240], size=n_samples)

# Employment Profile
employment_type = np.random.choice(['Salaried', 'Self-Employed', 'Business'], size=n_samples, p=[0.55, 0.30, 0.15])

# Collateral Value (0 for unsecured, up to 10M INR for secured)
has_collateral = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
collateral_value = (has_collateral * np.random.randint(200000, 10000001, size=n_samples)).astype(int)

# 2. Derive Underwriting Metric: FOIR (Fixed Obligation to Income Ratio)
proposed_emi = loan_amount / tenure_months
foir = (existing_emi + proposed_emi) / monthly_income

# 3. Apply Banking Risk Rules to Determine Loan Approval Status
status = []
for i in range(n_samples):
    # Prime Borrower: High CIBIL, low FOIR -> Strong Approval
    if cibil_score[i] >= 750 and foir[i] <= 0.50:
        status.append(1)
    # Secured Borrower: Moderate CIBIL, high collateral coverage -> Approval
    elif cibil_score[i] >= 670 and foir[i] <= 0.55 and collateral_value[i] >= loan_amount[i]:
        status.append(1)
    # High Risk: Low CIBIL or excessive FOIR (>65%) -> Rejection
    elif cibil_score[i] < 620 or foir[i] > 0.65:
        status.append(0)
    # Edge Cases: Probabilistic approval based on credit standing
    else:
        approval_prob = 0.7 if cibil_score[i] >= 700 else 0.3
        status.append(np.random.choice([1, 0], p=[approval_prob, 1 - approval_prob]))

# 4. Construct DataFrame
df = pd.DataFrame({
    'CIBIL': cibil_score,
    'MonthlyIncome': monthly_income,
    'ExistingEMI': existing_emi,
    'LoanAmount': loan_amount,
    'TenureMonths': tenure_months,
    'EmploymentType': employment_type,
    'CollateralValue': collateral_value,
    'Approved': status
})

# Save to CSV
output_filename = 'loan_approval.csv'
df.to_csv(output_filename, index=False)

print(f"Dataset successfully created with {len(df)} rows!")
print(f"Approval Distribution:\n{df['Approved'].value_counts(normalize=True)}")
print(f"\nFirst 5 rows of {output_filename}:")
print(df.head())