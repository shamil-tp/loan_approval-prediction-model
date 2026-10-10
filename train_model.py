import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load your pre-generated dataset
dataset_path = 'loan_approval.csv'  # Update with your file path if different
df = pd.read_csv(dataset_path)

print(f"Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# 2. Encode Categorical Features
le = LabelEncoder()
df['EmploymentType_Encoded'] = le.fit_transform(df['EmploymentType'])

# 3. Define Features (X) and Target (y)
feature_cols = [
    'CIBIL', 
    'MonthlyIncome', 
    'ExistingEMI', 
    'LoanAmount', 
    'TenureMonths', 
    'EmploymentType_Encoded', 
    'CollateralValue'
]

X = df[feature_cols]
y = df['Approved']

# 4. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Initialize and Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate Model Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save Artifacts for Flask API
joblib.dump(model, 'loan_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model saved as 'loan_model.pkl' and LabelEncoder as 'label_encoder.pkl'.")