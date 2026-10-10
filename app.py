from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and encoder
model = joblib.load('loan_model.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        cibil = float(data['cibil'])
        income = float(data['income'])
        emi = float(data['emi'])
        loan_amount = float(data['loan_amount'])
        tenure = float(data['tenure'])
        employment = data['employment']
        collateral = float(data['collateral'])

        # Calculate metrics
        proposed_emi = loan_amount / tenure
        foir = round(((emi + proposed_emi) / income) * 100, 2)
        
        # Encode Employment Type
        emp_encoded = le.transform([employment])[0]
        
        # Prepare Feature Array
        features = pd.DataFrame([[cibil, income, emi, loan_amount, tenure, emp_encoded, collateral]],
                                columns=['CIBIL', 'MonthlyIncome', 'ExistingEMI', 'LoanAmount', 'TenureMonths', 'EmploymentType_Encoded', 'CollateralValue'])
        
        # Model Prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Detailed breakdown feedback
        reasons = []
        if cibil < 650:
            reasons.append("CIBIL score is lower than the preferred threshold (650+).")
        if foir > 50:
            reasons.append(f"FOIR ({foir}%) exceeds the safe borrowing limit of 50%.")
        if collateral < loan_amount and cibil < 720:
            reasons.append("Insufficient collateral value for the requested risk profile.")

        return jsonify({
            'approved': bool(prediction == 1),
            'probability': round(float(probability) * 100, 2),
            'foir': foir,
            'reasons': reasons
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)