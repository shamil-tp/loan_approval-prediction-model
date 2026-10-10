# 🏦 Indian Loan Approval Prediction System

An end-to-end Machine Learning web application tailored to Indian banking and underwriting standards. The system evaluates applicant eligibility using financial indicators such as **CIBIL Score**, **FOIR (Fixed Obligation to Income Ratio)**, **Monthly Income**, **Collateral Value**, and **Existing Monthly EMIs**.

---

## 📌 Project Overview

This project provides an automated loan eligibility assessment tool backed by a **Random Forest Classifier**. It consists of:
1. **Model Pipeline**: Feature encoding, metric calculation (FOIR), and training on credit evaluation guidelines.
2. **REST API (Flask)**: Backend service for processing requests and serving model predictions.
3. **Web Interface**: Clean, responsive frontend built with HTML5, JavaScript, and Tailwind CSS.

---

## 🛠️ How It Works

### Underwriting Metrics & Logic
- **CIBIL Score**: Ranges from 300 to 900. Scores above 750 represent prime borrowers.
- **FOIR (Fixed Obligation to Income Ratio)**: Calculates total monthly debt obligations (existing EMIs + proposed EMI) against total monthly income:
  $$\text{FOIR} = \left( \frac{\text{Existing EMI} + \text{Proposed EMI}}{\text{Monthly Income}} \right) \times 100$$
  *Values below 50% are typically preferred by Indian lenders.*
- **Collateral Coverage**: Evaluates tangible assets pledged against unsecured credit risk.

### Application Workflow

```text
  [ User Form Input ]
          │
          ▼
    [ Flask API ] ──► Computes FOIR & Preprocesses Inputs
          │
          ▼
   [ Model Predict ] ──► Predicts Approval & Probability
          │
          ▼
 [ UI Result Render ] ──► Displays Status, FOIR %, & Risk Notes
```

---

## 📁 Directory Structure

```text
loan_project/
├── app.py                      # Flask API backend & web server
├── train_model.py              # ML model training pipeline
├── indian_loan_approval_2000.csv# Dataset (2,000 samples)
├── loan_model.pkl              # Saved Random Forest model
├── label_encoder.pkl           # Saved LabelEncoder artifact
├── templates/
│   └── index.html              # Frontend user interface
└── README.md                   # Project documentation
```

---

## 🚀 Setup & Local Installation

### Prerequisites
- **Python 3.8+** installed on your system.
- `pip` package manager.

### Step 1: Clone or Navigate to Project Directory
```bash
cd path/to/loan_project
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install flask pandas scikit-learn joblib
```

---

## 🏃 Execution Guide

### 1. Train the Machine Learning Model
Run `train_model.py` to train the classifier on your dataset and generate the necessary model artifacts (`loan_model.pkl` and `label_encoder.pkl`):

```bash
python train_model.py
```

**Output:**
```text
Dataset successfully loaded: 2000 rows, 8 columns.

--- Model Evaluation ---
Accuracy: ~92.00%

Model saved as 'loan_model.pkl' and LabelEncoder as 'label_encoder.pkl'.
```

---

### 2. Start the Flask Application Server
Run `app.py` to launch the local web server:

```bash
python app.py
```

**Output:**
```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

---

### 3. Open the Application
Navigate to `http://127.0.0.1:5000` in your web browser.

Input sample applicant data (CIBIL score, monthly income, existing EMIs, loan amount, tenure, employment type, collateral) and click **Assess Loan Eligibility**.

---

## 📡 API Endpoint Reference

### `POST /predict`
Processes loan application data and returns eligibility predictions.

#### Request Payload:
```json
{
  "cibil": 760,
  "income": 85000,
  "emi": 12000,
  "loan_amount": 1500000,
  "tenure": 60,
  "employment": "Salaried",
  "collateral": 2000000
}
```

#### Response Payload:
```json
{
  "approved": true,
  "probability": 88.45,
  "foir": 43.53,
  "reasons": []
}
```

---
