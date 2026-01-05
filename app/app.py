import os
import sys
import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model_path = os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl")
model = joblib.load(model_path)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Loan Approval Prediction")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to predict loan approval")

st.divider()

# ---------------- User Inputs ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])

st.divider()

# ---------------- Prediction ----------------
if st.button("Predict Loan Status"):
    input_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    df = pd.DataFrame([input_data])
    df = preprocess_data(df)
    df = feature_engineering(df, is_train=False)

    prediction = model.predict(df)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
