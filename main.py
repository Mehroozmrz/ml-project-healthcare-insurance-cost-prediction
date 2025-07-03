import streamlit as st
from prediction_helper import predict

import pandas as pd
# import joblib  # Uncomment when using a trained model

st.set_page_config(page_title="Health Insurance Cost Predictor", layout="centered")


st.title("Health Insurance Cost Predictor")

st.markdown("Provide your details below to get a prediction of your health insurance cost.")

# --------- Row 1: Categorical ---------
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
with col2:
    region = st.selectbox("Region", ['Northeast', 'Northwest', 'Southeast', 'Southwest'])
with col3:
    marital_status = st.selectbox("Marital Status", ['Unmarried', 'Married'])

# --------- Row 2: Categorical ---------
col4, col5, col6 = st.columns(3)
with col4:
    bmi_category = st.selectbox("BMI Category", ['Overweight', 'Underweight', 'Normal', 'Obesity'])
with col5:
    smoking_status = st.selectbox("Smoking Status", [
        'Regular', 'No Smoking', 'Occasional'
    ])
with col6:
    employment_status = st.selectbox("Employment Status", ['Self-Employed', 'Freelancer', 'Salaried'])

# --------- Row 3: Categorical ---------
col7, col8, col9 = st.columns(3)
# with col7:
#     income_level = st.selectbox("Income Level", ['> 40L', '<10L', '10L - 25L', '25L - 40L'])
with col7:
    medical_history = st.selectbox("Medical History", [
        'High blood pressure', 'No Disease', 'Diabetes & High blood pressure',
        'Diabetes & Heart disease', 'Diabetes', 'Diabetes & Thyroid',
        'Heart disease', 'Thyroid', 'High blood pressure & Heart disease'
    ])
with col8:
    insurance_plan = st.selectbox("Insurance Plan", ['Silver', 'Bronze', 'Gold'])

with col9:
    age = st.number_input("Age", min_value=18, max_value=100, value=18)

# --------- Row 4: Numeric ---------
col11, col12, col13 = st.columns(3)

with col11:
    number_of_dependants = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
with col12:
    income_lakhs = st.number_input("Income (in Lakhs)", min_value=0, step=1)
with col13:
    genital_risk = st.number_input("Genital Risk Score", min_value=0, max_value=10, step=1)
# --------- Row 5: Numeric ---------
col13, col14 = st.columns(2)

# with col14:
#     annual_premium_amount = st.number_input("Annual Premium Amount", min_value=0.0, value=20000.0, step=100.0)

st.markdown("---")


input_data = {
        "gender": gender,
        "region": region,
        "marital_status": marital_status,
        "bmi_category": bmi_category,
        "smoking_status": smoking_status,
        "employment_status": employment_status,
        "medical_history": medical_history,
        "insurance_plan": insurance_plan,
        "age": age,
        "number_of_dependants": number_of_dependants,
        "income_lakhs": income_lakhs,
        "genital_risk": genital_risk,
    }

# --------- Predict Button ---------
if st.button("🎯 Predict"):
    prediction = predict(input_data)

    # Preprocessing goes here if needed
    # prediction = model.predict(preprocessed_input)

    # Display dummy prediction (replace with real one)
    st.success(f"💰 Predicted Health Insurance Cost: ₹{prediction}")
