import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load trained model
model = joblib.load("burnout_model_rf.pkl")

# Set page configuration
st.set_page_config(page_title="Burnout Risk Predictor", layout="centered")

# === App Title and Description ===
st.title("Burnout Risk Prediction")
st.markdown("Use this form to predict whether an employee is at risk of burnout based on selected factors.")

# === Input Form ===
with st.form("burnout_form"):
    st.header("Employee Information")

    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=18, max_value=65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_role = st.selectbox("Job Role", ["Engineer", "Manager", "Analyst", "HR", "Sales"])
    experience = st.number_input("Years of Experience", min_value=0, max_value=40)
    work_hours = st.number_input("Work Hours Per Week", min_value=0, max_value=168, step=1)
    remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100)
    satisfaction = st.slider("Satisfaction Level", min_value=0.0, max_value=5.0, step=0.1)
    stress = st.slider("Stress Level", min_value=0, max_value=10)

    submitted = st.form_submit_button("Predict Burnout Risk")

# === Prediction ===
if submitted:
    input_data = pd.DataFrame([{
        'StressLevel': stress,
        'SatisfactionLevel': satisfaction,
        'WorkHoursPerWeek': work_hours
    }])

    prediction = model.predict(input_data)[0]

    st.header("Prediction Result")
    if prediction == 1:
        st.error("High Risk of Burnout")
    else:
        st.success("No Burnout Detected")

    # Save full data
    full_data = {
        'Name': name,
        'Age': age,
        'Gender': gender,
        'JobRole': job_role,
        'Experience': experience,
        'WorkHoursPerWeek': work_hours,
        'RemoteRatio': remote_ratio,
        'SatisfactionLevel': satisfaction,
        'StressLevel': stress,
        'Burnout': int(prediction)
    }

    csv_file = "burnout_predictions.csv"
    df = pd.DataFrame([full_data])
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)
