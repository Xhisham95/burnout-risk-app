import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load model trained on 3 features
model = joblib.load("burnout_model_rf.pkl")

st.title("💼 Burnout Risk Prediction App")
st.markdown("Fill in the employee details below to predict burnout risk.")

# === Input Form ===
with st.form("burnout_form"):
    st.subheader("👤 Employee Information")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=18, max_value=65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_role = st.selectbox("Job Role", ["Engineer", "Manager", "Analyst", "HR", "Sales"])
    experience = st.number_input("Experience (Years)", min_value=0, max_value=40)
    work_hours = st.number_input("Work Hours Per Week", min_value=0, max_value=168, step=1)
    remote_ratio = st.slider("Remote Ratio (%)", 0, 100)
    satisfaction = st.slider("Satisfaction Level (0.0 - 5.0)", 0.0, 5.0, step=0.1)
    stress = st.slider("Stress Level (0 - 10)", 0, 10)

    submitted = st.form_submit_button("Predict Burnout Risk")

# === Prediction ===
if submitted:
    # Prepare only the 3 features used in the model
    model_input = pd.DataFrame([{
        'StressLevel': stress,
        'SatisfactionLevel': satisfaction,
        'WorkHoursPerWeek': work_hours
    }])

    prediction = model.predict(model_input)[0]

    # Output the result
    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error("🧠 High Risk of Burnout!")
    else:
        st.success("✅ No Burnout Detected.")

    # Prepare full input data for saving
    full_input = {
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

    # Save entry to CSV
    csv_file = "burnout_predictions.csv"
    df_record = pd.DataFrame([full_input])
    if os.path.exists(csv_file):
        df_record.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df_record.to_csv(csv_file, index=False)

    st.info("📁 Entry saved to `burnout_predictions.csv`.")
