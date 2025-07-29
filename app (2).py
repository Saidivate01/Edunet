import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 75, 30)
education_num = st.sidebar.slider("Educational Number (5–16)", 5, 16, 10)
occupation = st.sidebar.slider("Occupation (Encoded 0–13)", 0, 13, 5)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
workclass = st.sidebar.slider("Workclass (Encoded 0–6)", 0, 6, 3)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

# Input data must match training feature names and order
input_df = pd.DataFrame({
    'age': [age],
    'educational-num': [education_num],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'workclass': [workclass],
    'gender': [gender]
})


st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

