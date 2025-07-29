import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

# Input from user
years_experience = st.number_input("Years of Experience", min_value=0.0, step=0.1)
age = st.number_input("Age", min_value=18, max_value=70, step=1)
education_level = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])

# One-hot encoding or manual mapping if required
education_mapping = {
    "High School": 0,
    "Bachelors": 1,
    "Masters": 2,
    "PhD": 3
}
education_level_encoded = education_mapping[education_level]

# Construct input DataFrame with EXACT same feature names used in training
input_df = pd.DataFrame([{
    "YearsExperience": years_experience,
    "Age": age,
    "EducationLevel": education_level_encoded
}])

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
