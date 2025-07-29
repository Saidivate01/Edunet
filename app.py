
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model_new.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Education mapping
education_mapping = {
    1: 'Preschool', 2: '1st-4th', 3: '5th-6th', 4: '7th-8th', 5: '9th',
    6: '10th', 7: '11th', 8: '12th', 9: 'HS-grad', 10: 'Some-college',
    11: 'Assoc-voc', 12: 'Assoc-acdm', 13: 'Bachelors', 14: 'Masters',
    15: 'Prof-school', 16: 'Doctorate'
}

# Occupation mapping
occupation_mapping = {
    0: '?', 1: 'Cambodia', 2: 'Canada', 3: 'China', 4: 'Columbia',
    5: 'Cuba', 6: 'Dominican-Republic', 7: 'Ecuador', 8: 'El-Salvador',
    9: 'England', 10: 'France', 11: 'Germany', 12: 'Greece',
    13: 'Guatemala', 14: 'Haiti'
}


age = st.sidebar.slider("Age", 18, 65, 30)
education_label = st.sidebar.selectbox("Education Level", list(education_mapping.values()))
occupation_label = st.sidebar.selectbox("Occupation", list(occupation_mapping.values()))
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Get the numerical value from the label
education_num = [k for k, v in education_mapping.items() if v == education_label][0]
occupation_num = [k for k, v in occupation_mapping.items() if v == occupation_label][0]


# Build input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'educational-num': [education_num],
    'occupation': [occupation_num],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
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
