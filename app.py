import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model1 = joblib.load("diabetes_logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="Diabetes Prediction",page_icon="🩺",layout="centered")

# Title
st.title("🩺 Diabetes Prediction using Logistic Regression")

st.write(
    "Enter the patient's information below to predict "
    "the likelihood of diabetes."
)

# User inputs
pregnancies = st.number_input("Pregnancies",min_value=0,max_value=20,value=1)

glucose = st.number_input("Glucose",min_value=0,max_value=300,value=120)

blood_pressure = st.number_input("Blood Pressure",min_value=0,max_value=200,value=70)

skin_thickness = st.number_input("Skin Thickness",min_value=0,max_value=100,value=20)

insulin = st.number_input("Insulin",min_value=0,max_value=900,value=80)

bmi = st.number_input("BMI",min_value=0.0,max_value=70.0,value=25.0)

diabetes_pedigree = st.number_input("Diabetes Pedigree Function",min_value=0.0,max_value=3.0,value=0.47)

age = st.number_input("Age",min_value=1,max_value=120,value=30)

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Scale input using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model1.predict(input_scaled)[0]

    # Probability
    probability = model1.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    st.write(
        f"Diabetes Probability: {probability:.2%}"
    )

    if prediction == 1:
        st.error("Prediction: Diabetes")
    else:
        st.success("Prediction: No Diabetes")