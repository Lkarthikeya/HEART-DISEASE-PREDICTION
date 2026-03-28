import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model, scaler = pickle.load(open("model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("🫀 Heart Disease Prediction")
st.write("Enter patient details")

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])

with col2:
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate")
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak")
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [1, 2, 3])

# Encode categorical
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Validation
def valid():
    return trestbps > 0 and chol > 0 and thalach > 0

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):

    if not valid():
        st.error("Enter valid values")
    else:
        # Correct feature order
        input_data = np.array([[age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach, exang,
                                oldpeak, slope, ca, thal]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]

        # IMPORTANT: dataset → 1 = disease
        disease_prob = probs[1]
        no_disease_prob = probs[0]

        st.subheader("Result")

        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease\n\nProbability: {disease_prob:.2f}")
            st.progress(int(disease_prob * 100))
        else:
            st.success(f"✅ Low Risk of Heart Disease\n\nProbability: {no_disease_prob:.2f}")
            st.progress(int(no_disease_prob * 100))