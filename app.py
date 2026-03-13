import streamlit as st
import pandas as pd
import joblib

# Load models
svm_model = joblib.load("svm_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
st.title("Childhood Asthma Prediction System")
st.write("Enter patient information:")

# Inputs
age = st.slider("Age",5,18)

bmi = st.slider("BMI",10.0,40.0)

pollution = st.slider("Pollution Exposure",0.0,10.0)

wheezing = st.selectbox("Wheezing",[0,1])

breath = st.selectbox("Shortness Of Breath",[0,1])

cough = st.selectbox("Coughing",[0,1])

night = st.selectbox("Nighttime Symptoms",[0,1])

# Prediction
if st.button("Predict Asthma Risk"):

    input_data = pd.DataFrame({
        "Age":[age],
        "BMI":[bmi],
        "PollutionExposure":[pollution],
        "Wheezing":[wheezing],
        "ShortnessOfBreath":[breath],
        "Coughing":[cough],
        "NighttimeSymptoms":[night]
    })

    # Probabilities
    svm_prob = svm_model.predict_proba(input_data)[0][1]
    xgb_prob = xgb_model.predict_proba(input_data)[0][1]

    hybrid_prob = (svm_prob + xgb_prob) / 2

    hybrid_pred = 1 if hybrid_prob > 0.5 else 0

    st.subheader("Prediction Results")

    st.write("SVM Probability:", round(svm_prob,3))
    st.write("XGBoost Probability:", round(xgb_prob,3))
    st.write("Hybrid Probability:", round(hybrid_prob,3))

    if hybrid_pred == 1:
        st.error("High Asthma Risk")
    else:
        st.success("Low Asthma Risk")