import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and label encoder
model = joblib.load("gradient_boosting_smote_nocategorical_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Emission Classification", layout="centered")
st.title("Emission Category Predictor (Numerical Features Only)")
st.markdown("Enter the measured values to classify the emission group.")

# Define all required numerical features
feature_names = [
    'E.EBP (mbar)', 'Cubic capacity (cm3)', 'Gross engine power in KW at 1500 rpm',
    'KW', 'A/F', 'IMT', 'IMP (mbar)', 'E.Nox (ppm)', 'E.HC (ppm)', 'E.CO (ppm)',
    'E.No2 (ppm)', 'CO2 (ppm)', 'Flow rate (kg/hr)', 'C.Nox (ppm)', 'C.HC (ppm)',
    'C.CO (ppm)', 'C.CO2 (ppm)', 'C.No2 (ppm)'
]

# Collect user inputs
inputs = {}
with st.form("input_form"):
    for feature in feature_names:
        inputs[feature] = st.number_input(feature, value=0.0, format="%.3f")
    submitted = st.form_submit_button("Predict")

if submitted:
    # Create DataFrame from input
    input_df = pd.DataFrame([inputs])
    
    # Predict
    prediction = model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([prediction])[0]
    pred_proba = model.predict_proba(input_df)[0]
    class_names = label_encoder.inverse_transform(np.arange(len(pred_proba)))

    # Display result
    st.success(f"**Predicted Category:** `{pred_label}`")
    
    # Show probability chart
    st.markdown("### 🔎 Prediction Confidence")
    proba_df = pd.DataFrame({
        "Class": class_names,
        "Probability": pred_proba
    }).sort_values("Probability", ascending=False)

    st.bar_chart(proba_df.set_index("Class"))

    # Optionally: raw probabilities
    with st.expander("See raw probability values"):
        st.dataframe(proba_df.reset_index(drop=True))
