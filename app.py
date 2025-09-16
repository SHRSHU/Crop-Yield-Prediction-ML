# ===============================
# app.py
# ===============================

import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load model, scaler, feature names, and encoders
# -------------------------
model = joblib.load("crop_yield_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
encoders = joblib.load("encoders.pkl")

# -------------------------
# App Title
# -------------------------
st.title("🌾 Crop Yield Prediction App")
st.write("Enter crop & environmental details to predict yield (kg/ha).")

# -------------------------
# User Inputs
# -------------------------
state = st.selectbox("State Name", encoders["State Name"].classes_)
district = st.text_input("District Name", encoders["Dist Name"].classes_[0])
crop = st.selectbox("Crop Name", encoders["Crop"].classes_)

area = st.number_input("Area (hectares)", min_value=1.0, value=1000.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=1200.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, value=70.0)
pH = st.number_input("Soil pH", min_value=0.0, value=6.5)
wind = st.number_input("Wind Speed (m/s)", min_value=0.0, value=2.0)
solar = st.number_input("Solar Radiation (MJ/m²/day)", min_value=0.0, value=18.0)

# -------------------------
# Encode categorical inputs using saved encoders
# -------------------------
state_code = encoders["State Name"].transform([state])[0]
district_code = encoders["Dist Name"].transform([district])[0]
crop_code = encoders["Crop"].transform([crop])[0]

# -------------------------
# Create input DataFrame
# -------------------------
input_data = pd.DataFrame([[
    state_code, district_code, crop_code,
    area, rainfall, temperature, humidity, pH, wind, solar
]], columns=feature_names)  # ✅ ensures correct order

# Scale input
input_scaled = scaler.transform(input_data)

# -------------------------
# Predict Yield
# -------------------------
if st.button("Predict Yield"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🌱 Predicted Crop Yield: {prediction:.2f} kg/ha")
