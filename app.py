import streamlit as st
import pandas as pd
import joblib

# 1. CROP SURVIVAL THRESHOLDS (Dictionary Configuration)
CROP_LIMITS = {
    'Wheat':  {'min_ph': 4.8, 'max_ph': 8.3, 'max_rain': 1200, 'min_temp': 5,  'max_temp': 38},
    'Rice':   {'min_ph': 4.5, 'max_ph': 8.5, 'max_rain': 2500, 'min_temp': 12, 'max_temp': 42},
    'Maize':  {'min_ph': 5.0, 'max_ph': 8.0, 'max_rain': 1500, 'min_temp': 10, 'max_temp': 40},
    'Cotton': {'min_ph': 5.5, 'max_ph': 8.5, 'max_rain': 1800, 'min_temp': 15, 'max_temp': 45},
}

# 2. BIOLOGICAL GUARDRAIL CHECKER
def check_biological_viability(crop_name, ph_val, rain_val, temp_val, fert_val, n_val, k_val):
    reasons = []
    limits = CROP_LIMITS.get(crop_name)
    
    if limits:
        if ph_val < limits['min_ph'] or ph_val > limits['max_ph']:
            reasons.append(f"Soil pH ({ph_val}) is outside the survival range for {crop_name} ({limits['min_ph']} - {limits['max_ph']}).")
        if rain_val > limits['max_rain']:
            reasons.append(f"Rainfall ({rain_val} mm) exceeds flood tolerance for {crop_name} ({limits['max_rain']} mm).")
        if temp_val < limits['min_temp'] or temp_val > limits['max_temp']:
            reasons.append(f"Temperature ({temp_val}°C) exceeds thermal limits for {crop_name}.")
            
    if fert_val < 15 and (n_val < 20 or k_val < 20):
        reasons.append("Severe Nutrient Depletion: Fertilizer and soil nutrients are too low to sustain growth.")
        
    return reasons

# 3. LOAD TRAINED MODEL
model = joblib.load('crop_yield_model.pkl')

# 4. STREAMLIT UI LAYOUT
st.title("🌾 Crop Yield Prediction System")
st.write("Enter soil, weather, and region details to predict crop yield.")

col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    state = st.selectbox("State", ['Punjab', 'Uttar Pradesh', 'West Bengal', 'Maharashtra', 'Tamil Nadu', 'Andhra Pradesh'])
    district = st.selectbox("District", ['Ludhiana', 'Patiala', 'Kanpur', 'Varanasi', 'Hooghly', 'Bardhaman', 'Nashik', 'Pune', 'Thanjavur', 'Kurnool'])
    crop = st.selectbox("Crop", ['Wheat', 'Rice', 'Maize', 'Cotton'])

with col_meta2:
    season = st.selectbox("Season", ['Kharif', 'Rabi', 'Whole Year'])
    soil_type = st.selectbox("Soil Type", ['Alluvial', 'Clay Loam', 'Black Soil', 'Red Loam', 'Red Soil'])

st.subheader("Nutrient & Weather Parameters")
col1, col2 = st.columns(2)

with col1:
    n = st.number_input("Nitrogen (N kg/ha)", 0, 300, 100)
    p = st.number_input("Phosphorus (P kg/ha)", 0, 150, 50)
    k = st.number_input("Potassium (K kg/ha)", 0, 150, 50)
    ph = st.number_input("Soil pH", 4.0, 10.0, 6.5)

with col2:
    temp = st.number_input("Avg Temperature (°C)", 10.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 10.0, 100.0, 70.0)
    rainfall = st.number_input("Rainfall (mm)", 10.0, 3000.0, 500.0)
    fertilizer = st.number_input("Fertilizer Used (kg/ha)", 0.0, 500.0, 150.0)

# 5. PREDICTION LOGIC
if st.button("Predict Yield"):
    failure_reasons = check_biological_viability(crop, ph, rainfall, temp, fertilizer, n, k)
    
    if failure_reasons:
        st.error("🚨 **Total Crop Failure Warning (0.00 Tons / Hectare)**")
        for reason in failure_reasons:
            st.write(f"• {reason}")
    else:
        input_data = pd.DataFrame([{
            'State': state,
            'District': district,
            'Crop': crop,
            'Season': season,
            'Soil_Type': soil_type,
            'N_kg_ha': n,
            'P_kg_ha': p,
            'K_kg_ha': k,
            'pH': ph,
            'Avg_Temperature_C': temp,
            'Humidity_pct': humidity,
            'Rainfall_mm': rainfall,
            'Fertilizer_Used_kg_ha': fertilizer
        }])
        
        prediction = model.predict(input_data)[0]
        st.success(f"🌱 Predicted Crop Yield: **{prediction:.2f} Tons / Hectare**")