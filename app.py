import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit as st


model=joblib.load('model.joblib')
encoder=joblib.load('encoder.joblib')
feature=joblib.load('feature.joblib')

st.set_page_config(
    page_title="AQI PREDICTION",
    layout="centered",
    initial_sidebar_state="auto"
)

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air quality is considered **satisfactory**, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Moderate", "Air quality is **acceptable**. However, there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may **experience health effects**. The general public is unlikely to be affected."
    elif aqi <= 200:
        return "Unhealthy", "**Everyone may begin to experience health effects**; members of sensitive groups may experience more serious health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: **Everyone may experience more serious health effects**."
    else:
        return "Hazardous", "Health warnings of **emergency conditions**. The entire population is more likely to be affected."

st.markdown("""
    <style>
    .reportview-container .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 30px;
    }
    .good { background-color: #00E400; }
    .moderate { background-color: #FFFF00; color: black !important; }
    .unhealthy_s { background-color: #FF7E00; }
    .unhealthy { background-color: #FF0000; }
    .very_unhealthy { background-color: #8F3F97; }
    .hazardous { background-color: #7E0023; }
    h1 {
        color: #0E1117;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)



st.title("AQI PREDICTION")
st.markdown("---")
st.subheader("Input Air Quality and Meteorological Parameters")


col1, col2, col3 = st.columns(3)

with col1:
    country= st.selectbox("Country", options=['US', 'GB', 'FR', 'DE', 'ES', 'IT', 'CA', 'MX', 'BR', 'AR', 'ZA',
       'EG', 'KE', 'NG', 'AE', 'SA', 'QA', 'IN', 'JP', 'KR', 'CN', 'HK',
       'SG', 'TH', 'MY', 'ID', 'AU', 'NZ', 'RU', 'TR', 'IR', 'PK', 'PH',
       'VN', 'PL', 'SE', 'FI', 'CH'], index=0)
    pm25 = st.slider('PM2.5 (µg/m³)', 0.0, 500.0, 50.0, step=0.1)
    no2 = st.slider('NO₂ (µg/m³)', 0.0, 200.0, 30.0, step=0.1)
    o3 = st.slider('O₃ (µg/m³)', 0.0, 250.0, 60.0, step=0.1)
    
with col2:
    pm10 = st.slider('PM10 (µg/m³)', 0.0, 600.0, 80.0, step=0.1)
    so2 = st.slider('SO₂ (µg/m³)', 0.0, 150.0, 10.0, step=0.1)
    co = st.slider('CO (mg/m³)', 0.0, 50.0, 5.0, step=0.1)

with col3:
    temperature = st.slider('Temperature (°C)', -20.0, 50.0, 25.0, step=0.1)
    humidity = st.slider('Humidity (%)', 0.0, 100.0, 50.0, step=0.1)
    wind_speed = st.slider('Wind Speed (m/s)', 0.0, 30.0, 5.0, step=0.1)

input=[[country, pm25,pm10,no2,so2,o3,co,temperature,humidity,wind_speed]]
x=pd.DataFrame(input,columns=feature.columns)
x=encoder.transform(x)
pred=model.predict(x)

if st.button('Predict Air Quality Index (AQI)'):
     st.success(f"AQI IN {country} : {pred[0]}")
     st.info(f"{get_aqi_category(pred[0])}")



