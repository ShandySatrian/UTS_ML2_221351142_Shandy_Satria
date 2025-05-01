import streamlit as st
import pandas as pd
import numpy as np
import pycountry
import joblib
import plotly.express as px
import tensorflow as tf

st.set_page_config(page_title="Sejukin – AQI Status Predictor", layout="centered")

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="model_ann.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load encoder dan scaler
scaler = joblib.load("scaler.pkl")
le_country = joblib.load("country_encoder.pkl")
le_status = joblib.load("status_encoder.pkl")

st.title("🌍 Sejukin – Air Quality Status Predictor")

# --- Input form
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Tanggal", pd.to_datetime("2023-01-01"))
        country = st.selectbox("Negara", sorted(le_country.classes_))
    with col2:
        aqi_value = st.number_input("AQI Value", min_value=0, max_value=500, value=100)

    submitted = st.form_submit_button("🔍 Prediksi")

# --- Saat tombol ditekan
if submitted:
    # Preprocessing
    year, month, day = date.year, date.month, date.day
    country_encoded = le_country.transform([country])[0]

    # Build input array
    input_arr = np.array([[year, month, day, country_encoded, aqi_value]])
    input_scaled = scaler.transform(input_arr).astype(np.float32)

    # Prediksi pakai TFLite
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output)
    status_predicted = le_status.inverse_transform([pred_class])[0]

    st.success(f"💡 Prediksi Status Kualitas Udara: **{status_predicted}**")

    # --- Visualisasi peta
    country_obj = pycountry.countries.get(name=country)
    if country_obj:
        iso_code = country_obj.alpha_3
        fig = px.choropleth(
            locations=[iso_code],
            locationmode="ISO-3",
            color=[aqi_value],
            color_continuous_scale="Reds",
            range_color=(0, 500),
            title=f"{country} | AQI: {aqi_value}"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Negara tidak ditemukan untuk ditampilkan di peta.")
