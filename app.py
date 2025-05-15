import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Modell betöltése
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("football_match_model.h5")

model = load_model()

# Feature oszlopok betöltése
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Scaler betöltése
scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy", allow_pickle=True)
scaler.scale_ = np.load("scaler_scale.npy", allow_pickle=True)

# Label Encoder osztályok betöltése
le_classes = np.load("label_encoder_classes.npy", allow_pickle=True)

# Streamlit App
st.title("Labdarúgó Mérkőzések Torna Predikciója")

home_team = st.text_input("Hazai csapat neve")
away_team = st.text_input("Vendég csapat neve")
year = st.number_input("Év", min_value=1872, max_value=2025, value=2024)
month = st.number_input("Hónap", min_value=1, max_value=12, value=6)
day_of_week = st.number_input("A hét napja (0=Hétfő, 6=Vasárnap)", min_value=0, max_value=6, value=3)
result = st.selectbox("Mérkőzés eredménye", ["home_win", "away_win", "draw"])

if st.button("Előrejelzés"):
    # Adatok előkészítése
    input_data = pd.DataFrame({'home_team': [home_team], 'away_team': [away_team], 'year': [year], 
                               'month': [month], 'day_of_week': [day_of_week], 'result': [result]})
    input_data = pd.get_dummies(input_data)  # Kategorikus adatok one-hot encodingja

    # Hiányzó oszlopok pótlása az eredeti feature struktúrához
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Rendezés az eredeti feature sorrendhez
    input_data = input_data[feature_columns]

    # Scaling
    input_scaled = scaler.fit_transform(input_data)

    # Predikció
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction)

    st.subheader(f"Előrejelzett torna típusa: {le_classes[predicted_class]}")
