import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Modell betöltése
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("football_model.h5")
    return model

model = load_model()

st.title("Nemzetközi Labdarúgó Mérkőzések Eredményelőrejelzése")

# Felhasználói bemenetek
year = st.number_input("Év", min_value=1872, max_value=2025, value=2024)
month = st.number_input("Hónap", min_value=1, max_value=12, value=6)
day_of_week = st.number_input("A hét napja (0=Hétfő, 6=Vasárnap)", min_value=0, max_value=6, value=3)

# Adatok előkészítése
input_data = pd.DataFrame({'year': [year], 'month': [month], 'day_of_week': [day_of_week]})
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_data)

# Predikció
prediction = model.predict(input_scaled)
predicted_class = np.argmax(prediction)

st.subheader(f"Előrejelzett torna típusa: {predicted_class}")
