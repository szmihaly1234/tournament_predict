import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# 📌 Modell betöltése
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("football_match_model.h5")

model = load_model()

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# 📌 Scaler betöltése
scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy", allow_pickle=True)
scaler.scale_ = np.load("scaler_scale.npy", allow_pickle=True)

# 📌 Felhasználói bemenet példája (weboldalról)
input_data = pd.DataFrame({'year': [2024], 'month': [6], 'day_of_week': [3], 
                           'home_team': ['Brazil'], 'away_team': ['Argentina'], 'result': ['home_win']})

# 📌 One-hot encoding a kategorikus változóknak
input_data = pd.get_dummies(input_data)

# 📌 Hiányzó oszlopok kitöltése a tanítási adatszerkezet alapján
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Hiányzó oszlopok pótlása nullával

# 📌 Feature-k rendezése az eredeti sorrend szerint
input_data = input_data[feature_columns]

# 📌 Input alak ellenőrzése
print("Input shape (ellenőrzés):", input_data.shape)  # Ennek (1,565)-nek kell lennie!

# 📌 Skálázás (az eredeti scaler-rel)
input_scaled = scaler.transform(input_data)
print("Scaled input shape:", input_scaled.shape)  # Ennek is (1,565)-nek kell lennie!

