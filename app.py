import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# Oldal konfiguráció
st.set_page_config(
    page_title="Labdarúgó Tournament Prediktor",
    page_icon="⚽",
    layout="wide"
)

# Modell és preprocesszor betöltése
@st.cache_resource
def load_components():
    # Modell betöltése
    model = load_model('football_match_model.h5')
    
    # Preprocesszor és label encoder betöltése
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Label encoder osztályainak betöltése
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, preprocessor, label_encoder

# Komponensek betöltése
try:
    model, preprocessor, label_encoder = load_components()
    classes = label_encoder.classes_
except Exception as e:
    st.error(f"Hiba történt a modell betöltésekor: {str(e)}")
    st.stop()

# Felhasználói interfész
st.title("⚽ Labdarúgó Tournament Prediktor")
st.markdown("""
Ez az alkalmazás előrejelzi, hogy egy adott mérkőzés milyen típusú versenyen (tournament) történik.
""")

# Űrlap létrehozása
with st.form("prediction_form"):
    st.header("Mérkőzés adatai")
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("Hazai csapat", "Hungary")
    with col2:
        away_team = st.text_input("Vendég csapat", "Germany")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        year = st.number_input("Év", min_value=1872, max_value=2100, value=2023)
    with col4:
        month = st.selectbox("Hónap", options=list(range(1,13)), index=5)
    with col5:
        day_of_week = st.selectbox("A hét napja", options=list(range(7)), 
                              format_func=lambda x: ["Hétfő", "Kedd", "Szerda", "Csütörtök", "Péntek", "Szombat", "Vasárnap"][x])
    
    result = st.radio("Eredmény", options=["home_win", "away_win", "draw"], 
                      format_func=lambda x: {"home_win": "Hazai győzelem", "away_win": "Vendég győzelem", "draw": "Döntetlen"}[x])
    
    submitted = st.form_submit_button("Előrejelzés készítése")

# Predikció készítése
if submitted:
    try:
        # Input adatok előkészítése DataFrame-be
        input_data = {
            'home_team': [home_team],
            'away_team': [away_team],
            'year': [year],
            'month': [month],
            'day_of_week': [day_of_week],
            'result': [result]
        }
        input_df = pd.DataFrame(input_data)
        
        # One-hot encoding a szöveges inputokhoz
        input_processed = pd.get_dummies(input_df)
        
        # Hiányzó oszlopok kezelése (ha a train-ben voltak olyan csapatok amik itt nincsenek)
        train_columns = preprocessor.get_feature_names_out()
        for col in train_columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        
        # Csak a szükséges oszlopok megtartása a megfelelő sorrendben
        input_processed = input_processed[train_columns]
        
        # Predikció
        predictions = model.predict(input_processed)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = np.max(predictions[0])
        
        # Eredmény megjelenítése
        st.success("Predikció kész!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediktált tournament típus", predicted_class)
            st.metric("Biztonsági szint", f"{confidence*100:.1f}%")
        
        with col2:
            st.subheader("Összes valószínűség")
            for i, class_name in enumerate(classes):
                st.progress(float(predictions[0][i]), 
                           text=f"{class_name}: {predictions[0][i]*100:.1f}%")
        
        # Dátum hatásának vizualizációja
        st.subheader("Hónap hatása a predikcióra")
        
        months = range(1, 13)
        month_probs = []
        
        for m in months:
            temp_data = input_data.copy()
            temp_data['month'] = [m]
            temp_df = pd.DataFrame(temp_data)
            temp_processed = pd.get_dummies(temp_df)
            
            # Hiányzó oszlopok kezelése
            for col in train_columns:
                if col not in temp_processed.columns:
                    temp_processed[col] = 0
            
            temp_processed = temp_processed[train_columns]
            month_pred = model.predict(temp_processed)
            month_probs.append(month_pred[0][predicted_class_idx])
        
        # Ábra készítése
        fig, ax = plt.subplots()
        ax.plot(months, month_probs, marker='o')
        ax.set_xlabel("Hónap")
        ax.set_ylabel("Valószínűség")
        ax.set_title(f"Valószínűség változása a hónapok szerint: {predicted_class}")
        ax.grid(True)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Hiba történt a predikció során: {str(e)}")

# Oldalsáv
with st.sidebar:
    st.header("ℹ Információ")
    st.markdown("""
    **Használati útmutató:**
    1. Töltse ki a mérkőzés adatait
    2. Kattintson az "Előrejelzés készítése" gombra
    3. Nézze meg az eredményeket
    
    **Modell információk:**
    - Modell típusa: Neurális háló
    - Pontosság: {:.1f}% (teszt adatokon)
    - Utolsó frissítés: {}
    """.format(0.85*100, "2023.11.15"))  # Itt helyettesítsd be a valós értékeket
