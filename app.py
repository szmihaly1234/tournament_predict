import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load all artifacts
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model('tournament_predictor.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return model, le, scaler, feature_columns

model, le, scaler, feature_columns = load_artifacts()

# Get unique teams from feature columns (extract from one-hot encoded features)
def get_unique_entities(prefix):
    entities = set()
    for col in feature_columns:
        if col.startswith(prefix + '_'):
            entity = col[len(prefix)+1:]
            entities.add(entity)
    return sorted(entities)

# Create the Streamlit UI
st.title("Football Tournament Predictor")
st.write("Predict which tournament a football match belongs to based on match details")

# Get available teams and results from feature columns
home_teams = get_unique_entities('home_team')
away_teams = get_unique_entities('away_team')
result_options = get_unique_entities('result')

# Input form
with st.form("match_details"):
    home_team = st.selectbox("Home Team", home_teams)
    away_team = st.selectbox("Away Team", away_teams)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    day_of_week = st.selectbox("Day of Week", 
                             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                             index=5)
    result = st.selectbox("Expected Result", result_options, index=0)
    
    submitted = st.form_submit_button("Predict Tournament")

if submitted:
    # Map day of week to number (Monday=0)
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    
    # Create input dataframe with raw values
    input_data = {
        'home_team': [home_team],
        'away_team': [away_team],
        'year': [year],
        'month': [month],
        'day_of_week': [day_map[day_of_week]],
        'result': [result]
    }
    
    # Convert to DataFrame
    df_input = pd.DataFrame(input_data)
    
    # One-hot encode the categorical features
    X_input = pd.get_dummies(df_input)
    
    # Create a DataFrame with all expected columns, initialized to 0
    X_processed = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Fill in the values we have
    for col in X_input.columns:
        if col in X_processed.columns:
            X_processed[col] = X_input[col]
    
    # Ensure numerical columns are properly set
    numerical_cols = ['year', 'month', 'day_of_week']
    for col in numerical_cols:
        if col in X_processed.columns:
            X_processed[col] = df_input[col].values[0]
    
    # Scale the features
    X_scaled = scaler.transform(X_processed)
    
    # Make prediction
    preds = model.predict(X_scaled)
    top3_idx = np.argsort(preds[0])[-3:][::-1]
    top3_tournaments = le.inverse_transform(top3_idx)
    top3_probs = preds[0][top3_idx]
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"Most likely tournament: **{top3_tournaments[0]}** ({(top3_probs[0]*100):.1f}%)")
    
    st.write("Top 3 predicted tournaments:")
    for tourn, prob in zip(top3_tournaments, top3_probs):
        st.write(f"- {tourn}: {(prob*100):.1f}%")

# Add some info
st.sidebar.markdown("""
**About this app:**
This app predicts which football tournament a match belongs to based on:
- Home and away teams
- Date information
- Match result

The model was trained on historical international football match data.
""")
