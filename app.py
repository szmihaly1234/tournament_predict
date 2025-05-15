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
    with open('onehot_columns.pkl', 'rb') as f:
        onehot_columns = pickle.load(f)
    
    # Load team lists from original data
    df = pd.read_csv('all_matches.csv')
    unique_teams = sorted(list(set(df['home_team'].unique()).union(set(df['away_team'].unique()))))
    
    return model, le, scaler, onehot_columns, unique_teams

model, le, scaler, onehot_columns, unique_teams = load_artifacts()

# Create the Streamlit UI
st.title("Football Tournament Predictor")
st.write("Predict which tournament a football match belongs to based on match details")

# Input form
with st.form("match_details"):
    home_team = st.selectbox("Home Team", unique_teams)
    away_team = st.selectbox("Away Team", unique_teams)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
    month = st.selectbox("Month", range(1, 13), format_func=lambda x: f"{x:02d}")
    day_of_week = st.selectbox("Day of Week", 
                             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                             index=5)
    result = st.selectbox("Expected Result", ["home_win", "away_win", "draw"], index=0)
    
    submitted = st.form_submit_button("Predict Tournament")

if submitted:
    try:
        # Map day of week to number (Monday=0)
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                 "Friday": 4, "Saturday": 5, "Sunday": 6}
        
        # Create a DataFrame with all columns from training, initialized to 0
        X = pd.DataFrame(0, index=[0], columns=onehot_columns)
        
        # Set the appropriate one-hot encoded columns
        home_col = f"home_team_{home_team}"
        away_col = f"away_team_{away_team}"
        result_col = f"result_{result}"
        
        if home_col in X.columns:
            X[home_col] = 1
        else:
            st.warning(f"Home team '{home_team}' not in training data. Using default values.")
        
        if away_col in X.columns:
            X[away_col] = 1
        else:
            st.warning(f"Away team '{away_team}' not in training data. Using default values.")
        
        if result_col in X.columns:
            X[result_col] = 1
        
        # Set numerical features
        X['year'] = year
        X['month'] = month
        X['day_of_week'] = day_map[day_of_week]
        
        # Ensure we only keep columns that were in training
        X = X[onehot_columns]
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
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
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please try different input values.")

# Add some info
st.sidebar.markdown("""
**About this app:**
This app predicts which football tournament a match belongs to based on:
- Home and away teams (selected from known teams)
- Date information
- Match result

The model was trained on historical international football match data.
""")
