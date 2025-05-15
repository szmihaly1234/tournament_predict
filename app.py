import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Tournament Predictor", layout="wide")

# Title
st.title("Football Match Tournament Predictor")

# Sidebar for user inputs
st.sidebar.header("Match Details")

# Load data, preprocess, and get feature columns
@st.cache_resource
def load_data_and_model():
    # Load data
    df = pd.read_csv('all_matches.csv')
    
    # Preprocessing
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['result'] = np.where(df['home_score'] > df['away_score'], 'home_win',
                           np.where(df['home_score'] < df['away_score'], 'away_win', 'draw'))
    
    tournament_counts = df['tournament'].value_counts()
    rare_tournaments = tournament_counts[tournament_counts < 500].index
    df['tournament'] = df['tournament'].replace(rare_tournaments, 'Other')
    
    # Prepare features and target
    features = ['home_team', 'away_team', 'year', 'month', 'day_of_week', 'result']
    X = pd.get_dummies(df[features])
    y = df['tournament']
    
    # Save the feature columns for later use
    feature_columns = X.columns.tolist()
    
    # Train model
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    
    # Build and train model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model (in a real deployment, you'd load a pre-trained model)
    model.fit(X_train, y_train_cat, epochs=10, batch_size=64, verbose=0)
    
    return model, le, scaler, features, feature_columns

# Load model and encoders
model, le, scaler, features, feature_columns = load_data_and_model()

# Get unique teams from the dataset
@st.cache_data
def get_unique_teams():
    df = pd.read_csv('all_matches.csv')
    teams = sorted(list(set(df['home_team'].unique()).union(set(df['away_team'].unique()))))
    return teams

teams = get_unique_teams()

# User input form
with st.sidebar.form("match_details"):
    home_team = st.selectbox("Home Team", teams)
    away_team = st.selectbox("Away Team", teams)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
    month = st.number_input("Month", min_value=1, max_value=12, value=6)
    day_of_week = st.selectbox("Day of Week", 
                              ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                              index=5)
    result = st.selectbox("Expected Result", ["home_win", "away_win", "draw"])
    
    submitted = st.form_submit_button("Predict Tournament")

def prepare_input(home_team, away_team, year, month, day_of_week, result, feature_columns):
    # Map day of week to number
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    day_num = day_map[day_of_week]
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'home_team': [home_team],
        'away_team': [away_team],
        'year': [year],
        'month': [month],
        'day_of_week': [day_num],
        'result': [result]
    })
    
    # One-hot encode
    X_input = pd.get_dummies(input_data)
    
    # Ensure all columns are present and in correct order
    for col in feature_columns:
        if col not in X_input.columns:
            X_input[col] = 0
    
    # Reorder columns to match training data
    X_input = X_input[feature_columns]
    
    return X_input

# When form is submitted
if submitted:
    try:
        # Prepare input data
        X_input = prepare_input(home_team, away_team, year, month, day_of_week, result, feature_columns)
        
        # Scale
        X_input_scaled = scaler.transform(X_input)
        
        # Predict
        prediction = model.predict(X_input_scaled)
        predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)
        
        # Display results
        st.subheader("Prediction Results")
        st.write(f"Predicted Tournament: **{predicted_class}**")
        st.write(f"Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        top3 = np.argsort(prediction[0])[-3:][::-1]
        st.write("\nTop 3 Predictions:")
        for i, idx in enumerate(top3):
            st.write(f"{i+1}. {le.inverse_transform([idx])[0]} ({prediction[0][idx]:.2%})")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
