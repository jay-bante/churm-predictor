import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

# --- Configuration ---
# Set the page title and layout
st.set_page_config(
    page_title="AI Crop Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ML Model Training (Cached for Performance) ---

@st.cache_resource
def load_and_train_model():
    """Loads the data, trains the Random Forest model, and caches it."""
    
    # Check for the data file
    if not os.path.exists('Crop_recommendation.csv'):
        st.error("🚨 ERROR: 'Crop_recommendation.csv' not found.")
        st.stop()

    try:
        data = pd.read_csv('Crop_recommendation.csv')
    except Exception as e:
        st.error(f"🚨 ERROR loading CSV: {e}")
        st.stop()

    # Separate Features (X) and Target (y)
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Get feature names for ordered input (N, P, K, Temp, Humidity, pH, Rainfall)
    feature_names = X.columns.tolist()

    # Train the model (Using all data for final app, but split is included for verification)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Validation (Optional: Display accuracy on test set in the app)
    val_model = RandomForestClassifier(n_estimators=100, random_state=42)
    val_model.fit(X_train, y_train)
    predictions = val_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, feature_names, accuracy

# --- Prediction Function ---

def make_prediction(model, feature_names, input_values):
    """
    Runs prediction and returns the top 3 crops with confidence scores.
    """
    # Convert input dictionary to a DataFrame in the correct feature order
    input_data = pd.DataFrame([input_values], columns=feature_names)
    
    # Get class probabilities
    probabilities = model.predict_proba(input_data)[0]
    crop_classes = model.classes_
    
    # Combine and sort
    confidence_scores = sorted(
        zip(crop_classes, probabilities), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return confidence_scores[:3] # Return top 3

# --- Streamlit App UI ---

# Load model and features only once
try:
    model, features, accuracy = load_and_train_model()
    st.sidebar.success(f"Model trained with {accuracy*100:.2f}% validation accuracy.")
except Exception as e:
    st.error("Failed to initialize the machine learning model.")
    st.stop()

st.title("🌾 AI Crop Recommender")
st.markdown("Enter the soil and climate metrics below to get the top 3 crop recommendations.")


# Input Card Style
st.markdown(
    """
    <style>
    /* Mimic the card style from the HTML */
    .stContainer {
        border: 1px solid #c8e6c9; /* Light green border */
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stTextInput label, .stNumberInput label {
        font-weight: 500;
        color: #1f2937; /* Gray-700 */
    }
    </style>
    """, unsafe_allow_html=True
)

st.subheader("Soil & Climate Metrics (7 Features)")

# Use columns to mimic the two-column grid layout (3 pairs + 1 wide)
input_values = {}
feature_labels = {
    'N': '1. Nitrogen (N) Content',
    'P': '2. Phosphorus (P) Content',
    'K': '3. Potassium (K) Content',
    'temperature': '4. Average Temperature (°C)',
    'humidity': '5. Relative Humidity (%)',
    'ph': '6. pH Value (0-14)',
    'rainfall': '7. Rainfall (mm)'
}
default_values = {
    'N': 90.0, 'P': 42.0, 'K': 43.0, 'temperature': 20.87, 
    'humidity': 82.00, 'ph': 6.50, 'rainfall': 202.93
}

# 3 Columns for the first 6 inputs
cols1 = st.columns(2)

# Input 1: N
with cols1[0]:
    input_values['N'] = st.number_input(feature_labels['N'], value=default_values['N'], format="%.2f", step=1.0)

# Input 2: P
with cols1[1]:
    input_values['P'] = st.number_input(feature_labels['P'], value=default_values['P'], format="%.2f", step=1.0)

# Input 3: K
cols2 = st.columns(2)
with cols2[0]:
    input_values['K'] = st.number_input(feature_labels['K'], value=default_values['K'], format="%.2f", step=1.0)

# Input 4: Temperature
with cols2[1]:
    input_values['temperature'] = st.number_input(feature_labels['temperature'], value=default_values['temperature'], format="%.2f", step=0.1)

# Input 5: Humidity
cols3 = st.columns(2)
with cols3[0]:
    input_values['humidity'] = st.number_input(feature_labels['humidity'], value=default_values['humidity'], format="%.2f", step=0.1)

# Input 6: pH
with cols3[1]:
    input_values['ph'] = st.number_input(feature_labels['ph'], value=default_values['ph'], format="%.2f", step=0.01)

# Input 7: Rainfall (Full width)
input_values['rainfall'] = st.number_input(feature_labels['rainfall'], value=default_values['rainfall'], format="%.2f", step=1.0)

# --- Prediction Button ---
if st.button("Get Top 3 Recommendations", type="primary", use_container_width=True):
    st.subheader("Prediction Results")
    
    # Create the input dictionary with feature names in the correct order
    ordered_input = {f: input_values[f] for f in features}
    
    with st.spinner('Analyzing data and calculating prediction...'):
        top_crops = make_prediction(model, features, ordered_input)
        
        st.success("Analysis Complete!")
        st.markdown("### ✨ Top 3 Most Suitable Crops:")
        
        # Display results in a structured list
        for i, (crop, confidence) in enumerate(top_crops):
            confidence_percent = confidence * 100
            
            if i == 0:
                # Highlight the best recommendation
                st.markdown(f"**🥇 1. {crop.upper()}** (Confidence: **{confidence_percent:.2f}%**)", unsafe_allow_html=True)
            else:
                st.markdown(f"&nbsp;&nbsp;&nbsp; {i+1}. {crop.upper()} (Confidence: {confidence_percent:.2f}%)")

# How to run the app
st.sidebar.info(
    "**To run this app locally:**\n"
    "1. Save this code as `crop_recommender_app.py`.\n"
    "2. Ensure `Crop_recommendation.csv` is in the same folder.\n"
    "3. Run the command: `streamlit run crop_recommender_app.py`"
)