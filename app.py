import streamlit as st
import pickle
import numpy as np
import pandas as pd
import math

# Load the XGBoost model from file "xgb.pkl"
@st.cache_resource
def load_model():
    with open("xgb.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model

model = load_model()

st.title("🚕 XGBoost Tip Prediction App")

# User Inputs
passenger_count = st.number_input("Passenger Count")
trip_distance = st.number_input("Trip Distance (miles)")
day_night = st.selectbox("Day or Night", [0, 1])  # 0 = Day, 1 = Night
tolls_amount = st.number_input("Tolls Amount ($)")
trip_duration_minutes = st.number_input("Trip Duration (minutes)")


if st.button("Predict Tip 💰"):
    # Prepare input features as a 2D array and convert to DataFrame
    input_features = np.array([[passenger_count, trip_distance, day_night, tolls_amount, trip_duration_minutes]])
    input_df = pd.DataFrame(
        input_features, 
        columns=['passenger_count', 'trip_distance', 'day_night', 'tolls_amount', 'trip_duration_minutes']
    )
    
    # Make prediction
    tip_pred = model.predict(input_features)[0]
    
    # Format predicted tip as a string (e.g., "5.79$")
    tip_str = f"{tip_pred:.2f}$"
    st.success(f"🚖 Taxi Tip Recommendation: Tip ${tip_str} 💵")

    
    meter_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=PT+Sans:wght@700&display=swap');
        .meter-container {{
            width: 600px;
            background-color: #000;
            color: white;
            border-radius: 20px;
            padding: 20px;
            margin: auto;
            box-shadow: 0 0 10px #333;
            font-family: 'PT Sans', sans-serif;
        }}
        .meter-header {{
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            letter-spacing: 2px;
            margin-bottom: 20px;
        }}
        .meter-body {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .meter-left {{
            font-size: 1rem;
            line-height: 1.8rem;
            text-align: left;
        }}
        .meter-right {{
            text-align: right;
            font-size: 1.2rem;
            color: #FFD700;
        }}
        .tip-value {{
            margin-top: 10px;
            font-size: 1.5rem;
            color: #FFD700;
            font-weight: bold;
        }}
    </style>

    <div class="meter-container">
    <p style="text-align:center; font-size: 0.8rem; color: #aaa;">
            Dubai • 29°C • 15:20 • 21-03-2025
        </p>
        <div class="meter-header">TAXI METER</div>
        <div class="meter-body">
            <div class="meter-left">
                Distance: {trip_distance:.2f} miles<br>
                Time: {trip_duration_minutes:.2f} min<br>
                Extras: ${tolls_amount:.2f}<br>
                Total: ${20:.2f}
            </div>
            <div class="meter-right">
                Tip {tip_pred:.2f}$? ✅ ❌ <br>
            </div>
        </div>
    </div>
    """
    
    st.markdown(meter_html, unsafe_allow_html=True)
 
