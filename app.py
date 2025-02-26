import pandas as pd
import streamlit as st
import pickle
import numpy as np

with open("xgb.pkl", "rb") as model_file:
    model = pickle.load(model_file)


    
# Streamlit App UI
st.title("🚕 XgBoost Tip Prediction App")

# User Inputs
passenger_count = st.number_input("Passenger Count")
trip_distance = st.number_input("Trip Distance (miles)")
day_night = st.selectbox("Day or Night", [0, 1])  # 0 = Day, 1 = Night
tolls_amount = st.number_input("Tolls_Amount")
trip_duration_minutes = st.number_input("Trip Duration (minutes)")

# Prediction Button
if st.button("Predict Tip 💰"):
    input_features = np.array([[passenger_count, trip_distance, day_night, tolls_amount, trip_duration_minutes]])
    input_features = input_features.reshape(1, -1)
    tip_pred = model.predict(input_features)[0]

    # Convert prediction to a string, e.g., "5.79$"
    tip_str = f"{tip_pred:.2f}$"

import streamlit as st
import pickle
import numpy as np

# 1) Load your model (example: xgb.pkl)
with open("xgb.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚕 Taxi Meter Emulation - Clean PT Sans Design")

# 2) Mock user inputs for demonstration
fare = st.number_input("Fare ($)", min_value=0.0, value=12.50)
extras = st.number_input("Extras ($)", min_value=0.0, value=3.00)
day_night = st.selectbox("Time of Day", ["Day", "Night"])
trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, value=5.0)
trip_duration = st.number_input("Trip Duration (minutes)", min_value=1, value=15)

# Convert day/night to 0/1
day_night_val = 1 if day_night == "Night" else 0

# 3) Compute total for mock display
total = fare + extras

# 4) Predict TIP from your model using relevant features
if st.button("Show Taxi Meter"):
    # Example features for the model: [fare, extras, day_night_val, trip_distance, trip_duration]
    # Adapt to your actual feature order!
    input_features = np.array([[fare, extras, day_night_val, trip_distance, trip_duration]])
    tip_pred = model.predict(input_features)[0]
    
    # Convert to string with two decimals
    tip_str = f"{tip_pred:.2f}"
    
    # 5) HTML + CSS for a digital taxi meter with PT Sans font
    meter_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=PT+Sans:wght@700&display=swap');
    </style>
    
    <div style="
        width: 500px;
        height: 270px;
        background-color: #000;
        border: 3px solid #333;
        border-radius: 30px;
        margin: auto;
        position: relative;
        box-shadow: 0 0 10px #333;
        font-family: 'PT Sans', sans-serif;
        color: #f00;
    ">
      <!-- FARE -->
      <div style="
          position: absolute;
          top: 20px;
          left: 40px;
          font-size: 2rem;
      ">
        FARE: ${20.00:.2f}
      </div>

      <!-- EXTRAS -->
      <div style="
          position: absolute;
          top: 70px;
          left: 40px;
          font-size: 2rem;
      ">
        EXTRAS: ${4.00:.2f}
      </div>

      <!-- TOTAL -->
      <div style="
          position: absolute;
          top: 120px;
          left: 40px;
          font-size: 2rem;
      ">
        TOTAL: ${14.00:.2f}
      </div>

      <!-- TIP -->
      <div style="
          position: absolute;
          top: 170px;
          left: 40px;
          font-size: 2.2rem;
      ">
        TIP: ${tip_str}
      </div>
    </div>
    """

    st.markdown(meter_html, unsafe_allow_html=True)
