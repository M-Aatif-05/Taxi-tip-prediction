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

    # HTML + CSS for the "digital" meter look
    # Uses a black background, red text, and a rounded border
    meter_html = f"""
    <style>
    /* Optional: a retro/digital-like font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    </style>

    <div style="
        width: 400px;
        height: 200px;
        background-color: #000;
        border: 4px solid #333;
        border-radius: 50px;
        margin: auto;
        position: relative;
        box-shadow: 0 0 15px #333;
    ">
      <!-- "HIRED" label (or you can replace with "FARE") -->
      <div style="
          position: absolute;
          top: 20px;
          left: 40px;
          color: red;
          font-family: 'VT323', monospace; /* fallback: monospace */
          font-size: 2.5rem;
      ">
        HIRED
      </div>

      <!-- TIP label & predicted tip -->
      <div style="
          position: absolute;
          top: 80px;
          left: 40px;
          color: red;
          font-family: 'VT323', monospace;
          font-size: 2rem;
      ">
        TIP:<br/>{tip_str}
      </div>
    </div>
    """

    # Render the custom meter in Streamlit
    st.markdown(meter_html, unsafe_allow_html=True)
