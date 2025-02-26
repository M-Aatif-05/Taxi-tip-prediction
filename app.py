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


# HTML + CSS for a custom taxi meter display
meter_html = f"""
<style>
  .meter-container {{
      width: 600px;
      background-color: #000;
      border-radius: 20px;
      padding: 20px;
      color: white;
      font-family: 'PT Sans', sans-serif;
      margin: auto;
  }}
  .meter-header {{
      text-align: center;
      font-size: 2.5rem;
      font-weight: bold;
      margin-bottom: 20px;
      letter-spacing: 2px;
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
      font-weight: bold;
  }}
  .meter-emojis {{
      font-size: 1.8rem;
      margin-top: 10px;
  }}
</style>

<div class="meter-container">
    <div class="meter-header">TAXI METER</div>
    <div class="meter-body">
        <div class="meter-left">
            Distance: {trip_distance:.2f} miles<br>
            Time: {trip_duration_minutes} min<br>
            Extras: ${tolls_amount:.2f}<br>
            Total: ${20.00:.2f}
        </div>
        <div class="meter-right">
            <div class="meter-emojis">
                😊 😐 😞
            Tip {tip_str: .2f}?:<br>
            </div>
        </div>
    </div>
</div>
"""

st.markdown(meter_html, unsafe_allow_html=True)
