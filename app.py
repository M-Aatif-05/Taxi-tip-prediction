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


# HTML + CSS for the custom taxi meter design
    meter_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=PT+Sans:wght@700&display=swap');
    
    .meter-container {{
        width: 600px;
        background-color: #000;
        border-radius: 20px;
        padding: 20px;
        margin: auto;
        box-shadow: 0 0 10px #333;
        color: white;
        font-family: 'PT Sans', sans-serif;
        position: relative;
    }}
    
    /* Top fake info row */
    .top-info {{
        display: flex;
        justify-content: space-around;
        font-size: 0.8rem;
        color: #aaa;
        margin-bottom: 10px;
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
        font-weight: bold;
    }}
    
    
    <div class="meter-container">
        <!-- Top Info Row -->
        <div class="top-info">
            <div>Dubai</div>
            <div>29°C</div>
            <div>15:30 PM</div>
            <div>2025-03-21</div>
        </div>
        
        <!-- Main Header -->
        <div class="meter-header">TAXI METER</div>
        
        <!-- Body: Left (ride details) & Right (tip prediction with emojis) -->
        <div class="meter-body">
            <div class="meter-left">
                Distance: {trip_distance:.2f} miles<br>
                Time: {trip_duration_minutes:.2f} min<br>
                Extras: ${tolls_amount:.2f}<br>
                Total: ${20:.2f}  <!-- Example total calculation -->
            </div>
            <div class="meter-right">
                Tip {tip_str}?<br>
                <div style="margin-top: 10px; font-size: 1.5rem;">
                    {tip_str}
                </div>
            </div>
        </div>
    </div>
    """
    
    st.markdown(meter_html, unsafe_allow_html=True)
