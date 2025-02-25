import streamlit as st
import joblib
import numpy as np

# Load the trained model from GitHub repo
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("lgbm.pkl")  # Model must be in the same repo

model = load_model()

# Streamlit App UI
st.title("🚕 LGBM Tip Prediction App")

# User Inputs
passenger_count = st.number_input("Passenger Count")
trip_distance = st.number_input("Trip Distance (miles)")
fare_amount = st.number_input("Fare Amount ($)")
day_night = st.selectbox("Day or Night", [0, 1])  # 0 = Day, 1 = Night
trip_duration = st.number_input("Trip Duration (minutes)")

# Prediction Button
if st.button("Predict Tip 💰"):
    input_features = np.array([[passenger_count, trip_distance, fare_amount, day_night, trip_duration]])
    prediction = model.predict(input_features)[0]
    
    st.success(f"💵 Predicted Tip Amount: ${prediction:.2f}")
