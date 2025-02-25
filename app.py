import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ✅ Load the LGBM model
@st.cache_resource
def load_model():
    model_path = "lgbm.pkl"  # Make sure this is the correct path
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("💰 Taxi Tip Prediction App")

# ✅ User Inputs
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
trip_distance = st.number_input("Trip Distance (miles/km)", min_value=0.0, value=1.0)
fare_amount = st.number_input("Fare Amount ($)", min_value=0.0, value=5.0)
day_night = st.selectbox("Time of Day", ["Day (0)", "Night (1)"])
trip_duration_minutes = st.number_input("Trip Duration (minutes)", min_value=0.0, value=10.0)

# ✅ Convert 'Day/Night' into integer
day_night = 1 if "Night" in day_night else 0

# ✅ Prediction Button
if st.button("Predict Tip 💰"):
    try:
        # ✅ Convert input into a NumPy array (ensure it's 2D)
        input_features = np.array([[passenger_count, trip_distance, fare_amount, day_night, trip_duration_minutes]])

        # ✅ Convert to DataFrame (LGBM sometimes needs it)
        input_df = pd.DataFrame(input_features, columns=['passenger_count', 'trip_distance', 'fare_amount', 'day_night', 'trip_duration_minutes'])

        # ✅ Make Prediction
        prediction = model.predict(input_df)[0]

        # ✅ Display Result
        st.success(f"💵 Predicted Tip Amount: ${prediction:.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

