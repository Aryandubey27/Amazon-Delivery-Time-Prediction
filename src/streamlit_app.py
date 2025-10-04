import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

MODEL_PATH = "/mnt/data/project_outputs_safe/best_pipeline.joblib"  # default; override if needed

@st.cache_data
def load_model(path=MODEL_PATH):
    return joblib.load(path)

model = load_model()

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="centered")
st.title("Amazon Delivery Time Prediction")
st.write("Estimate delivery time (hours) from order & pickup features.")

with st.expander("Model info"):
    st.write("Model loaded from:", MODEL_PATH)
    try:
        est = model.named_steps.get('model', None)
        if est is None:
            est = model
        if hasattr(est, 'feature_importances_'):
            fi = est.feature_importances_
            st.write("Model exposes feature_importances_. (See console for details)")
    except Exception as e:
        st.write("Could not retrieve feature importances:", e)

st.header("Input features")
col1, col2 = st.columns(2)
with col1:
    Agent_Age = st.number_input("Agent Age", min_value=18, max_value=80, value=30)
    Agent_Rating = st.slider("Agent Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
    distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.0, step=0.1)
with col2:
    pickup_delay_min = st.number_input("Pickup delay (minutes)", min_value=0.0, value=5.0, step=1.0)
    order_hour = st.slider("Order hour (0-23)", 0, 23, 12)
    order_dayofweek = st.slider("Order day of week (0=Mon,6=Sun)", 0, 6, 2)

# categorical defaults (adjust as per your dataset categories)
Weather = st.selectbox("Weather", ["Sunny","Rainy","Cloudy","Storm"])
Traffic = st.selectbox("Traffic", ["Low","Medium","High"])
Vehicle = st.selectbox("Vehicle", ["Bike","Motorbike","Car","Van"])
Area = st.selectbox("Area", ["Urban","Metropolitan","Rural"])
Category = st.selectbox("Category", ["Grocery","Electronics","Clothing","Other"])

input_df = pd.DataFrame([{
    "Agent_Age": Agent_Age,
    "Agent_Rating": Agent_Rating,
    "distance_km": distance_km,
    "pickup_delay_min": pickup_delay_min,
    "order_hour": order_hour,
    "order_dayofweek": order_dayofweek,
    "Weather": Weather,
    "Traffic": Traffic,
    "Vehicle": Vehicle,
    "Area": Area,
    "Category": Category
}])

st.subheader("Input preview")
st.table(input_df.T)

if st.button("Predict delivery time"):
    pred = model.predict(input_df)[0]
    st.success(f"Estimated delivery time: {pred:.2f} hours")
    st.write("Note: model trained on historical dataset; please validate results against business SLAs.")
