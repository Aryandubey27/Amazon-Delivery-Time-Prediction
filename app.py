import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Amazon Delivery Time Predictor",
    page_icon="📦",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

try:
    model, preprocessor = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    model_loaded = False

# Title and description
st.title("📦 Amazon Delivery Time Prediction System")
st.markdown("### Predict delivery times based on order and agent details")

# Sidebar for input
st.sidebar.header("📝 Input Features")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Order Details")
    distance_km = st.number_input("Distance (km)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    order_hour = st.slider("Order Hour (24h format)", 0, 23, 12)
    order_dayofweek = st.selectbox("Day of Week", 
                                     options=[0, 1, 2, 3, 4, 5, 6],
                                     format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
    pickup_delay_mins = st.number_input("Pickup Delay (minutes)", min_value=0.0, max_value=120.0, value=5.0, step=1.0)

with col2:
    st.subheader("👤 Agent Details")
    agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
    agent_rating = st.slider("Agent Rating", 1.0, 5.0, 4.0, 0.1)
    
    st.subheader("🌍 Context")
    weather = st.selectbox("Weather Conditions", ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Fog', 'Windy'])
    traffic = st.selectbox("Traffic Conditions", ['Low', 'Medium', 'High', 'Jam'])
    vehicle = st.selectbox("Vehicle Type", ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
    area = st.selectbox("Area Type", ['Urban', 'Semi-Urban', 'Metropolitian'])
    category = st.selectbox("Order Category", ['Snack', 'Meal', 'Drinks', 'Buffet'])

# Predict button
if st.button("🔮 Predict Delivery Time", type="primary"):
    if model_loaded:
        # Create input dataframe
        input_data = pd.DataFrame({
            'distance_km': [distance_km],
            'Agent_Age': [agent_age],
            'Agent_Rating': [agent_rating],
            'pickup_delay_mins': [pickup_delay_mins],
            'order_hour': [order_hour],
            'order_dayofweek': [order_dayofweek],
            'Weather': [weather],
            'Traffic': [traffic],
            'Vehicle': [vehicle],
            'Area': [area],
            'Category': [category]
        })
        
        try:
            # Transform and predict
            input_transformed = preprocessor.transform(input_data)
            prediction = model.predict(input_transformed)[0]
            
            # Display prediction
            st.success("### 🎯 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Delivery Time", f"{prediction:.1f} mins")
            
            with col2:
                estimated_time = datetime.now() + timedelta(minutes=prediction)
                st.metric("Estimated Arrival", estimated_time.strftime("%H:%M"))
            
            with col3:
                if prediction < 30:
                    st.metric("Status", "🟢 Fast", delta="Good")
                elif prediction < 45:
                    st.metric("Status", "🟡 Normal", delta="Average")
                else:
                    st.metric("Status", "🔴 Slow", delta="High")
            
            # Visualization
            st.markdown("---")
            st.subheader("📊 Prediction Analysis")
            
            # Feature contribution (approximate)
            feature_values = {
                'Distance': distance_km * 2,
                'Traffic': {'Low': 5, 'Medium': 10, 'High': 15, 'Jam': 20}[traffic],
                'Pickup Delay': pickup_delay_mins * 0.8,
                'Weather': {'Sunny': 2, 'Cloudy': 3, 'Fog': 5, 'Rainy': 7, 'Stormy': 10, 'Windy': 4}[weather],
                'Agent Rating': (5 - agent_rating) * 3
            }
            
            fig = go.Figure(data=[
                go.Bar(x=list(feature_values.keys()), 
                      y=list(feature_values.values()),
                      marker_color='lightblue')
            ])
            fig.update_layout(
                title="Approximate Time Contribution by Factor",
                xaxis_title="Factor",
                yaxis_title="Time Impact (mins)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model not loaded. Please ensure model files are present.")

# Footer
st.markdown("---")
st.markdown("### 📌 About")
st.info("""
This system predicts Amazon delivery times using machine learning. 
The model considers various factors including distance, traffic, weather, 
agent details, and order characteristics to provide accurate delivery time estimates.
""")

# Display feature importance if available
try:
    feature_imp = pd.read_csv('feature_importance.csv')
    with st.expander("📈 View Top Features Importance"):
        fig = px.bar(feature_imp.head(10), 
                     x='Importance', 
                     y='Feature', 
                     orientation='h',
                     title='Top 10 Most Important Features')
        st.plotly_chart(fig, use_container_width=True)
except:
    pass