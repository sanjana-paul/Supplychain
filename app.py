
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('best_classification_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_clf.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Re-initialize LabelEncoder to get the classes. It's important that the order of classes is the same.
# Based on the previous output, the original y_clf had 'Moderate Risk', 'High Risk', 'Low Risk'.
# The encoded values were 0, 1, 2 for High, Low, Moderate based on default alphabetical sorting.
# Let's ensure the encoder maps these back correctly.
label_encoder = LabelEncoder()
# Fit with the full set of possible classes in alphabetical order to maintain consistency
label_encoder.fit(['High Risk', 'Low Risk', 'Moderate Risk'])

# Streamlit App Title
st.title('Supply Chain Risk Classification Predictor')

st.write('Enter the feature values to predict the risk classification:')

# Input fields for all 23 features from the 'x' DataFrame
# The order of features must match the training data
input_features = {}
input_features['vehicle_gps_latitude'] = st.number_input('Vehicle GPS Latitude', value=40.0)
input_features['vehicle_gps_longitude'] = st.number_input('Vehicle GPS Longitude', value=-75.0)
input_features['fuel_consumption_rate'] = st.number_input('Fuel Consumption Rate', value=7.5)
input_features['eta_variation_hours'] = st.number_input('ETA Variation Hours', value=2.5)
input_features['traffic_congestion_level'] = st.number_input('Traffic Congestion Level', value=5.0)
input_features['warehouse_inventory_level'] = st.number_input('Warehouse Inventory Level', value=500.0)
input_features['loading_unloading_time'] = st.number_input('Loading/Unloading Time', value=2.0)
input_features['handling_equipment_availability'] = st.number_input('Handling Equipment Availability', value=0.5)
input_features['order_fulfillment_status'] = st.number_input('Order Fulfillment Status', value=0.7)
input_features['weather_condition_severity'] = st.number_input('Weather Condition Severity', value=0.6)
input_features['port_congestion_level'] = st.number_input('Port Congestion Level', value=0.4)
input_features['shipping_costs'] = st.number_input('Shipping Costs', value=100.0)
input_features['supplier_reliability_score'] = st.number_input('Supplier Reliability Score', value=0.8)
input_features['lead_time_days'] = st.number_input('Lead Time Days', value=7.0)
input_features['historical_demand'] = st.number_input('Historical Demand', value=150.0)
input_features['iot_temperature'] = st.number_input('IoT Temperature', value=20.0)
input_features['cargo_condition_status'] = st.number_input('Cargo Condition Status', value=0.9)
input_features['route_risk_level'] = st.number_input('Route Risk Level', value=6.0)
input_features['customs_clearance_time'] = st.number_input('Customs Clearance Time', value=1.5)
input_features['driver_behavior_score'] = st.number_input('Driver Behavior Score', value=0.75)
input_features['fatigue_monitoring_score'] = st.number_input('Fatigue Monitoring Score', value=0.6)
input_features['disruption_likelihood_score'] = st.number_input('Disruption Likelihood Score', value=0.85)
input_features['delay_probability'] = st.number_input('Delay Probability', value=0.7)


if st.button('Predict Risk Classification'):
    # Create a DataFrame from the input features
    input_df = pd.DataFrame([input_features])

    # Scale the input features
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction_encoded = model.predict(scaled_input)

    # Decode the prediction back to original class name
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    st.success(f'The predicted Risk Classification is: **{prediction_label[0]}**')
