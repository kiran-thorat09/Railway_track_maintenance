import streamlit as st
import joblib
import pandas as pd

#Loading model
model = joblib.load("railway_track_model.pkl")

#Get required feature names in correct order
required_features = list(model.feature_names_in_)

st.set_page_config(page_title="Track Maintenance Predictor", layout="centered")
st.title("Railway Track Maintenance Predictor")

segment_id = st.text_input("Segment ID", placeholder="e.g., SEG-102")

#Collect inputs
input_dict = {}

for feature in required_features:
    if "temperature" in feature:
        input_dict[feature] = st.slider("Temperature (°C)", -10.0, 40.0, step=0.2)
    elif "vibration" in feature:
        input_dict[feature] = st.slider("Vibration Level", 0.0, 2.0, step=0.05)
    elif "geometry" in feature:
        input_dict[feature] = st.slider("Track Geometry Deviation", 0.0, 4.0, step=0.05)
    elif "last_maintenance" in feature:
        input_dict[feature] = st.number_input("Months Since Last Maintenance", 0, 120)
    elif "traffic" in feature:
        input_dict[feature] = st.slider("Traffic Density (trains/day)", 0, 400)
    else:
        input_dict[feature] = st.number_input(f"Enter value for {feature}", step=1.0)

input_data = pd.DataFrame([[input_dict[feat] for feat in required_features]], columns=required_features)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # risk of failure (class 1)

    st.markdown(f"#### Segment ID: `{segment_id}`")
    st.metric(label="Failure Probability", value=f"{probability:.2%}")

    if probability >= 0.7:
        st.error(f" High Risk! Maintenance Required immediatly for **Segment {segment_id}** (Risk: {probability:.2f})")
    elif probability >= 0.5 and probability < 0.7:
        st.error(f" Medium Risk! Maintenance Required for **Segment {segment_id}** (Risk: {probability:.2f})")
    elif probability >= 0.3 and probability < 0.5:
        st.error(f" Low Risk! Maintenance Required in some time for **Segment {segment_id}** (Risk: {probability:.2f})")
    else:
        st.success(f" No Maintenance Needed Segment {segment_id} is Safe (Risk Score: {probability:.2f})")