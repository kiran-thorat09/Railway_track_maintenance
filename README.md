
# Railway Track Maintenance Predictor

This project is a Streamlit web application that predicts the risk of failure for railway track segments using a machine learning model trained on track condition features such as temperature, vibration levels, geometry deviations, and more.

##  Project Overview

Railway infrastructure requires timely and efficient maintenance to prevent accidents and ensure safety. This application leverages a trained ML model to:
- Predict the probability of failure for a given track segment
- Suggest maintenance urgency
- Help railway engineers make data-driven maintenance decisions

##  Project Workflow

### 1. Data Acquisition & Cleaning

- Collected raw data from `.csv` files
- Performed exploratory data analysis (EDA)
- Cleaned and preprocessed data for model training
- Dropped irrelevant or highly correlated columns (e.g., `install_year`)
- Converted `last_maintenance` date to number of months since last service
- Rounded numerical values for consistency
- Saved cleaned dataset as `train_data_clean.xls`

### 2.  Feature Engineering

- Extracted `last_maintenance_months` from datetime column
- Removed original `last_maintenance` column
- Prepared target variable: `failure`
- Final features included temperature, vibration, geometry deviation, traffic.

### 3.  Model Training

- Used `RandomForestClassifier` from scikit-learn
- Performed an 80/20 train-test split
- Evaluated model using accuracy and classification report
- Final model achieved strong performance on test data

### 4.  Model Saving

- Serialized the trained model using `joblib`
- Saved it as `railway_track_model.pkl` for later use in the web app

---

##  Web App: Streamlit Deployment

A simple Streamlit frontend allows users to:

- Input track segment conditions manually
- View failure risk score and recommended action
- Deploy the app using [Render](https://render.com)

###  App Features

- Real-time predictions
- User-friendly sliders and input fields
- Maintenance recommendations based on risk score

##  File Structure

railway-predictor/

├── app.py #Streamlit frontend

├── railway_track_model.pkl #Trained ML model

├── train_model #train_model

├── requirements.txt #Required packages

├── train_data_clean.xls #Cleaned data used for training

└── README.md #Project documentation


###  Install Req.txt 
pip install -r Req.txt

###  Run App
streamlit run app.py

## You can simply visit website where i have deployed the app

https://railway-predictor.onrender.com

###  Note
You can simply give data like :

Segment ID : 2098
Traffic Density (trains/day) : 100
Track Geometry Deviation : 0.4
Vibration Level : 0.2
Temperature (°C) : 15
Months Since Last Maintenance : 8

You will see :
Segment ID: 2098
Failure Probability : 45.00%
"Low Risk! Maintenance Required in some time for Segment 2098 (Risk: 0.45)"
