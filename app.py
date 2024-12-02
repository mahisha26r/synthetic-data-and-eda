!pip install xgboost
import streamlit as st
import pickle
import numpy as np
from datetime import date

# Load the model and preprocessing objects
with open('model_gbm.pkl', 'rb') as f:
    model_gbm = pickle.load(f)

with open('scaler_gbm.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder_gbm.pkl', 'rb') as f:
    ohe = pickle.load(f)

# Define the app
st.title("Hourly Pay Rate Prediction")
st.write("Provide details to predict the hourly pay rate for nurses.")

# Input fields
job_title = st.text_input("Job Title", placeholder="E.g., Registered Nurse")
location = st.text_input("Location", placeholder="E.g., New York")
hospital_name = st.text_input("Hospital Name", placeholder="E.g., City General Hospital")

col1, col2 = st.columns(2)
with col1:
    cost_of_living_index = st.number_input("Cost of Living Index", min_value=0.0, step=0.1)
    schools_rating = st.number_input("Schools Rating", min_value=0.0, step=0.1)
with col2:
    crime_rate = st.number_input("Crime Rate", min_value=0.0, step=0.1)
    public_transport_score = st.number_input("Public Transport Score", min_value=0.0, step=0.1)

start_date = st.date_input("Contract Start Date", value=date.today())
end_date = st.date_input("Contract End Date", value=date.today())

if st.button("Predict"):
    # Validate date input
    if end_date <= start_date:
        st.error("Contract End Date must be after Contract Start Date.")
    else:
        # Process contract duration
        contract_duration = (end_date - start_date).days

        # Preprocess continuous features
        continuous_features = np.array([cost_of_living_index, schools_rating, crime_rate, public_transport_score, contract_duration]).reshape(1, -1)
        continuous_features_scaled = scaler.transform(continuous_features)

        # One-hot encode hospital name
        if hospital_name in ohe.categories_[0]:
            hospital_one_hot = ohe.transform([[hospital_name]])
        else:
            st.warning(f"Hospital name '{hospital_name}' not recognized. Using default encoding.")
            hospital_one_hot = np.zeros((1, len(ohe.categories_[0])))

        # Combine all features
        input_features = np.hstack([continuous_features_scaled, hospital_one_hot])

        # Predict hourly rate
        predicted_rate = model_gbm.predict(input_features)[0]

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Job Title:** {job_title}")
        st.write(f"**Location:** {location}")
        st.write(f"**Hospital Name:** {hospital_name}")
        st.write(f"**Contract Start Date:** {start_date}")
        st.write(f"**Contract End Date:** {end_date}")
        st.success(f"**Predicted Hourly Rate:** ${predicted_rate:.2f}")
