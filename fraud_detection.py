import streamlit as st
import pandas as pd
import joblib

# Load the trained fraud detection model
model = joblib.load("fraud_detection_pipeline.pkl")
 
# Set the title of the web app
st.title("Fraud Detection Prediction App")

# Description for the user
st.markdown("Please enter the transaction details and use the predict button")

# Divider line
st.divider()

# Input widgets to collect user input for transaction data
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])

amount = st.number_input("Amount", min_value=0.0, value=1000.0)

# Set example values: Old balance = 10000.0, New balance = 9000.0
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)

# Add destination balances
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Create a DataFrame from user input to feed into the model
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Get the prediction result from the model (0 = Not Fraud, 1 = Fraud)
    prediction = model.predict(input_data)[0]

    # Display the prediction result
    st.subheader(f"Prediction : '{int(prediction)}'")

    # Show appropriate message based on prediction
    if prediction == 1:
        st.error("This transaction can be fraud")
    else:
        st.success("This transaction looks like it is not a fraud")
