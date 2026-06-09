
import streamlit as st
import pandas as pd
import joblib

# Load the complete pipeline model
model = joblib.load("model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Shipment Delivery Prediction",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Shipment Delivery Prediction")

st.write(
    "Enter shipment details below to predict whether the shipment will reach on time."
)

# User Inputs

id_value = st.number_input(
    "Shipment ID",
    min_value=1,
    value=1
)

warehouse_block = st.selectbox(
    "Warehouse Block",
    ["A", "B", "C", "D", "F"]
)

mode_of_shipment = st.selectbox(
    "Mode of Shipment",
    ["Flight", "Road", "Ship"]
)

customer_care_calls = st.number_input(
    "Customer Care Calls",
    min_value=0,
    value=2
)

customer_rating = st.slider(
    "Customer Rating",
    min_value=1,
    max_value=5,
    value=3
)

cost_of_the_product = st.number_input(
    "Cost of Product",
    min_value=0,
    value=200
)

prior_purchases = st.number_input(
    "Prior Purchases",
    min_value=0,
    value=3
)

product_importance = st.selectbox(
    "Product Importance",
    ["low", "medium", "high"]
)

gender = st.selectbox(
    "Gender",
    ["M", "F"]
)

discount_offered = st.number_input(
    "Discount Offered",
    min_value=0,
    value=10
)

weight_in_gms = st.number_input(
    "Weight in Grams",
    min_value=0,
    value=3000
)

# Prediction

if st.button("Predict"):

    input_df = pd.DataFrame({
        "ID": [id_value],
        "Warehouse_block": [warehouse_block],
        "Mode_of_Shipment": [mode_of_shipment],
        "Customer_care_calls": [customer_care_calls],
        "Customer_rating": [customer_rating],
        "Cost_of_the_Product": [cost_of_the_product],
        "Prior_purchases": [prior_purchases],
        "Product_importance": [product_importance],
        "Gender": [gender],
        "Discount_offered": [discount_offered],
        "Weight_in_gms": [weight_in_gms]
    })

    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.success("✅ Shipment is likely to reach on time.")
    else:
        st.error("⚠️ Shipment is likely to be delayed.")


