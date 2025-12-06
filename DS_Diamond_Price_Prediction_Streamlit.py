import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu

##Diamond Price Predict App
##Load the encoder and model
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

##Streamlit App
st.set_page_config(layout="wide")

##Define dictionary
cut_order = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
clarity_order = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
color_order = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
##Mapping
cut_map = {cat: idx for idx, cat in enumerate(cut_order)}
clarity_map = {cat: idx for idx, cat in enumerate(clarity_order)}
color_map = {cat: idx for idx, cat in enumerate(color_order)}

##Title 
st.markdown("<h1 style='text-align: center; color: purple'>💎Diamond Price Prediction and Market Segmentation💎</h1>", unsafe_allow_html=True)
st.write("Enter the features to predict the Price of the Diamond")

with st.sidebar:
    selected = option_menu(
        menu_title='Home', 
        options=['Price Prediction', 'Clustering']
    )
if selected == 'Price Prediction':

    ##Function to get user input
    carat = st.number_input("carat", min_value=0.20, max_value=5.00, value=0.54, step=0.01)
    depth = st.number_input("depth", min_value=50.00, max_value=80.00, value=60.08, step=0.1)
    table = st.number_input("table", min_value=50.0, max_value=100.0, value=57.0, step=0.1)
    x = st.number_input("Length (x)", min_value=3.0, max_value=10.0, value=5.75, step=0.01)
    y = st.number_input("Width (y)", min_value=3.0, max_value=10.0, value=5.76, step=0.01)
    z = st.number_input("Depth (z)", min_value=2.0, max_value=6.0, value=3.50, step=0.01)
    Volume = st.number_input("Volume", min_value=4.7, max_value=500.0, value=115.92, step=0.01)
    Price_per_Carat = st.number_input("Price per Carat", min_value=7.76, max_value=32.40, value=14.60, step=0.1)
    Dimension_ratio = st.number_input("Dimension ratio", min_value=2.40, max_value=53.20, value=20.14, step=0.1)

    cut = st.selectbox("cut", cut_order)
    clarity = st.selectbox("clarity", clarity_order)
    color = st.selectbox("color", color_order)

if st.button("Predict Price"):

    # Encode categorical features
    color_encoded = color_map[color]
    cut_encoded = cut_map[cut]
    clarity_encoded = clarity_map[clarity]

    # Log transform numerical features (if model was trained with log1p)
    carat_log = np.log1p(carat)

    # Create input array
    input_data = np.array([[carat_log, depth, table, x, y, z, Volume, Price_per_Carat, 
                            Dimension_ratio, color_encoded, cut_encoded, clarity_encoded]])

    # Predict
    price_log = rf_model.predict(input_data)[0]  # model predicts log(price)

    # Inverse log transform
    price = np.expm1(price_log)

    st.success(f"💎 Predicted Diamond Price: ${price:,.2f}")
    exchange_rate = 89.98
    Rupees = exchange_rate * price
    st.success(f"💎 Predicted Diamond Price in INR: Rs.{Rupees:,.2f}")



##Market Segmentation
with open('kmeans_final.pkl', 'rb') as h:
    kmeans_final = pickle.load(h)
with open('data.pkl', 'rb') as i:
    data = pickle.load(i)

if selected == 'Clustering':
    st.title('Market Segmentation')
    carat = st.number_input("carat", min_value=0.20, max_value=5.00, value=0.54, step=0.01)
    depth = st.number_input("depth", min_value=50.00, max_value=80.00, value=60.08, step=0.1)
    table = st.number_input("table", min_value=50.0, max_value=100.0, value=57.0, step=0.1)
    price = st.number_input("price", min_value=5.70, max_value=9.38, value=7.92, step=0.1)
    x = st.number_input("Length (x)", min_value=3.0, max_value=10.0, value=5.75, step=0.01)
    y = st.number_input("Width (y)", min_value=3.0, max_value=10.0, value=5.76, step=0.01)
    z = st.number_input("Depth (z)", min_value=2.0, max_value=6.0, value=3.50, step=0.01)
    Volume = st.number_input("Volume", min_value=4.7, max_value=500.0, value=115.92, step=0.01)
    Price_per_Carat = st.number_input("Price per Carat", min_value=7.76, max_value=32.40, value=14.60, step=0.1)
    Dimension_ratio = st.number_input("Dimension ratio", min_value=2.40, max_value=53.20, value=20.14, step=0.1)
    cut = st.number_input("cut_encoded", min_value=0.00, max_value=4.00, value=0.00, step=0.01)
    clarity = st.number_input("clarity_encoded", min_value=0.00, max_value=7.00, value=5.00, step=0.1)
    color = st.number_input("color_encoded", min_value=0.00, max_value=7.00, value=0.00, step=0.1)
    
    cluster_labels = {
    0: "Mid-range Balanced Diamonds",
    1: "Affordable Small Diamonds",
    2: "Premium Heavy Diamonds"}

    if st.button('Predict Cluster'):
        input_data = np.array([[carat, depth, table, price, x, y, z, Volume, Price_per_Carat, 
                            Dimension_ratio, cut, clarity, color]])

        cluster_assignment = kmeans_final.predict(input_data)[0]
        predicted_label = cluster_labels.get(cluster_assignment, 'Unkown Cluster')
        st.subheader("Prediction Result:")
        st.success(f"These diamonds belong to: **{predicted_label}**")

