import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5', compile=False)
st.write("Model loaded successfully")


# Load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = float(st.number_input('Credit score'))
estimated_salary = st.number_input('Estimated salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox('Has Credit card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography': [geography]  # Add Geography for later one-hot encoding
})

# One-hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate one-hot encoded geography with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Drop the 'Geography' column after encoding
input_data.drop(columns=['Geography'], inplace=True)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Display churn probability
st.write(f'Churn probability: {prediction_proba:.2f}')

# Display result based on the prediction
if prediction_proba > 0.5:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')
