import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

st.title('Diabetes Prediction')

# Load dataset
diab = pd.read_csv('diab.csv')

# Prepare features and target
X = diab.drop('CLASS', axis=1)
y = diab['CLASS']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split training into train and val
x_tr, x_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Feature scaling (fit on train, transform train and val)
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)
x_val = scaler.transform(x_val)

# Load model
model = joblib.load('XGB_Grid.pkl')

# User inputs
gender_text = st.selectbox('Gender', ['F', 'M'])
GENDER = 0 if gender_text == 'F' else 1

AGE = st.number_input('AGE')
Urea = st.number_input('Urea')
Cr = st.number_input('Cr')
HbA1c = st.number_input('HbA1c')
Chol = st.number_input('Chol')
TG = st.number_input('TG')
HDL = st.number_input('HDL')
LDL = st.number_input('LDL')
VLDL = st.number_input('VLDL')
BMI = st.number_input('BMI')

if st.button('Predict'):
    # Prepare input features in correct order
    input_features = np.array([[GENDER, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI]])
    
    # Scale input features using the scaler fitted on training data
    input_scaled = scaler.transform(input_features)
    
    # Predict using scaled features
    prediction = model.predict(input_scaled)
    
    if prediction == 0:
        st.write('The person is not diabetic')
    else:
        st.write('The person is diabetic')
