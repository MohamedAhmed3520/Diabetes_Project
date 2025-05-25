import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title('Diabetes Prediction')

# Load your model and dataset
diab = pd.read_csv('diab.csv')
x = diab.drop('CLASS', axis=1)
y = diab['CLASS']



model = joblib.load('XGB_Grid.pkl')

# User chooses text option
gender_text = st.selectbox('Gender', ['F', 'M'])

# Convert text to 0 or 1 as in your dataset
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
    # Pass the numeric gender to model
    prediction = model.predict([[GENDER, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI]])
    if prediction == 0:
        st.write('The person is not diabetic')
    else:
        st.write('The person is diabetic')
