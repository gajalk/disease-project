# -*- coding: utf-8 -*-
"""
Updated on Apr 21 with proper Three-Way Decision Making, Input Validation,
and inclusion of Diabetes Scaler for proper predictions.

@author: Deepk
"""

import os
import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load the saved models and scalers
model_path = 'C:/Users/kumar/OneDrive/Desktop/Multiple Disease Prediction System(1)/saved models(1)/'

diabetes_model = pickle.load(open(os.path.join(model_path, 'diabetes_model.pkl'), 'rb'))
diabetes_scaler = pickle.load(open(os.path.join(model_path, 'diabetes_scaler.pkl'), 'rb'))

heart_disease_model = pickle.load(open(os.path.join(model_path, 'heart_model.pkl'), 'rb'))
heart_scaler = pickle.load(open(os.path.join(model_path, 'heart_scaler.pkl'), 'rb'))

parkinsons_model = pickle.load(open(os.path.join(model_path, 'parkinsons_model.pkl'), 'rb'))
parkinsons_scaler = pickle.load(open(os.path.join(model_path, 'parkinsons_scaler.pkl'), 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            "Parkinson's Prediction"],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Helper function for consistent diagnosis messages
def get_diagnosis(prob, disease_name):
    if prob > 0.55:
        return f"🟢 Positive: The person has {disease_name} with a probability of {prob:.2f}"
    elif 0.45 <= prob <= 0.55:
        return f"🟡 Uncertain: Borderline case with a probability of {prob:.2f}, further tests recommended"
    else:
        return f"🔴 Negative: The person does not have {disease_name} with a probability of {1 - prob:.2f}"

# Input handler for all models
def handle_prediction(input_list, model, disease_name, scaler=None):
    try:
        input_list = [float(x) for x in input_list]
        if scaler:
            input_scaled = scaler.transform([input_list])
            prob = model.predict_proba(input_scaled)[0][1]
        else:
            prob = model.predict_proba([input_list])[0][1]

        st.write(f"Predicted probability: {prob:.4f}")
        diagnosis = get_diagnosis(prob, disease_name)
        if 'Positive' in diagnosis:
            st.success(diagnosis)
        elif 'Uncertain' in diagnosis:
            st.warning(diagnosis)
        else:
            st.error(diagnosis)

    except ValueError:
        st.error("⚠️ Please enter valid numeric values for all fields!")

# -------------------- Diabetes Prediction Page --------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        handle_prediction(user_input, diabetes_model, "diabetes", scaler=diabetes_scaler)

# -------------------- Heart Disease Prediction Page --------------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]
        handle_prediction(user_input, heart_disease_model, "heart disease", scaler=heart_scaler)

# -------------------- Parkinson's Prediction Page --------------------
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]
    inputs = [st.text_input(name) for name in feature_names]

    if st.button("Parkinson's Test Result"):
        handle_prediction(inputs, parkinsons_model, "Parkinson's disease", scaler=parkinsons_scaler)
