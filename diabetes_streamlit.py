import streamlit as st
import numpy as np
import joblib

# 1. Load the trained model parameters (w, b, means, devs)
@st.cache_resource
def load_diabetes_data():
    return joblib.load('diabetes_model.pkl')

data = load_diabetes_data()

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩸")

st.title("🩸 Zenitith AI: Diabetes Predictor")
st.write("This model uses **Multiple Linear Regression** with manual Gradient Descent and Mean Normalization.")

# 2. User Interface for Inputs
with st.form("diabetes_prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 85, 40)
        bmi = st.number_input("BMI", 15.0, 45.0, 24.5)
        bp = st.number_input("Blood Pressure", 50.0, 170.0, 120.0)

    with col2:
        glucose = st.number_input("Glucose Level", 70.0, 250.0, 120.0)
        exercise = st.slider("Exercise Time (min/week)", 0, 1260, 120)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)

    submit = st.form_submit_button("Predict Diabetes Score")

# 3. Prediction Logic
if submit:
    # Prepare the input array
    raw_inputs = np.array([age, bmi, bp, glucose, exercise, stress])
    
    # Feature Scaling: Must use the SAME means and devs from training!
    inputs_scaled = (raw_inputs - data['means']) / data['devs']
    
    # Model Inference: y = w . x + b
    prediction = np.dot(inputs_scaled, data['w']) + data['b']
    
    # 4. Results Display
    st.divider()
    st.metric(label="Predicted Diabetes Score", value=f"{prediction:.2f}")
    
    # Feedback based on your target limits
    if prediction > 450:
        st.warning("High score detected. Please consult a medical professional.")
    else:
        st.success("Your score is within the normal range.")
