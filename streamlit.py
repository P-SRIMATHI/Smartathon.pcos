import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import shap
import openai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from fpdf import FPDF
import streamlit.components.v1 as components
import time
import random

# Initialize session state variables
if "score" not in st.session_state:
    st.session_state.score = 0
if "posts" not in st.session_state:
    st.session_state.posts = []
if "health_points" not in st.session_state:
    st.session_state.health_points = 0
if "water_intake" not in st.session_state:
    st.session_state.water_intake = 0
if "steps_walked" not in st.session_state:
    st.session_state.steps_walked = 0

# Function to calculate BMI
def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

# Function to generate PDF report
def generate_report(prediction_prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "PCOS Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, "High probability of PCOS detected." if prediction_prob > 0.5 else "No significant risk of PCOS detected.")
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Lifestyle Changes:\n- Maintain a balanced diet\n- Exercise regularly\n- Manage stress\n- Get enough sleep")
    report_path = "PCOS_Report.pdf"
    pdf.output(report_path)
    return report_path

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("PCOS_data.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_")  
    return df

df = load_data()

# Feature selection
possible_features = ["AMH", "betaHCG", "FSH"]
selected_features = [col for col in df.columns if any(feature in col for feature in possible_features)]
if not selected_features:
    st.error("None of the selected features are found in the dataset!")
    st.stop()

df = df.dropna()
X = df[selected_features]
y = df[df.columns[df.columns.str.contains("PCOS")][0]]

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("PCOS Prediction Dashboard")

# PCOS Prediction Section
st.header("1. PCOS Prediction 🩺")
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
bmi = calculate_bmi(weight, height)
st.write(f"Calculated BMI: {bmi:.2f}")

user_input = {col: st.number_input(f"{col}", value=float(pd.to_numeric(X.iloc[:, i], errors="coerce").mean(skipna=True) or 0)) for i, col in enumerate(selected_features)}

if st.button("Submit Prediction"):
    input_df = pd.DataFrame([user_input])
    prediction_proba = model.predict_proba(input_df)
    prediction_prob = prediction_proba[0][1] if prediction_proba.shape[1] > 1 else prediction_proba[0]
    prediction = "PCOS Detected" if prediction_prob > 0.5 else "No PCOS Detected"
    st.success(prediction)
    report_path = generate_report(prediction_prob)
    with open(report_path, "rb") as file:
        st.download_button("Download Report", file, file_name="PCOS_Report.pdf")

# AI Alerts
st.header("2. AI Alerts")
if prediction_prob > 0.8:
    st.warning("High risk of PCOS detected. Consider consulting a healthcare professional.")
elif prediction_prob > 0.5:
    st.info("Moderate risk of PCOS detected. Lifestyle changes are recommended.")

# Health Gamification
st.header("3. Health Gamification 🎮")
water_glasses = st.slider("Glasses of water today", 0, 15)
st.session_state.water_intake = water_glasses
if water_glasses >= 8:
    st.session_state.health_points += 10
    st.success("Great job! +10 points")

steps = st.slider("Steps walked today", 0, 20000)
st.session_state.steps_walked = steps
if steps >= 10000:
    st.session_state.health_points += 20
    st.success("Awesome! +20 points")

st.write(f"Total Health Points: {st.session_state.health_points}")
if st.session_state.health_points > 40:
    st.balloons()

# Community Support
st.header("4. Community Support 💬")
new_post = st.text_area("Post your experience or ask a question:")
if st.button("Submit Post"):
    if new_post:
        st.session_state.posts.append(new_post)
        st.success("Post submitted successfully!")

if st.session_state.posts:
    st.write("### Community Posts:")
    for idx, post in enumerate(st.session_state.posts, 1):
        st.write(f"{idx}. {post}")

# Trivia Quiz
st.header("5. Trivia Quiz 🧠")
questions = {
    "What is a common symptom of PCOS?": ["Irregular periods", "Acne", "Hair loss"],
    "Which hormone is often imbalanced in PCOS?": ["Insulin", "Estrogen", "Progesterone"],
    "What lifestyle change helps manage PCOS?": ["Regular exercise", "Skipping meals", "High sugar diet"]
}
quiz_score = sum([1 for q, opts in questions.items() if st.radio(q, opts) == opts[0]])
st.write(f"Your quiz score: {quiz_score}/{len(questions)}")

# Mood Tracker
st.header("6. Mood Tracker 😊")
mood = st.selectbox("How do you feel today?", ["Happy", "Excited", "Neutral", "Sad", "Anxious"])
st.write(f"You are feeling: {mood}")

# 3D Model Display
st.header("7. PCOS 3D Model 🩺")
components.iframe("https://sketchfab.com/models/62bfb490ad344caaaea675da9df7ba34/embed", height=500)
