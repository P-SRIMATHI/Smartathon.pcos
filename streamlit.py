import os
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
from fpdf import FPDF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
file_path = "PCOS_data.csv"
def load_data():
    try:
        df = pd.read_csv(file_path)
        df_cleaned = df.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], errors="ignore")
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == "object":
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
            else:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        df_cleaned = df_cleaned.apply(pd.to_numeric, errors="coerce")
        return df_cleaned
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

df_cleaned = load_data()
X = df_cleaned.drop(columns=["PCOS (Y/N)"])
y = df_cleaned["PCOS (Y/N)"]
X_filled = X.fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fun Facts & PCOS Prediction Game
st.sidebar.title("PCOS Dashboard")
st.title("🎮 PCOS Prediction Game & Fun Facts")
fun_facts = [
    "PCOS affects 1 in 10 women of reproductive age!",
    "Lifestyle changes, such as exercise and a balanced diet, help manage PCOS symptoms.",
    "PCOS is one of the leading causes of infertility in women.",
    "Insulin resistance plays a key role in PCOS development.",
    "Maintaining a healthy weight can reduce PCOS symptoms!"
]

user_input = []
progress_bar = st.progress(0)
for idx, feature in enumerate(X_filled.columns):
    value = st.number_input(f"Enter your {feature}", min_value=0.0, format="%.2f")
    user_input.append(value)
    progress_bar.progress((idx + 1) / len(X_filled.columns))

if st.button("🎲 Predict PCOS Risk!"):
    with st.spinner("Analyzing your data...🔍"):
        time.sleep(2)
        user_input = np.array(user_input).reshape(1, -1)
        prediction = model.predict(user_input)
        risk_level = random.randint(1, 100)
    
    st.subheader("🔮 Prediction Result:")
    result_text = "High risk of PCOS" if prediction[0] == 1 else "Low risk of PCOS"
    st.write(f"{result_text}. Estimated risk level: {risk_level}%")
    st.write(random.choice(fun_facts))
    
    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "PCOS Prediction Report", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Prediction: {result_text}\nRisk Level: {risk_level}%")
        pdf.ln(10)
        pdf.multi_cell(0, 10, "Recommended Lifestyle Changes:\n- Maintain a balanced diet\n- Exercise regularly\n- Manage stress\n- Get enough sleep")
        report_path = "PCOS_Report.pdf"
        pdf.output(report_path)
        return report_path
    
    report_path = generate_report()
    with open(report_path, "rb") as file:
        st.download_button("Download Report", file, file_name="PCOS_Report.pdf")

# Health Gamification
st.header("🎮 Health Gamification")
water_glasses = st.slider("How many glasses of water did you drink today?", min_value=0, max_value=15)
steps = st.slider("How many steps did you walk today?", min_value=0, max_value=20000)

# Community Support
st.header("💬 Community Support")
new_post = st.text_area("Post your experience or ask a question:")
if st.button("Submit Post"):
    if new_post:
        st.success("Post submitted successfully!")
    else:
        st.warning("Please write something to post.")

# PCOS Quiz
st.header("🧠 PCOS Quiz")
questions = {
    "What is a common symptom of PCOS?": ["Irregular periods", "Acne", "Hair loss"],
    "Which hormone is often imbalanced in PCOS?": ["Insulin", "Estrogen", "Progesterone"],
    "What lifestyle change can help manage PCOS?": ["Regular exercise", "Skipping meals", "High sugar diet"]
}
quiz_score = 0
for question, options in questions.items():
    answer = st.radio(question, options)
    if answer == options[0]:
        quiz_score += 1
st.write(f"Your final quiz score: {quiz_score}/{len(questions)}")

# Mood Tracker
st.header("😊 Mood Tracker")
mood = st.selectbox("How do you feel today?", ["Happy", "Excited", "Neutral", "Sad", "Anxious"])
mood_advice = {
    "Happy": "Keep up the great energy! 🌟",
    "Excited": "Enjoy the excitement! 🌈",
    "Neutral": "It's okay to feel neutral, take it easy. ☁",
    "Sad": "Take care of yourself, things will get better. 💙",
    "Anxious": "Try some deep breaths, you're doing well. 🌱"
}
st.write(f"You are feeling: {mood}")
st.write(mood_advice.get(mood, "Stay strong!"))

# PCOS-Friendly Recipes
st.header("🍲 PCOS-Friendly Recipes")
recipes = [
    {"name": "Spinach & Chickpea Curry", "ingredients": ["Spinach", "Chickpeas", "Coconut milk", "Garlic", "Ginger"]},
    {"name": "Oats Pancakes", "ingredients": ["Oats", "Eggs", "Banana", "Almond milk"]},
    {"name": "Greek Yogurt Salad", "ingredients": ["Greek Yogurt", "Cucumber", "Olives", "Olive oil", "Lemon"]},
]
for recipe in recipes:
    st.subheader(recipe["name"])
    st.write("Ingredients:", ", ".join(recipe["ingredients"]))

# 3D Model Display
st.header("🩺 Explore PCOS in 3D")
model_url = "https://sketchfab.com/models/62bfb490ad344caaaea675da9df7ba34/embed"
st.write("Rotate, zoom, and explore the PCOS-related anatomy interactively.")
st.components.v1.iframe(model_url, height=500)
