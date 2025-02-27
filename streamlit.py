import numpy as np
import pandas as pd
import streamlit as st
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit App Title
st.set_page_config(page_title="PCOS Prediction Game", layout="wide")
st.title("🎮 PCOS Prediction Game")
st.markdown("Engage in an interactive way to assess your risk level for PCOS!")

# ================== Data Preprocessing ==================
@st.cache_data
def load_and_preprocess_data():
    file_path = "PCOS_data.csv"
    df = pd.read_csv(file_path)
    
    df_cleaned = df.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], errors="ignore")
    
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object":
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        else:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors="coerce")

    if "PCOS (Y/N)" not in df_cleaned.columns:
        st.error("Error: Target column 'PCOS (Y/N)' not found in dataset.")
        st.stop()
    
    X = df_cleaned.drop(columns=["PCOS (Y/N)"])
    y = df_cleaned["PCOS (Y/N)"]
    
    return X, y

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# ================== Personality Quiz Approach ==================
def personality_quiz():
    st.subheader("📝 Answer These Fun Questions!")
    risk_score = 0
    
    fatigue = st.radio("How often do you feel fatigued?", ["🔋 Rarely", "😴 Sometimes", "🛌 Often", "🛑 Always"])
    if fatigue == "🛑 Always": risk_score += 3
    elif fatigue == "🛌 Often": risk_score += 2
    elif fatigue == "😴 Sometimes": risk_score += 1
    
    diet = st.radio("How would you describe your diet?", ["🍏 Healthy", "🍔 Fast food sometimes", "🍕 Mostly unhealthy", "🥤 Poor diet"])
    if diet == "🥤 Poor diet": risk_score += 3
    elif diet == "🍕 Mostly unhealthy": risk_score += 2
    elif diet == "🍔 Fast food sometimes": risk_score += 1
    
    exercise = st.radio("How often do you exercise?", ["🏃‍♀️ Regularly", "🚶 Occasionally", "🛋️ Rarely", "❌ Never"])
    if exercise == "❌ Never": risk_score += 3
    elif exercise == "🛋️ Rarely": risk_score += 2
    elif exercise == "🚶 Occasionally": risk_score += 1
    
    return risk_score

# ================== Risk Reveal with Spin Wheel ==================
def risk_spin_wheel(risk_score):
    st.subheader("🎡 Your Risk Level!")
    
    risk_percentage = min(risk_score * 10, 100)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        title={"text": "PCOS Risk Level"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "red" if risk_percentage > 50 else "green"}}
    ))
    st.plotly_chart(fig)
    
    if risk_percentage < 30:
        st.balloons()
    elif risk_percentage > 70:
        st.warning("🚨 High Risk! Consider consulting a doctor.")
    
    return risk_percentage

# ================== Achievements & Badges ==================
def show_badges(risk_percentage):
    st.subheader("🏆 Your Achievement")
    
    if risk_percentage < 30:
        st.success("🥇 **Hormone Hero!** You’re doing great! Keep up the healthy lifestyle! 💪")
        st.audio("success_sound.mp3")
    elif risk_percentage < 60:
        st.warning("⚠️ **Balance Seeker!** You're on the right path but can make improvements!")
    else:
        st.error("🚨 **PCOS Warrior!** You might be at high risk. Consider consulting a doctor.")
        st.audio("alert_sound.mp3")

# ================== Lifestyle Recommendation System ==================
def lifestyle_recommendations(risk_percentage):
    st.subheader("💡 Personalized Lifestyle Recommendations")
    
    if risk_percentage < 30:
        st.write("✅ Maintain a balanced diet with whole foods and regular physical activity!")
    elif risk_percentage < 60:
        st.write("⚠️ Try increasing your exercise and reducing processed foods!")
    else:
        st.write("🚨 Prioritize health checkups and stress management techniques like yoga!")

# ================== Mini Health Challenges ==================
def health_challenges():
    st.subheader("🎯 Mini Health Challenge")
    challenge_list = [
        "🏃‍♀️ Do 10 jumping jacks now!",
        "🥗 Eat a fruit or veggie today!",
        "🧘 Take 3 deep breaths and relax!",
        "💧 Drink a glass of water now!"
    ]
    challenge = random.choice(challenge_list)
    st.write(f"**Your challenge:** {challenge}")
    if st.button("Complete Challenge ✅"):
        st.success("Great job! Keep up the healthy habits! 💪")

# ================== Main Execution ==================
X_data, y_data = load_and_preprocess_data()
pcos_model, model_acc = train_model(X_data, y_data)
st.sidebar.write(f"✅ **Model Accuracy:** {model_acc * 100:.2f}%")

risk_score = personality_quiz()
risk_percentage = risk_spin_wheel(risk_score)
show_badges(risk_percentage)
lifestyle_recommendations(risk_percentage)
health_challenges()
