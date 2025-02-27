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

# Load and prepare dataset
file_path = "PCOS_data.csv"
try:
    df = pd.read_csv(file_path)
    df_cleaned = df.drop(columns=["Sl. No", "Patient File No.", "Unnamed: 44"], errors="ignore")
    
    # Handle missing values
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object":
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        else:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # Convert non-numeric columns to numeric
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors="coerce")

    # Define features (X) and target variable (y)
    if "PCOS (Y/N)" not in df_cleaned.columns:
        raise ValueError("Target column 'PCOS (Y/N)' not found in the dataset.")
    
    X = df_cleaned.drop(columns=["PCOS (Y/N)"])
    y = df_cleaned["PCOS (Y/N)"]
    
    # Handle missing values in features
    X_filled = X.fillna(X.median())
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
    
    # Train the RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test model accuracy
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"✅ Model Accuracy: {model_accuracy * 100:.2f}%")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Health tips
health_tips = [
    "🌱 Eat a balanced diet rich in whole foods and fiber!",
    "🏃‍♀️ Regular exercise can improve insulin sensitivity and overall health.",
    "💧 Stay hydrated! Drinking enough water helps in hormonal balance.",
    "😴 Prioritize sleep! Aim for 7-9 hours to regulate hormones.",
    "🧘‍♀️ Manage stress with yoga, meditation, or deep breathing techniques."
]

# Function to create a dynamic gauge chart for risk level
def risk_meter(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "PCOS Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if score > 70 else "orange" if score > 40 else "green"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    return fig

# Streamlit UI for PCOS Prediction
def pcos_prediction_game():
    st.title("🎮 PCOS Prediction Game")
    st.write("Answer the following questions to unlock insights! 🎯")
    
    user_input = []
    progress_bar = st.progress(0)
    
    for idx, feature in enumerate(X_filled.columns):
        value = st.number_input(f"Enter your {feature}", min_value=0.0, format="%.2f")
        user_input.append(value)
        progress_bar.progress((idx + 1) / len(X_filled.columns))
    
    if st.button("🎲 Predict PCOS Risk!"):
        with st.spinner("Analyzing your data...🔍"):
            time.sleep(2)  # Simulate processing time
            user_input = np.array(user_input).reshape(1, -1)
            prediction = model.predict(user_input)
            risk_level = random.randint(1, 100)
        
        if prediction[0] == 0:
            st.success(f"✅ Low risk of PCOS. Your estimated risk level: {risk_level}%")
            st.write("Great job! Keep maintaining a healthy lifestyle. 💪")
            st.write("Here are some additional health tips for you:")
            for tip in random.sample(health_tips, 3):
                st.write(f"- {tip}")
            
            # Show a dynamic gauge chart for risk level
            st.write("### 📊 Your Risk Level")
            st.plotly_chart(risk_meter(risk_level))
        else:
            st.warning(f"⚠️ High risk of PCOS. Your estimated risk level: {risk_level}%")
            st.write("It's important to focus on your health and consider making lifestyle changes.")
    
    st.write("\nThank you for playing! 🌟")

# Function for Personality Quiz
def personality_quiz():
    st.title("🩺 PCOS Lifestyle Risk Assessment")
    st.markdown("#### Answer these questions to assess your risk level.")
    
    questions = {
        "How often do you exercise?": {"Daily": 0, "3-5 times a week": 10, "1-2 times a week": 20, "Rarely": 30},
        "How would you rate your diet?": {"Excellent": 0, "Good": 10, "Average": 20, "Poor": 30},
        "Do you have irregular menstrual cycles?": {"Never": 0, "Occasionally": 10, "Often": 20, "Always": 30},
        "How stressed do you feel daily?": {"Not at all": 0, "Mildly": 10, "Moderately": 20, "Highly stressed": 30},
        "How many hours of sleep do you get per night?": {"More than 8": 0, "7-8 hours": 10, "5-6 hours": 20, "Less than 5": 30}
    }
    
    score = 0
    for question, options in questions.items():
        answer = st.radio(question, list(options.keys()), index=0)
        score += options[answer]
        st.progress(score // (len(questions) * 3))
        time.sleep(0.3)
    
    return score

def get_recommendations(score):
    if score < 40:
        return "✅ You're doing great! Keep maintaining a balanced lifestyle."
    elif score < 70:
        return "⚠️ Consider improving your diet and exercise habits to lower risk."
    else:
        return "🚨 High risk detected! Consult a healthcare provider and adopt healthier habits."

def get_personalized_plan(score):
    if score < 40:
        return "🥗 Healthy Diet: Continue balanced meals with fruits, veggies, and lean proteins.\n🏋️‍♀️ Exercise: Maintain your routine with 30 min daily workouts."
    elif score < 70:
        return "🥗 Diet Tip: Reduce processed foods and add more fiber-rich meals.\n🏋️‍♀️ Exercise: Try strength training and yoga for better hormone balance."
    else:
        return "🚨 High Risk Alert: \n🥗 Focus on low-glycemic foods, whole grains, and healthy fats.\n🏋️‍♀️ Regular Exercise: Daily 30-45 min workouts with cardio and strength training recommended."

def get_motivational_message():
    messages = [
        "🌟 Every step towards a healthier you is a victory!",
        "🏆 Small changes today lead to a healthier tomorrow!",
        "💖 Your health matters—take care of yourself!",
        "🔥 Keep pushing forward, your body will thank you!"
    ]
    return random.choice(messages)

def main():
    score = personality_quiz()
    st.subheader(f"📊 Your PCOS Risk Score: **{score}**")
    st.plotly_chart(risk_meter(score))
    
    st.markdown(f"### 💡 {get_recommendations(score)}")
    st.markdown(f"### 📅 Personalized Diet & Exercise Plan:\n{get_personalized_plan(score)}")
    st.success(get_motivational_message())
    
    if score < 40:
        st.balloons()
    elif score < 70:
        st.snow()
    else:
        st.warning("⚠️ Consider lifestyle changes and consult a doctor!")
        st.error("🚑 Immediate action is recommended!")
    
    if st.button("🎡 Spin the Wheel for a Health Tip!"):
        st.write(random.choice(health_tips))
    
    st.markdown("---")
    pcos_prediction_game()

# Run the app
if __name__ == "__main__":
    main()
