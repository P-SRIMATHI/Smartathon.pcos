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
    progress_bar = st.progress(0)
    
    fatigue = st.radio("How often do you feel fatigued?", ["🔋 Rarely", "😴 Sometimes", "🛌 Often", "🛑 Always"])
    if fatigue == "🛑 Always": risk_score += 3
    elif fatigue == "🛌 Often": risk_score += 2
    elif fatigue == "😴 Sometimes": risk_score += 1
    progress_bar.progress(25)
    
    diet = st.radio("How would you describe your diet?", ["🍏 Healthy", "🍔 Fast food sometimes", "🍕 Mostly unhealthy", "🥤 Poor diet"])
    if diet == "🥤 Poor diet": risk_score += 3
    elif diet == "🍕 Mostly unhealthy": risk_score += 2
    elif diet == "🍔 Fast food sometimes": risk_score += 1
    progress_bar.progress(50)
    
    exercise = st.radio("How often do you exercise?", ["🏃‍♀️ Regularly", "🚶 Occasionally", "🛋️ Rarely", "❌ Never"])
    if exercise == "❌ Never": risk_score += 3
    elif exercise == "🛋️ Rarely": risk_score += 2
    elif exercise == "🚶 Occasionally": risk_score += 1
    progress_bar.progress(100)
    
    return risk_score

# ================== Risk Reveal with Animated Meter ==================
def risk_meter_animation(risk_score):
    st.subheader("🎡 Your Risk Level!")
    risk_percentage = min(risk_score * 10, 100)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i in range(0, risk_percentage + 1, 5):
        progress_text.text(f"Risk Level: {i}%")
        progress_bar.progress(i)
        time.sleep(0.05)
    
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

# ================== Lifestyle Recommendation System ==================
def lifestyle_recommendations(risk_percentage):
    st.subheader("💡 Personalized Lifestyle Recommendations")
    
    if risk_percentage < 30:
        st.success("✅ Maintain a balanced diet with whole foods and regular physical activity!")
    elif risk_percentage < 60:
        st.warning("⚠️ Try increasing your exercise and reducing processed foods!")
    else:
        st.error("🚨 Prioritize health checkups and stress management techniques like yoga!")
    
    health_quotes = [
        "🌟 Small changes lead to big results!", 
        "💪 Stay consistent, and you’ll see improvements!", 
        "🧘 Mind and body balance is key to health!"
    ]
    st.info(random.choice(health_quotes))

# ================== Main Execution ==================
X_data, y_data = load_and_preprocess_data()
pcos_model, model_acc = train_model(X_data, y_data)
st.sidebar.write(f"✅ **Model Accuracy:** {model_acc * 100:.2f}%")

risk_score = personality_quiz()
risk_percentage = risk_meter_animation(risk_score)
lifestyle_recommendations(risk_percentage)
import streamlit as st
import random
import time
import plotly.graph_objects as go

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

def spin_the_wheel():
    tips = [
        "Drink 8 glasses of water today! 💧", 
        "Do 10 minutes of stretching! 🧘", 
        "Try a sugar-free day! 🍎", 
        "Take a deep-breathing break! 😌"
    ]
    return random.choice(tips)

def community_forum():
    st.title("🌍 PCOS Community Forum")
    st.markdown("### Share experiences, tips, and support with others!")
    
    if "posts" not in st.session_state:
        st.session_state.posts = []
        st.session_state.upvotes = {}
    
    with st.form("new_post"):
        user_name = st.text_input("Your Name (or leave blank for anonymous):")
        user_message = st.text_area("Share your experience or ask a question:")
        submit_button = st.form_submit_button("Post")
        
        if submit_button and user_message:
            user_name = user_name if user_name else "Anonymous"
            post_id = len(st.session_state.posts)
            st.session_state.posts.append((post_id, user_name, user_message))
            st.session_state.upvotes[post_id] = 0
            st.success("✅ Post shared successfully!")
    
    st.markdown("---")
    
    if st.session_state.posts:
        for post_id, name, message in reversed(st.session_state.posts):
            st.markdown(f"**{name}:** {message}")
            if st.button(f"👍 {st.session_state.upvotes[post_id]}", key=f"upvote_{post_id}"):
                st.session_state.upvotes[post_id] += 1
            st.markdown("---")

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
        st.write(spin_the_wheel())
    
    st.markdown("---")
    community_forum()
    
if __name__ == "__main__":
    main()
