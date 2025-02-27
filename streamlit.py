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
    
    return risk_percentage

# ================== Achievements & Badges ==================
def show_badges(risk_percentage):
    st.subheader("🏆 Your Achievement")
    
    if risk_percentage < 30:
        st.success("🥇 **Hormone Hero!** You’re doing great! Keep up the healthy lifestyle! 💪")
    elif risk_percentage < 60:
        st.warning("⚠️ **Balance Seeker!** You're on the right path but can make improvements!")
    else:
        st.error("🚨 **PCOS Warrior!** You might be at high risk. Consider consulting a doctor.")

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

# ================== Main Execution ==================
X_data, y_data = load_and_preprocess_data()
pcos_model, model_acc = train_model(X_data, y_data)
st.sidebar.write(f"✅ **Model Accuracy:** {model_acc * 100:.2f}%")

risk_score = personality_quiz()
risk_percentage = risk_spin_wheel(risk_score)
show_badges(risk_percentage)
health_challenges()

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

 
